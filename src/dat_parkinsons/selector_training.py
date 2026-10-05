"""Patient-grouped slice-score regression from locally held annotations."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .contracts import require_columns
from .data import seed_everything
from .io import write_json
from .models import SliceScorerResNet18
from .preprocessing import scorer_input


class SliceDataset(Dataset):
    def __init__(self, frame, root, augment=False):
        self.frame, self.root, self.augment = frame.reset_index(drop=True), Path(root), augment

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = scorer_input(np.load(self.root / row.npy_filename, allow_pickle=False))
        if self.augment:
            if np.random.rand() < 0.5:
                image = image[:, :, ::-1].copy()
            if np.random.rand() < 0.3:
                image = np.clip(
                    image + np.random.normal(0, 0.02, image.shape).astype(np.float32), 0, 1
                )
            for kind in ["rotate", "scale"]:
                if np.random.rand() < 0.3:
                    angle = np.random.uniform(-10, 10) if kind == "rotate" else 0
                    scale = np.random.uniform(0.9, 1.1) if kind == "scale" else 1
                    matrix = cv2.getRotationMatrix2D((64, 64), angle, scale)
                    image = cv2.warpAffine(
                        image[0], matrix, (128, 128), borderMode=cv2.BORDER_REFLECT
                    )[None]
        return torch.from_numpy(image.copy()), torch.tensor(float(row.score), dtype=torch.float32)


def train_selector(
    labels_path, output, epochs=40, n_splits=5, seed=42, device="cpu", final=False, pretrained=True
):
    if epochs < 1 or n_splits < 2:
        raise ValueError("epochs >= 1 and n_splits >= 2 are required")
    path = Path(labels_path)
    frame = pd.read_csv(path, dtype={"patient_id": str})
    require_columns(frame, ["patient_id", "z", "score", "npy_filename"])
    if frame.patient_id.isna().any() or frame.patient_id.str.strip().eq("").any():
        raise ValueError("Missing annotation patient IDs")
    if frame.duplicated(["patient_id", "z"]).any():
        raise ValueError("Duplicate slice annotations")
    scores = frame.score.to_numpy(dtype=float)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Slice scores must lie in [0, 1]")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(device)
    if final:
        splits = [(np.arange(len(frame)), np.array([], dtype=int))]
    else:
        splits = list(GroupKFold(n_splits=n_splits).split(frame, groups=frame.patient_id))
    history, fold_records = [], []
    for fold, (train_idx, valid_idx) in enumerate(splits):
        seed_everything(seed + fold)
        train, valid = frame.iloc[train_idx], frame.iloc[valid_idx]
        if len(train) < 2:
            raise ValueError("Need at least two annotated training slices")
        bins = np.clip((train.score.to_numpy() * 10).astype(int), 0, 9)
        weights = 1 / np.maximum(np.bincount(bins, minlength=10), 1)[bins]
        sampler = WeightedRandomSampler(weights, num_samples=len(train), replacement=True)
        train_loader = DataLoader(
            SliceDataset(train, path.parent, True),
            batch_size=16,
            sampler=sampler,
            drop_last=len(train) % 16 == 1,
        )
        valid_loader = DataLoader(SliceDataset(valid, path.parent), batch_size=16)
        model = SliceScorerResNet18(pretrained=pretrained).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best, stale, state = float("inf"), 0, None
        for epoch in range(epochs):
            model.train()
            for images, targets in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(images.to(device)), targets.to(device))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite selector loss")
                loss.backward()
                optimizer.step()
            if final:
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                continue
            model.eval()
            total, count = 0.0, 0
            with torch.inference_mode():
                for images, targets in valid_loader:
                    error = (model(images.to(device)).cpu() - targets).square()
                    total += float(error.sum())
                    count += len(targets)
            mse = total / count
            baseline = float(np.mean((valid.score - train.score.mean()) ** 2))
            history.append({"fold": fold, "epoch": epoch + 1, "mse": mse, "baseline_mse": baseline})
            print(
                f"selector fold={fold} epoch={epoch + 1} mse={mse:.4f} baseline={baseline:.4f}",
                flush=True,
            )
            if mse < best:
                best, stale = mse, 0
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= 8:
                break
        name = "selector.pt" if final else f"fold_{fold}.pt"
        torch.save(
            {"state_dict": state, "normalization": "per_slice_50_99", "architecture": "resnet18"},
            out / name,
        )
        fold_records.extend({"patient_id": pid, "fold": fold} for pid in valid.patient_id.unique())
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    pd.DataFrame(fold_records, columns=["patient_id", "fold"]).to_csv(
        out / "folds.csv", index=False
    )
    write_json(
        out / "run.json",
        {
            "epochs": epochs,
            "n_splits": n_splits,
            "seed": seed,
            "final": final,
            "pretrained": pretrained,
            "annotation_contract": "rotated raw central 50% crops, resampled at 1 mm",
        },
    )
