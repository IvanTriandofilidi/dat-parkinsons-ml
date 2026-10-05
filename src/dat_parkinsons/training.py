"""Classifier training with persisted folds, preprocessing, and training-only imputation."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .contracts import metrics, patient_folds, unique_patients
from .data import ImageDataset, sbr_fill_values, seed_everything
from .io import file_hash, read_json, write_json
from .models import ParkinsonClassifier
from .preprocessing import ViewConfig


@torch.inference_mode()
def predict_loader(model, loader, device, tta="none"):
    model.eval()
    predictions = []
    for image, sbr, _ in loader:
        image, sbr = image.to(device), sbr.to(device)
        probs = torch.sigmoid(model(image, sbr))
        if tta == "hflip":
            probs = (probs + torch.sigmoid(model(image.flip(-1), sbr))) / 2
        elif tta != "none":
            raise ValueError(f"Unsupported TTA: {tta}")
        predictions.extend(probs.cpu().numpy().tolist())
    return np.asarray(predictions)


def train_classifier(manifest_path, config_path, output, device="cpu", folds_path=None):
    config = read_json(config_path)
    manifest_path = Path(manifest_path)
    frame = pd.read_csv(manifest_path, dtype={"patient_id": str})
    unique_patients(frame)
    preparation = read_json(manifest_path.parent / "preprocessing.json")
    if preparation["normalization"] != "per_slice_50_99":
        raise ValueError("Unsupported scorer preprocessing")
    view = ViewConfig(**preparation["view"])
    if view != ViewConfig(**config["view"]):
        raise ValueError("Training view configuration differs from prepared images")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    seed = config.get("seed", 42)
    if folds_path:
        folds = pd.read_csv(folds_path, dtype={"patient_id": str})
        unique_patients(folds)
        if set(folds.patient_id) != set(frame.patient_id):
            raise ValueError("Fold manifest and training manifest have different patients")
        frame = frame.merge(
            folds[["patient_id", "label", "fold"]],
            on=["patient_id", "label"],
            validate="one_to_one",
        )
        if len(frame) != len(folds):
            raise ValueError("Fold labels do not match")
        values = frame.fold.to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values != values.astype(int)):
            raise ValueError("Invalid fold IDs")
        frame["fold"] = values.astype(int)
        if sorted(frame.fold.unique()) != list(range(config.get("n_splits", 5))):
            raise ValueError("Fold IDs must be contiguous and match n_splits")
    else:
        folds = patient_folds(frame, config.get("n_splits", 5), seed)
        frame = frame.merge(folds[["patient_id", "fold"]], on="patient_id", validate="one_to_one")
    frame[["patient_id", "label", "fold"]].sort_values("patient_id").to_csv(
        out / "folds.csv", index=False
    )
    write_json(out / "config.json", config)
    write_json(
        out / "provenance.json",
        {"manifest_sha256": file_hash(manifest_path), "preprocessing": preparation},
    )
    model_args = config["model"]
    params = config["training"]
    epochs = params.get("epochs", 25)
    batch_size = params.get("batch_size", 16)
    if epochs < 1 or batch_size < 2:
        raise ValueError("epochs >= 1 and batch_size >= 2 are required")
    device = torch.device(device)
    tta = params.get("tta", "none")
    records, history = [], []
    for fold in sorted(frame.fold.unique()):
        seed_everything(seed + int(fold))
        train = frame.loc[frame.fold != fold].reset_index(drop=True)
        valid = frame.loc[frame.fold == fold].reset_index(drop=True)
        if min(train.label.nunique(), valid.label.nunique()) != 2:
            raise ValueError("Each train and validation fold needs both classes")
        fill = sbr_fill_values(train) if model_args.get("use_sbr", False) else None
        train_ds = ImageDataset(
            train, manifest_path.parent, True, model_args.get("use_sbr", False), fill
        )
        valid_ds = ImageDataset(
            valid, manifest_path.parent, False, model_args.get("use_sbr", False), fill
        )
        generator = torch.Generator().manual_seed(seed + int(fold))
        # num_workers=0 keeps the default portable to Windows and notebooks.
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            drop_last=len(train) % batch_size == 1,
        )
        valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
        model = ParkinsonClassifier(**model_args, pretrained=config.get("pretrained", True)).to(
            device
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": params.get("lr_backbone", 1e-4)},
                {"params": model.head.parameters(), "lr": params.get("lr_head", 1e-3)},
            ],
            weight_decay=params.get("weight_decay", 1e-3),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()
        best, stale, best_epoch, best_state = float("inf"), 0, None, None
        for epoch in range(epochs):
            model.train()
            total, count = 0.0, 0
            for image, sbr, label in train_loader:
                image, sbr, label = image.to(device), sbr.to(device), label.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(image, sbr), label)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(label)
                count += len(label)
            scheduler.step()
            val = metrics(valid.label, predict_loader(model, valid_loader, device, "none"))
            history.append(
                {
                    "fold": int(fold),
                    "epoch": epoch + 1,
                    "train_loss": total / count,
                    "val_log_loss": val["log_loss"],
                    "val_roc_auc": val["roc_auc"],
                }
            )
            print(
                f"fold={fold} epoch={epoch + 1} train_loss={total / count:.4f} val_log_loss={val['log_loss']:.4f}",
                flush=True,
            )
            if val["log_loss"] < best:
                best, stale, best_epoch = val["log_loss"], 0, epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= params.get("patience", 8):
                break
        model.load_state_dict(best_state)
        checkpoint = {
            "format_version": 1,
            "state_dict": best_state,
            "model": model_args,
            "sbr_fill": fill,
            "view": view.to_dict(),
            "tta": tta,
            "scorer_sha256": preparation["scorer_sha256"],
            "fold": int(fold),
            "seed": seed,
            "best_epoch": best_epoch,
        }
        torch.save(checkpoint, out / f"fold_{fold}.pt")
        preds = predict_loader(model, valid_loader, device, tta)
        records.extend(
            {
                "patient_id": str(pid),
                "label": int(label),
                "fold": int(fold),
                "pred_probability": float(pred),
            }
            for pid, label, pred in zip(valid.patient_id, valid.label, preds)
        )
    oof = pd.DataFrame(records).sort_values("patient_id")
    unique_patients(oof)
    oof.to_csv(out / "oof.csv", index=False)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    report = metrics(oof.label, oof.pred_probability)
    report["scope"] = (
        "OOF estimates; validation folds were also used for epoch selection. Not an independent test."
    )
    write_json(out / "metrics.json", report)
    return report
