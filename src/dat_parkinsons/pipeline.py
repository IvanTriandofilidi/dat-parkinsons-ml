"""Select a slice once, create configured views, and predict one examination at a time."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from .contracts import binary_labels, submission, unique_patients
from .data import image_transform, impute_sbr
from .ensemble import blend
from .io import file_hash, load_scorer, read_json, write_json
from .models import ParkinsonClassifier
from .preprocessing import (
    ViewConfig,
    discover_volumes,
    load_volume,
    make_view,
    raw_scorer_crop,
    scorer_input,
    validate_volume,
)


@torch.inference_mode()
def select_slice(volume, model, device="cpu", batch_size=32, step=1):
    volume = validate_volume(volume)
    if batch_size < 1 or step < 1:
        raise ValueError("Batch size and slice step must be positive")
    indices = list(range(0, volume.shape[2], step))
    scores = []
    model.eval()
    for start in range(0, len(indices), batch_size):
        batch = np.stack(
            [scorer_input(raw_scorer_crop(volume, z)) for z in indices[start : start + batch_size]]
        )
        scores.extend(model(torch.from_numpy(batch).to(device)).cpu().numpy().tolist())
    scores = np.asarray(scores)
    if scores.shape != (len(indices),) or not np.isfinite(scores).all():
        raise ValueError("Scorer returned invalid scores")
    winner = int(np.argmax(scores))
    return indices[winner], float(scores[winner])


def prepare_images(nifti_dir, labels_path, scorer_path, output, config, device="cpu"):
    volumes = discover_volumes(nifti_dir)
    labels = pd.read_csv(labels_path, dtype={"uid": str}).rename(
        columns={"uid": "patient_id", "is_pathologic": "label"}
    )
    unique_patients(labels)
    binary_labels(labels.label)
    if set(labels.patient_id) != set(volumes):
        raise ValueError("Volume IDs and label IDs must match exactly")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    model = load_scorer(scorer_path, device)
    rows = []
    for row in labels.sort_values("patient_id").itertuples():
        volume = load_volume(volumes[row.patient_id])
        best_z, score = select_slice(volume, model, device)
        image, features = make_view(volume, best_z, config)
        filename = f"{row.patient_id}.png"
        if not cv2.imwrite(str(out / filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise IOError("Failed to save prepared image")
        rows.append(
            {
                "patient_id": row.patient_id,
                "label": row.label,
                "image_path": filename,
                "best_z": best_z,
                "slice_score": score,
                "sbr_mean": features[0],
                "sbr_asymmetry": features[1],
            }
        )
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    write_json(
        out / "preprocessing.json",
        {
            "view": config.to_dict(),
            "scorer_sha256": file_hash(scorer_path),
            "normalization": "per_slice_50_99",
        },
    )


class Predictor:
    def __init__(self, bundle_path, device="cpu"):
        bundle_path = Path(bundle_path)
        self.bundle = read_json(bundle_path)
        self.device = torch.device(device)
        self.scorer = load_scorer(bundle_path.parent / self.bundle["scorer"], self.device)
        scorer_hash = file_hash(bundle_path.parent / self.bundle["scorer"])
        self.branches = []
        names = []
        for branch in self.bundle["branches"]:
            names.append(branch["name"])
            if not branch["checkpoints"]:
                raise ValueError("Each branch needs at least one checkpoint")
            members = []
            first_view = None
            for checkpoint_path in branch["checkpoints"]:
                checkpoint = torch.load(
                    bundle_path.parent / checkpoint_path,
                    map_location=self.device,
                    weights_only=True,
                )
                if checkpoint.get("format_version") != 1:
                    raise ValueError(
                        "Use a checkpoint produced by this package; legacy weights require explicit migration"
                    )
                if checkpoint["scorer_sha256"] != scorer_hash:
                    raise ValueError(
                        "Scorer differs from the one used to prepare classifier training images"
                    )
                view = ViewConfig(**checkpoint["view"])
                if first_view is not None and view != first_view:
                    raise ValueError("All members of a branch must use the same view")
                first_view = view
                model = ParkinsonClassifier(**checkpoint["model"], pretrained=False).to(self.device)
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                members.append(
                    (model.eval(), checkpoint["sbr_fill"], checkpoint.get("tta", "none"))
                )
            self.branches.append((first_view, members))
        if not names or len(names) != len(set(names)):
            raise ValueError("Branch names must be nonempty and unique")
        if names != self.bundle["model_order"]:
            raise ValueError("Branch order must match the blend artifact model_order")
        blend(np.full((1, len(names)), 0.5), self.bundle["weights"])
        self.transform = image_transform(False)

    @torch.inference_mode()
    def predict_volume(self, volume):
        volume = validate_volume(volume)
        z, _ = select_slice(volume, self.scorer, self.device)
        predictions = []
        for view, members in self.branches:
            image, features = make_view(volume, z, view)
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            fold_predictions = []
            for model, fill, tta in members:
                sbr = (
                    torch.from_numpy(impute_sbr(features, fill)).unsqueeze(0).to(self.device)
                    if model.use_sbr
                    else None
                )
                p = torch.sigmoid(model(tensor, sbr))
                if tta == "hflip":
                    p = (p + torch.sigmoid(model(tensor.flip(-1), sbr))) / 2
                elif tta != "none":
                    raise ValueError(f"Unsupported TTA policy: {tta}")
                fold_predictions.append(float(p.item()))
            predictions.append(float(np.mean(fold_predictions)))
        return float(blend([predictions], self.bundle["weights"])[0])

    def predict_directory(self, directory, output):
        volumes = discover_volumes(directory)
        probabilities = [self.predict_volume(load_volume(path)) for path in volumes.values()]
        result = submission(list(volumes), probabilities)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        return result
