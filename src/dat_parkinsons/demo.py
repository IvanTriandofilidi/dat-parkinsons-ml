"""No patient data, external weights, or network required."""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .data import seed_everything
from .io import file_hash, write_json
from .models import ParkinsonClassifier, SliceScorerResNet18
from .pipeline import Predictor
from .preprocessing import ViewConfig


def synthetic_volume(seed=42):
    rng = np.random.default_rng(seed)
    y, x, z = np.mgrid[-1:1:48j, -1:1:48j, -1:1:16j]
    base = np.exp(-3 * (x * x + y * y + z * z))
    left = np.exp(-45 * ((x - 0.2) ** 2 + y * y + (z - 0.1) ** 2))
    right = np.exp(-45 * ((x + 0.2) ** 2 + y * y + (z - 0.1) ** 2))
    return np.maximum(0, base + 2 * left + 1.5 * right + rng.normal(0, 0.01, base.shape)).astype(
        np.float32
    )


def run_demo(output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    volumes = out / "synthetic_niftis"
    volumes.mkdir()
    seed_everything(42)
    torch.set_num_threads(2)
    nib.save(nib.Nifti1Image(synthetic_volume(), np.eye(4)), volumes / "synthetic_000.nii.gz")
    scorer_path = out / "selector.pt"
    torch.save(
        {
            "state_dict": SliceScorerResNet18(pretrained=False).state_dict(),
            "normalization": "per_slice_50_99",
        },
        scorer_path,
    )
    model_args = {"model_name": "resnet18", "drop_rate": 0.3, "use_sbr": True}
    classifier = ParkinsonClassifier(**model_args, pretrained=False)
    torch.save(
        {
            "format_version": 1,
            "state_dict": classifier.state_dict(),
            "model": model_args,
            "view": ViewConfig().to_dict(),
            "sbr_fill": [0.0, 0.0],
            "tta": "hflip",
            "scorer_sha256": file_hash(scorer_path),
        },
        out / "classifier.pt",
    )
    write_json(
        out / "bundle.json",
        {
            "scorer": "selector.pt",
            "model_order": ["synthetic"],
            "weights": [1.0],
            "branches": [{"name": "synthetic", "checkpoints": ["classifier.pt"]}],
        },
    )
    predictor = Predictor(out / "bundle.json")
    result = predictor.predict_directory(volumes, out / "submission.csv")
    repeat = predictor.predict_directory(volumes, out / "submission_repeat.csv")
    if not np.array_equal(result.is_pathologic, repeat.is_pathologic):
        raise AssertionError("Inference was not deterministic")
    report = {
        "status": "passed",
        "examinations": len(result),
        "deterministic": True,
        "data": "synthetic",
        "weights": "random initialization",
        "meaning": "Software integration check only; prediction has no diagnostic meaning",
    }
    write_json(out / "report.json", report)
    return report
