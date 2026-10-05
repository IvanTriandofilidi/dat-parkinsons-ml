"""Tiny real optimization runs on synthetic data; no pretrained downloads."""

import json

import cv2
import numpy as np
import pandas as pd
import torch

from dat_parkinsons.io import file_hash, write_json
from dat_parkinsons.models import SliceScorerResNet18
from dat_parkinsons.pipeline import Predictor
from dat_parkinsons.preprocessing import ViewConfig
from dat_parkinsons.selector_training import train_selector
from dat_parkinsons.training import train_classifier


def test_classifier_training_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    scorer = tmp_path / "selector.pt"
    torch.save(
        {
            "state_dict": SliceScorerResNet18(pretrained=False).state_dict(),
            "normalization": "per_slice_50_99",
        },
        scorer,
    )
    rows = []
    for index in range(8):
        filename = f"synthetic_{index}.png"
        cv2.imwrite(str(prepared / filename), rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
        rows.append(
            {
                "patient_id": f"synthetic_{index}",
                "image_path": filename,
                "label": index % 2,
                "sbr_mean": float(index),
                "sbr_asymmetry": 0.1,
            }
        )
    pd.DataFrame(rows).to_csv(prepared / "manifest.csv", index=False)
    view = ViewConfig().to_dict()
    write_json(
        prepared / "preprocessing.json",
        {"view": view, "normalization": "per_slice_50_99", "scorer_sha256": file_hash(scorer)},
    )
    config = {
        "seed": 42,
        "n_splits": 2,
        "pretrained": False,
        "view": view,
        "model": {"model_name": "resnet18", "drop_rate": 0.1, "use_sbr": True},
        "training": {"epochs": 1, "batch_size": 2, "tta": "hflip"},
    }
    write_json(tmp_path / "config.json", config)
    result = train_classifier(
        prepared / "manifest.csv", tmp_path / "config.json", tmp_path / "trained"
    )
    assert result["n"] == 8
    oof = pd.read_csv(tmp_path / "trained/oof.csv")
    assert oof.patient_id.is_unique and len(oof) == 8
    artifact = torch.load(tmp_path / "trained/fold_0.pt", weights_only=True)
    assert artifact["sbr_fill"] is not None and artifact["scorer_sha256"] == file_hash(scorer)
    write_json(
        tmp_path / "bundle.json",
        {
            "scorer": "selector.pt",
            "weights": [1],
            "model_order": ["test"],
            "branches": [
                {"name": "test", "checkpoints": ["trained/fold_0.pt", "trained/fold_1.pt"]}
            ],
        },
    )
    predictor = Predictor(tmp_path / "bundle.json")
    p = predictor.predict_volume(rng.random((16, 16, 3), dtype=np.float32))
    assert 0 <= p <= 1


def test_selector_grouped_cv_and_final_refit(tmp_path):
    torch.set_num_threads(2)
    rng = np.random.default_rng(13)
    rows = []
    for patient in range(4):
        for z in range(2):
            filename = f"synthetic_{patient}_{z}.npy"
            np.save(tmp_path / filename, rng.random((16, 16), dtype=np.float32))
            rows.append(
                {
                    "patient_id": f"synthetic_{patient}",
                    "z": z,
                    "score": 0.25 + 0.5 * z,
                    "npy_filename": filename,
                }
            )
    labels = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(labels, index=False)
    train_selector(labels, tmp_path / "cv", epochs=1, n_splits=2, pretrained=False)
    folds = pd.read_csv(tmp_path / "cv/folds.csv")
    assert folds.patient_id.is_unique and len(folds) == 4
    train_selector(labels, tmp_path / "final", epochs=1, n_splits=2, final=True, pretrained=False)
    checkpoint = torch.load(tmp_path / "final/selector.pt", weights_only=True)
    assert checkpoint["normalization"] == "per_slice_50_99"
    assert json.loads((tmp_path / "final/run.json").read_text())["final"] is True
