"""Small explicit JSON and checkpoint helpers. No model downloads during inference."""

import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_hashes(ids):
    return sorted(hashlib.sha256(str(uid).encode()).hexdigest() for uid in ids)


def load_scorer(path, device):
    import torch
    from .models import SliceScorerResNet18

    model = SliceScorerResNet18(pretrained=False).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if "state_dict" in checkpoint:
        if checkpoint.get("normalization") != "per_slice_50_99":
            raise ValueError("Scorer normalization does not match the implemented input contract")
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=True)
    return model.eval()
