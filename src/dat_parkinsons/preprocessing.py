"""Shared image preparation for training exports and online inference.

Native array orientation is retained to match the source notebooks. Resampling
does not establish anatomical orientation; validate input affines separately.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import center_of_mass


@dataclass(frozen=True)
class ViewConfig:
    mode: str = "25d"
    offset_mm: int = 7
    crop_size: int = 200
    resample_mm: float = 1.0
    scorer_window: tuple = (50, 99)

    def __post_init__(self):
        if self.mode not in {"25d", "rgb", "grayscale"}:
            raise ValueError(f"Unknown view: {self.mode}")
        if self.resample_mm != 1.0:
            raise ValueError("This checkpoint contract requires 1 mm resampling")
        if self.crop_size < 2 or self.offset_mm < 0:
            raise ValueError("Invalid crop size or offset")
        if tuple(self.scorer_window) != (50, 99):
            raise ValueError("Only the per-slice 50-99 scorer contract is supported")
        object.__setattr__(self, "scorer_window", tuple(self.scorer_window))

    def to_dict(self):
        return asdict(self)


def validate_volume(volume):
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3 or min(volume.shape) < 2:
        raise ValueError("Expected a nonempty 3D volume with each axis >= 2")
    if not np.isfinite(volume).all():
        raise ValueError("Volume contains nonfinite voxels")
    return volume


def load_volume(path):
    import nibabel as nib
    import torchio as tio

    header = nib.load(str(path))
    if len(header.shape) != 3:
        raise ValueError("Only single-channel 3D NIfTI volumes are supported")
    image = tio.Resample(1.0)(tio.ScalarImage(str(path)))
    return validate_volume(image.data[0].numpy())


def discover_volumes(directory):
    files = sorted(p for p in Path(directory).rglob("*") if p.name.endswith((".nii", ".nii.gz")))
    if not files:
        raise FileNotFoundError("No NIfTI volumes found")
    by_id = {}
    for path in files:
        uid = path.name.split(".nii")[0]
        if uid in by_id:
            raise ValueError(f"Duplicate volume ID {uid}: resolve .nii/.nii.gz copies")
        by_id[uid] = path
    return by_id


def scorer_input(raw_crop, window=(50, 99)):
    raw = np.asarray(raw_crop, dtype=np.float32)
    if raw.ndim != 2 or raw.size == 0 or not np.isfinite(raw).all():
        raise ValueError("Expected a finite, nonempty 2D scorer crop")
    low, high = np.percentile(raw, window)
    clipped = np.clip(raw, low, high)
    span = clipped.max() - clipped.min()
    image = (clipped - clipped.min()) / span if span > 0 else np.zeros_like(clipped)
    return cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)[None].astype(np.float32)


def raw_scorer_crop(volume, z):
    h, w, _ = volume.shape
    return np.rot90(volume[int(h * 0.25) : int(h * 0.75), int(w * 0.25) : int(w * 0.75), z]).copy()


def brain_center(volume):
    mask = volume > np.percentile(volume, 70)
    if mask.any():
        y, x, _ = center_of_mass(mask)
        return int(y), int(x)
    return volume.shape[0] // 2, volume.shape[1] // 2


def crop_fixed(image, center, size=200):
    if size < 2:
        raise ValueError("Crop size must be >= 2")
    y, x = center
    y0, x0 = y - size // 2, x - size // 2
    y1, x1 = y0 + size, x0 + size
    out = np.zeros((size, size), dtype=image.dtype)
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(image.shape[0], y1), min(image.shape[1], x1)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return out


def sbr_features(raw):
    """Notebook geometric uptake proxies, not anatomically segmented clinical SBR."""
    h, w = raw.shape
    cy, cx, size = h // 2, w // 2, min(h, w)
    yy, xx = np.ogrid[:h, :w]
    distance = (xx - cx) ** 2 + (yy - cy) ** 2
    background = raw[(distance >= (size * 0.24) ** 2) & (distance <= (size * 0.34) ** 2)]
    background = background[background > 0]
    if len(background) < 10:
        return np.array([np.nan, np.nan], dtype=np.float32)
    median = float(np.median(background))
    specific = distance <= (size * 0.15) ** 2
    ratios = []
    for mask in [specific & (xx < cx), specific & (xx >= cx)]:
        values = raw[mask]
        values = values[values > 0]
        k = max(1, int(len(values) * 0.05))
        top = float(np.mean(np.partition(values, -k)[-k:])) if len(values) else 0.0
        ratios.append((top - median) / median)
    mean = float(np.mean(ratios))
    return np.array([mean, abs(ratios[0] - ratios[1]) / max(abs(mean), 1e-6)], dtype=np.float32)


def normalize_stack(stack):
    positive = stack[stack > 0]
    if not len(positive):
        return np.zeros_like(stack, dtype=np.float32)
    low, high = np.percentile(positive, [0.5, 99.8])
    return (
        ((np.clip(stack, low, high) - low) / (high - low)).astype(np.float32)
        if high > low
        else np.zeros_like(stack, dtype=np.float32)
    )


def multi_contrast(raw):
    positive = raw[raw > 0]
    if not len(positive):
        return np.zeros((*raw.shape, 3), dtype=np.float32)
    low, high = np.percentile(positive, [1, 99.5])
    base = (np.clip(raw, low, high) - low) / (high - low + 1e-8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply((base * 255).astype(np.uint8)).astype(np.float32) / 255
    ratio = np.clip(raw / (np.median(positive) + 1e-8), 0, 3) / 3
    return np.stack([base, contrast, ratio], axis=-1).astype(np.float32)


def make_view(volume, best_z, config):
    if not 0 <= best_z < volume.shape[2]:
        raise ValueError("Selected slice lies outside the volume")
    center = brain_center(volume)
    raw = crop_fixed(volume[:, :, best_z], center, config.crop_size)
    features = sbr_features(raw)
    if config.mode == "rgb":
        view = multi_contrast(np.rot90(raw))
    elif config.mode == "grayscale":
        gray = normalize_stack(np.rot90(raw))
        view = np.repeat(gray[..., None], 3, axis=-1)
    else:
        channels = []
        for offset in [-config.offset_mm, 0, config.offset_mm]:
            z = int(np.clip(best_z + offset, 0, volume.shape[2] - 1))
            slab = volume[:, :, max(0, z - 1) : min(volume.shape[2], z + 2)].mean(axis=2)
            channels.append(crop_fixed(slab, center, config.crop_size))
        medians = np.array([np.median(c[c > 0]) if (c > 0).any() else 0 for c in channels])
        target = np.median(medians[medians > 0]) if (medians > 0).any() else 0
        channels = [
            c * np.clip(target / m, 0.5, 2) if m > 0 else c for c, m in zip(channels, medians)
        ]
        view = np.rot90(normalize_stack(np.stack(channels, axis=-1)), axes=(0, 1))
    # Match the notebook's PNG quantization in both training and inference.
    return (np.clip(view, 0, 1) * 255).astype(np.uint8).copy(), features
