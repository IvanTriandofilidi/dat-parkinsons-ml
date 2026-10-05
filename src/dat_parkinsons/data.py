"""Image transforms and dataset contracts, shared by training and prediction."""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .contracts import binary_labels, require_columns, unique_patients

SBR_COLUMNS = ["sbr_mean", "sbr_asymmetry"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def image_transform(training=False):
    items = [transforms.ToPILImage(), transforms.Resize((224, 224))]
    if training:
        items += [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomAffine(10, scale=(0.9, 1.1)),
        ]
    return transforms.Compose(
        items
        + [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def sbr_fill_values(train_frame):
    require_columns(train_frame, SBR_COLUMNS)
    values = train_frame[SBR_COLUMNS].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Infinite SBR feature")
    fill = []
    for col in values.T:
        valid = col[np.isfinite(col)]
        if not len(valid):
            raise ValueError("SBR feature is missing for every training patient")
        fill.append(float(np.median(valid)))
    return fill


def impute_sbr(values, fill):
    x, f = np.asarray(values, dtype=np.float32), np.asarray(fill, dtype=np.float32)
    if x.shape[-1] != 2 or f.shape != (2,) or not np.isfinite(f).all() or np.isinf(x).any():
        raise ValueError("Invalid SBR features or stored training medians")
    return np.where(np.isnan(x), f, x).astype(np.float32)


class ImageDataset(Dataset):
    def __init__(self, frame, root, training=False, use_sbr=False, fill=None):
        unique_patients(frame)
        require_columns(frame, ["image_path", "label"])
        binary_labels(frame.label)
        self.frame = frame.reset_index(drop=True)
        self.root = Path(root)
        self.transform = image_transform(training)
        self.use_sbr = use_sbr
        if use_sbr and fill is None:
            raise ValueError("SBR training medians must be supplied explicitly")
        self.fill = fill

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        path = self.root / row.image_path
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unreadable image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        features = (
            impute_sbr(row[SBR_COLUMNS].to_numpy(dtype=float), self.fill)
            if self.use_sbr
            else np.zeros(0, dtype=np.float32)
        )
        return self.transform(image), torch.from_numpy(features), torch.tensor(float(row.label))
