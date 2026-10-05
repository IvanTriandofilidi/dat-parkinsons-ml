import numpy as np
import pytest
import torch

from dat_parkinsons.data import impute_sbr
from dat_parkinsons.pipeline import select_slice
from dat_parkinsons.preprocessing import (
    ViewConfig,
    crop_fixed,
    discover_volumes,
    make_view,
    raw_scorer_crop,
    sbr_features,
    scorer_input,
)


def test_crop_padding_preserves_center_and_odd_size():
    image = np.arange(20).reshape(4, 5)
    crop = crop_fixed(image, (0, 0), 5)
    assert crop.shape == (5, 5)
    assert crop[2, 2] == image[0, 0]
    assert crop[:2].sum() == 0
    assert crop_fixed(image, (100, 100), 5).sum() == 0


def test_scorer_constant_and_nonfinite_input():
    tensor = scorer_input(np.ones((7, 9)))
    assert tensor.shape == (1, 128, 128)
    assert np.isfinite(tensor).all() and tensor.sum() == 0
    with pytest.raises(ValueError):
        scorer_input(np.full((5, 5), np.nan))


@pytest.mark.parametrize("mode", ["rgb", "25d", "grayscale"])
@pytest.mark.parametrize("z", [0, 2])
def test_views_at_depth_boundaries(mode, z):
    volume = np.ones((20, 24, 3), dtype=np.float32)
    image, features = make_view(volume, z, ViewConfig(mode=mode, offset_mm=7))
    assert image.shape == (200, 200, 3)
    assert image.dtype == np.uint8
    assert features.shape == (2,)


def test_raw_scorer_crop_training_inference_contract():
    volume = np.arange(10 * 12 * 4, dtype=np.float32).reshape(10, 12, 4)
    expected = np.rot90(volume[2:7, 3:9, 1])
    np.testing.assert_array_equal(raw_scorer_crop(volume, 1), expected)


def test_selector_batching_and_tie_breaking():
    class Constant(torch.nn.Module):
        def forward(self, x):
            return torch.ones(len(x), device=x.device) * 0.5

    volume = np.ones((8, 8, 7), dtype=np.float32)
    for batch in [1, 3, 20]:
        z, score = select_slice(volume, Constant(), batch_size=batch)
        assert z == 0 and score == 0.5


def test_imputation_uses_stored_training_values_only():
    values = np.array([[np.nan, 7], [10000, np.nan]])
    filled = impute_sbr(values, [2, 3])
    np.testing.assert_array_equal(filled, [[2, 7], [10000, 3]])
    np.testing.assert_array_equal(impute_sbr(values[:1], [2, 3]), filled[:1])
    with pytest.raises(ValueError):
        impute_sbr(values, [np.nan, 3])


def test_sbr_constant_volume_has_zero_ratios():
    np.testing.assert_allclose(sbr_features(np.ones((200, 200))), [0, 0])
    assert np.isnan(sbr_features(np.zeros((200, 200)))).all()


def test_duplicate_nifti_ids_are_rejected(tmp_path):
    (tmp_path / "one.nii").touch()
    (tmp_path / "one.nii.gz").touch()
    with pytest.raises(ValueError, match="Duplicate"):
        discover_volumes(tmp_path)
