import pytest
import torch

from dat_parkinsons.models import ParkinsonClassifier, SliceScorerResNet18


@pytest.mark.parametrize(
    "name", ["seresnext26d_32x4d", "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"]
)
def test_example_backbones_accept_classifier_contract(name):
    torch.set_num_threads(2)
    model = ParkinsonClassifier(model_name=name, use_sbr=True, pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 224, 224), torch.zeros(1, 2))
    assert output.shape == (1,) and torch.isfinite(output).all()


def test_selector_requires_single_channel_input():
    torch.set_num_threads(2)
    model = SliceScorerResNet18(pretrained=False).eval()
    with torch.inference_mode():
        assert model(torch.zeros(1, 1, 128, 128)).shape == (1,)
        with pytest.raises(RuntimeError):
            model(torch.zeros(1, 3, 128, 128))
