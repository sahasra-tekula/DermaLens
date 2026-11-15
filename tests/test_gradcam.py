import torch
from torch import nn
from torchvision.models import resnet18

from src.gradcam import GradCAM


def test_gradcam_heatmap_matches_input_size_and_has_finite_values() -> None:
    torch.set_num_threads(1)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 7)
    images = torch.randn(1, 3, 128, 160)

    with GradCAM(model) as gradcam:
        with torch.inference_mode():
            heatmaps, logits, targets = gradcam(images)

    assert heatmaps.shape == (1, 128, 160)
    assert logits.shape == (1, 7)
    assert targets.shape == (1,)
    assert torch.isfinite(heatmaps).all()
    assert heatmaps.min().item() >= 0.0
    assert heatmaps.max().item() <= 1.0
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not model.layer4[-1]._forward_hooks
