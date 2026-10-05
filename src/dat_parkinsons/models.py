"""Model definitions preserve the selected notebook parameter names."""

import torch
from torch import nn


class SliceScorerResNet18(nn.Module):
    def __init__(self, pretrained=False, freeze_until_layer=3, head_dropout=0.4):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        old = backbone.conv1
        conv = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            conv.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = conv
        layers = [
            backbone.conv1,
            backbone.bn1,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        ]
        for layer in layers[:freeze_until_layer]:
            for parameter in layer.parameters():
                parameter.requires_grad = False
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(512, 64), nn.ReLU(), nn.Dropout(head_dropout), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


class ParkinsonClassifier(nn.Module):
    def __init__(self, model_name="seresnet34", pretrained=False, drop_rate=0.3, use_sbr=False):
        super().__init__()
        import timm

        self.use_sbr = use_sbr
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        count = self.backbone.num_features + (2 if use_sbr else 0)
        self.head = nn.Sequential(
            nn.Linear(count, 128), nn.ReLU(), nn.Dropout(drop_rate), nn.Linear(128, 1)
        )

    def forward(self, image, sbr=None):
        features = self.backbone(image)
        if self.use_sbr:
            if sbr is None or sbr.shape != (len(image), 2):
                raise ValueError("SBR-enabled classifiers require two features per image")
            features = torch.cat([features, sbr], dim=1)
        return self.head(features).squeeze(-1)
