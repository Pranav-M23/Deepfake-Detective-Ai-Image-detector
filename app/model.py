import torch
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """
    Improved CNN with BatchNorm and Dropout to reduce overfitting.
    Input: 224x224 RGB
    """
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),            # 112x112

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),            # 56x56

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),            # 28x28

            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # 1x1
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class EfficientNetDetector(nn.Module):
    """
    Transfer-learning model based on EfficientNet-B0.
    Much stronger than SimpleCNN for deepfake/AI detection.
    Input: 224x224 RGB
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetDetector, self).__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # Replace the final classifier — single linear layer is far more
        # sample-efficient when training data is limited (< a few hundred images)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)


def load_model(weights_path: str, device: str, model_type: str = "simple"):
    """
    Load a trained deepfake-detection model.

    Args:
        weights_path: Path to the .pt weights file.
        device:       'cuda' or 'cpu'.
        model_type:   'simple'  → SimpleCNN (default, compatible with existing weights)
                      'efficientnet' → EfficientNetDetector
    """
    if model_type == "efficientnet":
        model = EfficientNetDetector(num_classes=2, pretrained=False).to(device)
    else:
        model = SimpleCNN(num_classes=2).to(device)

    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"[model] Loaded weights from '{weights_path}' ({model_type})")
    except FileNotFoundError:
        print(f"[model] WARNING: weights file not found at '{weights_path}'. Using random weights.")
    except Exception as e:
        print(f"[model] WARNING: could not load weights — {e}. Using random weights.")

    model.eval()
    return model