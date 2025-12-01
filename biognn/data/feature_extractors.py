"""
Feature extraction modules for different biometric modalities
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class FaceFeatureExtractor(nn.Module):
    """Feature extractor for face images using pretrained CNN"""

    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        feature_dim: int = 512,
        freeze_backbone: bool = False
    ):
        """
        Args:
            backbone: Backbone architecture ('resnet50', 'resnet18', 'mobilenet_v2')
            pretrained: Use pretrained weights
            feature_dim: Output feature dimension
            freeze_backbone: Freeze backbone weights
        """
        super().__init__()

        self.feature_dim = feature_dim

        # Load backbone
        if backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            in_features = base_model.fc.in_features
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        elif backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            in_features = base_model.fc.in_features
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        elif backbone == 'mobilenet_v2':
            base_model = models.mobilenet_v2(pretrained=pretrained)
            in_features = base_model.classifier[1].in_features
            self.backbone = base_model.features
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, feature_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Face images [batch, 3, H, W]

        Returns:
            Feature vectors [batch, feature_dim]
        """
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        features = self.projection(features)
        return features


class FingerprintFeatureExtractor(nn.Module):
    """Feature extractor for fingerprint images"""

    def __init__(
        self,
        feature_dim: int = 512,
        input_channels: int = 1,
        backbone: str = 'custom',  # 'custom' or 'mobilenet_v2'
        pretrained: bool = False
    ):
        super().__init__()

        self.feature_dim = feature_dim

        if backbone == 'mobilenet_v2':
            # Use MobileNetV2 for fingerprint
            base_model = models.mobilenet_v2(pretrained=pretrained)

            # Modify first conv for grayscale input
            if input_channels == 1:
                original_conv = base_model.features[0][0]
                base_model.features[0][0] = nn.Conv2d(
                    1, original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=False
                )

            self.backbone = base_model.features
            in_features = base_model.last_channel

            self.fc_layers = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, feature_dim)
            )

        else:
            # Custom CNN for fingerprint
            self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, feature_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Fingerprint images [batch, 1, H, W]

        Returns:
            Feature vectors [batch, feature_dim]
        """
        if hasattr(self, 'backbone'):
            features = self.backbone(x)
            features = self.fc_layers(features)
        else:
            features = self.conv_layers(x)
            features = features.view(features.size(0), -1)
            features = self.fc_layers(features)
        return features


class IrisFeatureExtractor(nn.Module):
    """Feature extractor for iris images"""

    def __init__(
        self,
        feature_dim: int = 512,
        input_channels: int = 1,
        backbone: str = 'custom',  # 'custom' or 'densenet121'
        pretrained: bool = False
    ):
        super().__init__()

        self.feature_dim = feature_dim

        if backbone == 'densenet121':
            # Use DenseNet121 for iris
            base_model = models.densenet121(pretrained=pretrained)

            # Modify first conv for grayscale input
            if input_channels == 1:
                original_conv = base_model.features.conv0
                base_model.features.conv0 = nn.Conv2d(
                    1, original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=False
                )

            self.backbone = base_model.features
            in_features = base_model.classifier.in_features

            self.fc_layers = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, feature_dim)
            )

        else:
            # CNN layers adapted for iris images
            self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Iris images [batch, 1, H, W]

        Returns:
            Feature vectors [batch, feature_dim]
        """
        if hasattr(self, 'backbone'):
            features = self.backbone(x)
            features = self.fc_layers(features)
        else:
            features = self.conv_layers(x)
            features = features.view(features.size(0), -1)
            features = self.fc_layers(features)
        return features


class VoiceFeatureExtractor(nn.Module):
    """Feature extractor for voice/audio using CNN on spectrograms"""

    def __init__(
        self,
        feature_dim: int = 512,
        input_dim: int = 40  # MFCC features
    ):
        super().__init__()

        self.feature_dim = feature_dim

        # 1D CNN for temporal patterns
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        # Fusion and projection
        self.fc_layers = nn.Sequential(
            nn.Linear(512 + 512, 1024),  # CNN + LSTM features
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, feature_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: MFCC features [batch, n_mfcc, time_frames]

        Returns:
            Feature vectors [batch, feature_dim]
        """
        # CNN branch
        cnn_features = self.conv_layers(x)
        cnn_features = cnn_features.squeeze(-1)

        # LSTM branch
        x_transposed = x.transpose(1, 2)  # [batch, time_frames, n_mfcc]
        lstm_out, (h_n, _) = self.lstm(x_transposed)
        # Concatenate final hidden states from both directions
        lstm_features = torch.cat([h_n[-2], h_n[-1]], dim=1)

        # Combine features
        combined = torch.cat([cnn_features, lstm_features], dim=1)
        features = self.fc_layers(combined)

        return features


def get_feature_extractor(
    modality: str,
    feature_dim: int = 512,
    **kwargs
) -> nn.Module:
    """
    Get feature extractor for a specific modality

    Args:
        modality: Modality name ('face', 'fingerprint', 'iris', 'voice')
        feature_dim: Output feature dimension
        **kwargs: Additional arguments for the feature extractor

    Returns:
        Feature extractor module
    """
    extractors = {
        'face': FaceFeatureExtractor,
        'fingerprint': FingerprintFeatureExtractor,
        'finger': FingerprintFeatureExtractor,  # alias for fingerprint
        'iris': IrisFeatureExtractor,
        'voice': VoiceFeatureExtractor,
    }

    if modality not in extractors:
        raise ValueError(f"Unknown modality: {modality}")

    return extractors[modality](feature_dim=feature_dim, **kwargs)
