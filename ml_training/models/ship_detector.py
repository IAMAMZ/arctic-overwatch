"""
Advanced CNN Architectures for Ship Detection in SAR Imagery

This module implements state-of-the-art convolutional neural networks
optimized for detecting maritime vessels in Synthetic Aperture Radar (SAR)
imagery, with special focus on Arctic conditions and dark vessel detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision.models as models
from typing import Dict, List, Optional, Tuple
import numpy as np


class AttentionBlock(nn.Module):
    """
    Spatial Attention Block for focusing on ship-like features
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(AttentionBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.size()
        
        # Channel attention
        avg_out = self.global_avg_pool(x).view(batch_size, channels)
        max_out = self.global_max_pool(x).view(batch_size, channels)
        
        channel_att = self.fc(avg_out) + self.fc(max_out)
        channel_att = channel_att.view(batch_size, channels, 1, 1)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.spatial_conv(spatial_att))
        
        return x * spatial_att


class ResidualBlock(nn.Module):
    """
    Enhanced Residual Block with Batch Normalization and Dropout
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout_rate: float = 0.1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.dropout = nn.Dropout2d(dropout_rate)
        self.attention = AttentionBlock(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out = self.attention(out)
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out


class ShipDetectorCNN(nn.Module):
    """
    Advanced CNN for Ship Detection in SAR Imagery
    
    Features:
    - Multi-scale feature extraction
    - Attention mechanisms
    - Residual connections
    - Maritime-specific preprocessing
    - Arctic condition optimization
    """
    
    def __init__(
        self, 
        input_channels: int = 1,  # SAR imagery is typically single-channel
        num_classes: int = 2,     # Ship/No-Ship binary classification
        dropout_rate: float = 0.2,
        use_attention: bool = True
    ):
        super(ShipDetectorCNN, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks with increasing complexity
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 3, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        # Multi-scale feature pyramid
        self.fpn_conv1 = nn.Conv2d(512, 256, kernel_size=1)
        self.fpn_conv2 = nn.Conv2d(256, 256, kernel_size=1)
        self.fpn_conv3 = nn.Conv2d(128, 256, kernel_size=1)
        
        # Global features
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 3, 512),  # 3 scales from FPN
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
        
        # Ship confidence head (for regression of detection confidence)
        self.confidence_head = nn.Sequential(
            nn.Linear(256 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Initial feature extraction
        x = self.initial_conv(x)
        
        # Hierarchical feature extraction
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        
        # Feature Pyramid Network
        p4 = self.fpn_conv1(c4)
        p3 = self.fpn_conv2(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p2 = self.fpn_conv3(c2) + F.interpolate(p3, scale_factor=2, mode='nearest')
        
        # Global pooling on different scales
        f4 = self.global_avg_pool(p4).flatten(1)
        f3 = self.global_avg_pool(p3).flatten(1)
        f2 = self.global_avg_pool(p2).flatten(1)
        
        # Combine multi-scale features
        features = torch.cat([f4, f3, f2], dim=1)
        
        # Classification and confidence estimation
        logits = self.classifier(features)
        confidence = self.confidence_head(features)
        
        return {
            'logits': logits,
            'confidence': confidence,
            'features': features
        }


class AdvancedShipNet(nn.Module):
    """
    State-of-the-art Ship Detection Network with Transfer Learning
    
    Combines ResNet backbone with custom maritime detection heads
    Optimized for Arctic SAR imagery analysis
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        num_classes: int = 2,
        input_channels: int = 1,
        freeze_backbone: bool = False
    ):
        super(AdvancedShipNet, self).__init__()
        
        # Load pretrained backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Modify first layer for SAR imagery (single channel)
        if input_channels != 3:
            if hasattr(self.backbone, 'conv1'):
                self.backbone.conv1 = nn.Conv2d(
                    input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
            elif hasattr(self.backbone, 'features'):
                # EfficientNet case
                self.backbone.features[0][0] = nn.Conv2d(
                    input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
                )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Remove original classifier
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        
        # Maritime-specific detection heads
        self.ship_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
        # Ship size estimation head
        self.size_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)  # Small, Medium, Large
        )
        
        # Vessel type classifier (for advanced analysis)
        self.vessel_type = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 8)  # Cargo, Tanker, Fishing, Military, etc.
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features using backbone
        features = self.backbone(x)
        
        # Multiple prediction heads
        ship_logits = self.ship_classifier(features)
        size_logits = self.size_estimator(features)
        type_logits = self.vessel_type(features)
        
        return {
            'ship_logits': ship_logits,
            'size_logits': size_logits,
            'type_logits': type_logits,
            'features': features
        }


class MaritimeLoss(nn.Module):
    """
    Custom loss function for maritime detection
    Combines classification loss with spatial awareness and confidence estimation
    """
    
    def __init__(
        self, 
        class_weights: Optional[torch.Tensor] = None,
        alpha: float = 0.7,  # Weight for classification loss
        beta: float = 0.2,   # Weight for confidence loss
        gamma: float = 0.1   # Weight for spatial consistency
    ):
        super(MaritimeLoss, self).__init__()
        self.classification_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.confidence_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = 0.0
        
        # Classification loss
        if 'logits' in predictions and 'labels' in targets:
            cls_loss = self.classification_loss(predictions['logits'], targets['labels'])
            total_loss += self.alpha * cls_loss
        
        # Confidence loss
        if 'confidence' in predictions and 'confidence_targets' in targets:
            conf_loss = self.confidence_loss(predictions['confidence'].squeeze(), targets['confidence_targets'])
            total_loss += self.beta * conf_loss
        
        # Size estimation loss
        if 'size_logits' in predictions and 'size_targets' in targets:
            size_loss = self.classification_loss(predictions['size_logits'], targets['size_targets'])
            total_loss += self.gamma * size_loss
        
        return total_loss


# Model factory function
def create_ship_detector(
    model_type: str = 'advanced',
    input_channels: int = 1,
    num_classes: int = 2,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create ship detection models
    
    Args:
        model_type: Type of model ('basic', 'advanced', 'transfer')
        input_channels: Number of input channels
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        Configured model instance
    """
    
    if model_type == 'basic':
        return ShipDetectorCNN(
            input_channels=input_channels,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type == 'advanced':
        return AdvancedShipNet(
            input_channels=input_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model instantiation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test basic model
    model_basic = ShipDetectorCNN(input_channels=1, num_classes=2)
    model_basic.to(device)
    
    # Test advanced model
    model_advanced = AdvancedShipNet(input_channels=1, num_classes=2)
    model_advanced.to(device)
    
    # Test forward pass
    test_input = torch.randn(4, 1, 224, 224).to(device)
    
    with torch.no_grad():
        output_basic = model_basic(test_input)
        output_advanced = model_advanced(test_input)
    
    print("Basic Model Output Shapes:")
    for key, tensor in output_basic.items():
        print(f"  {key}: {tensor.shape}")
    
    print("\nAdvanced Model Output Shapes:")
    for key, tensor in output_advanced.items():
        print(f"  {key}: {tensor.shape}")
    
    print(f"\nModels successfully created and tested on {device}")
