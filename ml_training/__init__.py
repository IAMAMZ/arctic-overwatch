"""
Arctic Overwatch - Machine Learning Training Module

A comprehensive CNN-based ship detection system for SAR (Synthetic Aperture Radar) imagery
processing and dark vessel identification in Arctic waters.

This module provides:
- Advanced CNN architectures for ship detection
- Multi-scale feature extraction
- Data augmentation and preprocessing
- Transfer learning capabilities
- Real-time inference optimization
- Maritime domain-specific loss functions
"""

__version__ = "2.1.0"
__author__ = "Arctic Overwatch Team"

from .models.ship_detector import ShipDetectorCNN, AdvancedShipNet
from .training.trainer import ModelTrainer
from .data.preprocessing import SARImageProcessor
from .utils.metrics import DetectionMetrics

__all__ = [
    'ShipDetectorCNN',
    'AdvancedShipNet', 
    'ModelTrainer',
    'SARImageProcessor',
    'DetectionMetrics'
]
