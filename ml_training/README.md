# Arctic Overwatch - ML Training Module

Advanced CNN training pipeline for maritime vessel detection in SAR imagery, optimized for Arctic surveillance operations.

## 🚀 Features

- **State-of-the-art CNN Architectures**: Custom networks with attention mechanisms, residual connections, and feature pyramid networks
- **Advanced SAR Preprocessing**: Speckle noise reduction, sea ice interference removal, and maritime feature enhancement
- **Multi-scale Training**: Dynamic input scaling for robust detection across different image resolutions
- **Production-Ready Training**: Mixed precision, distributed training, gradient clipping, and advanced optimization
- **Comprehensive Evaluation**: mAP, precision-recall curves, ROC analysis, and maritime-specific metrics
- **Experiment Management**: TensorBoard, Weights & Biases integration, and automatic checkpointing

## 📁 Project Structure

```
ml_training/
├── models/                 # Neural network architectures
│   ├── ship_detector.py   # Advanced CNN models
│   └── __init__.py
├── data/                  # Data processing pipeline
│   ├── preprocessing.py   # SAR image preprocessing
│   └── __init__.py
├── training/              # Training framework
│   ├── trainer.py        # Main training logic
│   └── __init__.py
├── utils/                 # Utility functions
│   ├── metrics.py        # Evaluation metrics
│   ├── checkpoint.py     # Model checkpointing
│   ├── early_stopping.py # Early stopping implementation
│   └── __init__.py
├── configs/               # Configuration files
│   ├── base_config.json  # Base configuration
│   ├── production_config.json # Production settings
│   └── __init__.py
├── train.py              # Main training script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository_url>
cd arctic-overwatch/ml_training
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\\Scripts\\activate  # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🚀 Quick Start

### Basic Training

```bash
# Train with base configuration
python train.py --config configs/base_config.json

# Resume training from checkpoint
python train.py --config configs/base_config.json --resume

# Production training with advanced settings
python train.py --config configs/production_config.json
```

### Evaluation Only

```bash
# Evaluate best model on test set
python train.py --config configs/base_config.json --evaluate-only
```

## 📊 Model Architectures

### 1. ShipDetectorCNN
Custom CNN with advanced features:
- Multi-scale feature extraction
- Spatial attention mechanisms
- Residual connections with dropout
- Feature Pyramid Network (FPN)
- Maritime-specific loss functions

### 2. AdvancedShipNet
Transfer learning-based architecture:
- ResNet/EfficientNet backbones
- Multi-head prediction (ship detection, size estimation, vessel type)
- Fine-tuned for SAR imagery
- Arctic condition optimization

## 🔧 Configuration

### Base Configuration (`configs/base_config.json`)
Optimized for development and testing:
- Input size: 224×224
- Batch size: 32
- ResNet50 backbone
- 150 epochs with early stopping

### Production Configuration (`configs/production_config.json`)
Optimized for deployment:
- Input size: 380×380
- EfficientNet-B4 backbone
- Advanced augmentation
- Model optimization settings

### Custom Configuration
Create your own config by modifying the JSON files:

```json
{
  "experiment_name": "my_experiment",
  "model": {
    "model_type": "advanced",
    "backbone": "resnet50",
    "input_size": [224, 224]
  },
  "training": {
    "epochs": 100,
    "learning_rate": 0.001,
    "optimizer": "adamw"
  }
}
```

## 📈 Data Pipeline

### SAR Image Preprocessing
- **Speckle Noise Reduction**: Bilateral filtering, non-local means
- **Feature Enhancement**: CLAHE, edge enhancement
- **Sea Ice Removal**: Morphological operations
- **Normalization**: dB-scale conversion with maritime statistics

### Data Augmentation
- Geometric transforms (rotation, flipping, elastic deformation)
- Photometric transforms (brightness, contrast, noise)
- Arctic-specific augmentations

### Dataset Format
Supports multiple annotation formats:
- COCO JSON format
- Custom JSON format
- Automatic negative sample mining

## 🏋️ Training Features

### Advanced Optimization
- **Mixed Precision Training**: Faster training with FP16
- **Gradient Clipping**: Stable training for deep networks
- **Learning Rate Scheduling**: Cosine annealing, warmup
- **Advanced Optimizers**: AdamW, Ranger, RAdam

### Regularization
- **Early Stopping**: Patience-based with metric monitoring
- **Dropout**: Adaptive dropout rates
- **Weight Decay**: L2 regularization
- **Data Augmentation**: Comprehensive augmentation pipeline

### Model Checkpointing
- Automatic best model saving
- Resume capability
- Checkpoint cleanup
- Model version tracking

## 📊 Evaluation Metrics

### Standard Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, Average Precision
- Confusion Matrix analysis

### Maritime-Specific Metrics
- **Ship Detection Rate**: True positive rate for vessels
- **False Alarm Rate**: False positives per image
- **Detection Confidence**: High-confidence detection rate
- **Size Classification**: Small/Medium/Large vessel accuracy

### Visualization
- Training curves (loss, accuracy, mAP)
- Confusion matrix heatmaps
- ROC and PR curves
- Sample predictions with confidence

## 🔬 Advanced Features

### Model Optimization
- **Quantization**: INT8 inference optimization
- **Pruning**: Network compression
- **ONNX Export**: Cross-platform deployment
- **TensorRT**: GPU inference acceleration

### Distributed Training
- Multi-GPU training support
- Gradient synchronization
- Efficient data loading

### Experiment Tracking
- **TensorBoard**: Local visualization
- **Weights & Biases**: Cloud experiment tracking
- **MLflow**: Model registry and versioning

## 📝 Usage Examples

### Custom Training Loop
```python
from models.ship_detector import create_ship_detector
from training.trainer import ModelTrainer
from data.preprocessing import create_dataloaders

# Create model
model = create_ship_detector(
    model_type='advanced',
    input_channels=1,
    num_classes=2
)

# Create data loaders
train_loader, val_loader = create_dataloaders(
    train_dir='./data/train',
    val_dir='./data/val',
    train_annotations='./data/train/annotations.json',
    val_annotations='./data/val/annotations.json'
)

# Train model
trainer = ModelTrainer(model, train_loader, val_loader, config, device)
history = trainer.train(epochs=100)
```

### Custom Preprocessing
```python
from data.preprocessing import SARImageProcessor

processor = SARImageProcessor(
    target_size=(224, 224),
    normalize_db=True,
    denoise_method='bilateral',
    enhance_contrast=True
)

processed_image = processor.preprocess_image('path/to/sar_image.tif')
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Slow Training**:
   - Increase num_workers for data loading
   - Use SSD storage for dataset
   - Enable pin_memory for GPU training

3. **Poor Convergence**:
   - Adjust learning rate
   - Check data preprocessing
   - Verify label quality

### Performance Optimization

- **Data Loading**: Use multiple workers and pin memory
- **Model Optimization**: Enable mixed precision training
- **Storage**: Use SSD for dataset storage
- **Memory**: Monitor GPU memory usage

## 📚 References

- SAR Ship Detection: "Deep Learning for SAR Ship Detection"
- Attention Mechanisms: "Attention Is All You Need"
- Feature Pyramid Networks: "Feature Pyramid Networks for Object Detection"
- Mixed Precision Training: "Mixed Precision Training"

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is part of Arctic Overwatch maritime surveillance system.

## 🙋‍♂️ Support

For technical support or questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review configuration examples

---

**Arctic Overwatch v2.1** - Advanced Maritime Surveillance with Deep Learning
