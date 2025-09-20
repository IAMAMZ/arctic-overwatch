#!/usr/bin/env python3
"""
Arctic Overwatch - CNN Demo Script

Quick demonstration of the CNN training capabilities.
Shows model architecture, preprocessing pipeline, and basic training simulation.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Import local modules
from models.ship_detector import ShipDetectorCNN, AdvancedShipNet, create_ship_detector
from data.preprocessing import SARImageProcessor
from training.trainer import ModelTrainer
from utils.metrics import DetectionMetrics


def demo_model_architectures():
    """Demonstrate different model architectures"""
    print("🚢 Arctic Overwatch - CNN Model Architectures Demo")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test input (simulating SAR imagery)
    batch_size = 4
    input_channels = 1
    height, width = 224, 224
    test_input = torch.randn(batch_size, input_channels, height, width).to(device)
    
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Input represents: {batch_size} SAR images, {height}x{width} pixels")
    
    # 1. Basic Ship Detector CNN
    print("\n1. 🔍 ShipDetectorCNN (Custom Architecture)")
    print("-" * 40)
    
    basic_model = ShipDetectorCNN(
        input_channels=input_channels,
        num_classes=2,
        dropout_rate=0.2
    ).to(device)
    
    with torch.no_grad():
        basic_output = basic_model(test_input)
    
    total_params = sum(p.numel() for p in basic_model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Output keys: {list(basic_output.keys())}")
    for key, tensor in basic_output.items():
        print(f"  {key}: {tensor.shape}")
    
    # 2. Advanced Ship Net
    print("\n2. 🎯 AdvancedShipNet (Transfer Learning)")
    print("-" * 40)
    
    advanced_model = AdvancedShipNet(
        backbone='resnet50',
        input_channels=input_channels,
        num_classes=2,
        pretrained=False  # For demo, avoid downloading weights
    ).to(device)
    
    with torch.no_grad():
        advanced_output = advanced_model(test_input)
    
    total_params = sum(p.numel() for p in advanced_model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Output keys: {list(advanced_output.keys())}")
    for key, tensor in advanced_output.items():
        print(f"  {key}: {tensor.shape}")
    
    print("\n✅ Model architecture demo completed!")


def demo_preprocessing_pipeline():
    """Demonstrate SAR image preprocessing"""
    print("\n🖼️  SAR Image Preprocessing Pipeline Demo")
    print("=" * 60)
    
    # Create processor
    processor = SARImageProcessor(
        target_size=(224, 224),
        normalize_db=True,
        denoise_method='bilateral',
        enhance_contrast=True
    )
    
    print(f"Target size: {processor.target_size}")
    print(f"Normalization: {'dB scale' if processor.normalize_db else 'Linear scale'}")
    print(f"Denoising method: {processor.denoise_method}")
    
    # Simulate SAR image data (exponential distribution typical for SAR)
    print("\n📡 Simulating SAR imagery...")
    raw_sar_image = np.random.exponential(0.1, (512, 512)).astype(np.float32)
    
    print(f"Original image shape: {raw_sar_image.shape}")
    print(f"Original value range: [{raw_sar_image.min():.3f}, {raw_sar_image.max():.3f}]")
    
    # Apply preprocessing steps
    print("\n🔧 Applying preprocessing pipeline...")
    
    # Step-by-step demonstration
    denoised = processor.reduce_speckle_noise(raw_sar_image)
    enhanced = processor.enhance_maritime_features(denoised)
    normalized = processor.normalize_sar_image(enhanced)
    
    print(f"After denoising: [{denoised.min():.3f}, {denoised.max():.3f}]")
    print(f"After enhancement: [{enhanced.min():.3f}, {enhanced.max():.3f}]")
    print(f"After normalization: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # Full preprocessing
    processed_tensor = processor.preprocess_image(raw_sar_image, is_training=True)
    print(f"\nFinal tensor shape: {processed_tensor.shape}")
    print(f"Final tensor range: [{processed_tensor.min():.3f}, {processed_tensor.max():.3f}]")
    
    print("\n✅ Preprocessing pipeline demo completed!")


def demo_training_simulation():
    """Simulate a short training run"""
    print("\n🏋️ Training Pipeline Simulation")
    print("=" * 60)
    
    # Create small model for quick demo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ShipDetectorCNN(
        input_channels=1,
        num_classes=2,
        dropout_rate=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Simulate training data
    print("\n📊 Generating simulated training data...")
    
    batch_size = 8
    num_batches = 5
    
    # Simulate training metrics
    print("\n🔄 Simulating training epochs...")
    
    metrics_calculator = DetectionMetrics(num_classes=2)
    
    for epoch in range(3):
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        print(f"\nEpoch {epoch + 1}/3:")
        
        for batch in range(num_batches):
            # Simulate batch
            images = torch.randn(batch_size, 1, 224, 224).to(device)
            targets = torch.randint(0, 2, (batch_size,)).to(device)
            
            # Simulate forward pass
            with torch.no_grad():
                outputs = model(images)
                predictions = torch.argmax(outputs['logits'], dim=1)
                
                # Calculate simulated loss (decreasing over epochs)
                batch_loss = 1.0 - (epoch * 0.2) + np.random.normal(0, 0.1)
                epoch_loss += max(batch_loss, 0.1)
                
                # Calculate accuracy
                correct = (predictions == targets).sum().item()
                correct_predictions += correct
                total_samples += batch_size
                
                # Accumulate for metrics
                metrics_calculator.accumulate_batch(predictions, targets)
            
            print(f"  Batch {batch + 1}/{num_batches}: Loss = {batch_loss:.4f}")
        
        # Calculate epoch metrics
        epoch_metrics = metrics_calculator.calculate_batch_metrics(
            torch.randint(0, 2, (total_samples,)),
            torch.randint(0, 2, (total_samples,))
        )
        
        avg_loss = epoch_loss / num_batches
        accuracy = correct_predictions / total_samples
        
        print(f"  Epoch Summary:")
        print(f"    Loss: {avg_loss:.4f}")
        print(f"    Accuracy: {accuracy:.3f}")
        print(f"    Precision: {epoch_metrics['precision']:.3f}")
        print(f"    Recall: {epoch_metrics['recall']:.3f}")
        print(f"    F1-Score: {epoch_metrics['f1_score']:.3f}")
        
        # Reset for next epoch
        metrics_calculator.reset_accumulation()
    
    print("\n✅ Training simulation completed!")


def demo_evaluation_metrics():
    """Demonstrate evaluation metrics"""
    print("\n📈 Evaluation Metrics Demo")
    print("=" * 60)
    
    # Simulate prediction results
    np.random.seed(42)
    
    # Generate realistic maritime detection scenario
    total_samples = 1000
    true_ship_ratio = 0.3  # 30% of images contain ships
    
    # Ground truth (0=no ship, 1=ship)
    targets = np.random.choice([0, 1], size=total_samples, p=[1-true_ship_ratio, true_ship_ratio])
    
    # Simulate realistic predictions (ships are harder to detect than background)
    predictions = targets.copy()
    
    # Add some false negatives (missed ships)
    ship_indices = np.where(targets == 1)[0]
    missed_ships = np.random.choice(ship_indices, size=int(0.15 * len(ship_indices)), replace=False)
    predictions[missed_ships] = 0
    
    # Add some false positives (false alarms)
    no_ship_indices = np.where(targets == 0)[0]
    false_alarms = np.random.choice(no_ship_indices, size=int(0.05 * len(no_ship_indices)), replace=False)
    predictions[false_alarms] = 1
    
    # Simulate confidence scores
    confidences = np.random.beta(2, 1, size=total_samples)  # Higher confidence distribution
    confidences[targets == 0] *= 0.7  # Lower confidence for background
    
    print(f"Total samples: {total_samples}")
    print(f"True ships: {np.sum(targets == 1)}")
    print(f"Predicted ships: {np.sum(predictions == 1)}")
    
    # Calculate metrics
    metrics_calc = DetectionMetrics(num_classes=2, class_names=['No Ship', 'Ship'])
    
    # Convert to tensors and accumulate
    targets_tensor = torch.tensor(targets)
    predictions_tensor = torch.tensor(predictions) 
    confidences_tensor = torch.tensor(confidences)
    probs_tensor = torch.stack([1-confidences_tensor, confidences_tensor], dim=1)
    
    metrics_calc.accumulate_batch(predictions_tensor, targets_tensor, probs_tensor, confidences_tensor)
    
    # Calculate comprehensive metrics
    results = metrics_calc.calculate_epoch_metrics()
    
    print("\n📊 Detection Performance:")
    print("-" * 30)
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall: {results['recall']:.3f}")
    print(f"F1-Score: {results['f1_score']:.3f}")
    print(f"Specificity: {results['specificity']:.3f}")
    if 'roc_auc' in results:
        print(f"ROC-AUC: {results['roc_auc']:.3f}")
    if 'average_precision' in results:
        print(f"Average Precision: {results['average_precision']:.3f}")
    
    print("\n🚢 Maritime-Specific Metrics:")
    print("-" * 30)
    print(f"Ship Detection Rate: {results['ship_detection_rate']:.3f}")
    print(f"False Alarm Rate: {results['false_alarm_rate']:.3f}")
    print(f"Detection Confidence Rate: {results['detection_confidence_rate']:.3f}")
    
    # Confusion matrix
    print("\n🔢 Confusion Matrix:")
    cm = np.array(results['confusion_matrix'])
    print("       Predicted")
    print("       No Ship  Ship")
    print(f"No Ship   {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"Ship      {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    print("\n✅ Evaluation metrics demo completed!")


def demo_configuration():
    """Show configuration management"""
    print("\n⚙️  Configuration Management Demo")
    print("=" * 60)
    
    # Load configuration
    config_path = Path("configs/base_config.json")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("📋 Base Configuration Summary:")
        print(f"Experiment: {config['experiment_name']}")
        print(f"Model Type: {config['model']['model_type']}")
        print(f"Backbone: {config['model']['backbone']}")
        print(f"Input Size: {config['model']['input_size']}")
        print(f"Batch Size: {config['data']['batch_size']}")
        print(f"Epochs: {config['training']['epochs']}")
        print(f"Learning Rate: {config['training']['learning_rate']}")
        print(f"Optimizer: {config['training']['optimizer']}")
        
        print("\n🔧 Preprocessing Settings:")
        preproc = config['preprocessing']
        print(f"Normalize dB: {preproc['normalize_db']}")
        print(f"Denoise Method: {preproc['denoise_method']}")
        print(f"Enhance Contrast: {preproc['enhance_contrast']}")
        print(f"Multi-scale Training: {preproc['multi_scale_training']}")
        
        print("\n📊 Augmentation Settings:")
        aug = config['augmentation']
        print(f"Horizontal Flip: {aug['horizontal_flip_prob']}")
        print(f"Rotation Limit: {aug['rotation_limit']}°")
        print(f"Gaussian Noise: {aug['gaussian_noise_prob']}")
        
    else:
        print("⚠️  Configuration file not found, showing example structure:")
        example_config = {
            "experiment_name": "arctic_ship_detection_demo",
            "model": {
                "model_type": "advanced",
                "input_channels": 1,
                "num_classes": 2
            },
            "training": {
                "epochs": 100,
                "learning_rate": 0.001,
                "optimizer": "adamw"
            }
        }
        print(json.dumps(example_config, indent=2))
    
    print("\n✅ Configuration demo completed!")


def main():
    """Run complete demo"""
    print("🌊 ARCTIC OVERWATCH - CNN TRAINING SYSTEM DEMO")
    print("=" * 70)
    print("Advanced Maritime Vessel Detection in SAR Imagery")
    print("Powered by PyTorch & Deep Learning")
    print("=" * 70)
    
    try:
        # Run all demos
        demo_model_architectures()
        demo_preprocessing_pipeline()
        demo_evaluation_metrics()
        demo_configuration()
        demo_training_simulation()
        
        print("\n" + "=" * 70)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\n📚 Next Steps:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Prepare your SAR dataset")
        print("3. Configure training: edit configs/base_config.json")
        print("4. Start training: python train.py --config configs/base_config.json")
        print("\n🚢 Ready for Arctic Maritime Surveillance!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        print("This is normal if dependencies are not installed")
        print("Run: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
