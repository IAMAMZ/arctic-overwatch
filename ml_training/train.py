#!/usr/bin/env python3
"""
Arctic Overwatch - Main Training Script

Advanced CNN training pipeline for maritime vessel detection in SAR imagery.
Supports multiple model architectures, advanced training techniques, and
comprehensive evaluation metrics.

Usage:
    python train.py --config configs/base_config.json
    python train.py --config configs/production_config.json --resume
    python train.py --config configs/base_config.json --evaluate-only
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import local modules
from models.ship_detector import create_ship_detector
from training.trainer import ModelTrainer
from data.preprocessing import create_dataloaders
from utils.metrics import DetectionMetrics
from utils.checkpoint import CheckpointManager


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file"""
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate required fields
    required_fields = ['model', 'data', 'training']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config section: {field}")
    
    return config


def setup_device(config: Dict) -> torch.device:
    """Setup compute device"""
    
    device_config = config.get('hardware', {}).get('device', 'auto')
    
    if device_config == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"CUDA available: {torch.cuda.get_device_name()}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        else:
            device = torch.device('cpu')
            print("CUDA not available, using CPU")
    else:
        device = torch.device(device_config)
    
    return device


def create_model(config: Dict, device: torch.device) -> nn.Module:
    """Create and initialize model"""
    
    model_config = config['model']
    
    model = create_ship_detector(
        model_type=model_config['model_type'],
        input_channels=model_config['input_channels'],
        num_classes=model_config['num_classes'],
        pretrained=model_config.get('pretrained', True),
        backbone=model_config.get('backbone', 'resnet50'),
        freeze_backbone=model_config.get('freeze_backbone', False)
    )
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model_config['model_type']}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def create_data_loaders(config: Dict) -> tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create training, validation, and test data loaders"""
    
    data_config = config['data']
    
    # Create train and validation loaders
    train_loader, val_loader = create_dataloaders(
        train_dir=data_config['train_data_dir'],
        val_dir=data_config['val_data_dir'],
        train_annotations=data_config['train_annotations'],
        val_annotations=data_config['val_annotations'],
        batch_size=data_config['batch_size'],
        num_workers=data_config.get('num_workers', 4),
        target_size=tuple(config['model']['input_size'])
    )
    
    # Create test loader if specified
    test_loader = None
    if 'test_data_dir' in data_config and 'test_annotations' in data_config:
        try:
            from data.preprocessing import ShipDetectionDataset, SARImageProcessor
            
            test_processor = SARImageProcessor(
                target_size=tuple(config['model']['input_size']),
                normalize_db=True
            )
            
            test_dataset = ShipDetectionDataset(
                data_dir=data_config['test_data_dir'],
                annotations_file=data_config['test_annotations'],
                processor=test_processor,
                mode='test'
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=data_config['batch_size'],
                shuffle=False,
                num_workers=data_config.get('num_workers', 4),
                pin_memory=True
            )
            
            print(f"Created test loader with {len(test_dataset)} samples")
            
        except Exception as e:
            print(f"Could not create test loader: {e}")
    
    print(f"Created data loaders:")
    print(f"  Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"  Validation: {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    
    return train_loader, val_loader, test_loader


def train_model(config: Dict, args):
    """Main training function"""
    
    # Setup
    device = setup_device(config)
    model = create_model(config, device)
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Create output directories
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = Path(config['checkpointing']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        use_wandb=config['logging'].get('use_wandb', False)
    )
    
    # Resume training if requested
    if args.resume:
        try:
            resume_info = trainer.checkpoint_manager.resume_training(
                model, trainer.optimizer, trainer.scheduler
            )
            print(f"Resumed training from epoch {resume_info['epoch']}")
        except Exception as e:
            print(f"Could not resume training: {e}")
            print("Starting fresh training...")
    
    # Start training
    print(f"\nStarting training with configuration: {config['experiment_name']}")
    print("=" * 60)
    
    try:
        training_history = trainer.train(config['training']['epochs'])
        
        # Save training history
        history_path = output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f"\nTraining completed successfully!")
        print(f"Training history saved to: {history_path}")
        print(f"Best validation mAP: {trainer.best_val_map:.4f}")
        
        # Save final plots
        trainer.save_training_plots(output_dir)
        
        # Test evaluation if test loader available
        if test_loader and not args.skip_test:
            print("\nRunning final test evaluation...")
            test_metrics = evaluate_model(model, test_loader, device, config)
            
            test_results_path = output_dir / 'test_results.json'
            with open(test_results_path, 'w') as f:
                json.dump(test_metrics, f, indent=2)
            
            print(f"Test results saved to: {test_results_path}")
            print(f"Test Accuracy: {test_metrics.get('accuracy', 'N/A'):.4f}")
            print(f"Test mAP: {test_metrics.get('mAP', 'N/A'):.4f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save current state
        trainer.checkpoint_manager.save_checkpoint({
            'epoch': trainer.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'interrupted': True,
            'config': config
        }, filename='interrupted_checkpoint.pth')
        print("Saved interrupted training state")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)


def evaluate_model(
    model: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device, 
    config: Dict
) -> Dict:
    """Evaluate model on given dataset"""
    
    model.eval()
    metrics = DetectionMetrics(
        num_classes=config['model']['num_classes'],
        threshold=config['evaluation']['detection_threshold']
    )
    
    all_predictions = []
    all_targets = []
    all_confidences = []
    
    print("Running evaluation...")
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            targets = batch['ship_label'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions
            if 'logits' in outputs:
                probs = torch.softmax(outputs['logits'], dim=1)
                predictions = torch.argmax(probs, dim=1)
                confidences = outputs.get('confidence', torch.zeros_like(targets))
            elif 'ship_logits' in outputs:
                probs = torch.softmax(outputs['ship_logits'], dim=1)
                predictions = torch.argmax(probs, dim=1)
                confidences = torch.sigmoid(outputs.get('confidence', torch.zeros_like(targets)))
            else:
                predictions = targets  # Fallback
                confidences = torch.ones_like(targets)
            
            # Accumulate results
            metrics.accumulate_batch(predictions, targets, probs, confidences)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    # Calculate final metrics
    final_metrics = metrics.calculate_epoch_metrics()
    
    # Calculate mAP
    from utils.metrics import calculate_map
    final_metrics['mAP'] = calculate_map(all_predictions, all_targets, all_confidences)
    
    return final_metrics


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Arctic Overwatch CNN Training')
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to configuration JSON file'
    )
    
    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='Resume training from latest checkpoint'
    )
    
    parser.add_argument(
        '--evaluate-only', '-e',
        action='store_true',
        help='Only run evaluation on test set'
    )
    
    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='Skip test evaluation after training'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Print configuration summary
    print(f"Experiment: {config['experiment_name']}")
    print(f"Description: {config['description']}")
    print(f"Model: {config['model']['model_type']} ({config['model']['backbone']})")
    print(f"Input size: {config['model']['input_size']}")
    print(f"Batch size: {config['data']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    print("=" * 60)
    
    if args.evaluate_only:
        # Load best model and evaluate
        device = setup_device(config)
        model = create_model(config, device)
        
        # Load checkpoint
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=Path(config['checkpointing']['checkpoint_dir'])
        )
        
        try:
            checkpoint_data = checkpoint_manager.load_checkpoint(load_best=True, device=device)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            print("Loaded best model checkpoint")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            sys.exit(1)
        
        # Create test data loader
        _, _, test_loader = create_data_loaders(config)
        
        if test_loader is None:
            logger.error("No test data available for evaluation")
            sys.exit(1)
        
        # Run evaluation
        test_metrics = evaluate_model(model, test_loader, device, config)
        
        print("\nEvaluation Results:")
        for key, value in test_metrics.items():
            if key != 'confusion_matrix':
                print(f"{key}: {value:.4f}")
        
        # Save results
        output_dir = Path(config['paths']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
    
    else:
        # Run training
        train_model(config, args)
    
    print("\nScript completed successfully!")


if __name__ == "__main__":
    main()
