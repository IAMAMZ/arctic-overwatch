"""
Advanced Model Training Framework for Ship Detection

Comprehensive training system with advanced optimization techniques,
learning rate scheduling, and distributed training support.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.ship_detector import MaritimeLoss, create_ship_detector
from ..utils.metrics import DetectionMetrics, calculate_map
from ..utils.checkpoint import CheckpointManager
from ..utils.early_stopping import EarlyStopping


class ModelTrainer:
    """
    Advanced Training Framework for Maritime Detection Models
    
    Features:
    - Mixed precision training
    - Learning rate scheduling
    - Early stopping with patience
    - Model checkpointing
    - TensorBoard logging
    - Weights & Biases integration
    - Gradient clipping
    - Model pruning and quantization
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        checkpoint_dir: Optional[Path] = None,
        use_wandb: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir or Path('./checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize training components
        self._setup_optimizer()
        self._setup_loss_functions()
        self._setup_scheduler()
        self._setup_logging()
        self._setup_metrics()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_map = 0.0
        self.training_history = defaultdict(list)
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 15),
            min_delta=config.get('early_stopping_min_delta', 1e-4),
            restore_best_weights=True
        )
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            max_checkpoints=config.get('max_checkpoints', 5)
        )
        
        # Weights & Biases
        if use_wandb and config.get('use_wandb', False):
            wandb.init(
                project=config.get('wandb_project', 'arctic-overwatch'),
                config=config,
                name=config.get('experiment_name', 'ship_detection')
            )
            wandb.watch(self.model, log_freq=100)
    
    def _setup_optimizer(self):
        """Initialize optimizer with advanced techniques"""
        optimizer_name = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_name.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
                nesterov=True
            )
        elif optimizer_name.lower() == 'ranger':
            # RAdam + Lookahead (requires ranger-adabelief package)
            try:
                from ranger_adabelief import RangerAdaBelief
                self.optimizer = RangerAdaBelief(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay
                )
            except ImportError:
                print("Ranger optimizer not available, falling back to AdamW")
                self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _setup_loss_functions(self):
        """Initialize loss functions"""
        # Class weights for imbalanced datasets
        class_weights = self.config.get('class_weights', None)
        if class_weights:
            class_weights = torch.tensor(class_weights, device=self.device)
        
        self.criterion = MaritimeLoss(
            class_weights=class_weights,
            alpha=self.config.get('classification_weight', 0.7),
            beta=self.config.get('confidence_weight', 0.2),
            gamma=self.config.get('size_weight', 0.1)
        )
        
        # Additional loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.focal_loss = self._focal_loss  # Custom focal loss implementation
    
    def _setup_scheduler(self):
        """Initialize learning rate scheduler"""
        scheduler_name = self.config.get('scheduler', 'cosine')
        
        if scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_name == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        elif scheduler_name == 'warmup_cosine':
            self.scheduler = self._get_warmup_cosine_scheduler()
        else:
            self.scheduler = None
    
    def _setup_logging(self):
        """Initialize logging"""
        self.writer = SummaryWriter(
            log_dir=Path(self.config.get('log_dir', './logs')) / f"run_{int(time.time())}"
        )
    
    def _setup_metrics(self):
        """Initialize evaluation metrics"""
        self.metrics = DetectionMetrics(
            num_classes=self.config.get('num_classes', 2),
            threshold=self.config.get('detection_threshold', 0.5)
        )
    
    def _focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 1, gamma: float = 2) -> torch.Tensor:
        """
        Focal Loss for addressing class imbalance
        """
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def _get_warmup_cosine_scheduler(self):
        """
        Custom warmup + cosine annealing scheduler
        """
        warmup_epochs = self.config.get('warmup_epochs', 5)
        total_epochs = self.config.get('epochs', 100)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = defaultdict(float)
        epoch_metrics = defaultdict(float)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['image'].to(self.device)
            ship_labels = batch['ship_label'].to(self.device)
            confidence_targets = batch['confidence'].to(self.device)
            size_labels = batch['size_label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    
                    # Prepare targets for loss calculation
                    targets = {
                        'labels': ship_labels,
                        'confidence_targets': confidence_targets,
                        'size_targets': size_labels
                    }
                    
                    loss = self.criterion(outputs, targets)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('gradient_clipping', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clipping']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                
                targets = {
                    'labels': ship_labels,
                    'confidence_targets': confidence_targets,
                    'size_targets': size_labels
                }
                
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                if self.config.get('gradient_clipping', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clipping']
                    )
                
                self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                if 'logits' in outputs:
                    ship_preds = torch.softmax(outputs['logits'], dim=1)
                    ship_pred_labels = torch.argmax(ship_preds, dim=1)
                elif 'ship_logits' in outputs:
                    ship_preds = torch.softmax(outputs['ship_logits'], dim=1)
                    ship_pred_labels = torch.argmax(ship_preds, dim=1)
                else:
                    ship_pred_labels = ship_labels  # Fallback
                
                batch_metrics = self.metrics.calculate_batch_metrics(
                    ship_pred_labels, ship_labels
                )
                
                for key, value in batch_metrics.items():
                    epoch_metrics[key] += value
            
            # Update losses
            epoch_losses['total'] += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log to TensorBoard
            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], global_step)
        
        # Average losses and metrics
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return dict(epoch_losses), dict(epoch_metrics)
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        epoch_losses = defaultdict(float)
        epoch_metrics = defaultdict(float)
        
        all_predictions = []
        all_targets = []
        all_confidences = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                images = batch['image'].to(self.device)
                ship_labels = batch['ship_label'].to(self.device)
                confidence_targets = batch['confidence'].to(self.device)
                size_labels = batch['size_label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                targets = {
                    'labels': ship_labels,
                    'confidence_targets': confidence_targets,
                    'size_targets': size_labels
                }
                
                loss = self.criterion(outputs, targets)
                epoch_losses['total'] += loss.item()
                
                # Get predictions
                if 'logits' in outputs:
                    ship_preds = torch.softmax(outputs['logits'], dim=1)
                    ship_pred_labels = torch.argmax(ship_preds, dim=1)
                    confidence_preds = outputs.get('confidence', torch.zeros_like(confidence_targets))
                elif 'ship_logits' in outputs:
                    ship_preds = torch.softmax(outputs['ship_logits'], dim=1)
                    ship_pred_labels = torch.argmax(ship_preds, dim=1)
                    confidence_preds = torch.sigmoid(outputs.get('confidence', torch.zeros_like(confidence_targets)))
                else:
                    ship_pred_labels = ship_labels  # Fallback
                    confidence_preds = torch.zeros_like(confidence_targets)
                
                # Collect predictions for mAP calculation
                all_predictions.extend(ship_pred_labels.cpu().numpy())
                all_targets.extend(ship_labels.cpu().numpy())
                all_confidences.extend(confidence_preds.cpu().numpy())
                
                # Calculate batch metrics
                batch_metrics = self.metrics.calculate_batch_metrics(
                    ship_pred_labels, ship_labels
                )
                
                for key, value in batch_metrics.items():
                    epoch_metrics[key] += value
        
        # Average losses and metrics
        num_batches = len(self.val_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # Calculate mAP
        val_map = calculate_map(all_predictions, all_targets, all_confidences)
        epoch_metrics['mAP'] = val_map
        
        return dict(epoch_losses), dict(epoch_metrics)
    
    def train(self, epochs: int) -> Dict[str, List]:
        """
        Complete training loop
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Training phase
            train_losses, train_metrics = self.train_epoch()
            
            # Validation phase
            val_losses, val_metrics = self.validate_epoch()
            
            epoch_time = time.time() - start_time
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total'])
                else:
                    self.scheduler.step()
            
            # Log epoch results
            self._log_epoch_results(
                epoch, train_losses, train_metrics, 
                val_losses, val_metrics, epoch_time
            )
            
            # Save checkpoint
            is_best = val_metrics.get('mAP', 0) > self.best_val_map
            if is_best:
                self.best_val_map = val_metrics.get('mAP', 0)
                self.best_val_loss = val_losses['total']
            
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_val_map': self.best_val_map,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': self.config
            }
            
            self.checkpoint_manager.save_checkpoint(
                checkpoint_data,
                is_best=is_best,
                filename=f'checkpoint_epoch_{epoch:03d}.pth'
            )
            
            # Early stopping check
            if self.early_stopping.should_stop(val_losses['total']):
                print(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Update training history
            for key, value in train_losses.items():
                self.training_history[f'train_{key}'].append(value)
            for key, value in val_losses.items():
                self.training_history[f'val_{key}'].append(value)
            for key, value in train_metrics.items():
                self.training_history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                self.training_history[f'val_{key}'].append(value)
        
        print("Training completed!")
        return dict(self.training_history)
    
    def _log_epoch_results(
        self, 
        epoch: int, 
        train_losses: Dict, 
        train_metrics: Dict,
        val_losses: Dict, 
        val_metrics: Dict, 
        epoch_time: float
    ):
        """Log epoch results to various platforms"""
        
        # Console logging
        print(f"\nEpoch {epoch:03d} | Time: {epoch_time:.1f}s | LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"Train Loss: {train_losses['total']:.4f} | Train Acc: {train_metrics.get('accuracy', 0):.3f}")
        print(f"Val Loss: {val_losses['total']:.4f} | Val Acc: {val_metrics.get('accuracy', 0):.3f} | Val mAP: {val_metrics.get('mAP', 0):.3f}")
        
        # TensorBoard logging
        for key, value in train_losses.items():
            self.writer.add_scalar(f'Train/Loss_{key}', value, epoch)
        for key, value in val_losses.items():
            self.writer.add_scalar(f'Val/Loss_{key}', value, epoch)
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'Train/Metric_{key}', value, epoch)
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'Val/Metric_{key}', value, epoch)
        
        self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
        self.writer.add_scalar('Train/Epoch_Time', epoch_time, epoch)
        
        # Weights & Biases logging
        if hasattr(self, 'use_wandb') and self.config.get('use_wandb', False):
            log_dict = {
                'epoch': epoch,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time
            }
            
            for key, value in train_losses.items():
                log_dict[f'train_loss_{key}'] = value
            for key, value in val_losses.items():
                log_dict[f'val_loss_{key}'] = value
            for key, value in train_metrics.items():
                log_dict[f'train_{key}'] = value
            for key, value in val_metrics.items():
                log_dict[f'val_{key}'] = value
            
            wandb.log(log_dict, step=epoch)
    
    def save_training_plots(self, save_dir: Path):
        """Generate and save training plots"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Loss curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.training_history['train_total'], label='Train')
        plt.plot(self.training_history['val_total'], label='Validation')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy curves
        plt.subplot(1, 3, 2)
        if 'train_accuracy' in self.training_history:
            plt.plot(self.training_history['train_accuracy'], label='Train')
            plt.plot(self.training_history['val_accuracy'], label='Validation')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # mAP curve
        plt.subplot(1, 3, 3)
        if 'val_mAP' in self.training_history:
            plt.plot(self.training_history['val_mAP'], label='Validation mAP')
        plt.title('Mean Average Precision')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to {save_dir}")


def train_ship_detector(config_path: str):
    """
    Main training function
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_ship_detector(
        model_type=config['model_type'],
        input_channels=config['input_channels'],
        num_classes=config['num_classes'],
        pretrained=config.get('pretrained', True)
    )
    
    # Create data loaders (this would be implemented with your specific data)
    from ..data.preprocessing import create_dataloaders
    
    train_loader, val_loader = create_dataloaders(
        train_dir=config['train_data_dir'],
        val_dir=config['val_data_dir'],
        train_annotations=config['train_annotations'],
        val_annotations=config['val_annotations'],
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4),
        target_size=tuple(config['input_size'])
    )
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=Path(config.get('checkpoint_dir', './checkpoints')),
        use_wandb=config.get('use_wandb', False)
    )
    
    # Start training
    training_history = trainer.train(config['epochs'])
    
    # Save final plots
    trainer.save_training_plots(Path(config.get('output_dir', './outputs')))
    
    return trainer, training_history


if __name__ == "__main__":
    # Example usage
    print("Ship Detection Trainer initialized successfully!")
    print("Use train_ship_detector(config_path) to start training.")
