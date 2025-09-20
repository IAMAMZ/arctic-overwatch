"""
Early Stopping Implementation

Prevents overfitting by monitoring validation metrics and stopping
training when improvement plateaus.
"""

import numpy as np
from typing import Optional
import logging


class EarlyStopping:
    """
    Early stopping to halt training when validation metric stops improving
    """
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0,
        restore_best_weights: bool = True,
        monitor_metric: str = 'val_loss',
        mode: str = 'min',
        baseline: Optional[float] = None,
        verbose: bool = True
    ):
        """
        Args:
            patience: Number of epochs to wait after last time validation metric improved
            min_delta: Minimum change in monitored quantity to qualify as improvement
            restore_best_weights: Whether to restore model weights from best epoch
            monitor_metric: Metric to monitor
            mode: 'min' for metrics where lower is better, 'max' for higher is better
            baseline: Baseline value for the metric. Training stops if model doesn't show improvement over baseline
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.baseline = baseline
        self.verbose = verbose
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_metric = None
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best_metric = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best_metric = -np.Inf
        else:
            raise ValueError(f"Mode {mode} is unknown, use 'min' or 'max'")
        
        if self.baseline is not None:
            if mode == 'min':
                self.baseline_met = lambda current: current <= self.baseline
            else:
                self.baseline_met = lambda current: current >= self.baseline
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, current_metric: float, model_weights: Optional = None) -> bool:
        """
        Check if training should stop
        
        Args:
            current_metric: Current value of the monitored metric
            model_weights: Current model weights to save if this is the best epoch
            
        Returns:
            True if training should stop, False otherwise
        """
        return self.should_stop(current_metric, model_weights)
    
    def should_stop(self, current_metric: float, model_weights: Optional = None) -> bool:
        """
        Determine if early stopping criteria are met
        
        Args:
            current_metric: Current value of monitored metric
            model_weights: Current model state dict
            
        Returns:
            Boolean indicating whether to stop training
        """
        
        # Check if baseline is met
        if self.baseline is not None and not self.baseline_met(current_metric):
            if self.verbose:
                self.logger.info(
                    f"Epoch: metric {current_metric:.4f} did not reach baseline {self.baseline:.4f}"
                )
        
        # Check for improvement
        if self._is_improvement(current_metric):
            self.best_metric = current_metric
            self.wait = 0
            
            # Save best weights
            if self.restore_best_weights and model_weights is not None:
                self.best_weights = model_weights.copy()
            
            if self.verbose:
                self.logger.info(
                    f"Metric improved to {current_metric:.4f}"
                )
            
            return False
        
        else:
            self.wait += 1
            if self.verbose:
                self.logger.info(
                    f"Metric did not improve from {self.best_metric:.4f}. "
                    f"Patience: {self.wait}/{self.patience}"
                )
            
            if self.wait >= self.patience:
                self.stopped_epoch = self.wait
                if self.verbose:
                    self.logger.info(
                        f"Early stopping triggered after {self.patience} epochs without improvement"
                    )
                return True
        
        return False
    
    def _is_improvement(self, current_metric: float) -> bool:
        """Check if current metric is an improvement over the best so far"""
        if self.mode == 'min':
            return current_metric < (self.best_metric - self.min_delta)
        else:
            return current_metric > (self.best_metric + self.min_delta)
    
    def get_best_metric(self) -> float:
        """Get the best metric value seen so far"""
        return self.best_metric
    
    def get_best_weights(self):
        """Get the best model weights"""
        return self.best_weights
    
    def reset(self):
        """Reset early stopping state"""
        self.wait = 0
        self.stopped_epoch = 0
        if self.mode == 'min':
            self.best_metric = np.Inf
        else:
            self.best_metric = -np.Inf
        self.best_weights = None


class AdvancedEarlyStopping(EarlyStopping):
    """
    Advanced early stopping with additional features
    """
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0,
        restore_best_weights: bool = True,
        monitor_metric: str = 'val_loss',
        mode: str = 'min',
        baseline: Optional[float] = None,
        verbose: bool = True,
        warmup_epochs: int = 0,
        cooldown_epochs: int = 0,
        reduce_lr_on_plateau: bool = False,
        lr_reduction_factor: float = 0.5,
        min_lr: float = 1e-7
    ):
        """
        Extended early stopping with learning rate reduction and warmup
        
        Args:
            warmup_epochs: Number of initial epochs to ignore (useful for noisy start)
            cooldown_epochs: Number of epochs to wait after LR reduction before monitoring again
            reduce_lr_on_plateau: Whether to reduce learning rate when plateau is detected
            lr_reduction_factor: Factor to reduce learning rate by
            min_lr: Minimum learning rate threshold
        """
        super().__init__(
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=restore_best_weights,
            monitor_metric=monitor_metric,
            mode=mode,
            baseline=baseline,
            verbose=verbose
        )
        
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_reduction_factor = lr_reduction_factor
        self.min_lr = min_lr
        
        self.current_epoch = 0
        self.last_lr_reduction_epoch = -1
        self.lr_reductions = 0
    
    def should_stop(
        self, 
        current_metric: float, 
        model_weights: Optional = None,
        optimizer: Optional = None
    ) -> bool:
        """
        Advanced early stopping with learning rate scheduling
        
        Args:
            current_metric: Current value of monitored metric
            model_weights: Current model state dict
            optimizer: Optimizer to potentially adjust learning rate
            
        Returns:
            Boolean indicating whether to stop training
        """
        self.current_epoch += 1
        
        # Skip monitoring during warmup
        if self.current_epoch <= self.warmup_epochs:
            if self.verbose:
                self.logger.info(f"Warmup epoch {self.current_epoch}/{self.warmup_epochs}")
            return False
        
        # Skip monitoring during cooldown after LR reduction
        if (self.current_epoch - self.last_lr_reduction_epoch) <= self.cooldown_epochs:
            if self.verbose:
                self.logger.info(
                    f"Cooldown epoch {self.current_epoch - self.last_lr_reduction_epoch}/{self.cooldown_epochs}"
                )
            return False
        
        # Check for improvement
        if self._is_improvement(current_metric):
            self.best_metric = current_metric
            self.wait = 0
            
            if self.restore_best_weights and model_weights is not None:
                self.best_weights = model_weights.copy()
            
            if self.verbose:
                self.logger.info(f"Metric improved to {current_metric:.4f}")
            
            return False
        
        else:
            self.wait += 1
            
            # Check if we should reduce learning rate
            if (self.reduce_lr_on_plateau and 
                optimizer is not None and 
                self.wait >= self.patience // 2):  # Reduce LR at half patience
                
                current_lr = optimizer.param_groups[0]['lr']
                new_lr = current_lr * self.lr_reduction_factor
                
                if new_lr >= self.min_lr:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    
                    self.last_lr_reduction_epoch = self.current_epoch
                    self.lr_reductions += 1
                    self.wait = 0  # Reset patience after LR reduction
                    
                    if self.verbose:
                        self.logger.info(
                            f"Reduced learning rate from {current_lr:.2e} to {new_lr:.2e}"
                        )
                    
                    return False
            
            if self.verbose:
                self.logger.info(
                    f"Metric did not improve. Patience: {self.wait}/{self.patience}"
                )
            
            if self.wait >= self.patience:
                if self.verbose:
                    self.logger.info(
                        f"Early stopping triggered. Best metric: {self.best_metric:.4f}"
                    )
                return True
        
        return False
    
    def get_lr_reduction_info(self) -> dict:
        """Get information about learning rate reductions"""
        return {
            'num_reductions': self.lr_reductions,
            'last_reduction_epoch': self.last_lr_reduction_epoch
        }


if __name__ == "__main__":
    # Test early stopping
    import torch
    import torch.nn as nn
    
    # Mock validation losses (decreasing then increasing)
    val_losses = [1.0, 0.8, 0.6, 0.5, 0.45, 0.44, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48]
    
    early_stopping = EarlyStopping(patience=3, min_delta=0.01, verbose=True)
    
    # Mock model
    model = nn.Linear(10, 1)
    
    for epoch, val_loss in enumerate(val_losses):
        should_stop = early_stopping.should_stop(val_loss, model.state_dict())
        print(f"Epoch {epoch}: Val Loss = {val_loss:.3f}, Should Stop = {should_stop}")
        
        if should_stop:
            break
    
    print(f"Best metric achieved: {early_stopping.get_best_metric():.4f}")
    print("Early stopping test completed successfully!")
