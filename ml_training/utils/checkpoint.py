"""
Model Checkpointing System

Advanced checkpointing with automatic cleanup, best model tracking,
and resume capability for robust training sessions.
"""

import torch
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import logging


class CheckpointManager:
    """
    Advanced checkpoint management with automatic cleanup and best model tracking
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        max_checkpoints: int = 5,
        save_best_only: bool = False,
        monitor_metric: str = 'val_loss',
        monitor_mode: str = 'min'
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        
        self.best_metric_value = float('inf') if monitor_mode == 'min' else float('-inf')
        self.checkpoint_history = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(
        self,
        checkpoint_data: Dict,
        is_best: bool = False,
        filename: Optional[str] = None
    ) -> Path:
        """Save model checkpoint with metadata"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_{timestamp}.pth"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Add timestamp and metadata
        checkpoint_data.update({
            'timestamp': datetime.now().isoformat(),
            'checkpoint_path': str(checkpoint_path),
            'pytorch_version': torch.__version__
        })
        
        # Save checkpoint
        try:
            torch.save(checkpoint_data, checkpoint_path)
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Update checkpoint history
            self.checkpoint_history.append({
                'path': checkpoint_path,
                'timestamp': checkpoint_data['timestamp'],
                'epoch': checkpoint_data.get('epoch', 0),
                'is_best': is_best,
                'metrics': {
                    key: value for key, value in checkpoint_data.items()
                    if 'metric' in key.lower() or key in ['train_losses', 'val_losses', 'train_metrics', 'val_metrics']
                }
            })
            
            # Save best model separately
            if is_best:
                best_model_path = self.checkpoint_dir / 'best_model.pth'
                shutil.copy2(checkpoint_path, best_model_path)
                self.logger.info(f"Best model updated: {best_model_path}")
            
            # Cleanup old checkpoints
            if not self.save_best_only:
                self._cleanup_old_checkpoints()
            
            # Save checkpoint history
            self._save_checkpoint_history()
            
            return checkpoint_path
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise e
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[Path] = None,
        load_best: bool = False,
        device: Optional[torch.device] = None
    ) -> Dict:
        """Load model checkpoint"""
        
        if load_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
        elif checkpoint_path is None:
            # Load latest checkpoint
            checkpoint_path = self.get_latest_checkpoint()
        
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            if device is not None:
                checkpoint_data = torch.load(checkpoint_path, map_location=device)
            else:
                checkpoint_data = torch.load(checkpoint_path)
            
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise e
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to the latest checkpoint"""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_*.pth'))
        if not checkpoints:
            return None
        
        # Sort by modification time
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return latest
    
    def list_checkpoints(self) -> List[Dict]:
        """List all available checkpoints with metadata"""
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob('checkpoint_*.pth'):
            try:
                # Load minimal info without full checkpoint
                checkpoint_data = torch.load(checkpoint_file, map_location='cpu')
                checkpoints.append({
                    'path': checkpoint_file,
                    'epoch': checkpoint_data.get('epoch', 0),
                    'timestamp': checkpoint_data.get('timestamp', 'unknown'),
                    'val_loss': checkpoint_data.get('val_losses', {}).get('total', 0),
                    'val_metrics': checkpoint_data.get('val_metrics', {})
                })
            except Exception as e:
                self.logger.warning(f"Could not read checkpoint {checkpoint_file}: {e}")
        
        # Sort by epoch
        checkpoints.sort(key=lambda x: x['epoch'])
        return checkpoints
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit"""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_*.pth'))
        
        if len(checkpoints) > self.max_checkpoints:
            # Sort by modification time
            checkpoints.sort(key=lambda p: p.stat().st_mtime)
            
            # Remove oldest checkpoints
            to_remove = checkpoints[:-self.max_checkpoints]
            for checkpoint in to_remove:
                try:
                    checkpoint.unlink()
                    self.logger.info(f"Removed old checkpoint: {checkpoint}")
                except Exception as e:
                    self.logger.warning(f"Could not remove checkpoint {checkpoint}: {e}")
    
    def _save_checkpoint_history(self):
        """Save checkpoint history metadata"""
        history_file = self.checkpoint_dir / 'checkpoint_history.json'
        
        try:
            # Convert Path objects to strings for JSON serialization
            serializable_history = []
            for item in self.checkpoint_history:
                serializable_item = item.copy()
                serializable_item['path'] = str(item['path'])
                serializable_history.append(serializable_item)
            
            with open(history_file, 'w') as f:
                json.dump(serializable_history, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Could not save checkpoint history: {e}")
    
    def resume_training(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                       scheduler: Optional = None) -> Dict:
        """Resume training from latest checkpoint"""
        
        latest_checkpoint_path = self.get_latest_checkpoint()
        if latest_checkpoint_path is None:
            self.logger.info("No checkpoint found, starting fresh training")
            return {'epoch': 0, 'best_metric': self.best_metric_value}
        
        checkpoint_data = self.load_checkpoint(latest_checkpoint_path)
        
        # Load model state
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # Load scheduler state if available
        if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        
        # Update best metric
        if 'best_val_map' in checkpoint_data:
            self.best_metric_value = checkpoint_data['best_val_map']
        
        resume_info = {
            'epoch': checkpoint_data.get('epoch', 0) + 1,
            'best_metric': self.best_metric_value,
            'resume_from': str(latest_checkpoint_path)
        }
        
        self.logger.info(f"Training resumed from epoch {resume_info['epoch']}")
        return resume_info
