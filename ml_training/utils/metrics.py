"""
Evaluation Metrics for Maritime Detection

Comprehensive evaluation metrics including precision, recall, F1-score,
Average Precision (AP), and Mean Average Precision (mAP) optimized
for maritime vessel detection applications.
"""

import torch
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class DetectionMetrics:
    """
    Comprehensive metrics calculator for ship detection tasks
    """
    
    def __init__(
        self, 
        num_classes: int = 2, 
        threshold: float = 0.5,
        class_names: Optional[List[str]] = None
    ):
        self.num_classes = num_classes
        self.threshold = threshold
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        
        # Accumulate predictions for epoch-level metrics
        self.reset_accumulation()
    
    def reset_accumulation(self):
        """Reset accumulated predictions for new epoch"""
        self.accumulated_predictions = []
        self.accumulated_targets = []
        self.accumulated_confidences = []
        self.accumulated_probabilities = []
    
    def accumulate_batch(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None,
        confidences: Optional[torch.Tensor] = None
    ):
        """Accumulate batch predictions for epoch-level metrics"""
        self.accumulated_predictions.extend(predictions.cpu().numpy())
        self.accumulated_targets.extend(targets.cpu().numpy())
        
        if probabilities is not None:
            self.accumulated_probabilities.extend(probabilities.cpu().numpy())
        
        if confidences is not None:
            self.accumulated_confidences.extend(confidences.cpu().numpy())
    
    def calculate_batch_metrics(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate metrics for a single batch"""
        
        # Convert to numpy
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        # Basic metrics
        accuracy = accuracy_score(target_np, pred_np)
        
        # Handle binary and multiclass cases
        if self.num_classes == 2:
            precision = precision_score(target_np, pred_np, zero_division=0)
            recall = recall_score(target_np, pred_np, zero_division=0)
            f1 = f1_score(target_np, pred_np, zero_division=0)
        else:
            precision = precision_score(target_np, pred_np, average='weighted', zero_division=0)
            recall = recall_score(target_np, pred_np, average='weighted', zero_division=0)
            f1 = f1_score(target_np, pred_np, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def calculate_epoch_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive metrics for entire epoch"""
        if not self.accumulated_predictions:
            return {}
        
        predictions = np.array(self.accumulated_predictions)
        targets = np.array(self.accumulated_targets)
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(targets, predictions)
        
        if self.num_classes == 2:
            # Binary classification metrics
            metrics['precision'] = precision_score(targets, predictions, zero_division=0)
            metrics['recall'] = recall_score(targets, predictions, zero_division=0)
            metrics['f1_score'] = f1_score(targets, predictions, zero_division=0)
            metrics['specificity'] = self._calculate_specificity(targets, predictions)
            
            # ROC-AUC if probabilities available
            if self.accumulated_probabilities:
                probs = np.array(self.accumulated_probabilities)
                if len(probs.shape) > 1 and probs.shape[1] == 2:
                    probs = probs[:, 1]  # Take positive class probability
                metrics['roc_auc'] = roc_auc_score(targets, probs)
                
                # Average Precision (AP)
                metrics['average_precision'] = average_precision_score(targets, probs)
        else:
            # Multiclass metrics
            metrics['precision'] = precision_score(targets, predictions, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(targets, predictions, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(targets, predictions, average='weighted', zero_division=0)
            
            # Per-class metrics
            per_class_precision = precision_score(targets, predictions, average=None, zero_division=0)
            per_class_recall = recall_score(targets, predictions, average=None, zero_division=0)
            per_class_f1 = f1_score(targets, predictions, average=None, zero_division=0)
            
            for i, class_name in enumerate(self.class_names):
                metrics[f'precision_{class_name}'] = per_class_precision[i]
                metrics[f'recall_{class_name}'] = per_class_recall[i]
                metrics[f'f1_{class_name}'] = per_class_f1[i]
        
        # Confusion matrix derived metrics
        cm = confusion_matrix(targets, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Maritime-specific metrics
        if self.num_classes == 2:
            metrics.update(self._calculate_maritime_metrics(targets, predictions))
        
        return metrics
    
    def _calculate_specificity(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate specificity (True Negative Rate)"""
        cm = confusion_matrix(targets, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            return specificity
        return 0.0
    
    def _calculate_maritime_metrics(
        self, 
        targets: np.ndarray, 
        predictions: np.ndarray
    ) -> Dict[str, float]:
        """Calculate maritime-specific detection metrics"""
        
        # Ship detection rate (recall for ship class)
        ship_mask = targets == 1
        if np.sum(ship_mask) > 0:
            ship_detection_rate = np.sum(predictions[ship_mask] == 1) / np.sum(ship_mask)
        else:
            ship_detection_rate = 0.0
        
        # False alarm rate (false positives per image)
        total_images = len(targets)
        false_positives = np.sum((targets == 0) & (predictions == 1))
        false_alarm_rate = false_positives / total_images
        
        # Dark vessel detection efficiency
        # (Assuming dark vessels are harder to detect)
        if hasattr(self, 'accumulated_confidences') and self.accumulated_confidences:
            confidences = np.array(self.accumulated_confidences)
            high_confidence_detections = np.sum((predictions == 1) & (confidences > 0.8))
            total_detections = np.sum(predictions == 1)
            detection_confidence_rate = high_confidence_detections / total_detections if total_detections > 0 else 0.0
        else:
            detection_confidence_rate = 0.0
        
        return {
            'ship_detection_rate': ship_detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'detection_confidence_rate': detection_confidence_rate
        }
    
    def plot_confusion_matrix(
        self, 
        save_path: Optional[str] = None,
        normalize: bool = False
    ) -> plt.Figure:
        """Plot confusion matrix"""
        
        if not self.accumulated_predictions:
            print("No accumulated predictions to plot")
            return None
        
        predictions = np.array(self.accumulated_predictions)
        targets = np.array(self.accumulated_targets)
        
        cm = confusion_matrix(targets, predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names[:cm.shape[1]],
            yticklabels=self.class_names[:cm.shape[0]]
        )
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_roc_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot ROC curve for binary classification"""
        
        if self.num_classes != 2 or not self.accumulated_probabilities:
            print("ROC curve only available for binary classification with probabilities")
            return None
        
        targets = np.array(self.accumulated_targets)
        probs = np.array(self.accumulated_probabilities)
        
        if len(probs.shape) > 1 and probs.shape[1] == 2:
            probs = probs[:, 1]  # Take positive class probability
        
        fpr, tpr, thresholds = roc_curve(targets, probs)
        auc_score = roc_auc_score(targets, probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_precision_recall_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot Precision-Recall curve"""
        
        if self.num_classes != 2 or not self.accumulated_probabilities:
            print("PR curve only available for binary classification with probabilities")
            return None
        
        targets = np.array(self.accumulated_targets)
        probs = np.array(self.accumulated_probabilities)
        
        if len(probs.shape) > 1 and probs.shape[1] == 2:
            probs = probs[:, 1]  # Take positive class probability
        
        precision, recall, thresholds = precision_recall_curve(targets, probs)
        ap_score = average_precision_score(targets, probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {ap_score:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add baseline (random classifier)
        baseline = np.sum(targets) / len(targets)
        plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline (AP = {baseline:.3f})')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()


def calculate_map(
    predictions: List[int],
    targets: List[int], 
    confidences: List[float],
    iou_thresholds: List[float] = [0.5, 0.75, 0.9]
) -> float:
    """
    Calculate Mean Average Precision (mAP) for detection
    Simplified version for classification-style detection
    """
    
    if not confidences:
        # If no confidences provided, use simple accuracy
        return accuracy_score(targets, predictions)
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)
    confidences = np.array(confidences)
    
    # For binary classification, calculate AP
    if len(np.unique(targets)) == 2:
        try:
            ap = average_precision_score(targets, confidences)
            return ap
        except ValueError:
            return accuracy_score(targets, predictions)
    
    # For multiclass, calculate weighted AP
    aps = []
    for class_id in np.unique(targets):
        binary_targets = (targets == class_id).astype(int)
        if len(np.unique(binary_targets)) == 2:
            ap = average_precision_score(binary_targets, confidences)
            aps.append(ap)
    
    return np.mean(aps) if aps else accuracy_score(targets, predictions)


def calculate_detection_metrics(
    predictions: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate object detection metrics (if bounding boxes available)
    This is a placeholder for full object detection evaluation
    """
    
    # This would implement full object detection metrics
    # including IoU-based matching, precision-recall curves, etc.
    # For now, return classification metrics
    
    metrics = {}
    
    if 'classifications' in predictions and 'classifications' in ground_truth:
        pred_classes = predictions['classifications']
        true_classes = ground_truth['classifications']
        
        metrics['accuracy'] = accuracy_score(true_classes, pred_classes)
        metrics['precision'] = precision_score(true_classes, pred_classes, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(true_classes, pred_classes, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(true_classes, pred_classes, average='weighted', zero_division=0)
    
    return metrics


if __name__ == "__main__":
    # Test metrics calculation
    metrics_calculator = DetectionMetrics(num_classes=2, class_names=['No Ship', 'Ship'])
    
    # Generate test data
    np.random.seed(42)
    test_targets = np.random.randint(0, 2, 1000)
    test_predictions = np.random.randint(0, 2, 1000)
    test_probabilities = np.random.random((1000, 2))
    test_probabilities = test_probabilities / test_probabilities.sum(axis=1, keepdims=True)
    
    # Accumulate test data
    for i in range(0, 1000, 100):
        batch_targets = torch.tensor(test_targets[i:i+100])
        batch_predictions = torch.tensor(test_predictions[i:i+100])
        batch_probabilities = torch.tensor(test_probabilities[i:i+100])
        
        metrics_calculator.accumulate_batch(
            batch_predictions, batch_targets, batch_probabilities
        )
    
    # Calculate metrics
    epoch_metrics = metrics_calculator.calculate_epoch_metrics()
    
    print("Test Metrics:")
    for key, value in epoch_metrics.items():
        if key != 'confusion_matrix':
            print(f"{key}: {value:.4f}")
    
    print("\nMetrics calculation test completed successfully!")
