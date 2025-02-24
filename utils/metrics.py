"""Evaluation metrics for deepfake detection."""
from typing import Tuple

import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import torch


def calculate_eer(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Calculate Equal Error Rate (EER) and threshold."""
    # Check if all samples belong to one class
    if len(np.unique(y_true)) == 1:
        return 0.0, 0.5  # Return default values for single-class case
    
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        fnr = 1 - tpr
        
        # Find threshold where FPR = FNR
        eer_threshold_idx = np.nanargmin(np.absolute(fnr - fpr))
        eer_threshold = thresholds[eer_threshold_idx]
        eer = fpr[eer_threshold_idx]
        
        return eer, eer_threshold
    except:
        return 0.0, 0.5  # Fallback values if calculation fails

def get_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> dict:
    """Calculate EER and AUC metrics with edge case handling.
    
    Args:
        predictions: Model predictions (after softmax)
        labels: Ground truth labels
    
    Returns:
        Dictionary containing EER, AUC and optimal threshold
    """
    # Convert to numpy arrays
    y_true = labels.cpu().numpy()
    y_score = predictions.cpu().numpy()
    
    if y_score.shape[1] == 2:  # If using softmax outputs
        y_score = y_score[:, 1]  # Take probability of positive class
    
    # Handle edge cases
    unique_labels = np.unique(y_true)
    if len(unique_labels) == 1:
        # All samples belong to the same class
        return {
            'eer': 0.0,
            'auc': 1.0 if y_true[0] == 1 else 0.0,
            'threshold': 0.5,
            'warning': f'All samples belong to class {y_true[0]}'
        }
    
    # Calculate metrics
    try:
        eer, eer_threshold = calculate_eer(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
    except Exception as e:
        print(f"Warning: Metric calculation failed - {str(e)}")
        eer, eer_threshold = 0.0, 0.5
        roc_auc = 0.5
    
    return {
        'eer': float(eer * 100),  # Convert to percentage
        'auc': float(roc_auc),
        'threshold': float(eer_threshold)
    }
