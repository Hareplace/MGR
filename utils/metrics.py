# metric.py

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(logits, labels, average='macro'):
    """
    Computes classification metrics from logits and labels.

    Args:
        logits (Tensor): [B, C] predicted logits
        labels (Tensor): [B] ground truth labels
        average (str): averaging method for multi-class ('macro', 'micro', 'weighted')

    Returns:
        dict: accuracy, f1, precision, recall
    """
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average=average)
    precision = precision_score(labels, preds, average=average)
    recall = recall_score(labels, preds, average=average)

    return {
        'accuracy': acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }
