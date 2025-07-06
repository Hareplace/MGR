import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationLoss(nn.Module):
    """
    Task-specific classification loss (CrossEntropy).
    """
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.loss_fn(logits, labels)

class DistanceCorrelationLoss(nn.Module):
    """
    Distance correlation loss to enforce diversity between unimodal and multimodal features.
    Adapted from Szekely et al. (2007).
    """
    def __init__(self):
        super(DistanceCorrelationLoss, self).__init__()

    def forward(self, h_u, h_m):
        """
        h_u: [B, d] - unimodal representation
        h_m: [B, d] - multimodal representation
        """
        def pdist(x):
            x_square = (x ** 2).sum(dim=1, keepdim=True)
            dist = x_square + x_square.t() - 2 * x @ x.t()
            dist = torch.sqrt(F.relu(dist) + 1e-8)
            return dist

        def dcov(A, B):
            A_mean = A.mean(dim=0, keepdim=True)
            B_mean = B.mean(dim=0, keepdim=True)
            A_centered = A - A_mean
            B_centered = B - B_mean
            return (A_centered * B_centered).mean()

        a = pdist(h_u)
        b = pdist(h_m)

        A = a - a.mean(dim=0) - a.mean(dim=1, keepdim=True) + a.mean()
        B = b - b.mean(dim=0) - b.mean(dim=1, keepdim=True) + b.mean()

        dcov_xy = (A * B).mean()
        dcov_xx = (A * A).mean()
        dcov_yy = (B * B).mean()

        dcor = dcov_xy / (torch.sqrt(dcov_xx * dcov_yy + 1e-8))
        return 1 - dcor  # 1 - distance correlation (maximize diversity)

class MGRLoss(nn.Module):
    """
    Full MGR loss: L = L_task + λdiv * L_div
    """
    def __init__(self, lambda_div=1.0):
        super(MGRLoss, self).__init__()
        self.task_loss = ClassificationLoss()
        self.div_loss = DistanceCorrelationLoss()
        self.lambda_div = lambda_div

    def forward(self, logits, labels, h_u, h_m):
        """
        logits: [B, C] - classification predictions
        labels: [B] - target labels
        h_u: [B, d] - pooled unimodal features
        h_m: [B, d] - pooled multimodal features
        """
        L_task = self.task_loss(logits, labels)
        L_div = self.div_loss(h_u, h_m)
        L = L_task + self.lambda_div * L_div
        return L, {'L_task': L_task.item(), 'L_div': L_div.item()}


def compute_loss(logits, labels, h_m, hu_dict, modality_mask, lambda_div=1.0):
    """
    Wrapper for loss computation.
    Inputs:
        logits: model predictions
        labels: ground-truth labels
        h_m: [B, N, d] - full multimodal representation
        hu_dict: {mod: [B, N_mod, d]} - unimodal features
        modality_mask: dict like {'V':1, 'A':1, 'T':0}
    """
    mgr_loss = MGRLoss(lambda_div=lambda_div)

    # Pool multimodal features (mean across tokens)
    h_m_pooled = h_m.mean(dim=1)

    # Pool unimodal features (only for available modalities)
    h_u_list = [hu.mean(dim=1) for mod, hu in hu_dict.items() if modality_mask[mod]]
    h_u_mean = torch.stack(h_u_list, dim=0).mean(dim=0)  # [B, d]

    loss, loss_dict = mgr_loss(logits, labels, h_u_mean, h_m_pooled)
    return loss, loss_dict