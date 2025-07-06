import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, z):
        """
        Args:
            z: refined features from MGR, shape (B, N, d)
        Returns:
            logits: class logits for each sample, shape (B, num_classes)
        """
        # Pool the token features (e.g., mean pooling)
        pooled = z.mean(dim=1)  # shape: (B, d)

        h = F.relu(self.fc1(pooled))
        h = F.relu(self.fc2(h))
        logits = self.classifier(h)  # shape: (B, num_classes)

        return logits
