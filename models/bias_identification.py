import torch
import torch.nn as nn
import torch.nn.functional as F

class BiasIdentifier(nn.Module):
    def __init__(self, hidden_dim, alpha_f=0.5, keep_ratio=0.5):
        """
        Args:
            hidden_dim: dimensionality of token features (d)
            alpha_f: weight balancing feature-level and correlation-level bias estimation
            keep_ratio: top-ρ ratio of tokens to keep as unbiased
        """
        super().__init__()
        self.alpha_f = alpha_f
        self.keep_ratio = keep_ratio

        # Learnable projector fΨ for computing ξ_f from token features r
        self.feature_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # Output scalar per token
        )

    def forward(self, r, e):
        """
        Args:
            r: conditionally enhanced representations, shape (B, N, d)
            e: correlation matrix, shape (B, N, N)

        Returns:
            idx_unbiased: indices of unbiased tokens
            idx_biased: indices of biased tokens
            bias_scores: token-wise bias scores ξ ∈ ℝ^{B,N}
        """
        B, N, _ = r.shape

        # Feature-level ranking ξ_f ∈ ℝ^{B,N}
        xi_f = self.feature_projector(r).squeeze(-1)  # (B, N)

        # Correlation-level ranking ξ_e ∈ ℝ^{B,N}
        xi_e = torch.sum(e, dim=-1)  # Sum across token correlations
        xi_e = torch.tanh(xi_e)      # Nonlinear mapping

        # Final ranking ξ
        xi_f = torch.tanh(xi_f)      # Add tanh here too (optional depending on range)
        xi = self.alpha_f * xi_f + (1 - self.alpha_f) * xi_e

        # Sort indices by descending ξ (high ξ = low bias)
        sorted_idx = torch.argsort(xi, dim=-1, descending=True)

        k = int(self.keep_ratio * N)
        idx_unbiased = sorted_idx[:, :k]   # Top-ρ tokens retained
        idx_biased = sorted_idx[:, k:]     # Remaining tokens discarded

        return idx_unbiased, idx_biased, xi
