import torch
import torch.nn as nn
import torch.nn.functional as F


class RedistributionRefiner(nn.Module):
    def __init__(self, hidden_dim, num_iterations=1, activation=nn.ReLU()):
        """
        Args:
            hidden_dim: feature dimension of each token
            num_iterations: number of refinement steps (LT in the paper)
            activation: nonlinearity σ in the update rule
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.activation = activation

        # Learnable transformation matrices
        self.Wc1 = nn.Linear(hidden_dim, hidden_dim)
        self.Wc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, r, e, idx_unbiased, idx_biased):
        """
        Args:
            r: original token representations, shape (B, N, d)
            e: correlation matrix, shape (B, N, N)
            idx_unbiased: indices of unbiased tokens, shape (B, k)
            idx_biased: indices of biased tokens, shape (B, N-k)

        Returns:
            z: refined token representations, shape (B, k, d)
        """
        B, N, d = r.shape
        device = r.device

        # Iterative refinement
        for _ in range(self.num_iterations):
            z = []
            for b in range(B):
                ru = r[b, idx_unbiased[b]]  # shape: (k, d)
                rb = r[b, idx_biased[b]]  # shape: (N-k, d)
                eb = e[b][idx_biased[b]][:, idx_unbiased[b]]  # (N-k, k)

                # Normalize adjacency matrix (optional)
                A = eb  # no normalization for simplicity

                # Propagate information from ru to rb
                propagated = torch.matmul(A, self.Wc2(ru))  # (N-k, d)

                # Update biased tokens using redistribution rule
                updated_biased = self.activation(self.Wc1(rb) + propagated)  # (N-k, d)

                # Merge updated biased tokens and original unbiased tokens
                merged = torch.cat([ru, updated_biased], dim=0)  # (N, d)

                # Sort back by original order (optional)
                z.append(merged[:len(idx_unbiased[b])])  # Keep top-ρ^{LT} tokens only

            # Stack batch
            r = torch.stack(z, dim=0)

        return r  # Final refined representation z (B, k, d)
