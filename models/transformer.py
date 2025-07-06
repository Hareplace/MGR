# transformer.py
import torch
import torch.nn as nn
from einops import rearrange

# Basic Transformer Encoder using PyTorch's built-in encoder layers
class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim=768, num_layers=6, num_heads=8, dropout=0.1):
        super().__init__()
        # A single encoder block with self-attention and feedforward network
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,              # feature dimension
            nhead=num_heads,                 # number of attention heads
            dim_feedforward=hidden_dim * 4,  # hidden size in feedforward layer
            dropout=dropout,
            batch_first=True                 # [B, N, D] input format
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [B, N, d] token sequence
        return self.encoder(x)  # output: [B, N, d]


# The main multimodal transformer that fuses vision, audio, and text tokens
class MultimodalTransformer(nn.Module):
    def __init__(self, hidden_dim=768, num_layers=6, num_heads=8):
        super().__init__()
        self.encoder = TransformerEncoder(hidden_dim, num_layers, num_heads)

    def forward(self, x_dict):
        # x_dict: dict containing tokens for each modality
        # Example: {'V': [B, N_v, d], 'A': [B, N_a, d], 'T': [B, N_t, d]}

        # Concatenate all modality tokens into a single sequence
        x_cat = torch.cat([x_dict[mod] for mod in ['V', 'A', 'T']], dim=1)  # [B, N, d]

        # Apply the transformer to the concatenated sequence
        h = self.encoder(x_cat)  # [B, N, d]

        # Get number of tokens for each modality
        N_v, N_a, N_t = x_dict['V'].shape[1], x_dict['A'].shape[1], x_dict['T'].shape[1]

        # Slice the output back to per-modality representations
        h_V = h[:, :N_v, :]
        h_A = h[:, N_v:N_v+N_a, :]
        h_T = h[:, N_v+N_a:, :]

        # Return full sequence and each modality-specific output
        return h, {'V': h_V, 'A': h_A, 'T': h_T}


# Simple transformer for unimodal inputs (no need to slice output)
class UnimodalTransformer(nn.Module):
    def __init__(self, hidden_dim=768, num_layers=6, num_heads=8):
        super().__init__()
        self.encoder = TransformerEncoder(hidden_dim, num_layers, num_heads)

    def forward(self, tokens):
        # tokens: [B, N, d] input for a single modality
        return self.encoder(tokens)  # [B, N, d]
