# mixed_prompt.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedPromptModule(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        # Learnable modality-specific prompt embeddings (p_V, p_A, p_T)
        self.prompt_embeddings = nn.ParameterDict({
            mod: nn.Parameter(torch.randn(hidden_dim)) for mod in ['V', 'A', 'T']
        })

        # Context estimator fΦ: average pooling over tokens
        self.context_pooler = nn.AdaptiveAvgPool1d(1)

        # Context mixing function fMix: a learnable MLP
        self.context_mixer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Attention parameters (multi-head attention)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.W_O = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, he, modality_mask):
        """
        he: [B, N, d] - enhanced tokens after unimodal generalization
        modality_mask: List of available modalities ['V', 'A'] for example
        """
        B, N, d = he.size()

        # Step 1: Gather available prompts
        available_prompts = [self.prompt_embeddings[m] for m in modality_mask]  # each [d]
        prompt_stack = torch.stack(available_prompts, dim=0).mean(dim=0)  # [d]
        prompt_stack = prompt_stack.unsqueeze(0).expand(B, -1)  # [B, d]

        # Step 2: Pool contextual info from he
        pooled_context = self.context_pooler(he.transpose(1, 2)).squeeze(-1)  # [B, d]

        # Step 3: fMix(prompt + pooled_context)
        mixed_input = torch.cat([prompt_stack, pooled_context], dim=-1)  # [B, 2d]
        mixed_prompt = self.context_mixer(mixed_input)  # [B, d]

        # Step 4: Append prompt as new token
        prompt_token = mixed_prompt.unsqueeze(1)  # [B, 1, d]
        hp = torch.cat([he, prompt_token], dim=1)  # [B, N+1, d]

        # Step 5: Self-attention over [he || P]
        Q = self.W_Q(hp)  # [B, N+1, d]
        K = self.W_K(hp)  # [B, N+1, d]
        V = self.W_V(hp)  # [B, N+1, d]

        # Reshape for multi-head attention
        Q = Q.view(B, N+1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N+1, d/h]
        K = K.view(B, N+1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N+1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, h, N+1, N+1]
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Final output r
        r_heads = torch.matmul(attn_weights, V)  # [B, h, N+1, d/h]
        r = r_heads.transpose(1, 2).contiguous().view(B, N+1, d)  # [B, N+1, d]
        r = self.W_O(r)  # [B, N+1, d]

        # Return only the original token part and correlation matrix
        return r[:, :-1, :], attn_weights.mean(dim=1)[:, :-1, :-1]  # r [B, N, d], e [B, N, N]
