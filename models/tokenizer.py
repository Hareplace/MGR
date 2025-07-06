# tokenizer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionTokenizer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, hidden_dim=768):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, num_patches, hidden_dim]
        x = self.patch_embed(x)  # [B, hidden_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_dim]
        return x

class AudioTokenizer(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=768):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # x: [B, T, input_dim] -> [B, T, hidden_dim]
        return self.linear(x)

class TextTokenizer(nn.Module):
    def __init__(self, vocab_size=30522, hidden_dim=768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, x):
        # x: [B, seq_len] -> [B, seq_len, hidden_dim]
        return self.embedding(x)

class ModalityEmbedder(nn.Module):
    def __init__(self, num_tokens_dict, hidden_dim):
        super().__init__()
        self.E_miss = nn.ParameterDict({
            k: nn.Parameter(torch.randn(v, hidden_dim)) for k, v in num_tokens_dict.items()
        })
        self.E_avail = nn.ParameterDict({
            k: nn.Parameter(torch.randn(v, hidden_dim)) for k, v in num_tokens_dict.items()
        })

    def forward(self, x_dict, modality_mask, tokenizer_dict):
        # x_dict: {'V': img, 'A': audio, 'T': text} (some may be None)
        # modality_mask: e.g., {'V': 1, 'A': 0, 'T': 1} (1 for available, 0 for missing)
        token_outputs = {}
        for mod in ['V', 'A', 'T']:
            if modality_mask[mod]:
                tokens = tokenizer_dict[mod](x_dict[mod])  # B x Ni x d
                tokens = tokens + self.E_avail[mod]  # add availability embedding
            else:
                # Repeat E_miss to match batch size
                batch_size = x_dict[[k for k in x_dict if x_dict[k] is not None][0]].shape[0]
                tokens = self.E_miss[mod].unsqueeze(0).repeat(batch_size, 1, 1)
            token_outputs[mod] = tokens
        return token_outputs
