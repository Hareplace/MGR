# mgr.py

import torch
import torch.nn as nn

from models.transformer import MultimodalTransformer, UnimodalTransformer
from models.mixed_prompt import MixedPromptModule
from models.bias_identification import BiasIdentifier
from models.redistribution import RedistributionRefiner
from models.head import ClassificationHead
from models.loss import compute_loss
from models.tokenizer import ModalityEmbedder, VisionTokenizer, AudioTokenizer, TextTokenizer

class MGR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config['hidden_dim']
        self.num_tokens_dict = config['num_tokens_dict']
        self.alpha_u = config['alpha_u']
        self.alpha_f = config['alpha_f']
        self.keep_ratio = config['keep_ratio']
        self.iterations = config['iterations']
        self.num_classes = config['num_classes']

        # Tokenizers
        self.tokenizers = {
            'V': VisionTokenizer(hidden_dim=self.hidden_dim),
            'A': AudioTokenizer(hidden_dim=self.hidden_dim),
            'T': TextTokenizer(hidden_dim=self.hidden_dim)
        }

        # Modality Embedder (learnable embeddings for availability/absence)
        self.embedder = ModalityEmbedder(self.num_tokens_dict, self.hidden_dim)

        # Transformers
        self.mm_transformer = MultimodalTransformer(self.hidden_dim)
        self.um_transformers = nn.ModuleDict({
            'V': UnimodalTransformer(self.hidden_dim),
            'A': UnimodalTransformer(self.hidden_dim),
            'T': UnimodalTransformer(self.hidden_dim)
        })

        # Mixed prompt module
        self.mixed_prompt = MixedPromptModule(self.hidden_dim)

        # Bias identifier and redistribution
        self.bias_identifier = BiasIdentifier(self.hidden_dim, self.alpha_f, self.keep_ratio)
        self.redistributor = RedistributionRefiner(self.hidden_dim)

        # Task head
        self.head = ClassificationHead(self.hidden_dim, self.num_classes)

    def forward(self, x_dict, modality_mask, labels=None):
        """
        Args:
            x_dict: Dict with modality raw inputs, keys: V, A, T
            modality_mask: Dict with binary indicators (1=available, 0=missing)
            labels: Optional, for supervised loss
        Returns:
            output: Dict with 'logits' and optional 'loss' and 'loss_dict'
        """
        # Step 1: Tokenize inputs and apply modality-specific embeddings
        token_dict = self.embedder(x_dict, modality_mask, self.tokenizers)
        tokens_cat = torch.cat([token_dict[mod] for mod in ['V', 'A', 'T']], dim=1)

        # Step 2: Multimodal Transformer
        h = self.mm_transformer(tokens_cat)

        # Step 3: Split h by modality
        h_dict = {}
        start = 0
        for mod in ['V', 'A', 'T']:
            length = token_dict[mod].shape[1]
            h_dict[mod] = h[:, start:start + length]
            start += length

        # Step 4: Unimodal enhancement
        h_e_dict = {}
        hu_dict = {}
        for mod in ['V', 'A', 'T']:
            if modality_mask[mod]:
                hu = self.um_transformers[mod](token_dict[mod])
                hu_dict[mod] = hu
                h_e_dict[mod] = h_dict[mod] + self.alpha_u * hu
            else:
                h_e_dict[mod] = h_dict[mod]
        h_e = torch.cat([h_e_dict[mod] for mod in ['V', 'A', 'T']], dim=1)

        # Step 5: Iterative context perception, bias identification and redistribution
        for _ in range(self.iterations):
            r, e = self.mixed_prompt(h_e, modality_mask)
            r, idxu, idxb = self.bias_identifier(r, e)
            h_e = self.redistributor(r, e, idxu, idxb)

        # Step 6: Classification head
        logits = self.head(h_e)

        # Step 7: Output
        if labels is not None:
            loss, loss_dict = compute_loss(logits, labels, h, hu_dict, modality_mask, lambda_div=1.0)
            return {'logits': logits, 'loss': loss, 'loss_dict': loss_dict}
        else:
            return {'logits': logits}

