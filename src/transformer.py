"""
TODO: Here we will use PyTorch's transformer decoder layers to 
build the transformer architecture.

TODO: Integrate the memory layer using the same Pytorch transformer implementation.

TODO: Make a transformer model from scratch. No pre-built transformer layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=384,
        n_layers=6,
        n_heads=6,
        d_ff=1536,
        max_seq_len=512,
        dropout=0.1,
        tie_weights=True,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.head.weight = self.token_emb.weight

    def forward(self, input_ids):
        B, T = input_ids.size()
        positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(input_ids.device)
        for layer in self.layers:
            x = layer(x, memory=None, tgt_mask=tgt_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# Example usage:
# model = DecoderOnlyTransformer(vocab_size=50257)