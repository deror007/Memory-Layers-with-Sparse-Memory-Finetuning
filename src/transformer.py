"""
TODO: Here we will use PyTorch's transformer decoder layers to 
build the transformer architecture.

TODO: Integrate the memory layer using the same Pytorch transformer implementation.

TODO: Make a transformer model from scratch. No pre-built transformer layer.
"""

import torch
import torch.nn as nn
# import torch.nn.functional as F

# NOTE: Some transformer layer parameter values for future reference

#         self,
#         vocab_size=50257, # GPT-2 tokenizer vocab size
#         d_model=384,
#         n_layers=6,
#         n_heads=6,
#         d_ff=1536,
#         max_seq_len=512,
#         dropout=0.1,
#         tie_weights=True,



# NOTE: NOT FINALISED ARCHITECTURE!
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.ln1(x + attn_output)
        ff_output = self.ff(x)
        x = self.ln2(x + ff_output)
        return x