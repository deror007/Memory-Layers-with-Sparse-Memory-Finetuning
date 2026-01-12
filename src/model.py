import torch
import torch.nn as nn
from src.transformer import DecoderBlock
from torch.nn import functional as F

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

class DecoderOnlyTransformerStack(nn.Module):
    def __init__(self, vocab_size= 50257, d_model=384, n_layers=6, n_heads=6, d_ff=1536, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        # NOTE: Below Layer norm is not neccessariy with current transformer architecture:
        # self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    
    def forward(self, x, attn_mask=None):
        # TODO: Need to understnad tiny stories dataset better.
        # NOTE: nn.cross-entrophy-loss requires logits, not softmaxed outputs.
        pass





