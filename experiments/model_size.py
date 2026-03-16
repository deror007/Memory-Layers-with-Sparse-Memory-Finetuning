import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import math

import torch.nn as nn
import torch
import torch.nn.functional as F

from src.memory_layer import MemoryPlusLayer

import timm


device = torch.device("mps")
model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10, cache_dir = "./models_dir").to(device)
d_model = model.embed_dim
memory_slots = 256**2
model.blocks[6].mlp = MemoryPlusLayer(d_model=d_model, memory_slots=memory_slots).to(device)



model.load_state_dict(torch.load("./models_dir/tiny_vit_memory_plus.pt", map_location=torch.device("mps")))

for name, param in model.named_parameters():
    print(f"{name:40} | {param.numel():>10} params | {param.element_size() * param.numel() / 1e6:.2f} MB")



"""
Memory layer MB sizes:

blocks.6.norm1.weight                    |        192 params | 0.00 MB
blocks.6.norm1.bias                      |        192 params | 0.00 MB
blocks.6.attn.qkv.weight                 |     110592 params | 0.44 MB
blocks.6.attn.qkv.bias                   |        576 params | 0.00 MB
blocks.6.attn.proj.weight                |      36864 params | 0.15 MB
blocks.6.attn.proj.bias                  |        192 params | 0.00 MB
blocks.6.norm2.weight                    |        192 params | 0.00 MB
blocks.6.norm2.bias                      |        192 params | 0.00 MB
blocks.6.mlp.subkey_one                  |      12288 params | 0.05 MB
blocks.6.mlp.subkey_two                  |      12288 params | 0.05 MB
blocks.6.mlp.values                      |   12582912 params | 50.33 MB
blocks.6.mlp.query.0.weight              |     147456 params | 0.59 MB
blocks.6.mlp.query.0.bias                |        768 params | 0.00 MB
blocks.6.mlp.query.2.weight              |      73728 params | 0.29 MB
blocks.6.mlp.query.2.bias                |         96 params | 0.00 MB
blocks.6.mlp.W1.weight                   |      36864 params | 0.15 MB
blocks.6.mlp.W2.weight                   |      36864 params | 0.15 MB
blocks.6.mlp.qk_norm.weight              |         48 params | 0.00 MB

TODO: Ablation study on value matrix sizes.

"""