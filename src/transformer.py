"""
TODO: Here we will use PyTorch's transformer decoder layers to 
build the transformer architecture.

TODO: Integrate the memory layer using the same Pytorch transformer implementation.

TODO: Make a transformer model from scratch. No pre-built transformer layer.
"""

import torch
import torch.nn as nn

# NOTE: Maybe be unneccessary but is a good placeholder for future customizations.
# THIS IS THE ATTENTION IS ALL YOU NEED PAPER IMPLEMENTATION
TransformerDecoderLayer = nn.TransformerDecoderLayer