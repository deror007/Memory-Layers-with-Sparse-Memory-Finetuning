import torch.nn as nn
import torch

class MemoryPlusLayer(nn.Module):

    def __init__(self, d_model, memory_slots, top_k = 32):
        # Define your memory mechanism here
        # Using Berges et al. (2024) "Memory Layers at scale" as a reference for the memory layer design
        
        super().__init__()

        self.key_dim = d_model // 2
        self.subkey_dim = self.key_dim // 2
        self.value_dim = d_model # <-- May experiment with this value, as it may affect performance and memory usage.
        # ... ALso being kept as d_model kinda makes the attribute redundant, but it is more explicit this way.
        

        # Total memory_slots = |C| * |C'|. Therefore, sub-key matrices have sqrt(memory_slots) rows.
        self.num_subkeys = int(torch.sqrt(memory_slots))
        assert self.num_subkeys ** 2 == memory_slots, "memory_slots must be a perfect square (e.g., 1024^2)"

        # Query MLP
        self.query = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(), # <-- Should match whatever the base models FFN activation function is.
            nn.Linear(d_model * 4, self.key_dim)
        )
        
        # Sub-Key Matrix One and Two
        # NOTE: Don't use nn.linear here, due to sparse key retrieval mechanism in forward pass.
        self.subkey_one = nn.parameter(torch.randn(self.num_subkeys, self.subkey_dim))
        self.subkey_two = nn.parameter(torch.randn(self.num_subkeys, self.subkey_dim))

        # Value Matrix
        self.values = nn.Parameter(torch.empty(memory_slots, self.value_dim))

        # Weight Matrix One
        self.W1 = nn.Linear(d_model, self.value_dim, bias=False)

        # Weight Matrix Two
        self.W2 = nn.Linear(self.value_dim, d_model, bias=False)

        # kaiming initialisation
        # NOTE: this is my own addition to the sub-key design to aid spreading latent information across sub-key spaces.
        nn.init.kaiming_uniform_(self.subkey_one)
        nn.init.kaiming_uniform_(self.subkey_two)
        nn.init.normal_(self.values, std=0.02)  # apparently from lample et al 2019

        # Silu activation function
        self.silu = nn.SiLU()

        # qk-normalisation, I think its more a general backbone design choice for memory layer, maybe tricky for adapter then and consider placing this somewhere
        # NOTE: potentially place this after residual connection as we are using interleaved architecture (at end of this gated memory layer)
        """
        NOTE: This is a technique used to stabilize training and improve convergence in transformer models. 
        """
        self.qk_norm = nn.RMSNorm() 
        

        # top-k selection
        """
        NOTE: Can experiment with this value, as it may affect performance and memory usage. 
        ... A smaller top-k may lead to faster computation but potentially less accurate results, 
        ... while a larger top-k may improve accuracy but increase computational cost.
        """
        self.top_k = top_k

        # 
        

    def forward(self, x):
        # TODO:Implement memory logic here
        return x

class gated_memory_layer(nn.Module):
    def __init__(self):
        super().__init__()
        pass