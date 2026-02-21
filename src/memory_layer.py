import torch.nn as nn
import torch

class MemoryPlusLayer(nn.Module):

    def __init__(self, d_model, memory_slots, top_k = 32):
        # Define your memory mechanism here
        # Using Berges et al. (2024) "Memory Layers at scale" as a reference for the memory layer design
        
        super().__init__()

        self.key_dim = d_model // 2
        self.subkey_dim = self.key_dim // 2
        self.value_dim = d_model # <-- NOTE: May experiment with this value, as it may affect performance and memory usage.
        
        # Total memory_slots = |C| * |C'|. Sub-key matrices have sqrt(memory_slots) rows.
        self.num_subkeys = int(torch.sqrt(memory_slots))
        assert self.num_subkeys ** 2 == memory_slots, "memory_slots must be a perfect square (e.g., 1024^2)"

        # Query MLP
        self.query = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(), # <-- Should match whatever the base models FFN activation function is.
            nn.Linear(d_model * 4, self.subkey_dim)
        )
        
        # Sub-Key Matrix One and Two
        # NOTE: Don't use nn.linear here, due to sparse key retrieval mechanism in forward pass.
        self.subkey_one = nn.Parameter(torch.empty(self.num_subkeys, self.subkey_dim))
        self.subkey_two = nn.Parameter(torch.empty(self.num_subkeys, self.subkey_dim))
        nn.init.uniform_(self.subkey_one, a = -1, b = 1)
        nn.init.uniform_(self.subkey_two, a = -1, b = 1)

        # Value Matrix
        self.values = nn.Parameter(torch.empty(memory_slots, self.value_dim))
        nn.init.normal_(self.values, std=0.02)  # apparently from lample et al 2019, CAN't FIND ITS REFERENCE

        # Weight Matrix One
        self.W1 = nn.Linear(d_model, self.value_dim, bias=False)

        # Weight Matrix Two
        self.W2 = nn.Linear(self.value_dim, d_model, bias=False)

        # Silu Activation Function
        self.silu = nn.SiLU()

        # QK-Normalisation, 
        # NOTE:I think its more a general backbone design choice for memory layer, potentially place this after residual connection as we are using interleaved architecture (at end of this gated memory layer)
        """
        NOTE: This is a technique used to stabilize training and improve convergence in transformer models. 
        """
        self.qk_norm = nn.RMSNorm() 
        
        # Top-K Selection
        """
        NOTE: Can experiment with this value, as it may affect performance and memory usage. 
        """
        self.top_k = top_k

        # Softmax
        self.softmax = nn.Softmax(dim=-1)


    def lookup_memory(self, query):
            # TODO: Compute Product Key Mechanism and retrieval

            k1 = self.subkey_one()
            k1 = self.qk_norm(k1)

            k2 = self.subkey_two()
            k2 = self.qk_norm(k2)

            k1q = query @ k1.T
            k2q = query @ k2.T

            kq = k1q + k2q

                # TODO: Top-k over kq

                # TODO: Softmax over top-k kq = s

                # TODO: s @ values of top-k = y

    def forward(self, x):

        q = self.query(x)
        q = self.qk_norm(q)

        y = self.lookup_memory(q)
                
        m_plus = self.silu(self.W1(x))
        m_plus = y * m_plus
        m_plus = self.W2(m_plus)


       
class gated_memory_layer(nn.Module):
    def __init__(self):
        super().__init__()
        pass