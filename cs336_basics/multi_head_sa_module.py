import torch
import math

from torch import Tensor
from einops import einsum, reduce, rearrange
from .linear_module import Linear
from .utils import scaled_dot_product_attention

class MultiHeadSA(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, dtype=None, device=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.Q_proj = Linear(d_model, d_model, dtype=dtype, device=device)
        self.K_proj = Linear(d_model, d_model, dtype=dtype, device=device)
        self.V_proj = Linear(d_model, d_model, dtype=dtype, device=device)
        self.O_proj = Linear(d_model, d_model, dtype=dtype, device=device)
    
    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[-2]
        K, Q, V = self.K_proj(x), self.Q_proj(x), self.V_proj(x)
        K_heads = rearrange(K, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads=self.num_heads)
        Q_heads = rearrange(Q, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads=self.num_heads)
        V_heads = rearrange(V, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads=self.num_heads)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=bool, device=x.device))
        output_heads = scaled_dot_product_attention(K_heads, Q_heads, V_heads, mask=causal_mask)
        output = rearrange(output_heads, "... num_heads seq_len d_head -> ... seq_len (num_heads d_head)")
        return self.O_proj(output)
