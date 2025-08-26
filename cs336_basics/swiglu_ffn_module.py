import torch
import math

from torch import Tensor
from einops import einsum, reduce
from .linear_module import Linear

class SwiGLUFFN(torch.nn.Module):
    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        d_ff = int(((d_model * (8 / 3)) // 64) * 64)
        self.linear1 = Linear(d_model, d_ff, device=None, dtype=None)
        self.linear2 = Linear(d_model, d_ff, device=None, dtype=None)
        self.linear3 = Linear(d_ff, d_model, device=None, dtype=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.linear1(x)
        b = self.linear2(x)
        # Apply SiLU to a
        a = torch.sigmoid(a) * a
        out = self.linear3(a * b)
        return out
