import torch
import math

from torch import Tensor
from einops import einsum, reduce

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.G = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        You should upcast your input to torch.float32 to prevent overflow when you square the input.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(reduce(x**2, '... d_model -> ... 1', 'sum') / self.d_model + self.eps)
        result = (x / rms) * self.G
        # Return the result in the original dtype
        return result.to(in_dtype)