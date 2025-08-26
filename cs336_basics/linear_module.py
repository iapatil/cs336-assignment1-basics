import torch
import math

from torch import Tensor
from einops import einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        sigma_sq = 2 / (in_features + out_features)
        sigma = math.sqrt(sigma_sq)
        self.W = torch.nn.Parameter(
            data=torch.nn.init.trunc_normal_(torch.empty(out_features, in_features, device=device, dtype=dtype), 
                                             mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma),
            requires_grad=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        return einsum(x, self.W, "... in_features, out_features in_features -> ... out_features")


# module = Linear(512, 32)
# print(module.state_dict().keys())