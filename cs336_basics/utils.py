import torch
from torch import Tensor
from einops import reduce, einsum, rearrange
import math

def softmax(x: Tensor, dim: int) -> Tensor:
    """
        x -> [4, 3]
        x[:, 0]
    """
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max
    x_exp = torch.exp(x_shifted)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    x_softmax = x_exp / x_exp_sum
    return x_softmax

def scaled_dot_product_attention(keys: Tensor, queries: Tensor, values: Tensor, mask: Tensor = None) -> Tensor:
    """
        keys/ queries -> (batch_size, ..., seq_len, d_k)
        values -> (batch_size, ..., seq_len, d_v)
        return an output with the shape (batch_size, ..., d_v)
    """
    q_k_dot = einsum(queries, keys, "... n d_k, ... m d_k -> ... n m")
    pre_softmax = q_k_dot / math.sqrt(keys.shape[-1])
    if mask is not None:
        pre_softmax[..., torch.logical_not(mask)] = float('-inf')
    post_softmax = softmax(pre_softmax, dim=-1)
    output = einsum(post_softmax, values, "... n m, ... m d_v -> ... n d_v")
    return output

def cross_entropy_loss(predicted_logits: Tensor, targets: Tensor) -> float:
    max_logit = reduce(predicted_logits, "... n -> ... 1", "max")
    predicted_logits = predicted_logits - max_logit
    predicted_logits_exp = torch.exp(predicted_logits)
    predicted_logits_exp_sum = reduce(predicted_logits_exp, "... n -> ... 1", "sum")
    predicted_logits_log_exp_sum = torch.log(predicted_logits_exp_sum)

    targets = rearrange(targets, "... -> ... 1")
    target_logits = torch.take_along_dim(predicted_logits, targets, dim=-1)

    loss = -reduce(target_logits - predicted_logits_log_exp_sum, "b ... -> ...", "mean")
    return loss
