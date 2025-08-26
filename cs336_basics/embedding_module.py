import torch
import math

from torch import Tensor
from einops import einsum, rearrange

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.E = torch.nn.Parameter(
            data=torch.nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype), 
                                             mean=0.0, std=1, a=-3, b=3),
            requires_grad=True,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        """
            The forward method should select the embedding vector for each token ID by indexing into an embedding matrix of shape (vocab_size, d_model) using a
            torch.LongTensor of token IDs with shape (batch_size, sequence_length)
        """
        # batch_size, _ = token_ids.shape
        # token_ids_flattened = rearrange(token_ids, "b seq_len -> (b seq_len)")
        # embeddings = self.E[token_ids_flattened]
        # return rearrange(embeddings, "(b seq_len) d_model -> b seq_len d_model", b=batch_size)
        # Don't need above code, since torch already supports Direct indexing works with multi-dimensional indices
        return self.E[token_ids]