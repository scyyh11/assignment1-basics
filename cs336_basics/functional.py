from __future__ import annotations

import torch
from einops import einsum
from jaxtyping import Bool, Float
from torch import Tensor


def softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    x = x - x.max(dim=dim, keepdim=True).values
    exp = torch.exp(x)
    return exp / exp.sum(dim=dim, keepdim=True)


def silu(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return x * torch.sigmoid(x)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") * d_k**-0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = softmax(scores, dim=-1)
    return weights @ V
