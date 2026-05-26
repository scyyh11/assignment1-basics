from __future__ import annotations
from collections.abc import Iterable

import torch
from einops import einsum
from jaxtyping import Bool, Float, Int
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
    return einsum(weights, V, "... queries keys, ... keys d_v -> ... queries d_v")


def cross_entropy(
    inputs: Float[Tensor, "... vocab_size"], 
    targets: Int[Tensor, ...]
) -> Float[Tensor, ""]:
    max_val = inputs.max(dim=-1,keepdim=True).values
    shifted = inputs - max_val
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))
    target_shifted = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - target_shifted).mean()


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], 
    max_l2_norm: float
) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))

    eps = 1e-6
    coef = max_l2_norm / (total_norm + eps)
    coef = torch.clamp(coef, max=1.0)
    for g in grads:
        g.mul_(coef)

