from __future__ import annotations

import torch
from einops import einsum, rearrange
from jaxtyping import Float, Int
from torch import Tensor, nn

from cs336_basics.functional import scaled_dot_product_attention, silu


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.empty(d_out, d_in))

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        rms_inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms_inv * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        gate = silu(self.w1(x))
        up = self.w3(x)
        return self.w2(gate * up)


class RoPE(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int) -> None:
        super().__init__()
        self.d_k = d_k
        half = d_k // 2
        i = torch.arange(half, dtype=torch.float32)
        inv_freq = theta ** (-2 * i / d_k)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = positions.unsqueeze(-1) * inv_freq
        # Buffers move with .to(device) but aren't trainable params.
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
    ) -> Float[Tensor, " ... sequence_length d_k"]:
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack([out_even, out_odd], dim=-1).flatten(-2)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        max_seq_len: int | None = None,
        theta: float | None = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

        self.use_rope = use_rope
        if use_rope:
            assert max_seq_len is not None and theta is not None
            self.rope = RoPE(self.d_head, theta, max_seq_len)

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        seq_len = x.shape[-2]

        # Stack Q/K/V projections into one matmul.
        W_qkv = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
        qkv = einsum(x, W_qkv, "... s d_in, d_out d_in -> ... s d_out")
        Q, K, V = qkv.chunk(3, dim=-1)

        Q = rearrange(Q, "... s (h d) -> ... h s d", h=self.num_heads)
        K = rearrange(K, "... s (h d) -> ... h s d", h=self.num_heads)
        V = rearrange(V, "... s (h d) -> ... h s d", h=self.num_heads)

        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(Q, K, V, mask=mask)

        out = rearrange(out, "... h s d -> ... s (h d)")
        return self.o_proj(out)
