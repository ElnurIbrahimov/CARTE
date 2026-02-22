"""
TRM building blocks: RMSNorm, SwiGLU, RoPE, Attention, Block, ReasoningModule.

Follows Samsung TRM paper specs:
- hidden_size=512, 8 heads, SwiGLU expansion=4, bfloat16
- Post-norm (norm after residual add)
- Non-causal attention (puzzles are bidirectional)
- ReasoningModule = 2 Blocks, shared across all recursion steps
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: gate branch with SiLU activation."""

    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        d_ff = d_model * expansion
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, d_head: int, max_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, max_len: int):
        t = torch.arange(max_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        seq_len = x.shape[-2]
        if offset + seq_len > self.cos_cache.shape[0]:
            self._build_cache(offset + seq_len)
        cos = self.cos_cache[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
        # Rotate pairs
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin


class Attention(nn.Module):
    """Multi-head attention with RoPE. Non-causal (bidirectional)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryPositionalEncoding(self.d_head)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, D_h]
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.rope(q)
        k = self.rope(k)

        # Scaled dot-product attention (non-causal)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class Block(nn.Module):
    """Transformer block: Attention + SwiGLU FFN with post-norm."""

    def __init__(self, d_model: int, n_heads: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = Attention(d_model, n_heads, dropout)
        self.ffn = SwiGLU(d_model, expansion)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Post-norm: add then norm
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x


class ReasoningModule(nn.Module):
    """
    Shared 2-layer transformer used for both z_L and z_H updates.
    This is the TRM's L_level function — called repeatedly in the recursive loop.
    """

    def __init__(self, d_model: int, n_heads: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.block1 = Block(d_model, n_heads, expansion, dropout)
        self.block2 = Block(d_model, n_heads, expansion, dropout)

    def forward(self, z_target: torch.Tensor, z_condition: torch.Tensor) -> torch.Tensor:
        """
        Update z_target conditioned on z_condition.
        TRM: z_L = L_level(z_L, z_H + input_emb) or z_H = L_level(z_H, z_L)

        Args:
            z_target: tensor to update [B, S, D]
            z_condition: conditioning tensor [B, S, D]
        Returns:
            updated z_target [B, S, D]
        """
        x = z_target + z_condition
        x = self.block1(x)
        x = self.block2(x)
        return x
