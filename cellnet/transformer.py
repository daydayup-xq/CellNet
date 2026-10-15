from __future__ import annotations
from typing import Dict, Optional
import torch
from torch import nn
from torch.nn import functional as F


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


def _rotate_half(x: torch.Tensor, half: bool = False) -> torch.Tensor:
    if half:
        first, second = x.chunk(2, dim=-1)
        return torch.cat([-second, first], dim=-1)
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def _apply_rot_embed_cat(
    x: torch.Tensor, rope: torch.Tensor, half: bool = False
) -> torch.Tensor:
    rope_dim = rope.shape[-1] // 2
    sin_emb, cos_emb = (rope[..., :rope_dim], rope[..., rope_dim:])
    while sin_emb.dim() < x.dim():
        sin_emb = sin_emb.unsqueeze(1)
        cos_emb = cos_emb.unsqueeze(1)
    return x * cos_emb + _rotate_half(x, half=half) * sin_emb


def _drop_path_packed(
    x: torch.Tensor, drop_prob: float, training: bool
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    random_tensor = keep_prob + torch.rand(1, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class PackedDropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path_packed(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self._qk_monitor: Optional[Dict[str, torch.Tensor]] = None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        rope: Optional[torch.Tensor] = None,
        rotate_half: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if rope is not None:
            rope = rope.to(dtype=q.dtype)
            q = _apply_rot_embed_cat(q, rope, half=rotate_half).type_as(v)
            k = _apply_rot_embed_cat(k, rope, half=rotate_half).type_as(v)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1.0,
        qkv_bias: bool = True,
        norm_eps: float = 1e-06,
        rope_rotate_half: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias)
        self.ls1 = LayerScale(dim, layer_scale_init_value)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = MLP(dim, mlp_ratio)
        self.ls2 = LayerScale(dim, layer_scale_init_value)
        self.drop_path = PackedDropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.rope_rotate_half = rope_rotate_half

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        rope: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.drop_path(
            self.ls1(
                self.attn(
                    self.norm1(x),
                    attn_mask,
                    rope=rope,
                    rotate_half=self.rope_rotate_half,
                )
            )
        )
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x
