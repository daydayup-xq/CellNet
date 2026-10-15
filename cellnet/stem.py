"""The deployed CellNet expression / gene identity / mean identity stem."""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class CellNetStem(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config["patch_size"]
        self.grid_size = tuple(n // self.patch_size for n in config["image_size"])
        self.num_patches = math.prod(self.grid_size)
        dim = config["embedding_dim"]
        self.zero_expression_floor = config["zero_expression_floor"]
        self.gene_identity_embed = nn.Embedding(
            config["vocabulary_size"] + 1, 256, padding_idx=0
        )
        self.gene_identity_embed.weight.requires_grad_(False)
        self.gene_projector = nn.Linear(256, 64)
        self.fresh_conv_proj = nn.Conv2d(64, dim, self.patch_size, self.patch_size)
        self.fresh_token_norm = nn.LayerNorm(dim)
        self.mean_gene_projector = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 64))
        self.id_patch_proj = nn.Linear(64, dim)
        self.id_token_norm = nn.LayerNorm(dim)
        self.expr_patch_proj = nn.Conv2d(2, dim, self.patch_size, self.patch_size)
        self.expr_token_norm = nn.LayerNorm(dim)

    def project_gene_ids(self, ids):
        """Unnormalized 64-D gene queries used by the expression decoder."""
        return self.gene_projector(
            self.gene_identity_embed(ids).to(self.gene_projector.weight.dtype)
        )

    def forward(self, x, *, gene_ids, mask=None):
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError(
                "image must be [batch, 2, height, width]: expression and measurement mask"
            )
        if gene_ids is None or gene_ids.shape != x[:, 0].shape:
            raise ValueError("gene_ids must be [batch, height, width]")
        if tuple(x.shape[-2:]) != tuple(n * self.patch_size for n in self.grid_size):
            raise ValueError("image shape differs from the pretrained layout")
        if mask is None:
            mask = x[:, 1:2]
        elif mask.ndim == 3:
            mask = mask[:, None]
        if (
            (gene_ids < 0) | (gene_ids >= self.gene_identity_embed.num_embeddings)
        ).any():
            raise ValueError("gene ID outside the pretrained vocabulary")
        valid = (mask[:, 0] > 0) & (gene_ids > 0)
        visible_ids = torch.where(valid, gene_ids, 0).long()
        expr = x[:, 0] * valid.to(x.dtype)
        # Training-only local jitter moves expression and identity together within patches.
        expr_jittered, visible_ids = self._fresh_local_layout_jitter_expression_gene(
            expr[:, None], visible_ids
        )
        expr = expr_jittered[:, 0]
        # Include padding explicitly so compact row zero is always the sentinel.
        unique, inverse = torch.unique(
            torch.cat([visible_ids.new_zeros(1), visible_ids.flatten()]),
            return_inverse=True,
        )
        compact = inverse[1:].reshape_as(visible_ids)
        rows = self.gene_identity_embed(unique).to(self.gene_projector.weight.dtype)
        table = F.normalize(self.gene_projector(rows).float(), dim=-1, eps=1e-8).to(
            x.dtype
        )
        coefficient = (expr + self.zero_expression_floor) * (visible_ids > 0).to(
            x.dtype
        )
        flat = coefficient.flatten()
        positions = (flat != 0).nonzero(as_tuple=True)[0]
        conv_dtype = (
            torch.get_autocast_dtype(x.device.type)
            if torch.is_autocast_enabled(x.device.type)
            else x.dtype
        )
        semantic = torch.zeros(flat.numel(), 64, device=x.device, dtype=conv_dtype)
        values = table[compact.flatten()[positions]] * flat[positions, None].to(
            table.dtype
        )
        semantic = semantic.index_copy(0, positions, values.to(conv_dtype))
        semantic = semantic.reshape(*expr.shape, 64).permute(0, 3, 1, 2)
        identity_tokens = self.fresh_conv_proj(semantic).flatten(2).transpose(1, 2)
        expr_input = torch.stack([expr, (visible_ids > 0).to(x.dtype)], dim=1)
        expression_tokens = self.expr_patch_proj(expr_input).flatten(2).transpose(1, 2)
        batch = x.shape[0]
        gh, gw = self.grid_size
        p = self.patch_size
        patch_ids = (
            compact.reshape(batch, gh, p, gw, p)
            .permute(0, 1, 3, 2, 4)
            .reshape(-1, p * p)
        )
        visible = patch_ids > 0
        counts = visible.sum(dim=1)
        offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        mean_table = self.mean_gene_projector(rows).float()
        sums = F.embedding_bag(
            patch_ids[visible],
            mean_table,
            offsets,
            mode="sum",
            include_last_offset=True,
        )
        means = (sums / counts.clamp_min(1)[:, None]).reshape(
            batch, self.num_patches, 64
        )
        mean_tokens = self.id_token_norm(self.id_patch_proj(means.to(x.dtype)))
        self._cached_id_tokens = mean_tokens
        tokens = (
            self.expr_token_norm(expression_tokens)
            + self.fresh_token_norm(identity_tokens)
            + mean_tokens
        ) / math.sqrt(3)
        return tokens, counts.reshape(batch, self.num_patches) > 0

    def _fresh_local_layout_jitter_expression_gene(self, expr, gene_ids):
        if not self.training:
            return expr, gene_ids
        batch, _, height, width = expr.shape
        gh, gw = self.grid_size
        p = self.patch_size
        selected = torch.rand(batch, gh, gw, 1, 1, device=expr.device) < 0.5
        pairs = torch.rand(batch, gh, gw, p, p // 2, device=expr.device) < 0.30
        swap = (
            (pairs & selected).permute(0, 1, 3, 2, 4).reshape(batch, height, width // 2)
        )
        out_expr = torch.empty_like(expr)
        out_ids = torch.empty_like(gene_ids)
        for out, source in [(out_expr[:, 0], expr[:, 0]), (out_ids, gene_ids)]:
            out[:, :, 0::2] = torch.where(swap, source[:, :, 1::2], source[:, :, 0::2])
            out[:, :, 1::2] = torch.where(swap, source[:, :, 0::2], source[:, :, 1::2])
        return out_expr, out_ids
