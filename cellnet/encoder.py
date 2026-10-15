from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import torch
from torch import nn
from timm.layers.pos_embed_sincos import RotaryEmbeddingDinoV3

from .transformer import Block
from .stem import CellNetStem


class CellNetEncoder(nn.Module):
    """CellNet ViT-B encoder with packed multi-view SDPA attention."""

    def __init__(self, config):
        super().__init__()
        dim = config["embedding_dim"]
        self.embed_dim = self.num_features = dim
        self.fixed_total_tokens = config["tokens_per_cell"]
        self.drop_empty_patches = True
        self.num_reg_tokens = 4
        self.num_prefix_tokens = 5
        self.cls_last_k_layers = config["readout_layers"]
        self.cls_layer_from_end = 0
        self.collect_patch_layer_features = True
        self.patch_embed = CellNetStem(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.reg_token = nn.Parameter(torch.zeros(1, 4, dim))
        self.rope = RotaryEmbeddingDinoV3(
            dim=dim // config["attention_heads"],
            temperature=100.0,
            feat_shape=list(self.patch_embed.grid_size),
            grid_indexing="ij",
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim,
                    config["attention_heads"],
                    qkv_bias=False,
                    norm_eps=1e-5,
                    layer_scale_init_value=1e-5,
                    rope_rotate_half=True,
                    drop_path=config["drop_path_rate"] * i / (config["depth"] - 1),
                )
                for i in range(config["depth"])
            ]
        )
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.reg_token, std=0.02)

    @staticmethod
    def _sort_valid_tokens(
        tokens: torch.Tensor, valid: torch.Tensor, randomize: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, dim = tokens.shape
        positions = (
            torch.arange(num_tokens, device=tokens.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        if randomize:
            random_scores = torch.rand(
                batch_size, num_tokens, device=tokens.device, dtype=torch.float32
            )
            scores = torch.where(
                valid,
                random_scores + float(num_tokens + 1),
                -positions.to(dtype=torch.float32),
            )
        else:
            scores = valid.to(dtype=torch.long) * (num_tokens + 1) - positions
        order = scores.argsort(dim=1, descending=True)
        return (tokens.gather(1, order.unsqueeze(-1).expand(-1, -1, dim)), order)

    def _get_rope_embed(
        self, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        if self.rope is None:
            return None
        rope = self.rope.get_embed(shape=self.patch_embed.grid_size)
        if isinstance(rope, tuple):
            rope = torch.cat(rope, dim=-1)
        return rope.to(device=device, dtype=dtype)

    def _allocate_tokens(
        self, valid_counts: Sequence[torch.Tensor]
    ) -> List[torch.Tensor]:
        """固定总 token 数分配模式。

        每个样本所有视图的 token 总和恒定 = fixed_total_tokens。
        全局视图（有效 token 多）自然获得更多，局部视图（掩码重）获得更少。
        序列长度完全确定性：n_views * num_prefix_tokens + fixed_total_tokens。

        重要：每个视图的分配不会超过其有效 token 数量。
        """
        caps = torch.stack(valid_counts, dim=0).to(dtype=torch.long)
        num_views = caps.shape[0]
        budget = self.fixed_total_tokens
        total_valid = caps.sum(dim=0)
        safe_total = total_valid.clamp_min(1)
        numerators = caps * budget
        base = torch.div(numerators, safe_total.unsqueeze(0), rounding_mode="floor")
        remainders = numerators.to(dtype=torch.float64) / safe_total.unsqueeze(0).to(
            dtype=torch.float64
        ) - base.to(dtype=torch.float64)
        reverse_view_ids = (
            torch.arange(num_views - 1, -1, -1, device=caps.device)
            .unsqueeze(1)
            .expand_as(caps)
        )
        reverse_order = remainders.flip(0).argsort(dim=0, descending=True, stable=True)
        priority = reverse_view_ids.gather(0, reverse_order)
        ranks = torch.empty_like(priority).scatter_(
            0,
            priority,
            torch.arange(num_views, device=caps.device).unsqueeze(1).expand_as(caps),
        )
        leftover = budget - base.sum(dim=0)
        proportional = base + ranks.lt(leftover.unsqueeze(0))
        alloc = torch.where((total_valid > budget).unsqueeze(0), proportional, caps)
        alloc[:, total_valid == 0] = 0
        alloc = torch.minimum(alloc, caps)
        return list(alloc.unbind(dim=0))

    def _view_widths(self, alloc: Sequence[torch.Tensor]) -> List[int]:
        max_alloc = torch.stack(alloc, dim=0).amax(dim=1).tolist()
        return [self.num_prefix_tokens + int(value) for value in max_alloc]

    @staticmethod
    def _build_attn_mask(
        alloc: Sequence[torch.Tensor],
        widths: Sequence[int],
        num_prefix_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = alloc[0].shape[0]
        total_len = sum(widths)
        offsets: List[int] = []
        running = 0
        for width in widths:
            offsets.append(running)
            running += width
        is_valid = torch.zeros(batch_size, total_len, dtype=torch.bool, device=device)
        same_view = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
        for view_idx, width in enumerate(widths):
            start = offsets[view_idx]
            end = start + width
            same_view[start:end, start:end] = True
            is_valid[:, start : start + num_prefix_tokens] = True
            patch_slots = width - num_prefix_tokens
            if patch_slots > 0:
                indices = torch.arange(patch_slots, device=device).unsqueeze(0)
                is_valid[:, start + num_prefix_tokens : end] = indices < alloc[
                    view_idx
                ].unsqueeze(1)
        allow = same_view.unsqueeze(0) & is_valid.unsqueeze(1) & is_valid.unsqueeze(2)
        diag = torch.arange(total_len, device=device)
        allow[:, diag, diag] = True
        mask = torch.zeros(batch_size, 1, total_len, total_len, device=device)
        mask.masked_fill_(~allow.unsqueeze(1), float("-inf"))
        return mask

    def forward(
        self,
        views: List[torch.Tensor] | torch.Tensor,
        masks: Optional[List[Optional[torch.Tensor]] | torch.Tensor] = None,
        gene_ids: Optional[List[Optional[torch.Tensor]] | torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(views, torch.Tensor):
            views = [views]
        if isinstance(masks, torch.Tensor):
            masks = [masks]
        if isinstance(gene_ids, torch.Tensor):
            if gene_ids.dim() == 4 and gene_ids.shape[0] == len(views):
                gene_ids = [gene_ids[i] for i in range(gene_ids.shape[0])]
            else:
                gene_ids = [gene_ids]
        if masks is None:
            masks = [None] * len(views)
        if gene_ids is None:
            raise ValueError("gene_ids are required")
        if len(views) != len(masks) or len(views) != len(gene_ids):
            raise ValueError("views, masks and gene_ids must have the same length")
        batch_size = views[0].shape[0]
        device = views[0].device
        embedded = []
        cached_id_tokens_list = []
        for view, mask, gid in zip(views, masks, gene_ids):
            tokens, valid = self.patch_embed(view, mask=mask, gene_ids=gid)
            sorted_tokens, order = self._sort_valid_tokens(
                tokens, valid, randomize=self.training
            )
            embedded.append((sorted_tokens, valid.sum(dim=1), order))
            raw_id = self.patch_embed._cached_id_tokens
            cached_id_tokens_list.append(
                raw_id.gather(1, order.unsqueeze(-1).expand_as(raw_id))
            )
        valid_counts = [item[1] for item in embedded]
        alloc = self._allocate_tokens(valid_counts)
        widths = self._view_widths(alloc)
        sequences: List[torch.Tensor] = []
        rope_embed = self._get_rope_embed(device, views[0].dtype)
        rope_parts: List[torch.Tensor] = []
        for view_idx, width in enumerate(widths):
            patch_tokens = embedded[view_idx][0]
            order = embedded[view_idx][2]
            patch_slots = width - self.num_prefix_tokens
            if patch_slots > 0:
                selected = patch_tokens[:, :patch_slots, :]
                valid_mask = torch.arange(patch_slots, device=device).unsqueeze(
                    0
                ) < alloc[view_idx].unsqueeze(1)
                selected = selected * valid_mask.unsqueeze(-1).to(dtype=selected.dtype)
            else:
                selected = patch_tokens.new_zeros(batch_size, 0, self.embed_dim)
            cls = self.cls_token.expand(batch_size, 1, self.embed_dim)
            prefix = [cls]
            if self.reg_token is not None:
                prefix.append(self.reg_token.expand(batch_size, -1, -1))
            sequences.append(torch.cat(prefix + [selected], dim=1))
            if rope_embed is not None:
                rope_dim = rope_embed.shape[-1] // 2
                prefix_rope = rope_embed.new_zeros(
                    batch_size, self.num_prefix_tokens, rope_embed.shape[-1]
                )
                prefix_rope[..., rope_dim:] = 1.0
                if patch_slots > 0:
                    gather_index = order.unsqueeze(-1).expand(
                        -1, -1, rope_embed.shape[-1]
                    )
                    sorted_rope = (
                        rope_embed.unsqueeze(0)
                        .expand(batch_size, -1, -1)
                        .gather(1, gather_index)
                    )
                    selected_rope = sorted_rope[:, :patch_slots, :]
                    selected_rope = selected_rope * valid_mask.unsqueeze(-1).to(
                        dtype=selected_rope.dtype
                    )
                else:
                    selected_rope = rope_embed.new_zeros(
                        batch_size, 0, rope_embed.shape[-1]
                    )
                rope_parts.append(torch.cat([prefix_rope, selected_rope], dim=1))
        packed = torch.cat(sequences, dim=1)
        attn_mask = self._build_attn_mask(alloc, widths, self.num_prefix_tokens, device)
        packed_rope = torch.cat(rope_parts, dim=1) if rope_parts else None
        offsets: List[int] = []
        running = 0
        for width in widths:
            offsets.append(running)
            running += width
        cls_deep_list: List[torch.Tensor] = []
        cls_layer_list: List[torch.Tensor] = []
        patch_layer_sum_by_view: List[Optional[torch.Tensor]] = [None] * len(widths)
        total_blocks = len(self.blocks)
        for i, block in enumerate(self.blocks):
            packed = block(packed, attn_mask, rope=packed_rope)
            if (
                self.cls_layer_from_end > 0
                and i == total_blocks - self.cls_layer_from_end
            ):
                cls_layer_list.append(
                    torch.stack([packed[:, off, :] for off in offsets], dim=0)
                )
            elif (
                self.cls_last_k_layers > 1
                and i >= total_blocks - self.cls_last_k_layers
            ):
                cls_layer = torch.stack([packed[:, off, :] for off in offsets], dim=0)
                cls_deep_list.append(cls_layer)
                if not getattr(self, "collect_patch_layer_features", False):
                    continue
                for view_idx, (offset, width) in enumerate(zip(offsets, widths)):
                    patch_slots = width - self.num_prefix_tokens
                    valid_mask = torch.arange(patch_slots, device=device).unsqueeze(
                        0
                    ) < alloc[view_idx].unsqueeze(1)
                    patch_layer = packed[
                        :, offset + self.num_prefix_tokens : offset + width, :
                    ]
                    patch_layer = self.norm(patch_layer)
                    patch_layer = patch_layer * valid_mask.unsqueeze(-1).to(
                        dtype=patch_layer.dtype
                    )
                    if patch_layer_sum_by_view[view_idx] is None:
                        patch_layer_sum_by_view[view_idx] = patch_layer
                    else:
                        patch_layer_sum_by_view[view_idx] = (
                            patch_layer_sum_by_view[view_idx] + patch_layer
                        )
        packed = self.norm(packed)
        cls_all = torch.stack([packed[:, offset, :] for offset in offsets], dim=0)
        if self.cls_last_k_layers > 1 and cls_deep_list:
            cls_deep_raw = torch.stack(cls_deep_list, dim=0).sum(dim=0)
            cls_deep = self.norm(cls_deep_raw)
        else:
            cls_deep = cls_all
        cls_selected_layer = self.norm(cls_layer_list[0]) if cls_layer_list else None
        mean_patches: List[torch.Tensor] = []
        patch_tokens_by_view: List[torch.Tensor] = []
        patch_ids_by_view: List[torch.Tensor] = []
        patch_mask_by_view: List[torch.Tensor] = []
        id_tokens_by_view: List[torch.Tensor] = []
        for view_idx, (offset, width) in enumerate(zip(offsets, widths)):
            patch_slots = width - self.num_prefix_tokens
            if patch_slots == 0:
                mean_patches.append(packed.new_zeros(batch_size, self.embed_dim))
                patch_tokens_by_view.append(
                    packed.new_zeros(batch_size, 0, self.embed_dim)
                )
                patch_ids_by_view.append(
                    torch.zeros(batch_size, 0, dtype=torch.long, device=device)
                )
                patch_mask_by_view.append(
                    torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
                )
                id_tokens_by_view.append(
                    packed.new_zeros(batch_size, 0, self.embed_dim)
                )
                continue
            patch_slice = packed[:, offset + self.num_prefix_tokens : offset + width, :]
            valid_mask = torch.arange(patch_slots, device=device).unsqueeze(0) < alloc[
                view_idx
            ].unsqueeze(1)
            patch_slice = patch_slice * valid_mask.unsqueeze(-1).to(
                dtype=patch_slice.dtype
            )
            patch_ids = embedded[view_idx][2][:, :patch_slots]
            denom = (
                alloc[view_idx].clamp(min=1).unsqueeze(-1).to(dtype=patch_slice.dtype)
            )
            mean_patches.append(patch_slice.sum(dim=1) / denom)
            patch_tokens_by_view.append(patch_slice)
            patch_ids_by_view.append(patch_ids)
            patch_mask_by_view.append(valid_mask)
            id_tok = cached_id_tokens_list[view_idx]
            if id_tok is not None:
                id_slice = id_tok[:, :patch_slots, :]
                id_slice = id_slice * valid_mask.unsqueeze(-1).to(dtype=id_slice.dtype)
                id_tokens_by_view.append(id_slice)
            else:
                id_tokens_by_view.append(
                    packed.new_zeros(batch_size, 0, self.embed_dim)
                )
        return {
            "cls": cls_all,
            "cls_deep": cls_deep,
            "cls_selected_layer": cls_selected_layer,
            "cls_layers": torch.stack(
                [self.norm(layer) for layer in cls_deep_list], dim=0
            )
            if cls_deep_list
            else None,
            "cls_layers_raw": torch.stack(cls_deep_list, dim=0)
            if cls_deep_list
            else None,
            "patches": packed,
            "offsets": offsets,
            "mean_patch": torch.stack(mean_patches, dim=0),
            "patch_tokens_by_view": patch_tokens_by_view,
            "patch_ids_by_view": patch_ids_by_view,
            "patch_mask_by_view": patch_mask_by_view,
            "id_tokens_by_view": id_tokens_by_view,
            "patch_tokens_by_view_layer_mean": [
                value / float(len(cls_deep_list)) for value in patch_layer_sum_by_view
            ]
            if cls_deep_list
            and all((value is not None for value in patch_layer_sum_by_view))
            else None,
        }
