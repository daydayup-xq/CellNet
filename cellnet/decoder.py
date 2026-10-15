from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import nn

DECODER_ERROR_BIN_EDGES = (
    0.0,
    0.01,
    0.02,
    0.035,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.5,
    1.0,
)


class CellNetDecoder(nn.Module):
    """CellNet patch cross-attention expression and confidence decoder."""

    DEFAULT_ERROR_BIN_EDGES = DECODER_ERROR_BIN_EDGES
    uses_patch_context = True

    def __init__(
        self,
        cls_dim: int,
        gene_dim: int,
        hidden_dim: int = 512,
        depth: int = 3,
        num_context_tokens: int = 32,
        num_heads: int = 8,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_context_tokens = int(num_context_tokens)
        effective_heads = min(int(num_heads), self.hidden_dim)
        while effective_heads > 1 and self.hidden_dim % effective_heads != 0:
            effective_heads -= 1
        self.cls_context = nn.Linear(cls_dim, self.hidden_dim, bias=False)
        self.patch_context = nn.Linear(cls_dim, self.hidden_dim, bias=False)
        self.gene_query = nn.Sequential(
            nn.LayerNorm(gene_dim), nn.Linear(gene_dim, self.hidden_dim, bias=False)
        )
        self.blocks = nn.ModuleList()
        for _ in range(int(depth)):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(
                            self.hidden_dim,
                            effective_heads,
                            batch_first=True,
                            bias=False,
                        ),
                        "norm": nn.LayerNorm(self.hidden_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(self.hidden_dim, self.hidden_dim * 2, bias=False),
                            nn.GELU(),
                            nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=False),
                        ),
                    }
                )
            )
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.magnitude_hidden = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.magnitude_activation = nn.GELU()
        self.out = nn.Linear(self.hidden_dim, 1)
        self.presence_hidden = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.presence_activation = nn.GELU()
        self.presence_out = nn.Linear(self.hidden_dim, 1)
        error_bin_edges = torch.tensor(
            self.DEFAULT_ERROR_BIN_EDGES, dtype=torch.float32
        )
        self.register_buffer("error_bin_edges", error_bin_edges, persistent=False)
        self.num_error_bins = int(error_bin_edges.numel() - 1)
        self.confidence_hidden = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.confidence_activation = nn.GELU()
        self.conf_head = nn.Linear(self.hidden_dim, self.num_error_bins)
        self._initialize_private_output_projections()
        self.use_query_interaction = True
        self.interaction_proj = nn.Linear(
            self.hidden_dim * 2, self.hidden_dim, bias=False
        )
        self.interaction_norm = nn.LayerNorm(self.hidden_dim)
        nn.init.normal_(self.interaction_proj.weight, mean=0.0, std=0.001)

    def _initialize_private_output_projections(self) -> None:
        """Start new private projections as near-identity transforms.

        The final output projections retain their constructor initialization.
        This method initializes only the private hidden transforms.
        """
        for layer in (
            self.magnitude_hidden,
            self.presence_hidden,
            self.confidence_hidden,
        ):
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        cls_token: torch.Tensor,
        gene_identity: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        context_tokens: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        zero_cell_branch: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = cls_token.shape[0]
        query = self.gene_query(gene_identity)
        if context_tokens is None:
            context_tokens = cls_token.new_zeros(
                B, 1 + self.num_context_tokens, cls_token.shape[-1]
            )
            context_mask = torch.zeros(
                B,
                1 + self.num_context_tokens,
                dtype=torch.bool,
                device=cls_token.device,
            )
            context_mask[:, 0] = True
        else:
            context_tokens = context_tokens.to(dtype=cls_token.dtype)
            if context_mask is None:
                context_mask = torch.ones(
                    context_tokens.shape[:2],
                    dtype=torch.bool,
                    device=context_tokens.device,
                )
        if zero_cell_branch:
            context_tokens = torch.zeros_like(context_tokens)
            context_mask = torch.zeros_like(context_mask)
            context_mask[:, 0] = True
        kv = context_tokens.new_zeros(
            context_tokens.shape[0], context_tokens.shape[1], self.hidden_dim
        )
        kv[:, :1] = self.cls_context(context_tokens[:, :1])
        if context_tokens.shape[1] > 1:
            kv[:, 1:] = self.patch_context(context_tokens[:, 1:])
        h = torch.zeros_like(query)
        key_padding = ~context_mask.to(dtype=torch.bool)
        for block in self.blocks:
            attended, _ = block["attn"](
                query + h, kv, kv, key_padding_mask=key_padding, need_weights=False
            )
            h = h + attended
            h = h + block["ffn"](block["norm"](h))
        h = self.out_norm(h)
        if getattr(self, "use_query_interaction", False):
            h = self.interaction_norm(
                self.interaction_proj(torch.cat([h, h * query], dim=-1))
            )
        magnitude_feature = self.magnitude_activation(self.magnitude_hidden(h))
        pred = self.out(magnitude_feature).squeeze(-1)
        presence_feature = self.presence_activation(self.presence_hidden(h))
        presence_logit = self.presence_out(presence_feature).squeeze(-1)
        confidence_feature = self.confidence_activation(
            self.confidence_hidden(h.detach())
        )
        error_logits = self.conf_head(confidence_feature)
        mask = target_mask.to(dtype=pred.dtype)
        return (
            pred * mask,
            presence_logit * mask,
            error_logits * mask.unsqueeze(-1),
            h,
        )
