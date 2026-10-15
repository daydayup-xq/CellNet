"""CellNet public loading, cell encoding and expression decoding API."""

from pathlib import Path
import json
import math
import torch
from torch import nn
from .encoder import CellNetEncoder
from .decoder import CellNetDecoder
from .expression import tweedie_cpe_params


class CellNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.backbone = CellNetEncoder(config)
        self.decoder = CellNetDecoder(
            cls_dim=config["embedding_dim"],
            gene_dim=64,
            hidden_dim=config["decoder_hidden_dim"],
            depth=config["decoder_depth"],
            num_context_tokens=config["decoder_context_tokens"],
            num_heads=config["decoder_heads"],
        )

    @classmethod
    def from_pretrained(cls, directory, device="cpu"):
        directory = Path(directory)
        config = json.loads((directory / "weights/config.json").read_text())
        if config.pop("model_name") != "CellNet":
            raise ValueError("Expected a CellNet weights directory")
        with torch.device("meta"):
            model = cls(config)
        state = torch.load(
            directory / "weights/cellnet.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        model.load_state_dict(state, strict=True, assign=True)
        # Nonpersistent confidence-bin buffer is deliberately absent from pure weights.
        from .decoder import DECODER_ERROR_BIN_EDGES

        model.decoder.error_bin_edges = torch.tensor(
            DECODER_ERROR_BIN_EDGES, dtype=torch.float32
        )
        return model.to(device).eval()

    def encode(self, image, gene_ids):
        """Return native cell readout, individual layers and visible patch tokens.

        A 4-D input is one view; 5-D is [batch, views, 2, height, width].
        The native readout is LayerNorm(sum(raw CLS from the last eight layers)).
        """
        if image.ndim == 4:
            images = [image]
            ids = [gene_ids]
        elif image.ndim == 5:
            images = list(image.unbind(1))
            ids = list(gene_ids.unbind(1))
        else:
            raise ValueError("image must be 4-D or 5-D")
        if not all(torch.isfinite(x).all() for x in images):
            raise ValueError("images contain nonfinite values")
        return self.backbone(images, masks=[x[:, 1:2] for x in images], gene_ids=ids)

    def forward(self, image, gene_ids):
        """Native 768-D cell embeddings, [batch, dim] or [batch, views, dim]."""
        out = self.encode(image, gene_ids)["cls_deep"].permute(1, 0, 2)
        return out[:, 0] if image.ndim == 4 else out

    def spatial_context(self, encoding, view=0):
        cls = encoding["cls_deep"][view]
        tokens = encoding["patch_tokens_by_view_layer_mean"][view]
        ids = encoding["patch_ids_by_view"][view]
        valid = encoding["patch_mask_by_view"][view]
        b, p, d = tokens.shape
        if p == 0:
            return None, None
        n = min(self.config["decoder_context_tokens"], p)
        gh, gw = self.backbone.patch_embed.grid_size
        rows, cols = min(
            ((r, n // r) for r in range(1, n + 1) if n % r == 0),
            key=lambda rc: abs(math.log((rc[1] / rc[0]) / (gw / gh))),
        )
        y = ids // gw
        x = ids % gw
        bucket = (y * rows // gh).clamp_max(rows - 1) * cols + (
            x * cols // gw
        ).clamp_max(cols - 1)
        pooled = tokens.new_zeros(b, n, d)
        pooled.scatter_add_(
            1, bucket[..., None].expand(-1, -1, d), tokens * valid[..., None]
        )
        counts = tokens.new_zeros(b, n)
        counts.scatter_add_(1, bucket, valid.to(tokens.dtype))
        pooled = pooled / counts.clamp_min(1)[..., None]
        mask = torch.cat(
            [torch.ones(b, 1, dtype=torch.bool, device=cls.device), counts > 0], dim=1
        )
        return torch.cat([cls[:, None], pooled], dim=1), mask

    def decode(self, encoding, target_gene_ids, view=0):
        """Decode targets from observed-only inputs; outputs are in log1p units."""
        cls = encoding["cls_deep"][view]
        ids = torch.as_tensor(target_gene_ids, device=cls.device, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids[None].expand(cls.shape[0], -1)
        if ids.ndim != 2 or ids.shape[0] != cls.shape[0] or ids.numel() == 0:
            raise ValueError(
                "target IDs must be nonempty [targets] or [batch, targets]"
            )
        if ((ids <= 0) | (ids > self.config["vocabulary_size"])).any():
            raise ValueError("target IDs must belong to the pretrained vocabulary")
        query = self.backbone.patch_embed.project_gene_ids(ids).to(cls.dtype)
        context, context_mask = self.spatial_context(encoding, view)
        mask = torch.ones_like(ids, dtype=torch.bool)
        raw, presence, error, hidden = self.decoder(
            cls, query, mask, context_tokens=context, context_mask=context_mask
        )
        magnitude, rate, probability, theta = tweedie_cpe_params(raw, presence)
        edges = self.decoder.error_bin_edges
        centers = (edges[:-1] + edges[1:]) * 0.5
        return {
            "expression_log1p": probability * magnitude * self.config["expression_max"],
            "positive_magnitude_log1p": magnitude * self.config["expression_max"],
            "presence_probability": probability,
            "expected_absolute_error": (error.softmax(-1) * centers).sum(-1)
            * self.config["expression_max"],
            "hidden": hidden,
        }

    def predict(self, image, gene_ids, target_gene_ids):
        """Hide target expression, mask and identity before encoding and decoding."""
        if image.ndim != 4:
            raise ValueError("predict expects a single-view 4-D image batch")
        target = torch.as_tensor(
            target_gene_ids, device=gene_ids.device, dtype=torch.long
        )
        if target.ndim == 1:
            target = target[None].expand(image.shape[0], -1)
        if target.ndim != 2 or target.shape[0] != image.shape[0] or target.numel() == 0:
            raise ValueError(
                "target IDs must be nonempty [targets] or [batch, targets]"
            )
        # Search each cell separately to avoid a batch x pixels x targets allocation.
        hidden = torch.stack([torch.isin(g, t) for g, t in zip(gene_ids, target)])
        visible_image = image.masked_fill(hidden[:, None], 0)
        visible_ids = gene_ids.masked_fill(hidden, 0)
        return self.decode(self.encode(visible_image, visible_ids), target)
