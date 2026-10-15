"""Minimal CellNet pretraining without species labels or prototype state.

Callers supply aligned, independently augmented views of the same cells.
This module retains view alignment and the deployed local SIGReg equation.
"""

from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .model import CellNet
from .expression import tweedie_cpe_weighted_nll


class CellNetPretrainer(nn.Module):
    def __init__(self, model, *, regularization_weight=0.05, decoder_weight=8.0):
        super().__init__()
        self.model = model
        d = model.config["embedding_dim"]
        self.projector = nn.Sequential(
            nn.Linear(d, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0),
            nn.Linear(1024, 256),
            nn.Dropout(0),
        )
        self.regularization_weight = regularization_weight
        self.decoder_weight = decoder_weight
        t = torch.linspace(0, 3, 17)
        dt = 3 / 16
        weights = torch.full((17,), 2 * dt)
        weights[[0, -1]] = dt
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("phi", torch.exp(-t.square() / 2), persistent=False)
        self.register_buffer(
            "integration_weights",
            weights * torch.exp(-t.square() / 2),
            persistent=False,
        )

    @classmethod
    def from_pretrained(cls, directory, device="cpu"):
        model = CellNet.from_pretrained(directory)
        trainer = cls(model)
        state = torch.load(
            Path(directory) / "weights/projector.pt",
            map_location="cpu",
            weights_only=True,
        )
        trainer.projector.load_state_dict(state, strict=True)
        return trainer.to(device)

    @classmethod
    def from_dinov3(cls, directory, checkpoint, device="cpu"):
        """Start with DINOv3 blocks, fresh CellNet layers and frozen gene identities."""
        import json
        from .initialization import load_dinov3_backbone

        directory = Path(directory)
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        config = json.loads((directory / "weights/config.json").read_text())
        if config.pop("model_name") != "CellNet":
            raise ValueError("Expected a CellNet model configuration")
        model = CellNet(config)
        report = load_dinov3_backbone(model.backbone, checkpoint)
        # Reuse only the frozen gene-identity resource, not trained CellNet layers.
        state = torch.load(directory / "weights/cellnet.pt", map_location="cpu",
                           weights_only=True, mmap=True)
        identity = state["backbone.patch_embed.gene_identity_embed.weight"]
        if identity.shape != model.backbone.patch_embed.gene_identity_embed.weight.shape:
            raise ValueError("Gene identity table does not match the vocabulary")
        model.backbone.patch_embed.gene_identity_embed = nn.Embedding.from_pretrained(
            identity, freeze=True, padding_idx=0)
        trainer = cls(model)
        trainer.initialization_report = report
        return trainer.to(device)

    def regularization(self, projected):
        p = projected.flatten(0, 1).float()
        directions = F.normalize(
            torch.randn(p.shape[-1], 1024, device=p.device), dim=0, eps=1e-12
        )
        phase = (p @ directions)[..., None] * self.t
        error = (phase.cos().mean(0) - self.phi).square() + phase.sin().mean(0).square()
        return ((error @ self.integration_weights) * p.shape[0]).mean()

    def forward(self, image, gene_ids, *, n_global=2, targets=None):
        if image.ndim != 5 or image.shape[0] < 2:
            raise ValueError("Pretraining expects [batch>=2, views, 2, H, W]")
        if not 1 <= n_global <= image.shape[1]:
            raise ValueError("Invalid number of global views")
        # Decoder truths are separate tensors; targets never enter visible context.
        if targets is not None:
            ids = targets["gene_ids"].to(gene_ids.device)
            if ids.ndim != 2 or ids.shape[0] != image.shape[0]:
                raise ValueError("Target IDs must be [batch, targets]")
            hidden = torch.stack(
                [torch.isin(g, t[t > 0]) for g, t in zip(gene_ids, ids)]
            )
            image = image.masked_fill(hidden[:, :, None], 0)
            gene_ids = gene_ids.masked_fill(hidden, 0)
        encoding = self.model.encode(image, gene_ids)
        features = encoding["cls_deep"]
        v, b, d = features.shape
        projected = (
            self.projector(features.permute(1, 0, 2).reshape(b * v, d))
            .reshape(b, v, -1)
            .permute(1, 0, 2)
        )
        target = projected[:n_global].mean(0, keepdim=True)
        alignment = (projected - target).square().mean()
        reg = self.regularization(projected)
        total = (
            1 - self.regularization_weight
        ) * alignment + self.regularization_weight * reg
        result = {"alignment": alignment, "regularization": reg}
        if targets is not None:
            valid = targets["mask"].to(image.device).bool()
            query_ids = targets["gene_ids"].to(image.device).long()
            if (
                (query_ids[valid] <= 0)
                | (query_ids[valid] > self.model.config["vocabulary_size"])
            ).any():
                raise ValueError("Training target outside pretrained vocabulary")
            # Padding queries have no loss or visible input contribution.
            query_ids = torch.where(valid, query_ids, torch.ones_like(query_ids))
            query = self.model.backbone.patch_embed.project_gene_ids(query_ids)
            context, mask = self.model.spatial_context(encoding)
            raw, presence, error, _ = self.model.decoder(
                features[0], query, valid, context_tokens=context, context_mask=mask
            )
            truth = targets["expression_normalized"].to(image.device)
            nll, _ = tweedie_cpe_weighted_nll(
                raw,
                presence,
                truth,
                valid,
                positive_threshold=0.03,
                pos_pool_count=targets["positive_pool_count"].to(image.device),
                zero_pool_count=targets["zero_pool_count"].to(image.device),
            )
            result["expression_nll"] = nll
            total = total + self.decoder_weight * nll
        result["loss"] = total
        return result
