"""Apply the fitted CellNet species layouts without refitting or external services."""

from pathlib import Path
import json
import numpy as np
import torch
from scipy import sparse


class CellNetImageConverter:
    def __init__(self, directory, species):
        root = Path(directory)
        self.config = json.loads((root / "weights/config.json").read_text())
        self.vocabulary = json.loads((root / "assets/gene_vocab.json").read_text())
        self.species = species
        self.gene_to_id = {key: index + 1 for key, index in self.vocabulary.items()}
        with np.load(root / "assets/species_layouts.npz", allow_pickle=False) as data:
            names = data["species"].tolist()
            if species not in names:
                raise ValueError(f"Unknown species {species!r}. Available: {names}")
            i = names.index(species)
            self.layout_ids = data[f"ids_{i}"].copy()
            self.layout_pixels = data[f"pixels_{i}"].copy()
        self.id_to_pixel = dict(
            zip(self.layout_ids.tolist(), self.layout_pixels.tolist())
        )
        self.last_report = None

    def resolve_gene_ids(self, genes):
        ids = []
        for gene in genes:
            gene = str(gene)
            key = gene if "::" in gene else f"{self.species}::{gene}"
            if not key.startswith(self.species + "::"):
                raise ValueError(f"Gene {gene!r} belongs to a different species")
            ids.append(self.gene_to_id.get(key, 0))
        return np.asarray(ids, dtype=np.int64)

    def transform(
        self,
        expression,
        genes,
        *,
        input_scale,
        measured=None,
        allow_unknown=False,
        storage_dtype="float16",
    ):
        """Convert cells x genes into expression/mask images and 1-based gene maps.

        input_scale must be 'counts' or 'log1p'. Counts use library-size 1e4
        normalization over the supplied measured genes. Subsets of a full count
        matrix must be normalized BEFORE subsetting (pass input_scale='log1p').
        All supplied genes are measured by default, including measured zeros.
        measured may be [genes] or [cells, genes]; False removes both expression
        and identity. Duplicate gene IDs are rejected. Unknowns raise by default.
        Expression uses a float16 storage round-trip before float32 inference.
        Use storage_dtype="float32" explicitly for unquantized images.
        """
        if input_scale not in {"counts", "log1p"}:
            raise ValueError("input_scale must be 'counts' or 'log1p'")
        if storage_dtype not in {"float16", "float32"}:
            raise ValueError("storage_dtype must be 'float16' or 'float32'")
        if len(expression.shape) != 2 or expression.shape[1] != len(genes):
            raise ValueError(
                "expression must be cells x genes with matching gene names"
            )
        ids = self.resolve_gene_ids(genes)
        known = ids > 0
        if len(np.unique(ids[known])) != known.sum():
            raise ValueError(
                "Duplicate gene IDs; combine duplicate columns before conversion"
            )
        mapped = known & np.asarray([int(i) in self.id_to_pixel for i in ids])
        unknown = [str(g) for g, keep in zip(genes, mapped) if not keep]
        if unknown and not allow_unknown:
            raise ValueError(
                f"{len(unknown)} genes absent from fitted layout, e.g. {unknown[:5]}"
            )
        if not mapped.any():
            raise ValueError("No genes match the fitted layout")
        n, g = expression.shape
        measured = (
            np.ones((n, g), dtype=bool)
            if measured is None
            else np.asarray(measured, dtype=bool)
        )
        if measured.shape == (g,):
            measured = np.broadcast_to(measured, (n, g))
        if measured.shape != (n, g):
            raise ValueError("measured must have shape [genes] or [cells, genes]")
        x = (
            expression.toarray()
            if sparse.issparse(expression)
            else np.asarray(expression)
        )
        x = np.array(x, dtype=np.float32, copy=True)
        if not np.isfinite(x[measured]).all() or (x[measured] < 0).any():
            raise ValueError("Measured expression must be finite and nonnegative")
        x[~measured] = 0
        if input_scale == "counts":
            counts = np.asarray(x, dtype=np.float64)
            total = counts.sum(axis=1, keepdims=True)
            scaled = np.divide(
                counts, total, out=np.zeros_like(counts), where=total > 0
            )
            x = np.log1p(np.clip(scaled * 10000.0, 0, None)).astype(np.float32)
        x = np.clip(x / self.config["expression_max"], 0, 1)
        x = x.astype(storage_dtype).astype(np.float32)
        h, w = self.config["image_size"]
        pixels = np.array(
            [self.id_to_pixel[int(i)] for i in ids[mapped]], dtype=np.int64
        )
        if len(np.unique(pixels)) != len(pixels):
            raise ValueError("Fitted layout has a pixel collision")
        image = np.zeros((n, 2, h * w), dtype=np.float32)
        gene_image = np.zeros((n, h * w), dtype=np.int64)
        visible = measured[:, mapped]
        image[:, 0, pixels] = x[:, mapped] * visible
        image[:, 1, pixels] = visible
        gene_image[:, pixels] = np.where(visible, ids[mapped][None], 0)
        self.last_report = {
            "species": self.species,
            "input_genes": g,
            "mapped_genes": int(mapped.sum()),
            "unknown_genes": unknown,
            "input_scale": input_scale,
            "expression_storage_dtype": storage_dtype,
        }
        return {
            "image": torch.from_numpy(image.reshape(n, 2, h, w)),
            "gene_ids": torch.from_numpy(gene_image.reshape(n, h, w)),
        }

    @classmethod
    def from_embeddings(
        cls, directory, model, species, genes, embeddings, *, projection=None
    ):
        """Register a new gene panel using the frozen fitter and identity IPCA.

        Mutates this model's frozen identity table in memory. Embeddings must
        use the same raw 1,920-D feature space as the pretrained identities.
        Native genes retain their original coordinates and identity rows.
        Use a fresh pretrained model for each independently extended panel.
        """
        from torch import nn
        from .projection import CellNetProjection
        from .coordinates import balanced_greedy_kdtree_assign_coords

        root = Path(directory)
        vocab = json.loads((root / "assets/gene_vocab.json").read_text())
        if model.config["vocabulary_size"] != len(vocab):
            raise ValueError(
                "Load a fresh pretrained model before registering a gene panel"
            )
        genes = [str(g) for g in genes]
        keys = [g if "::" in g else f"{species}::{g}" for g in genes]
        if (
            not genes
            or len(set(keys)) != len(keys)
            or any(not g.startswith(species + "::") for g in keys)
        ):
            raise ValueError(
                "Provide unique genes from one consistent species namespace"
            )
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.shape != (len(genes), 1920) or not np.isfinite(vectors).all():
            raise ValueError(
                "Expected finite [genes, 1920] compatible raw identity embeddings"
            )
        with np.load(root / "assets/species_layouts.npz", allow_pickle=False) as z:
            native = species in z["species"].tolist()
        if native:
            converter = cls(root, species)
        else:
            converter = cls.__new__(cls)
            converter.config = dict(model.config)
            converter.vocabulary = vocab
            converter.gene_to_id = {k: i + 1 for k, i in vocab.items()}
            converter.species = species
            converter.id_to_pixel = {}
            converter.last_report = None
        unknown = np.array([g not in vocab for g in keys])
        unmapped = np.array(
            [converter.gene_to_id.get(g, 0) not in converter.id_to_pixel for g in keys]
        )
        if unmapped.sum() > 384 * 384 - len(converter.id_to_pixel):
            raise ValueError("Gene panel exceeds the available pixels")
        # Compute and validate all projected assets before updating model or converter.
        if unmapped.any():
            fitter = projection if projection is not None else CellNetProjection(root)
            proposed = fitter.project(vectors[unmapped])
            occupied = np.zeros((384, 384), dtype=bool)
            for pixel in converter.id_to_pixel.values():
                occupied.flat[pixel] = True
            coords = balanced_greedy_kdtree_assign_coords(
                proposed, (384, 384), 16, occupied
            )
            pixels = coords[:, 0] * 384 + coords[:, 1]
            if len(np.unique(pixels)) != len(pixels):
                raise RuntimeError("New layout contains collisions")
        if unknown.any():
            table = model.backbone.patch_embed.gene_identity_embed.weight
            with np.load(
                root / "assets/identity_projection.npz", allow_pickle=False
            ) as z:
                reduced = (vectors[unknown] - z["mean"]) @ z["components"]
            new_rows = torch.as_tensor(reduced, device=table.device, dtype=table.dtype)
            extended = nn.Embedding.from_pretrained(
                torch.cat([table.detach(), new_rows], dim=0), freeze=True, padding_idx=0
            )
            for key in np.asarray(keys)[unknown]:
                converter.vocabulary[str(key)] = len(converter.vocabulary)
                converter.gene_to_id[str(key)] = len(converter.vocabulary)
            model.backbone.patch_embed.gene_identity_embed = extended
            model.config["vocabulary_size"] = extended.num_embeddings - 1
        for key, pixel in zip(
            np.asarray(keys)[unmapped], pixels if unmapped.any() else []
        ):
            converter.id_to_pixel[converter.gene_to_id[str(key)]] = int(pixel)
        converter.config = dict(model.config)
        converter.layout_ids = np.asarray(list(converter.id_to_pixel), dtype=np.int32)
        converter.layout_pixels = np.asarray(
            list(converter.id_to_pixel.values()), dtype=np.int32
        )
        return converter
