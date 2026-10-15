"""Load rendered pretraining shards and make paired masked views."""

from pathlib import Path
import csv
import hashlib
import json
import numpy as np


def iter_pretraining_cells(directory):
    """Yield image [2,H,W], gene_ids [H,W] and descriptive metadata as NumPy arrays.

    Images already contain normalized expression. Do not normalize
    them again. Species and cell labels are descriptive; they are not losses.
    """
    directory = Path(directory).resolve()
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["format"] != "cellnet-pretraining-npz-v1":
        raise ValueError("Unsupported pretraining data format")
    if metadata["status"] == "not_included":
        raise FileNotFoundError(
            "Full pretraining samples are not included; use example/"
        )
    with (directory / metadata["cells"]).open(newline="") as handle:
        records = list(csv.DictReader(handle))
    by_id = {row["sample_id"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate sample IDs in cells.csv")
    seen = set()
    for shard in metadata["shards"]:
        path = (directory / shard["path"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Shard path must stay inside the dataset directory")
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != shard["sha256"]:
            raise ValueError(f"Shard checksum mismatch: {shard['path']}")
        with np.load(path, allow_pickle=False) as data:
            expression = data["expression"]
            masks = data["measurement_mask"]
            gene_ids = data["gene_ids"]
            layout = data["layout_index"]
            names = data["sample_ids"]
        shape = tuple(metadata["image_size"])
        n = len(expression)
        if (
            expression.dtype != np.float16
            or expression.shape != (n, *shape)
            or masks.dtype != np.bool_
            or masks.ndim != 3
            or masks.shape[1:] != shape
            or gene_ids.shape != masks.shape
            or gene_ids.dtype != np.int32
            or layout.dtype != np.int32
            or layout.shape != (n,)
            or names.shape != (n,)
            or names.dtype.kind != "U"
            or n != shard["samples"]
        ):
            raise ValueError("Invalid shard array shapes or dtypes")
        if (
            not np.isfinite(expression).all()
            or (expression < 0).any()
            or (expression > 1).any()
            or (layout < 0).any()
            or (layout >= len(masks)).any()
            or (gene_ids < 0).any()
            or not np.array_equal(gene_ids > 0, masks)
        ):
            raise ValueError("Invalid expression, layout, mask or gene ID values")
        for i, sample_id in enumerate(names.tolist()):
            if sample_id in seen or sample_id not in by_id:
                raise ValueError("Missing or duplicate sample metadata")
            row = by_id[sample_id]
            index = int(layout[i])
            if int(row["layout_index"]) != index:
                raise ValueError("CSV and shard layout indices differ")
            visible = masks[index]
            if np.any(expression[i][~visible]):
                raise ValueError("Expression is nonzero at an unmeasured pixel")
            seen.add(sample_id)
            yield {
                "image": np.stack([expression[i].astype(np.float32), visible]),
                "gene_ids": gene_ids[index].astype(np.int64),
                "metadata": row,
            }
    if len(seen) != metadata["samples"] or seen != set(by_id):
        raise ValueError("Dataset sample count and cells.csv do not match")


def make_pretraining_views(cells, views=2, keep_fraction=0.75, seed=0):
    """Create independent measured-gene masks while keeping each cell paired.

    Each view independently masks measured genes. Hidden expression is never
    returned as input.
    """
    if len(cells) < 2 or views < 2 or not 0 < keep_fraction <= 1:
        raise ValueError("Need >=2 cells, >=2 views and keep_fraction in (0,1]")
    rng = np.random.default_rng(seed)
    images, ids = [], []
    for cell in cells:
        measured = cell["image"][1].astype(bool)
        cell_images, cell_ids = [], []
        for _ in range(views):
            visible = measured & (rng.random(measured.shape) < keep_fraction)
            cell_images.append(np.stack([cell["image"][0] * visible, visible]))
            cell_ids.append(np.where(visible, cell["gene_ids"], 0))
        images.append(cell_images)
        ids.append(cell_ids)
    return {
        "image": np.asarray(images, np.float32),
        "gene_ids": np.asarray(ids, np.int64),
    }
