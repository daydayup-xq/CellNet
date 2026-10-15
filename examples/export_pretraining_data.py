"""Repack native rendered tar shards into bounded CellNet NPZ shards."""

from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import tarfile
import numpy as np


def records(path):
    with tarfile.open(path, "r|gz") as archive:
        key, fields = None, {}
        for member in archive:
            if not member.isfile():
                continue
            current, field = member.name.rsplit(".", 1)
            if key is not None and current != key:
                yield fields
                fields = {}
            key = current
            fields[field] = archive.extractfile(member).read()
        if key is not None:
            yield fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--release", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-classes", nargs="+")
    parser.add_argument("--per-class", type=int)
    parser.add_argument("--shard-size", type=int, default=256)
    args = parser.parse_args()
    if args.shard_size < 1 or (args.per_class is not None and args.per_class < 1):
        raise ValueError("shard-size and per-class must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source
    buckets = source / "class_shards"
    if not buckets.is_dir():
        raise FileNotFoundError("Native DataBank class_shards directory is absent")
    chosen = (
        [buckets / name for name in args.storage_classes]
        if args.storage_classes
        else sorted(p for p in buckets.iterdir() if p.is_dir())
    )
    if not all(p.is_dir() and p.parent == buckets for p in chosen):
        raise ValueError(
            "Storage class must name a direct directory under class_shards"
        )
    index = json.loads((source / "masks/dataset_index.json").read_text())
    datasets = {int(i): name for name, i in index.items()}
    masks = np.load(source / "masks/gene_masks.npy", mmap_mode="r", allow_pickle=False)
    labels = json.loads((source / "cell_type_vocab.json").read_text())["names"]
    if isinstance(labels, dict):
        labels = {int(k): v for k, v in labels.items()}
    # A mismatched vocabulary would assign the wrong model identities.
    if json.loads((source / "gene_vocab.json").read_text()) != json.loads(
        (args.release / "assets/gene_vocab.json").read_text()
    ):
        raise ValueError("Source and release vocabularies differ")
    args.output.mkdir(parents=True)
    (args.output / "shards").mkdir()
    pending, layouts, shards = [], {}, []
    total_samples = 0
    species_seen = []
    checked_species = set()
    with np.load(
        args.release / "assets/species_layouts.npz", allow_pickle=False
    ) as data:
        names = data["species"].tolist()
        native = {
            s: (data[f"ids_{i}"].copy(), data[f"pixels_{i}"].copy())
            for i, s in enumerate(names)
        }
    source_maps = np.load(
        source / "pixel_maps/gene_to_flat_pixel.npy", mmap_mode="r", allow_pickle=False
    )
    source_species = json.loads((source / "pixel_maps/dataset_index.json").read_text())

    def flush():
        if not pending:
            return
        path = args.output / f"shards/shard-{len(shards):05d}.npz"
        np.savez_compressed(
            path,
            expression=np.stack([item[0] for item in pending]),
            measurement_mask=np.stack([value[0] for value in layouts.values()]),
            gene_ids=np.stack([value[1] for value in layouts.values()]),
            layout_index=np.array([item[1] for item in pending], np.int32),
            sample_ids=np.array([item[2] for item in pending]),
        )
        shards.append(
            {
                "path": str(path.relative_to(args.output)),
                "samples": len(pending),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        pending.clear()
        layouts.clear()

    with (args.output / "cells.csv").open("w", newline="") as handle:
        columns = [
            "sample_id",
            "species",
            "cell_type",
            "source_dataset",
            "layout_index",
            "measured_genes",
            "detected_genes",
            "mean_expression_measured",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for bucket in chosen:
            accepted = 0
            for path in sorted(bucket.glob("*.tar*")):
                for fields in records(path):
                    if args.per_class is not None and accepted >= args.per_class:
                        break
                    if (
                        not {"npy", "species", "dataset_id", "cell_type"}
                        <= fields.keys()
                    ):
                        raise ValueError("Incomplete native sample")
                    species = fields["species"].decode()
                    dataset_id = int(fields["dataset_id"])
                    dataset = datasets[dataset_id]
                    ids, pixels = native[species]
                    if species not in checked_species:
                        if not np.array_equal(
                            source_maps[source_species[species], ids - 1], pixels
                        ):
                            raise ValueError("Source and release native layouts differ")
                        checked_species.add(species)
                    image = np.load(io.BytesIO(fields["npy"]), allow_pickle=False)
                    if image.shape != (384, 384) or image.dtype != np.float16:
                        raise ValueError("Expected native 384x384 float16 images")
                    key = (species, dataset_id)
                    if key not in layouts:
                        selected = masks[dataset_id, ids - 1].astype(bool)
                        visible = np.zeros(384 * 384, bool)
                        visible[pixels[selected]] = True
                        gene_ids = np.zeros(384 * 384, np.int32)
                        gene_ids[pixels[selected]] = ids[selected]
                        layouts[key] = (
                            visible.reshape(384, 384),
                            gene_ids.reshape(384, 384),
                        )
                    visible, gene_ids = layouts[key]
                    if (
                        not np.isfinite(image).all()
                        or image.min() < 0
                        or image.max() > 1
                        or np.any(image[~visible])
                    ):
                        raise ValueError(
                            "Invalid expression or mismatched measurement mask"
                        )
                    layout_index = list(layouts).index(key)
                    sample_id = f"pretrain_{total_samples:04d}"
                    label = labels[int(fields["cell_type"])].split("::", 1)[-1]
                    writer.writerow(
                        {
                            "sample_id": sample_id,
                            "species": species,
                            "cell_type": label,
                            "source_dataset": dataset,
                            "layout_index": layout_index,
                            "measured_genes": int(visible.sum()),
                            "detected_genes": int(np.count_nonzero(image)),
                            "mean_expression_measured": float(
                                image[visible].astype(np.float64).mean()
                            ),
                        }
                    )
                    pending.append((image, layout_index, sample_id))
                    accepted += 1
                    total_samples += 1
                    if species not in species_seen:
                        species_seen.append(species)
                    if len(pending) == args.shard_size:
                        flush()
                if args.per_class is not None and accepted >= args.per_class:
                    break
            if args.per_class is not None and accepted != args.per_class:
                raise ValueError(f"Not enough complete records in {bucket.name}")
    flush()
    if total_samples == 0:
        raise ValueError("No complete records exported")
    metadata = {
        "format": "cellnet-pretraining-npz-v1",
        "status": "available",
        "samples": total_samples,
        "image_size": [384, 384],
        "cells": "cells.csv",
        "shards": shards,
        "species": species_seen,
        "terms": "Upstream dataset terms apply; source identifiers are recorded per cell. License review is required before external distribution.",
    }
    # The manifest is written last so partial exports cannot look complete.
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Exported {total_samples} cells in {len(shards)} shards")


if __name__ == "__main__":
    main()
