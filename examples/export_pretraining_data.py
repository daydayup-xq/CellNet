"""Repack native rendered tar shards into bounded CellNet NPZ shards."""

from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import numpy as np
from cellnet.pretraining_data import _NativeAssets, _native_records


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
    source = args.source.resolve()
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
    assets = _NativeAssets(
        source, json.loads((source / "metadata.json").read_text()), args.release
    )
    args.output.mkdir(parents=True)
    (args.output / "shards").mkdir()
    pending, layouts, shards = [], {}, []
    total_samples = 0
    species_seen = []

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
                for record_key, fields in _native_records(path):
                    if args.per_class is not None and accepted >= args.per_class:
                        break
                    if (
                        not {"npy", "species", "dataset_id", "cell_type"}
                        <= fields.keys()
                    ):
                        raise ValueError("Incomplete native sample")
                    species = fields["species"].decode()
                    dataset_id = assets.source_dataset(
                        int(fields["dataset_id"]), record_key, path.relative_to(source)
                    )
                    dataset = assets.datasets[dataset_id]
                    image = np.load(io.BytesIO(fields["npy"]), allow_pickle=False)
                    if image.shape != (384, 384) or image.dtype != np.float16:
                        raise ValueError("Expected native 384x384 float16 images")
                    key = (species, dataset_id)
                    if key not in layouts:
                        visible, gene_ids = assets.layout(species, dataset_id)
                        layouts[key] = (
                            visible,
                            gene_ids.astype(np.int32),
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
                    label = assets.labels[int(fields["cell_type"])].split("::", 1)[-1]
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
