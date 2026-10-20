"""Prepare real masked multi-view cells for the minimal pretraining entry point."""

from pathlib import Path
import argparse
from itertools import islice
import torch
from cellnet.pretraining_data import (
    default_pretraining_directory, iter_pretraining_cells, make_pretraining_views,
)


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path, default=root,
                   help="CellNet clone containing assets/ and weights/")
    p.add_argument("--data", type=Path,
                   help="Defaults to installed datasets/pretraining/full/, otherwise example/")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--storage-classes", nargs="+",
                           help="Native storage class directories for a partial download")
    selection.add_argument("--shards", nargs="+",
                           help="Native shard paths relative to --data for a partial download")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cells", type=int, default=2)
    p.add_argument("--views", type=int, default=2)
    p.add_argument("--keep-fraction", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=20261015)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.cells < 2:
        p.error("--cells must be at least 2")
    data = args.data if args.data is not None else default_pretraining_directory(args.release)
    cells = list(islice(iter_pretraining_cells(
        data, release=args.release, storage_classes=args.storage_classes, shards=args.shards,
    ), args.cells))
    if len(cells) != args.cells:
        raise ValueError("Not enough cells in the dataset")
    arrays = make_pretraining_views(cells, args.views, args.keep_fraction, args.seed)
    batch = {key: torch.from_numpy(value) for key, value in arrays.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(batch, args.output)
    print({key: tuple(value.shape) for key, value in batch.items()})
    print("Data:", data.resolve())
    print("Fitted release:", args.release.resolve())
    print("Saved real independently masked views:", args.output)


if __name__ == "__main__":
    main()
