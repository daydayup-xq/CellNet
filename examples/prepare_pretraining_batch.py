"""Prepare real masked multi-view cells for the minimal pretraining entry point."""

from pathlib import Path
import argparse
from itertools import islice
import torch
from cellnet.pretraining_data import iter_pretraining_cells, make_pretraining_views


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=root / "datasets/pretraining/example")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cells", type=int, default=2)
    p.add_argument("--views", type=int, default=2)
    p.add_argument("--keep-fraction", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=20261015)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cells = list(islice(iter_pretraining_cells(args.data), args.cells))
    if len(cells) != args.cells:
        raise ValueError("Not enough cells in the dataset")
    arrays = make_pretraining_views(cells, args.views, args.keep_fraction, args.seed)
    batch = {key: torch.from_numpy(value) for key, value in arrays.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(batch, args.output)
    print({key: tuple(value.shape) for key, value in batch.items()})
    print("Saved real independently masked views:", args.output)


if __name__ == "__main__":
    main()
