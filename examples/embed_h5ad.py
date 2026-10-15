"""Extract CellNet cell embeddings from an AnnData file in bounded batches."""

import argparse
from pathlib import Path
import numpy as np
import torch
from cellnet import CellNet, CellNetImageConverter


def read_rows(dataset, start, end, n_genes):
    """Read dense/CSR/CSC H5AD expression without backed SciPy subclasses."""
    import h5py
    from scipy import sparse

    if isinstance(dataset, h5py.Dataset):
        return dataset[start:end]
    encoding = dataset.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    if encoding == "csr_matrix":
        pointers = dataset["indptr"][start : end + 1]
        lo, hi = int(pointers[0]), int(pointers[-1])
        return sparse.csr_matrix(
            (dataset["data"][lo:hi], dataset["indices"][lo:hi], pointers - lo),
            shape=(end - start, n_genes),
        )
    if encoding == "csc_matrix":
        # Bounded memory, although CSR is substantially faster for row batches.
        pointers = dataset["indptr"][:]
        rows, columns, values = [], [], []
        for column in range(n_genes):
            lo, hi = int(pointers[column]), int(pointers[column + 1])
            indices = dataset["indices"][lo:hi]
            selected = (indices >= start) & (indices < end)
            rows.extend((indices[selected] - start).tolist())
            columns.extend([column] * int(selected.sum()))
            values.extend(dataset["data"][lo:hi][selected].tolist())
        return sparse.csr_matrix(
            (values, (rows, columns)), shape=(end - start, n_genes)
        )
    raise ValueError(f"Unsupported H5AD matrix encoding: {encoding!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .npy array, same cell order as input",
    )
    p.add_argument(
        "--species", required=True, help="Exact fitted namespace, e.g. Human or Mouse"
    )
    p.add_argument("--input-scale", choices=["counts", "log1p"], required=True)
    p.add_argument("--layer", choices=["native", "layer8", "layer6", "layer8_normalized", "last"], default="native")
    p.add_argument(
        "--layer-key", help="AnnData layer containing expression; defaults to X"
    )
    p.add_argument(
        "--gene-column", help="var column with stable IDs; defaults to var_names"
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--allow-unknown", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1 or args.output.suffix != ".npy":
        raise ValueError("Use positive batch size and .npy output")
    import anndata as ad
    import h5py

    data = ad.read_h5ad(args.input, backed="r")
    model = CellNet.from_pretrained(args.release, args.device)
    converter = CellNetImageConverter(args.release, args.species)
    genes = (
        data.var[args.gene_column].astype(str).tolist()
        if args.gene_column
        else data.var_names.tolist()
    )
    if not data.n_obs:
        raise ValueError("Input has no cells")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    result = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float32, shape=(data.n_obs, 768)
    )
    try:
        with h5py.File(args.input, "r") as handle, torch.inference_mode():
            matrix = handle[f"layers/{args.layer_key}" if args.layer_key else "X"]
            for start in range(0, data.n_obs, args.batch_size):
                end = min(start + args.batch_size, data.n_obs)
                batch = converter.transform(
                    read_rows(matrix, start, end, len(genes)),
                    genes,
                    input_scale=args.input_scale,
                    allow_unknown=args.allow_unknown,
                )
                encoded = model.encode(
                    **{k: v.to(args.device) for k, v in batch.items()}
                )
                if args.layer == "native":
                    features = encoded["cls_deep"][0]
                elif args.layer == "last":
                    features = encoded["cls"][0]
                elif args.layer == "layer8":
                    features = encoded["cls_layers_raw"][0, 0]
                elif args.layer == "layer6":
                    features = encoded["cls_layers_raw"][2, 0]
                elif args.layer == "layer8_normalized":
                    features = encoded["cls_layers"][0, 0]
                result[start:end] = features.float().cpu().numpy()
                print(f"{end}/{data.n_obs}", flush=True)
        result.flush()
        del result
        partial.replace(args.output)
        import json

        args.output.with_suffix(".json").write_text(
            json.dumps(
                dict(converter.last_report, readout=args.layer, cells=data.n_obs),
                indent=2,
            )
            + "\n"
        )
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    finally:
        data.file.close()


if __name__ == "__main__":
    main()
