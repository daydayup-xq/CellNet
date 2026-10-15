"""Shared loading and inference for the small PBMC task examples."""

from pathlib import Path
import argparse
import json
import numpy as np
import torch
from cellnet import CellNet, CellNetImageConverter

RELEASE = Path(__file__).resolve().parents[1]
DATA = RELEASE / "examples/data"


def parser(task, description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--release", type=Path, default=RELEASE)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--output", type=Path, default=Path("example_outputs") / task)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=2)
    return p


def load_data(directory, filename="totalx_pbmc_96.h5ad"):
    import anndata as ad

    return ad.read_h5ad(Path(directory) / filename)


def read_panel(directory):
    import pandas as pd

    return pd.read_csv(Path(directory) / "target_panel.csv")


def gene_pcc(prediction, truth):
    prediction = np.asarray(prediction, np.float64)
    truth = np.asarray(truth, np.float64)
    p = prediction - prediction.mean(0)
    y = truth - truth.mean(0)
    denominator = np.sqrt((p * p).sum(0) * (y * y).sum(0))
    pcc = np.divide(
        (p * y).sum(0),
        denominator,
        out=np.full(len(denominator), np.nan),
        where=denominator > 1e-12,
    )
    return pcc


def expression_metrics(prediction, truth):
    prediction = np.asarray(prediction, np.float64)
    truth = np.asarray(truth, np.float64)
    pcc = gene_pcc(prediction, truth)
    return {
        "cells": len(truth),
        "genes": truth.shape[1],
        "finite_pcc_genes": int(np.isfinite(pcc).sum()),
        "mean_gene_pcc": float(np.nanmean(pcc)) if np.isfinite(pcc).any() else None,
        "mae_log1p": float(np.mean(np.abs(prediction - truth))),
        "rmse_log1p": float(np.sqrt(np.mean((prediction - truth) ** 2))),
    }


def save_metrics(directory, metrics):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    text = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    (directory / "metrics.json").write_text(text)
    print(text, flush=True)


class Runtime:
    def __init__(self, args):
        if args.batch_size < 1:
            raise ValueError("batch size must be positive")
        torch.set_num_threads(4)
        self.args = args
        self.model = CellNet.from_pretrained(args.release, args.device)
        self.converter = CellNetImageConverter(args.release, "Human")
        self.cache = {}

    def batch(self, data, rows):
        scale = data.uns.get("cellnet_example", {}).get("input_scale", "log1p")
        batch = self.converter.transform(
            data.X[rows], data.var_names.tolist(), input_scale=scale
        )
        return {key: value.to(self.args.device) for key, value in batch.items()}

    def embeddings(self, data, key="main"):
        """Readouts share one actual encoder pass, cached only within this run."""
        if key not in self.cache:
            features = {name: [] for name in ["native", "layer8", "layer6", "last"]}
            with torch.inference_mode():
                for start in range(0, data.n_obs, self.args.batch_size):
                    rows = np.arange(
                        start, min(start + self.args.batch_size, data.n_obs)
                    )
                    output = self.model.encode(**self.batch(data, rows))
                    for name, value in [
                        ("native", output["cls_deep"][0]),
                        ("layer8", output["cls_layers_raw"][0, 0]),
                        ("layer6", output["cls_layers_raw"][2, 0]),
                        ("last", output["cls"][0]),
                    ]:
                        features[name].append(value.float().cpu().numpy())
                    print(f"Encoded {key}: {rows[-1] + 1}/{data.n_obs}", flush=True)
            self.cache[key] = {
                name: np.concatenate(value) for name, value in features.items()
            }
        return self.cache[key]

    def decode(self, data, genes, *, hide_targets):
        ids = self.converter.resolve_gene_ids(genes)
        if (ids == 0).any():
            raise ValueError("Target panel contains an unsupported gene")
        predictions, hidden = [], []
        with torch.inference_mode():
            for start in range(0, data.n_obs, self.args.batch_size):
                rows = np.arange(start, min(start + self.args.batch_size, data.n_obs))
                batch = self.batch(data, rows)
                if hide_targets:
                    result = self.model.predict(**batch, target_gene_ids=ids)
                else:
                    result = self.model.decode(self.model.encode(**batch), ids)
                predictions.append(result["expression_log1p"].float().cpu().numpy())
                hidden.append(result["hidden"].float().cpu().numpy())
                print(f"Decoded {rows[-1] + 1}/{data.n_obs}", flush=True)
        return np.concatenate(predictions), np.concatenate(hidden)
