"""Evaluate a frozen representation with Fig2 graph and metric settings."""
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scib
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from task_common import Runtime, load_data, parser, save_metrics


def run(args, runtime=None):
    runtime = runtime or Runtime(args)
    data = load_data(args.data)
    features = runtime.embeddings(data)["layer6"]
    if len(features) <= 90:
        raise ValueError("The fixed 90-neighbor protocol requires more than 90 cells")
    obs = data.obs[["cell_type", "batch"]].copy()
    for name in obs:
        obs[name] = pd.Categorical(obs[name].astype(str))
    result = ad.AnnData(X=features, obs=obs)
    result.obsm["X_emb"] = features
    sc.pp.neighbors(result, n_neighbors=90, use_rep="X_emb", metric="euclidean", random_state=0)
    sc.tl.leiden(result, resolution=0.05, random_state=0, key_added="leiden_metrics",
                 flavor="leidenalg", n_iterations=-1)
    labels = obs.cell_type.astype(str).to_numpy()
    clusters = result.obs.leiden_metrics.astype(str).to_numpy()
    nmi = float(normalized_mutual_info_score(labels, clusters))
    ari = float(adjusted_rand_score(labels, clusters))
    bio_asw = float((silhouette_score(features, labels) + 1) / 2)
    batch_asw = float(scib.metrics.silhouette_batch(result, batch_key="batch", label_key="cell_type",
                                                  embed="X_emb", metric="euclidean", scale=True,
                                                  return_all=False, verbose=False))
    connectivity = float(scib.metrics.graph_connectivity(result, label_key="cell_type"))
    metrics = {"task": "batch_integration", "cells": len(features), "readout": "raw CLS layer6",
               "n_neighbors": 90, "metric": "euclidean", "leiden_resolution": 0.05, "seed": 0,
               "nmi": nmi, "ari": ari, "cell_type_asw_scaled": bio_asw,
               "batch_asw_scaled": batch_asw, "graph_connectivity": connectivity,
               "avg_bio": (nmi + ari + bio_asw) / 3,
               "avg_batch": (batch_asw + connectivity) / 2,
               "scope": "One 96-cell PBMC cohort; full Fig2 aggregates metrics equally across tissues"}
    save_metrics(args.output, metrics)
    result.write_h5ad(args.output / "representation_and_graph.h5ad", compression="gzip")
    return metrics


if __name__ == "__main__":
    run(parser("batch_integration", __doc__).parse_args())
