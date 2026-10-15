"""Compute contextual gene similarities from decoder hidden states in PBMC cell-type contexts."""

import numpy as np
import pandas as pd
from task_common import Runtime, load_data, read_panel, parser, save_metrics


def run(args, runtime=None):
    runtime = runtime or Runtime(args)
    data = load_data(args.data)
    labels = data.obs.cell_type.astype(str).to_numpy()
    rows = np.array(
        [
            row
            for label in sorted(set(labels))
            for row in np.flatnonzero(labels == label)[:100]
        ]
    )
    selected = data[rows].copy()
    genes = read_panel(args.data).gene_id.astype(str).tolist()
    # Association uses the full measured context; queried genes remain visible.
    _, hidden = runtime.decode(selected, genes, hide_targets=False)
    contexts, matrices, edges = [], [], []
    for context in sorted(set(labels)):
        vectors = hidden[selected.obs.cell_type.astype(str).to_numpy() == context].mean(0)
        vectors = vectors - vectors.mean(0, keepdims=True)
        vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)
        score = vectors @ vectors.T
        contexts.append(context)
        matrices.append(score)
        upper = np.triu_indices(len(genes), 1)
        order = np.argsort(score[upper], kind="stable")[::-1][:20]
        edges.extend(
            {
                "context": context,
                "gene_a": genes[upper[0][i]],
                "gene_b": genes[upper[1][i]],
                "similarity": float(score[upper][i]),
            }
            for i in order
        )
    metrics = {
        "task": "gene_association",
        "cells": len(rows),
        "genes": len(genes),
        "contexts": contexts,
        "aggregation": "Mean hidden states over cells, center across panel genes, L2 normalize, cosine",
        "max_cells_per_context": 100,
        "query_genes_visible": True,
        "panel_scope": "64-gene demonstration panel; centering depends on panel composition",
        "scope": "Undirected similarity demonstration; no causal or regulatory validation",
    }
    save_metrics(args.output, metrics)
    np.savez_compressed(
        args.output / "contextual_similarities.npz",
        scores=np.stack(matrices),
        gene_ids=np.asarray(genes),
        contexts=np.asarray(contexts),
    )
    pd.DataFrame(edges).to_csv(args.output / "top_pairs.csv", index=False)
    return metrics


if __name__ == "__main__":
    run(parser("gene_association", __doc__).parse_args())
