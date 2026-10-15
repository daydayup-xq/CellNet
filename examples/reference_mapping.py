"""Within-species mapping with the fixed Fig2 KNN protocol on a small PBMC cohort."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score
from task_common import Runtime, load_data, parser, save_metrics


def prior_corrected_vote(distances, neighbors, reference_labels, gamma=0.25):
    classes, counts = np.unique(reference_labels, return_counts=True)
    codes = np.searchsorted(classes, reference_labels)[neighbors]
    with np.errstate(divide="ignore"):
        weights = 1.0 / distances
    zero_rows = np.isinf(weights).any(axis=1)
    weights[zero_rows] = np.isinf(weights[zero_rows]).astype(float)
    scores = np.zeros((len(neighbors), len(classes)), dtype=np.float64)
    np.add.at(scores, (np.arange(len(neighbors))[:, None], codes), weights)
    scores *= (counts / counts.sum()) ** (-gamma)
    return classes[scores.argmax(axis=1)]


def run(args, runtime=None):
    runtime = runtime or Runtime(args)
    data = load_data(args.data)
    features = runtime.embeddings(data)["layer8"].astype(np.float64)
    labels = data.obs.cell_type.astype(str).to_numpy()
    folds, predictions = [], []
    for seed in [42, 123, 2024, 2025, 20261008]:
        reference, query = train_test_split(np.arange(len(labels)), test_size=0.2,
                                          stratify=labels, random_state=seed)
        assert not set(reference) & set(query)
        scaler = StandardScaler().fit(features[reference])
        ref, qry = scaler.transform(features[reference]), scaler.transform(features[query])
        graph = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute").fit(ref)
        distance, neighbors = graph.kneighbors(qry)
        prediction = prior_corrected_vote(distance, neighbors, labels[reference])
        folds.append({"seed": seed, "reference_cells": len(reference), "query_cells": len(query),
                      "accuracy": float(accuracy_score(labels[query], prediction)),
                      "macro_f1": float(f1_score(labels[query], prediction, average="macro"))})
        predictions.extend({"seed": seed, "cell_id": data.obs_names[i], "truth": labels[i],
                            "mapped_label": p} for i, p in zip(query, prediction))
    metrics = {"task": "reference_mapping", "readout": "raw CLS layer8",
               "classifier": "reference-fitted StandardScaler + cosine distance KNN",
               "k": 10, "prior_gamma": 0.25, "query_fraction": 0.2,
               "self_neighbors_allowed": False, "folds": folds,
               "accuracy": float(np.mean([f["accuracy"] for f in folds])),
               "macro_f1": float(np.mean([f["macro_f1"] for f in folds])),
               "scope": "Five-seed within-species protocol on 96 PBMC cells; not full Fig2 cohorts or cross-species mapping"}
    save_metrics(args.output, metrics)
    pd.DataFrame(predictions).to_csv(args.output / "predictions.csv", index=False)
    return metrics


if __name__ == "__main__":
    run(parser("reference_mapping", __doc__).parse_args())
