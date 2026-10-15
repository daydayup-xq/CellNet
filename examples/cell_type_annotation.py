"""Native CellNet features with train-fitted scaling and shrinkage LDA."""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
from task_common import Runtime, load_data, parser, save_metrics


def run(args, runtime=None):
    runtime = runtime or Runtime(args)
    data = load_data(args.data, "tabula_sapiens_192.h5ad")
    features = runtime.embeddings(data, key="tabula_sapiens")["native"]
    labels = data.obs.cell_type.astype(str).to_numpy()
    tissues = data.obs.tissue.astype(str).to_numpy()
    donors = data.obs.donor_id.astype(str).to_numpy()
    folds, predictions, tissue_scores = [], [], []
    for tissue in sorted(set(tissues)):
        per_tissue = []
        for donor in sorted(set(donors[tissues == tissue])):
            reference = np.flatnonzero((tissues == tissue) & (donors != donor))
            query = np.flatnonzero((tissues == tissue) & (donors == donor))
            if not len(reference) or not len(query):
                raise ValueError("Each tissue needs at least two donors")
            assert not set(donors[reference]) & set(donors[query])
            clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
                solver="lsqr", shrinkage="auto"))
            clf.fit(features[reference], labels[reference])
            prediction = clf.predict(features[query])
            score = {"tissue": tissue, "held_out_donor": donor,
                     "reference_cells": len(reference), "query_cells": len(query),
                     "accuracy": float(accuracy_score(labels[query], prediction)),
                     "macro_f1": float(f1_score(labels[query], prediction, average="macro"))}
            folds.append(score); per_tissue.append(score)
            predictions.extend({"cell_id": data.obs_names[i], "tissue": tissue,
                                "held_out_donor": donor, "truth": labels[i], "prediction": p}
                               for i, p in zip(query, prediction))
        tissue_scores.append({"tissue": tissue, **{k: float(np.mean([f[k] for f in per_tissue]))
                                                   for k in ["accuracy", "macro_f1"]}})
    metrics = {"task": "cell_type_annotation", "cells": data.n_obs,
               "readout": "native: LayerNorm(sum of last 8 raw CLS tokens)",
               "classifier": "StandardScaler(train only) + LDA(lsqr, shrinkage=auto)",
               "protocol": "Within-tissue leave-one-donor-out; fold mean then equal tissue mean",
               "folds": folds, "tissues": tissue_scores,
               **{k: float(np.mean([f[k] for f in tissue_scores])) for k in ["accuracy", "macro_f1"]},
               "scope": "192 real Tabula Sapiens cells; same estimator and split protocol, reduced cohort"}
    save_metrics(args.output, metrics)
    pd.DataFrame(predictions).to_csv(args.output / "predictions.csv", index=False)
    np.savez_compressed(args.output / "embeddings.npz", native=features,
                        cell_ids=data.obs_names.to_numpy(str))
    return metrics


if __name__ == "__main__":
    run(parser("cell_type_annotation", __doc__).parse_args())
