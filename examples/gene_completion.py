"""Five-fold gene-axis completion; each full held-out fold is absent from its query file."""
import json
import numpy as np
from task_common import Runtime, load_data, parser, expression_metrics, gene_pcc, save_metrics


def run(args, runtime=None):
    runtime = runtime or Runtime(args)
    folder = args.data / "completion_fivefold"
    folds, pccs = [], []
    seen = set()
    for fold in range(1, 6):
        genes = json.loads((folder / f"fold{fold}_targets.json").read_text())
        query = load_data(folder, f"fold{fold}_query.h5ad")
        assert not set(query.var_names) & set(genes)
        assert not seen & set(genes)
        seen.update(genes)
        prediction, hidden = runtime.decode(query, genes, hide_targets=True)
        del hidden
        # Only the scorer opens truth, after inference is complete.
        truth = np.load(folder / f"fold{fold}_truth.npz", allow_pickle=False)
        assert np.array_equal(truth["cell_ids"], query.obs_names.to_numpy(str))
        assert np.array_equal(truth["gene_ids"], np.asarray(genes))
        metrics = expression_metrics(prediction, truth["expression_log1p"])
        metrics.update(fold=fold, observed_genes=query.n_vars, hidden_targets_in_query=0)
        pccs.extend(gene_pcc(prediction, truth["expression_log1p"]).tolist())
        save_metrics(args.output / f"fold{fold}", metrics)
        np.savez_compressed(args.output / f"fold{fold}" / "predictions.npz",
                            prediction_log1p=prediction, truth_log1p=truth["expression_log1p"],
                            cell_ids=truth["cell_ids"], gene_ids=truth["gene_ids"])
        folds.append(metrics)
    finite = np.asarray(pccs)[np.isfinite(pccs)]
    metrics = {"task": "gene_completion", "n_splits": 5, "seed": 42, "folds": folds,
               "genes_hidden_exactly_once": len(seen), "finite_pcc_genes": len(finite),
               "mean_gene_pcc": float(finite.mean()) if len(finite) else None,
               "prediction": "native expected expression p*M",
               "normalization": "Observed-only query library; separate scorer truth uses full measured supported-gene library",
               "scope": "Eight real PBMC cells, complete five-fold gene-axis split; not full Fig3 metrics. Positive-cell PCC requiring 20 cells is not estimated."}
    save_metrics(args.output, metrics)
    return metrics


if __name__ == "__main__":
    run(parser("gene_completion", __doc__).parse_args())
