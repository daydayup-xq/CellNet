"""Native-noise demonstration with count thinning and separate clean truth."""

import numpy as np
from task_common import (
    Runtime,
    load_data,
    read_panel,
    parser,
    expression_metrics,
    gene_pcc,
    save_metrics,
)


def run(args, runtime=None):
    runtime = runtime or Runtime(args)
    noisy = load_data(args.data, "denoising_noisy.h5ad")
    genes = read_panel(args.data).gene_id.astype(str).tolist()
    # These are noisy measured genes, so retain their noisy measurements.
    prediction, _ = runtime.decode(noisy, genes, hide_targets=False)
    truth = np.load(args.data / "denoising_truth.npz", allow_pickle=False)
    assert np.array_equal(truth["cell_ids"], noisy.obs_names.to_numpy(str))
    metrics = expression_metrics(prediction, truth["expression_log1p"])
    columns = noisy.var_names.get_indexer(genes)
    counts = noisy.X.toarray().astype(np.float64)
    library = counts.sum(1, keepdims=True)
    observed = np.clip(
        np.log1p(
            np.divide(
                counts[:, columns],
                library,
                out=np.zeros((len(counts), len(columns))),
                where=library > 0,
            )
            * 10000
        ),
        0,
        6,
    )
    metrics.update(
        task="denoising",
        noisy_baseline=expression_metrics(observed, truth["expression_log1p"]),
        scope="50% count thinning of eight real PBMC cells; not the OpenProblems protocol",
    )
    common = np.isfinite(gene_pcc(prediction, truth["expression_log1p"])) & np.isfinite(
        gene_pcc(observed, truth["expression_log1p"])
    )
    metrics["common_finite_pcc_targets"] = {
        "genes": int(common.sum()),
        "CellNet": expression_metrics(
            prediction[:, common], truth["expression_log1p"][:, common]
        ),
        "noisy_input": expression_metrics(
            observed[:, common], truth["expression_log1p"][:, common]
        ),
    }
    save_metrics(args.output, metrics)
    np.savez_compressed(
        args.output / "predictions.npz",
        prediction_log1p=prediction,
        noisy_log1p=observed,
        clean_log1p=truth["expression_log1p"],
        cell_ids=truth["cell_ids"],
        gene_ids=truth["gene_ids"],
    )
    return metrics


if __name__ == "__main__":
    run(parser("denoising", __doc__).parse_args())
