"""Run the representative PBMC examples, sharing one model and cell-feature cache."""

from copy import copy
import cell_type_annotation
import reference_mapping
import batch_integration
import gene_completion
import gene_association
import denoising
from task_common import Runtime, parser, save_metrics


def main():
    p = parser("all_tasks", __doc__)
    p.add_argument(
        "--tasks",
        nargs="+",
        choices=[
            "cell_type_annotation",
            "reference_mapping",
            "batch_integration",
            "gene_completion",
            "gene_association",
            "denoising",
        ],
        default=[
            "cell_type_annotation",
            "reference_mapping",
            "batch_integration",
            "gene_completion",
            "gene_association",
            "denoising",
        ],
    )
    args = p.parse_args()
    runtime = Runtime(args)
    modules = {
        module.__name__: module
        for module in [
            cell_type_annotation,
            reference_mapping,
            batch_integration,
            gene_completion,
            gene_association,
            denoising,
        ]
    }
    results = {}
    for name in args.tasks:
        task_args = copy(args)
        task_args.output = args.output / name
        results[name] = modules[name].run(task_args, runtime)
    save_metrics(
        args.output,
        {
            "tasks": results,
            "dataset": "Real Tabula Sapiens and PBMC subsets; representative protocol validation",
        },
    )


if __name__ == "__main__":
    main()
