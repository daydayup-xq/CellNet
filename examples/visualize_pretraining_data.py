"""Render the bundled real pretraining images, masks and cohort diagnostics."""

from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from cellnet.pretraining_data import iter_pretraining_cells


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=root / "datasets/pretraining/example"
    )
    parser.add_argument("--output", type=Path, default=Path("pretraining_figures"))
    args = parser.parse_args()
    cells = list(iter_pretraining_cells(args.data))
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    species = list(dict.fromkeys(c["metadata"]["species"] for c in cells))
    first = [next(c for c in cells if c["metadata"]["species"] == s) for s in species]
    fig, axes = plt.subplots(
        len(first),
        2,
        figsize=(8.2, 3.0 * len(first)),
        squeeze=False,
        layout="constrained",
    )
    cmap = plt.colormaps["magma"].copy()
    cmap.set_bad("#cbd5e1")
    for row, cell in enumerate(first):
        expression, measured = cell["image"]
        measured = measured.astype(bool)
        values = np.ma.array(expression, mask=~measured)
        image = axes[row, 0].imshow(
            values, cmap=cmap, vmin=0, vmax=1, interpolation="nearest"
        )
        axes[row, 1].imshow(
            measured, cmap="Greys", vmin=0, vmax=1, interpolation="nearest"
        )
        label = cell["metadata"]
        axes[row, 0].set_title(
            f"{label['species']} | {label['cell_type']}\nExpression: {label['sample_id']}",
            fontsize=9,
        )
        axes[row, 1].set_title(
            f"Measured genes: {int(measured.sum()):,}\nNonzero genes: {int(np.count_nonzero(expression)):,}",
            fontsize=9,
        )
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(
        image,
        ax=axes[:, 0].tolist(),
        label="Stored expression: clip(log1p, 0, 6) / 6",
        shrink=0.65,
    )
    axes[-1, 1].legend(
        handles=[
            Patch(facecolor="black", label="Measured, including zero"),
            Patch(facecolor="white", edgecolor="gray", label="Unmeasured"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "Real pretraining examples | native 384 × 384 layouts\nGray in expression panels = unmeasured; black = measured zero",
        fontsize=11,
    )
    for suffix in ["png", "svg"]:
        fig.savefig(args.output / f"expression_and_masks.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), layout="constrained")
    colors = ["#2563eb", "#059669", "#d97706", "#7c3aed"]
    measured_all = np.array([c["image"][1].sum() for c in cells])
    detected_all = np.array([np.count_nonzero(c["image"][0]) for c in cells])
    counts = [sum(c["metadata"]["species"] == s for c in cells) for s in species]
    axes[0].bar(species, counts, color=colors[: len(species)])
    axes[0].set_ylabel("Example cells")
    axes[0].set_ylim(0, max(counts) + 2)
    axes[0].tick_params(axis="x", rotation=20)
    for i, n in enumerate(counts):
        axes[0].text(i, n + 0.1, str(n), ha="center")
    for i, s in enumerate(species):
        selection = np.array([c["metadata"]["species"] == s for c in cells])
        axes[1].scatter(
            measured_all[selection],
            detected_all[selection],
            s=35,
            alpha=0.8,
            color=colors[i],
            label=s,
        )
        # Measured zeros stay in the denominator; unmeasured pixels do not.
        mean = [
            float(c["image"][0][c["image"][1].astype(bool)].mean())
            for c in cells
            if c["metadata"]["species"] == s
        ]
        axes[2].scatter(np.full(len(mean), i), mean, color=colors[i], s=30, alpha=0.6)
    axes[1].set_xlabel("Measured genes (mask)")
    axes[1].set_ylabel("Nonzero-expression genes")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].set_xticks(range(len(species)), species, rotation=20)
    axes[2].set_ylabel("Mean stored expression over measured genes")
    axes[2].set_ylim(bottom=0)
    fig.suptitle(
        "Fixed 24-cell format demonstration | not a population or performance comparison",
        fontsize=11,
    )
    for suffix in ["png", "svg"]:
        fig.savefig(args.output / f"sample_summary.{suffix}", dpi=180)
    plt.close(fig)
    summary = {
        "samples": len(cells),
        "species_counts": dict(zip(species, counts)),
        "measured_genes_range": [int(measured_all.min()), int(measured_all.max())],
        "nonzero_genes_range": [int(detected_all.min()), int(detected_all.max())],
        "scope": "Stored images and measurement masks; not raw counts, UMAP, model quality or biological validation",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "figure_source_data.csv").open("w", newline="") as handle:
        rows = [c["metadata"] for c in cells]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
