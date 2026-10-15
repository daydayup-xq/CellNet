# Runnable representative tasks

Install from the release directory with `pip install -e '.[examples]'`, then run:

```bash
python examples/run_all_tasks.py --device cpu --batch-size 2 --output example_outputs
```

Inputs are actual Tabula Sapiens and TotalX PBMC cells. [Figure protocols](FIGURE_PROTOCOLS.md) specify model readouts, downstream hyperparameters and the reduced-cohort boundaries. [Dataset sources](../README.md) lists original studies and official links.

| Task | Individual command | Input and output |
| --- | --- | --- |
| Cell annotation | `python examples/cell_type_annotation.py` | 192 Tabula cells; tissue-conditioned donor-LODO; native CLS + train-fitted scaler + LDA; per-fold labels and metrics |
| Reference mapping | `python examples/reference_mapping.py` | 96 PBMC cells; five 80/20 splits; raw layer8 + scaler + cosine KNN k=10, prior exponent 0.25 |
| Batch representation | `python examples/batch_integration.py` | 96 PBMC cells; raw layer6; 90-neighbor graph; Leiden 0.05; NMI/ARI/scaled ASW/connectivity |
| Gene completion | `python examples/gene_completion.py` | Eight PBMC cells; five full gene-axis folds; physically absent targets; separate predictions and truth |
| Gene association | `python examples/gene_association.py` | 24 PBMC cells per type, 64 genes; mean hidden states, gene centering, cosine similarities |
| Additional denoising API example | `python examples/denoising.py` | Eight thinned real-count cells; independent clean truth; noisy-input baseline; outside Fig2–5 |

All commands accept `--release`, `--data`, `--output`, `--device`, and `--batch-size`. `--data` selects a directory following the included file/schema layout. The default paths are relative to the script location, and examples use the CellNet package. For arbitrary expression files, `embed_h5ad.py` accepts an H5AD path, species, input scale, layer and stable-ID column. For example:

```bash
python examples/embed_h5ad.py --release . --input cells.h5ad \
  --output native_features.npy --species Human --input-scale log1p \
  --gene-column standard_gene_id --layer native --batch-size 2
```

Use stable IDs matching the selected species. A source `ensembl_id` column may include version suffixes; the Tabula example uses its existing unversioned `standard_gene_id` column. Unknown IDs fail by default, and no gene-symbol-to-ID guessing occurs.

The five-fold completion queries contain only permitted observed counts. Each target fold is absent before normalization and inference. The scorer alone reads separate target truth, normalized using the full supported measured-gene library.

`examples/expected/` contains tested CPU outputs. Validation results and scope are recorded in `docs/FIGURE_PROTOCOL_VALIDATION.json`.

The small denoising example did not outperform its noisy-input baseline. Contextual gene similarity is undirected and does not establish causality. These examples validate implementation and data loading, not full-cohort biological performance.
