# Runnable representative tasks

Set the absolute clone path and install the model assets and task dependencies:

```bash
export CELLNET_RELEASE="/absolute/path/to/CellNet"
git -C "$CELLNET_RELEASE" lfs pull
python -m pip install -e "${CELLNET_RELEASE}[examples]"
python "$CELLNET_RELEASE/examples/run_all_tasks.py" \
  --release "$CELLNET_RELEASE" --device cpu --batch-size 2 \
  --output "/path/to/example_outputs"
```

The supplied tasks and H5AD converter read `weights/` and the fitted mapping in `assets/` from `--release`. Keep the release vocabulary and `species_layouts.npz` together with its weights. For native species, gene identities and pixel positions are selected automatically from these fixed assets. Downloading the full Image Atlas is a separate step for workflows using pretraining cells; its [download directory guide](../datasets/pretraining/README.md) explains that setup.

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
python "$CELLNET_RELEASE/examples/embed_h5ad.py" \
  --release "$CELLNET_RELEASE" --input "/path/to/cells.h5ad" \
  --output "/path/to/native_features.npy" --species Human --input-scale counts \
  --gene-column standard_gene_id --layer native --batch-size 2
```

Use stable IDs matching the exact species namespace, such as `Human` or `Mouse`. A source `ensembl_id` column may include version suffixes; the Tabula example uses its existing unversioned `standard_gene_id` column. Omit `--gene-column` when `var_names` already contains stable IDs. Use `--input-scale counts` for counts, or `--input-scale log1p` for values already normalized to the model's 10,000-library-size, log1p scale. `--layer-key counts` selects a stored counts layer when needed. Unknown IDs fail by default, and no gene-symbol-to-ID guessing occurs.

The five-fold completion queries contain only permitted observed counts. Each target fold is absent before normalization and inference. The scorer alone reads separate target truth, normalized using the full supported measured-gene library.

`examples/expected/` contains tested CPU outputs. Validation results and scope are recorded in `docs/FIGURE_PROTOCOL_VALIDATION.json`.

The small denoising example did not outperform its noisy-input baseline. The validation report records the cohorts and metrics for these runnable examples.
