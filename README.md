# CellNet

[中文说明](README.zh-CN.md)

CellNet is a gene-identity-aware single-cell foundation model. It converts a cell's expression profile into a two-channel image and extracts 768-dimensional cell representations. Its expression decoder predicts genes from an observed-only cell context.

The main model weights occupy about 946 MB. The optional fitted projection index occupies about 992 MB and supports out-of-sample genes without refitting.

The project provides pretrained models, fitted gene-to-image converters, pretraining tools and examples for cell annotation, reference mapping, batch integration and gene-level analysis.

## Contents

| Path | Purpose |
| --- | --- |
| `cellnet/` | Model, expression decoder, fitted image conversion and pretraining core |
| `weights/cellnet.pt` | Pretrained encoder, frozen gene identity table and expression decoder |
| `weights/projector.pt` | Pretrained projection head used by the pretraining example |
| `weights/config.json` | Active model dimensions and input/readout parameters |
| `assets/` | Stable gene vocabulary, native species layouts and frozen new-gene projection assets |
| `examples/` | Runnable downstream tasks, real pretraining batch preparation and visualization |
| `datasets/pretraining/` | 24 real training examples, data schema, visualizations and a reserved full-data location |
| `tests/` | Inference, preprocessing and gradient checks |

## Installation

Install [Git LFS](https://git-lfs.com/) to download the model weights, fitted assets and example data, then clone the repository. Use Python 3.10 or newer.

```bash
git lfs install
git clone https://github.com/daydayup-xq/CellNet.git
cd CellNet
git lfs pull
pip install -e '.[h5ad]'
```

Weights and layouts are distributed alongside the Python package. Pass the release directory explicitly when loading them. The Python wheel contains code; retain `weights/` and `assets/` when moving the release. CPU inference uses PyTorch SDPA. No pretrained weights are downloaded automatically.

Validated environment: Python 3.11.15, PyTorch 2.9.1, timm 1.0.22, NumPy 2.3.0 and SciPy 1.17.0. Other allowed dependency versions have not all been tested.

## Representative real-data tasks

[Task examples](docs/TASK_EXAMPLES.md) provide runnable uses of **Tabula Sapiens** (192 real cells) and **TotalX PBMC** (96 real cells), with separate completion and denoising truths. The examples cover cell annotation, within-species reference mapping, batch representation diagnostics, hidden-gene completion, contextual gene association and native-noise denoising.

```bash
pip install -e '.[examples]'
python examples/run_all_tasks.py --device cpu --batch-size 2 --output example_outputs
```

Read [the data provenance](examples/data/README.md) and [the task validation report](docs/FIGURE_PROTOCOL_VALIDATION.json) for exact input scales, splits and validation scope. The examples are small demonstrations, not full publication benchmarks.

## Pretraining dataset examples

[Pretraining data](datasets/pretraining/README.md) includes 24 real Mouse, Zebrafish and Fruit_fly cells from the existing 100,000-cell training sample, with original expression, measured-gene masks and native gene IDs. It also provides a data reader, independent masked-view batch preparation, PNG/SVG visualizations and an explicit empty location for the future full dataset. Full pretraining samples are not included.

## Quick start

```python
from pathlib import Path
import numpy as np
import torch
from cellnet import CellNet, CellNetImageConverter

release = Path('/path/to/CellNet')
converter = CellNetImageConverter(release, species='Human')
model = CellNet.from_pretrained(release, device='cpu')

# Cells x genes. Use stable IDs belonging to the selected species.
genes = ['ENSG00000121410', 'ENSG00000268895']
expression = np.array([[1.2, 0.0], [0.0, 2.1]], dtype=np.float32)
batch = converter.transform(expression, genes, input_scale='log1p')

with torch.inference_mode():
    embedding = model(**batch)  # [cells, 768]
```

Species names are exact namespaces such as `Human` and `Mouse`. The 40 fitted layouts are stored in `assets/species_layouts.npz`; list their namespaces with `np.load(release / 'assets/species_layouts.npz', allow_pickle=False)['species'].tolist()`. Unknown genes raise an error by default. `allow_unknown=True` explicitly drops unmatched genes and records them in `converter.last_report`; it does not invent identity vectors or pixel positions.

## Expression and measurement semantics

Use `input_scale='counts'` for nonnegative counts: the converter applies library-size normalization to 10,000, `log1p`, clipping to [0, 6], and division by 6. By default expression values then make the same float16 storage round-trip as the model input pipeline, returning float32 tensors for inference. Set `storage_dtype="float32"` explicitly for unquantized images. Use `input_scale='log1p'` for values already normalized to the same expression scale. No automatic scale guessing is performed.

Every supplied gene is measured unless marked otherwise. A measured zero has a mask value of one and retains its identity. An unavailable gene has zero expression, zero measurement mask and gene ID zero. Pass `measured=` as a boolean vector or cells-by-genes matrix when necessary. Duplicated gene IDs must be combined before conversion.

For gene-completion evaluation, hide targets **before** normalization and conversion; compute size factors from permitted observed inputs or supply a separately justified fixed normalization. Normalizing with hidden target counts would leak target information. Keep truth values in separate arrays. The bundled completion example physically omits target columns and stores target truth in a separate file.

## Apply the fitted converter to new genes or species

For genes already in a fitted species layout, `CellNetImageConverter` uses the exact native positions without loading the larger projection index. To register genes outside that vocabulary, install the optional projection dependencies:

```bash
pip install -e '.[projection]'
```

```python
# vectors: [genes, 1920] compatible raw gene identity embeddings.
# The feature space must match the pretrained gene identities.
converter = CellNetImageConverter.from_embeddings(
    release, model, species='NewSpecies', genes=gene_names, embeddings=vectors
)
batch = converter.transform(expression, gene_names, input_scale='log1p')
with torch.inference_mode():
    embedding = model(**batch)
```

This applies the **original fitted** L2 normalization, PCA, t-SNE reference space, nearest-neighbor index and identity IPCA; it does not refit a new map. It uses the original balanced pixel-assignment algorithm to resolve collisions. Existing native genes keep their positions and identity rows. The model gains frozen identity rows for new genes in memory, while its learned encoder/decoder parameters remain fixed. Use a fresh model for each independently registered panel.

The raw 1,920-D vectors must come from the same feature space and preprocessing used for the pretrained identities. Gene symbols alone or embeddings from a different model cannot substitute for them. Generation of those raw embeddings is outside this release. The projection assets are numeric arrays and a native index, with no Python object pickle or training affinity graph. Synthetic out-of-sample projection is tested; this is not a biological validation of arbitrary new species.

## Cell embeddings and readouts

```bash
python examples/embed_h5ad.py \
  --release . --input cells.h5ad --output embeddings.npy \
  --species Human --input-scale counts --batch-size 8 --device cpu
```

The input defaults to `adata.X` with stable gene IDs in `var_names`. Select a layer with `--layer-key` or an ID column with `--gene-column`. The output preserves input cell order. Process large datasets in batches because images and integer gene maps are dense. The example reads dense, CSR and CSC expression directly from HDF5; CSR is preferred for efficient row batches. Output is published only after every batch succeeds.

The default `native` readout is **LayerNorm of the sum of the raw CLS tokens from the last eight blocks**. `--layer layer8` and `--layer layer6` select raw CLS tokens eight and six blocks from the end, respectively. `--layer layer8_normalized` selects the individually normalized eighth-from-last token; `--layer last` selects final normalized CLS. See [figure protocols](docs/FIGURE_PROTOCOLS.md) for the actual downstream estimators and hyperparameters.

## Gene completion

```python
target_ids = converter.resolve_gene_ids(['ENSG00000121410'])
with torch.inference_mode():
    prediction = model.predict(**batch, target_gene_ids=target_ids)
    expression_log1p = prediction['expression_log1p']
```

`predict` removes target expression, measurement masks and identities before encoding. Targets must belong to the released vocabulary. `encode` plus `decode` supports repeated queries on a precomputed **observed-only** context; callers are responsible for hiding all requested targets before creating that context. Confidence output is a predicted absolute-error estimate, not a guarantee of accuracy.

## Pretraining

CellNet supports initialization from [DINOv3 ViT-B/16](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) and continued training from the provided CellNet weights. Download DINOv3 separately from Hugging Face; the training command reads your local `model.safetensors` file.

```bash
pip install -e '.[pretraining]'
hf auth login
hf download facebook/dinov3-vitb16-pretrain-lvd1689m model.safetensors \
  --local-dir ../pretrained_models/dinov3-vitb16
python examples/prepare_pretraining_batch.py --output prepared_views.pt
python examples/pretrain.py --release . --init dinov3 \
  --dinov3-weights ../pretrained_models/dinov3-vitb16/model.safetensors \
  --batch prepared_views.pt --steps 1 --device cpu --output trained_weights
```

Request access on the model page before downloading. See [the pretraining guide](docs/PRETRAINING.md) for the compatible timm download, initialization details, batch format and continued-training command. DINOv3 weights are external training inputs; inference uses `weights/cellnet.pt`.

## Verification

```bash
python -m unittest discover -s tests -v
python examples/verify_release.py --release .
# Optional: validate the large fitted index and new-gene inference.
CELLNET_TEST_PROJECTION=1 python -m unittest discover -s tests -p test_projection.py -v
```

[Task validation](docs/FIGURE_PROTOCOL_VALIDATION.json) records representative real-data checks, and [DINOv3 initialization validation](docs/DINOV3_INITIALIZATION_VALIDATION.json) records training initialization and update checks. `SHA256SUMS` covers all delivered files except itself. Large binary files are covered by the included Git LFS patterns. Use Git LFS or a model artifact host when publishing them. Code is licensed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Datasets used in Figures 2–5

The following studies supply the main and associated supplementary analyses. Links point to the original study or its official data repository; full datasets are obtained from these sources.

| Figure | Study / dataset | Official links |
| --- | --- | --- |
| 2 | Tabula Sapiens 2.0 — human multi-organ atlas | [Figshare](https://figshare.com/articles/dataset/Tabula_Sapiens_v2/27921984) |
| 2 | An atlas of healthy and injured cell states and niches in the human kidney | [Study](https://www.nature.com/articles/s41586-023-05769-3) · [CELLxGENE](https://cellxgene.cziscience.com/collections/bcb61471-2a44-4d00-a0af-ff085512674c) |
| 2 | A molecular cell atlas of the human lung from single-cell RNA sequencing (Travaglini et al.) | [Study](https://www.nature.com/articles/s41586-020-2922-4) · [CELLxGENE](https://cellxgene.cziscience.com/collections/5d445965-6f1a-4b68-ba3a-b8f765155d3a) |
| 2, 4 | Single-cell transcriptomics identifies an effectorness gradient shaping the response of CD4+ T cells to cytokines (Cano-Gamez et al.) | [Study](https://www.nature.com/articles/s41467-020-15543-y) · [BioStudies S-BSST2978](https://www.ebi.ac.uk/biostudies/studies/S-BSST2978) |
| 2 | Single-cell transcriptomic analysis uncovers diverse and dynamic senescent cell populations (WI-38) | [Study](https://www.aging-us.com/article/204666/text) · [GSE226225](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE226225) |
| 2 | Seven-species circulating immune-cell atlas (author H5ad collection) | [Figshare 28008611](https://doi.org/10.6084/m9.figshare.28008611) |
| 2, 3 | The molecular evolution of spermatogenesis across mammals | [Study and data availability](https://www.nature.com/articles/s41586-022-05547-7) · [Author portal](https://apps.kaessmannlab.org/SpermEvol/) |
| 2 | Recruitment, rewiring, and deep conservation in flowering plant gene regulation (Brassicaceae multi-tissue atlas) | [GSE297576](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE297576) |
| 2 | Evolutionary diversity of cell-type-specific expression and stress response in Brassicaceae roots | [GSE268881](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE268881) |
| 3, 4, 5 | Scalable single-cell total RNA sequencing unifies coding and noncoding transcriptomics (TotalX: PBMC, developing brain, HEK293T and Huh7, including DENV conditions) | [Study](https://www.nature.com/articles/s41587-026-03068-6) · [Processed matrices](https://doi.org/10.6084/m9.figshare.29803628) · [Cell-line data GSE315939](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE315939) |
| 3 | Xenium Prime: Fresh Frozen Mouse Brain Hemisphere with 5K Mouse Pan Tissue and Pathways Panel | [10x Genomics dataset](https://www.10xgenomics.com/datasets/xenium-prime-fresh-frozen-mouse-brain) |
| 5 | Massively parallel digital transcriptional profiling of single cells (Zheng et al.; purified PBMC) | [Study](https://www.nature.com/articles/ncomms14049) · [Processed data loader](https://docs.scvi-tools.org/en/1.5.0/api/reference/scvi.data.purified_pbmc_dataset.html) |
| 5 | Human Brain Cell Atlas v1.0 — perirhinal cortex | [CELLxGENE collection](https://cellxgene.cziscience.com/collections/283d65eb-dd53-496d-adb7-7570c7caa443) · [scGPT processed data](https://github.com/bowang-lab/scGPT/blob/main/data/README.md) |
| 5 | High-throughput total RNA sequencing in single cells using VASA-seq | [Study](https://www.nature.com/articles/s41587-022-01361-8) · [GSE176588](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176588) |
| 5 | Dual genome-wide coding and lncRNA screens in neural induction of induced pluripotent stem cells | [Study](https://doi.org/10.1016/j.xgen.2022.100177) · [GSE150062](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE150062) |
