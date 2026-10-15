# Representative real datasets

The TotalX PBMC fixture contains 96 cells: four annotated cell types × three technical assays × eight cells, selected with seed 20261015 before inference. Cell identifiers are anonymous example IDs. The Tabula Sapiens fixture contains 192 cells from two tissues with three donors per tissue and four shared cell types per tissue.

| File | Contents |
| --- | --- |
| `totalx_pbmc_96.h5ad` | 96 cells × 36,499 supported genes; source log1p `X` and matching `layers/counts` |
| `tabula_sapiens_192.h5ad` | 192 cells × 59,556 supported genes; source log1p expression, donor and tissue labels |
| `target_panel.csv` | 48 coding and 16 lncRNA targets for association and denoising examples |
| `completion_fivefold/` | Five eight-cell query files, target-ID lists and separate truth arrays; whole target folds are physically absent from inputs |
| `denoising_noisy.h5ad` | Eight real PBMC profiles after deterministic 50% binomial thinning |
| `denoising_truth.npz` | Separate clean targets for the denoising profiles |
| `PROVENANCE.json` | PBMC cohort, assay, seed and input-scale records |
| `TABULA_PROVENANCE.json` | Tabula cohort selection and source expression details |
| `COMPLETION_FIVEFOLD.json` | Gene-axis partition and separate input/scoring normalization definitions |

Tabula uses the source `standard_gene_id` column for unversioned IDs. Annotation uses within-tissue leave-one-donor-out evaluation. Reference mapping uses five stratified 80/20 PBMC splits. The PBMC `obs/split` field defines the fixed cell groups used for panel selection and completion sampling.

Completion input normalization uses observed counts only. Scorer truth is normalized using the full supported measured-gene library and remains separate from model input. These are reduced-cohort task examples, not the full publication benchmarks.

Data sources: [TotalX study](https://www.nature.com/articles/s41587-026-03068-6), [TotalX processed matrices](https://doi.org/10.6084/m9.figshare.29803628), and [Tabula Sapiens 2.0](https://figshare.com/articles/dataset/Tabula_Sapiens_v2/27921984). Third-party data retain their upstream terms and attribution; the code license does not relicense these data. No specific TotalX data-license identifier is asserted here.
