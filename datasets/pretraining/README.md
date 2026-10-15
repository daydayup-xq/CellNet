# Pretraining data

This directory reserves a clear place for the full CellNet pretraining dataset and includes a small real format demonstration. **The full dataset is not included.**

| Path | Status / contents |
| --- | --- |
| `example/` | Available: 24 real cells extracted from the existing 100,000-cell training sample, with expression, measured-gene masks, gene IDs and anonymous sample IDs |
| `full/` | Reserved: empty shard list and zero published samples; add the full data here later |
| `figures/` | Rendered real expression/mask examples, cohort diagnostics, SVG copies and plotting source data |
| `FORMAT.md` | Array schema and instructions for adding future shards |
| `dataset_card.json` | Data scope, provenance, terms and limitations |

The examples contain eight cells each from Mouse, Zebrafish and Fruit_fly: two source cell annotations per species, four cells per annotation. Selection uses fixed source classes and record order, before inspecting model outputs. This is a tiny technical example, not a representative sample of the full pretraining distribution. Source annotations are retained without independent biological reannotation.

The original float16 expression arrays are preserved bitwise. Dataset-specific measured-gene masks and native gene IDs are materialized for six layouts. The package uses a compact NPZ format rather than copying the large original global mask table or embedding machine-specific paths. Existing `assets/` contains the shared vocabulary and fitted species layouts; no soft links are required.

## Read the data

After installing CellNet, run from the release directory:

```python
from cellnet.pretraining_data import iter_pretraining_cells

for sample in iter_pretraining_cells('datasets/pretraining/example'):
    print(sample['metadata']['sample_id'], sample['metadata']['species'])
    print(sample['image'].shape, sample['gene_ids'].shape)
    # image: float32 [2, 384, 384], channels expression and measurement mask
    # gene_ids: int64 [384, 384], zero only at unmeasured pixels
```

Stored expression is already normalized to [0,1]. Do not apply counts normalization or log1p again. Measured zero expression retains mask=1 and a positive gene ID. Unmeasured pixels have expression=0, mask=0 and ID=0. The reader validates shard SHA256, shapes, dtypes, masks, IDs and sample metadata.

## Make a real multi-view batch

```bash
python examples/prepare_pretraining_batch.py --output prepared_views.pt \
  --cells 2 --views 2 --keep-fraction 0.75 --seed 20261015
python examples/pretrain.py --release . --batch prepared_views.pt \
  --n-global 2 --steps 1 --device cpu --output continued_weights
```

Each view independently masks measured genes while keeping image/mask/ID aligned. View index refers to the same cell. This demonstrates independent masked-view preparation for CellNet pretraining. The preparation command alone does not train the model. Full-size model optimization needs substantial memory and compute; choose the appropriate device for actual training. See [the minimal training documentation](../../docs/PRETRAINING.md).

## Recreate the figures

```bash
pip install -e '.[visualization]'
python examples/visualize_pretraining_data.py --output pretraining_figures
```

![Real stored expression images and measurement masks](figures/expression_and_masks.png)

![Small example cohort diagnostics](figures/sample_summary.png)

The images show native pixel layouts, not microscopy. Gray expression pixels are unmeasured; measured zero is black. The plots use the stored normalized scale and do not infer raw counts or make model-performance claims. Measured/detected-gene counts refer only to genes represented in the model layout.

## Export native training shards

The portable exporter accepts a native DataBank with the matching vocabulary/layouts, reads its large mask tables through memory mapping, and writes bounded NPZ shards plus streamed CSV metadata. It never modifies source files and refuses to overwrite an output directory. A manifest is written only after successful export.

To reproduce the bundled cell selection from your existing 100,000-cell sample:

```bash
python examples/export_pretraining_data.py --source /path/to/100k_databank \
  --output exported_examples --storage-classes \
  class_536 class_537 class_1020 class_1035 class_146 class_170 \
  --per-class 4 --shard-size 8
```

This produces the same 24 images/masks/IDs in three shards; per-shard layout indices differ from the single bundled shard. All 24 were checked for exact equality after loading. To export a future full matching DataBank, omit the class and per-class limits, choose a new output directory, and use e.g. `--shard-size 256`. After checking source provenance/terms and output integrity, its actual files can populate `full/`. Do not change the placeholder's status until the data are present and verified.

## Full-data placeholder and upstream terms

`full/metadata.json` deliberately has `status: not_included`, `samples: 0`, and `shards: []`; the loader raises an explicit error if asked to read it. Fill `full/shards/`, `full/cells.csv` and the manifest following [FORMAT.md](FORMAT.md) when the actual data are available.

Original dataset identifiers are recorded in `example/cells.csv`. The 100,000-cell source pool and the 24-cell extraction are distinct datasets. No full-pretraining availability or accession is implied. Upstream datasets retain their own terms and attribution; the software Apache-2.0 license does not relicense these measurements. Individual dataset licenses have not yet been verified, so that check remains necessary before external distribution of third-party data.
