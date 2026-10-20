# CellNet Image Atlas

[Download the complete Image Atlas](https://pan.baidu.com/s/1FuoHnEJN9FRvOBio5uutYQ?pwd=xabe) · Extraction code: **xabe**.

A collection of single-cell transcriptome images with the gene identities, measurement masks and fitted pixel maps needed by CellNet. Source metadata describes 231,190,557 cell profiles, 26,467 source datasets and 1,015,163 gene identities across 40 species.

## Put the download in the right directory

Clone [CellNet](https://github.com/daydayup-xq/CellNet), download its model assets with Git LFS, and place **the contents of this download** directly in `CellNet/datasets/pretraining/full/`:

```text
CellNet/                         # --release points here
├── assets/                      # Published vocabulary and fitted layouts
├── weights/                     # Encoder, DNA-derived identities and decoder
└── datasets/pretraining/full/
    ├── README.md
    ├── README.zh-CN.md
    ├── DATASET_CARD.json
    ├── metadata.json
    ├── gene_vocab.json
    ├── cell_type_vocab.json
    ├── cell_type_storage_vocab.json
    ├── normalization_stats.json
    ├── noncoding_gene_ids.json
    ├── release_manifest.json
    ├── shards_manifest.json.gz
    ├── cell_sources.json.gz
    ├── dataset_sources.tsv
    ├── class_index.json
    ├── class_index.tsv
    ├── class_shards/class_*/shard-*.tar.gz
    ├── masks/
    │   ├── dataset_index.json
    │   ├── gene_coords.npy
    │   └── gene_masks.npy
    └── pixel_maps/
        ├── metadata.json
        ├── dataset_index.json
        ├── dataset_to_pixel_map.json
        ├── global_gene_coords.npy
        └── gene_to_flat_pixel.npy
```

`full/metadata.json` must exist directly at that path; remove an extra wrapper directory after downloading. Keep all published paths and names. Shards may be `.tar` or `.tar.gz`; **leave them packed** because the reader streams their records. A `class_shards/unlabeled/` directory, if supplied, is part of the same collection. `release_manifest.json` lists the shared assets and references `shards_manifest.json.gz`, the compressed inventory of all 512,835 native shards. Keep both manifests with the download.

## Pretraining compatibility

Keep `cell_sources.json.gz` alongside `metadata.json`. For the merged collection of **44,489,429 cells from 189 source datasets**, this compact index identifies each cell's source. The native reader verifies the index against `release_manifest.json` and automatically selects the corresponding measurement mask.

Pretraining reuses the published gene-to-image mapping. Downstream applications use the matching model `assets/` and `weights/` for your H5AD input; the full Image Atlas is optional for those workflows.

## Verify and prepare a pretraining batch

Set the absolute clone path once. The following commands work from any current directory:

```bash
export CELLNET_RELEASE="/absolute/path/to/CellNet"
git -C "$CELLNET_RELEASE" lfs pull
python -m pip install -e "$CELLNET_RELEASE[h5ad]"

python "$CELLNET_RELEASE/examples/verify_image_atlas.py" \
  --release "$CELLNET_RELEASE" \
  --data "$CELLNET_RELEASE/datasets/pretraining/full"

python "$CELLNET_RELEASE/examples/prepare_pretraining_batch.py" \
  --release "$CELLNET_RELEASE" \
  --data "$CELLNET_RELEASE/datasets/pretraining/full" \
  --output "$CELLNET_RELEASE/prepared_views.pt" \
  --cells 2 --views 2

python "$CELLNET_RELEASE/examples/pretrain.py" \
  --release "$CELLNET_RELEASE" --init cellnet \
  --batch "$CELLNET_RELEASE/prepared_views.pt" \
  --n-global 2 --steps 1 --device cpu \
  --output "$CELLNET_RELEASE/continued_weights"
```

This demonstrates one update on a small prepared batch. For training, stream batches and choose the device and schedule for your workload; see the repository's [pretraining guide](https://github.com/daydayup-xq/CellNet/blob/main/docs/PRETRAINING.md). The batch preparer also selects `datasets/pretraining/full/` automatically when it is installed under `--release`; otherwise it uses the separate 24-cell example.

The verifier checks the published file inventory and sizes, fitted mapping compatibility and a bounded sample of real records. `--checksums` reads the catalogued files and compares any published verified digests; see its report for the checksum coverage. It does not infer exhaustive cell counts from a short sample.

## One fitted mapping for pretraining and downstream use

Both workflows read the **already fitted** assets from `--release`. The native reader checks its vocabulary and species pixel positions against `assets/gene_vocab.json` and `assets/species_layouts.npz`. The downstream converter reuses those same fixed positions automatically. Keep `assets/` and `weights/` from the same release; no layout refitting is required.

For your own H5AD, the full cell collection is optional. Use the release assets directly:

```bash
python "$CELLNET_RELEASE/examples/embed_h5ad.py" \
  --release "$CELLNET_RELEASE" \
  --input "/path/to/cells.h5ad" --output "/path/to/embeddings.npy" \
  --species Human --input-scale counts --batch-size 8 --device cpu
```

Use the exact species namespace and stable gene IDs. `counts` means nonnegative counts; `log1p` means the model-compatible library-size-10,000, log1p scale. See [task examples](https://github.com/daydayup-xq/CellNet/blob/main/docs/TASK_EXAMPLES.md) for expression prediction, gene association and other downstream applications.

DNA-derived, frozen **256-dimensional gene identities** are included in `weights/cellnet.pt`, aligned to the released vocabulary. This Image Atlas contains rendered expression, masks and mappings; it does not contain the original 1,920-dimensional DNA embedding matrix. Existing genes use the frozen identities automatically. Registering new genes requires compatible raw embeddings and the optional fitted projection assets described in the model repository.

## Data format and source information

Stored expression is float16 in [0,1] and is already normalized. Do not normalize it again. The reader returns `image` as float32 `[2,384,384]` (expression and measurement mask) and `gene_ids` as int64 `[384,384]`. A measured zero has mask=1 and a positive gene ID; an unmeasured pixel has expression=0, mask=0 and ID=0.

`masks/dataset_index.json` maps source IDs to mask rows. `pixel_maps/dataset_index.json` maps species to fitted pixel-map rows. `dataset_sources.tsv` joins the numeric dataset ID, public source identifier and species without changing any indices. `cell_type_vocab.json` defines biological labels; `class_index.json` and `.tsv` help navigate storage buckets, which may combine several labels. Blank bucket fields mean the historical index did not record the relation; the native reader uses the labels within each cell record.

The headline cell count is the original metadata total; the release manifest records the scope of the upload and integrity checks. Source datasets retain their attribution and terms. The model repository's Apache-2.0 license applies to software and does not replace the original data licenses.

## Try 24 real cells first

`CellNet_pretraining_example_24cells.zip` is a separate starter package: eight cells each from Mouse, Zebrafish and Fruit_fly, with expression, measurement masks, gene IDs, vocabulary and fitted layouts. Extract it to use `CellNet/datasets/pretraining/example/`. The complete native atlas is available through the download link above.

This website's `SHA256SUMS` and `FILE_CATALOG.json` cover the starter ZIP and accompanying website documentation. The complete atlas carries its own `release_manifest.json` and `shards_manifest.json.gz` in the download.
