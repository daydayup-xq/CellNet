# CellNet pretraining NPZ format v1

A dataset directory contains `metadata.json`, `cells.csv` and `shards/*.npz`. Both the bundled example and future full release use the same reader. All files are real files, with no Python object pickle or absolute machine paths.

## Dataset metadata

Required fields are:

- `format`: `cellnet-pretraining-npz-v1`.
- `status`: `example` or `available`; `not_included` means an empty future-data placeholder and fails explicitly on read.
- `samples`: exact published cell count, not the original population size.
- `image_size`: `[384, 384]`.
- `cells`: relative CSV path.
- `shards`: list of `{ "path": "shards/shard-00000.npz", "samples": N, "sha256": "..." }` records, in read order.

`cells.csv` has a unique anonymous `sample_id` for every cell, plus `species`, `cell_type`, `source_dataset` and `layout_index`. The bundled CSV also records measured/detected-gene counts and mean expression over measured pixels for the figures. `layout_index` is local to the cell's shard. Cell type and species are descriptive fields; they do not introduce classification, prototype or alignment loss terms.

## Per-shard arrays

Write with `np.savez_compressed`; read with `allow_pickle=False`.

| Key | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `expression` | float16 | `[N,H,W]` | Stored production expression, clipped log1p / 6 in [0,1] |
| `measurement_mask` | bool | `[L,H,W]` | Unique measured-gene layouts used by this shard |
| `gene_ids` | int32 | `[L,H,W]` | Release vocabulary index + 1; zero at unmeasured pixels |
| `layout_index` | int32 | `[N]` | Index in `[0,L)` selecting each cell's mask/ID layout |
| `sample_ids` | Unicode | `[N]` | Unique anonymous IDs matching `cells.csv` |

Deduplicating masks/IDs within a shard avoids storing identical layout arrays for every cell. Do not replace dataset-specific measurement masks with full species masks: that would falsely mark unmeasured genes as measured. Do not remove gene IDs merely because expression is zero.

The reader returns float32 two-channel images `[2,H,W]` and int64 gene IDs `[H,W]`. The first channel is expression, the second is the measurement mask. It checks finite expression in [0,1], mask/ID equivalence, zero expression outside measured pixels, layout references, checksums and metadata coverage. Global gene identity rows must match the existing CellNet weights and `assets/gene_vocab.json`.

## Add full data later

1. Render using the same fixed normalization, frozen vocabulary and fitted layouts. Preserve float16 storage precision. Do not refit the converter.
2. Write bounded shards (for example 256 cells per shard) and matching unique CSV rows. Ensure every cell's expression, mask and IDs refer to the same layout.
3. Hash each final NPZ and record its count/path/SHA256. Set `samples` to the sum of actual included cells, `status` to `available`, and supply actual source attribution/license records.
4. Verify the complete dataset through `iter_pretraining_cells('datasets/pretraining/full')`. For large releases, keep the full data separately downloadable instead of putting it in the software wheel.
5. Regenerate release checksums after adding files. The original manifest's population counts cannot establish that corresponding data files are present.

Files in `full/` currently declare no available full data. No large training shard or the original approximately 27-GB global gene-mask table is bundled.
