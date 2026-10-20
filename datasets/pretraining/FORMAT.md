# CellNet pretraining data formats

`iter_pretraining_cells(directory, release=release)` reads both the native Image Atlas and the bundled portable NPZ example. Both return the same model inputs: expression/mask images and gene identities using the release's fixed vocabulary and fitted native layouts.

## Native Image Atlas

Download the published dataset into `datasets/pretraining/full/`. `metadata.json` sits at that directory's root with `class_shards/`, `masks/`, `pixel_maps/`, `gene_vocab.json` and `cell_type_vocab.json`. Preserve their relative paths and keep the `.tar.gz` shards compressed. See the [directory guide](README.md#directory-layout) for the tree.

Native data use compact species-injective layouts and dense float16 images (`sample_storage: dense_npy`). Release metadata may identify this format as `cellnet-image-atlas-native-v1`; the reader also accepts the original native metadata. The root `release_manifest.json` records shared assets and may reference `shards_manifest.json.gz` via `shard_manifest`. The compressed JSON has format `cellnet-image-atlas-shards-v1` and a `files` list of `{path, bytes, ...}` entries under `class_shards/`. Include the compressed inventory itself in the root manifest with its byte size and SHA256. `examples/verify_image_atlas.py` checks both lists; remote server fingerprints are provenance, not independently verified local checksums.

Each cell has adjacent tar members sharing a record key:

| Member suffix | Contents |
| --- | --- |
| `.npy` | float16 expression `[384,384]`, already normalized to [0,1] |
| `.species` | Exact native species namespace |
| `.dataset_id` | Source dataset index; declared aggregate IDs are resolved through the ownership index |
| `.cell_type` | Integer resolved through `cell_type_vocab.json` |

The shared files reconstruct model inputs:

- `gene_vocab.json` fixes global gene order. It must match `assets/gene_vocab.json` in `--release`.
- `masks/dataset_index.json` maps source datasets to rows in `masks/gene_masks.npy`. These masks mark measured genes for each source dataset.
- `pixel_maps/dataset_index.json` maps species to rows in `pixel_maps/gene_to_flat_pixel.npy`. The selected native gene positions must match `assets/species_layouts.npz`.
- The reader combines the dataset's measured-gene mask with its species' native gene positions to construct a per-cell visibility map and gene IDs. Vocabulary index + 1 is the model identity; zero is reserved for unmeasured pixels.

`metadata.json` can declare `ownership_file: "cell_sources.json.gz"` and `ownership_source_ids: [26278]`. The compressed JSON uses format `cellnet-image-atlas-ownership-v1`, `source_dataset_ids`, `key_stride: 10000000`, total `samples`, and `buckets`. Each `class_N` entry has `intervals` containing `[start, stop, dataset_mask_row, dataset_uuid]`, with half-open cell ordinals. The record key is `N * key_stride + ordinal`. Intervals are contiguous from zero and cannot exceed the key stride; their total must equal the declared sample count. Each UUID must match its exact row in `masks/dataset_index.json`.

The ownership file must appear in `release_manifest.json` with its SHA256. Only records whose stored source ID is declared in `ownership_source_ids` use this index. The reader retains the original value as `stored_dataset_id` and returns the resolved row as `dataset_id`. Missing coverage, inconsistent keys, incorrect hashes and unknown dataset references raise errors. A release may explicitly mark incomplete sources in `unresolved_dataset_ids`; those records are rejected before constructing measurement masks.

The reader streams cells from the native shards and memory maps the large shared arrays. Existing image coordinates are retained throughout loading and batch preparation. The source dataset's measurement mask distinguishes a measured zero from an unavailable gene.

Verify a download before using it:

```bash
export CELLNET_RELEASE="/absolute/path/to/CellNet"
python "$CELLNET_RELEASE/examples/verify_image_atlas.py" \
  --release "$CELLNET_RELEASE" \
  --data "$CELLNET_RELEASE/datasets/pretraining/full"
```

The default check compares the file inventory and byte sizes, validates mapping identity and loads a small sample. Add `--checksums` for full file hashing. The published data inventory supplies the number and scope of downloaded records.

## Portable NPZ format v1

The 24-cell `example/` contains `metadata.json`, `cells.csv` and `shards/*.npz`. This format is also useful for exporting small subsets of native data. It stores real numeric arrays and relative paths.

Required metadata fields are:

- `format`: `cellnet-pretraining-npz-v1`.
- `status`: `example` or `available` for populated datasets.
- `samples`: exact included cell count.
- `image_size`: `[384,384]`.
- `cells`: relative CSV path.
- `shards`: list of `{ "path": "shards/shard-00000.npz", "samples": N, "sha256": "..." }` records, in read order.

`cells.csv` has a unique anonymous `sample_id` for every cell, plus `species`, `cell_type`, `source_dataset` and `layout_index`. `layout_index` is local to the cell's shard. The bundled CSV also records measured/detected-gene counts and mean measured expression for its figures.

Write shards with `np.savez_compressed`; read with `allow_pickle=False`:

| Key | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `expression` | float16 | `[N,H,W]` | Stored normalized expression in [0,1] |
| `measurement_mask` | bool | `[L,H,W]` | Unique measured-gene layouts used by this shard |
| `gene_ids` | int32 | `[L,H,W]` | Release vocabulary index + 1, zero at unmeasured pixels |
| `layout_index` | int32 | `[N]` | Index in `[0,L)` selecting each cell's mask/ID layout |
| `sample_ids` | Unicode | `[N]` | Unique anonymous IDs matching `cells.csv` |

Deduplicating masks and identities within a shard keeps the package compact. Keep dataset-specific measurement masks: a full species layout also contains genes that may not have been measured in a given experiment. Measured zero expression retains a positive ID.

The reader checks shapes, dtypes, finite normalized expression, mask/ID alignment, layout references, shard hashes and CSV coverage. It returns float32 images `[2,H,W]` with expression and measurement-mask channels, plus int64 gene IDs `[H,W]`.

## Export a compact subset

The full Image Atlas is read natively. To create a portable subset, use the exporter with the same release assets:

```bash
python "$CELLNET_RELEASE/examples/export_pretraining_data.py" \
  --release "$CELLNET_RELEASE" \
  --source "$CELLNET_RELEASE/datasets/pretraining/full" \
  --output "/path/to/exported_examples" \
  --storage-classes class_536 class_537 class_1020 class_1035 class_146 class_170 \
  --per-class 4 --shard-size 8
```

This example writes up to four cells per selected storage class using the existing expression, masks and identities. Choose classes available in your downloaded atlas. The exporter writes its manifest after completing the output and records hashes and counts for the actual subset. Keep source attribution and upstream terms with redistributed subsets.
