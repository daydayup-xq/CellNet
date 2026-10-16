# CellNet Image atlas

## Available now: 24 real pretraining examples

Download `CellNet_pretraining_example_24cells.zip` and extract it alongside the CellNet code. It contains eight Mouse, eight Zebrafish and eight Fruit_fly cells, shared vocabulary and fitted species layouts, format documentation and two previews. The archive provides transcriptome expression renderings, dataset-specific measurement masks and native gene IDs from the repository examples.

After installing CellNet, read the extracted data with:

```python
from cellnet.pretraining_data import iter_pretraining_cells
for cell in iter_pretraining_cells('CellNet/datasets/pretraining/example'):
    image, gene_ids = cell['image'], cell['gene_ids']
```

Expression is already normalized to [0,1]. Measured zero retains mask=1; unmeasured pixels have mask=0 and gene ID=0. Preserve this distinction. SHA256SUMS and FILE_CATALOG.json describe every public file and the archive members.

## Image atlas: coming soon

The full CellNet Image atlas will be available here. Source metadata describes an atlas of 231,190,557 samples, 26,467 datasets and 1,015,163 vocabulary entries across 40 species. The starter package provides the 24 real examples described above. `FULL_DATA_STATUS.json` records the full collection's availability as `coming_soon`.

## Provenance and terms

Public sample IDs in the example are anonymous; the CSV retains species, cell type and source-dataset attribution. These 24 examples support exploring the CellNet data format and using the data reader. Third-party datasets retain their original terms and attribution. Apache-2.0 applies to CellNet software, not automatically to measurements; per-source data license verification remains pending in the repository documentation.

## Clean directory policy

The starter package is organized into documented examples, vocabulary, layouts, previews and release metadata. FILE_CATALOG.json provides the file inventory. The full Image atlas release will provide an exact file inventory and checksums when available.
