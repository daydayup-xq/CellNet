# CellNet Image Atlas

## Image Atlas

The CellNet Image Atlas is available through [Baidu Netdisk](https://pan.baidu.com/s/1J9rDKjT3yEqelqt3kaSIeQ?pwd=xabe). Extraction code: **xabe**.

Source metadata describes a pretraining collection of 231,190,557 cell profiles, 26,467 datasets and 1,015,163 gene identities across 40 species. `FULL_DATA_STATUS.json` records the full collection's access details and availability.

## 24 real pretraining examples

Download `CellNet_pretraining_example_24cells.zip` and extract it alongside the CellNet code. It contains eight Mouse, eight Zebrafish and eight Fruit_fly cells, shared vocabulary and fitted species layouts, format documentation and two previews. The archive provides transcriptome expression renderings, dataset-specific measurement masks and native gene IDs from the repository examples.

After installing CellNet, read the extracted data with:

```python
from cellnet.pretraining_data import iter_pretraining_cells
for cell in iter_pretraining_cells('CellNet/datasets/pretraining/example'):
    image, gene_ids = cell['image'], cell['gene_ids']
```

Expression is already normalized to [0,1]. Measured zero retains mask=1; unmeasured pixels have mask=0 and gene ID=0. Preserve this distinction.

## Files and metadata

`SHA256SUMS` covers the website's starter archive and accompanying metadata. `FILE_CATALOG.json` lists these files and all nine members of the starter archive. The full Image Atlas is accessed through the Baidu Netdisk link above.

The starter package is organized into documented examples, vocabulary, fitted layouts, previews and release metadata. `DATASET_CARD.json` describes the example package and access to the full collection.

## Provenance and terms

Public sample IDs in the example are anonymous; the CSV retains species, cell type and source-dataset attribution. These 24 examples support exploring the CellNet data format and using the data reader. Third-party datasets retain their original terms and attribution. Apache-2.0 applies to CellNet software, not automatically to measurements; per-source data license verification remains pending in the repository documentation.
