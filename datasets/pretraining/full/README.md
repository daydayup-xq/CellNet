# Install the Image Atlas here

Download the **[Image Atlas](https://pan.baidu.com/s/1FuoHnEJN9FRvOBio5uutYQ?pwd=xabe)** and put the contents of the release directory here. The installed paths must be `full/metadata.json`, `full/class_shards/`, `full/masks/`, `full/pixel_maps/`, `full/gene_vocab.json` and `full/cell_type_vocab.json`. Keep `cell_sources.json.gz`, the accompanying data card, source catalog and file manifest. The source index selects each merged cell's original measurement mask automatically. Leave the native `.tar.gz` shards compressed.

The `metadata.json` checked into Git is an empty installation marker. Replace it with the downloaded release's `metadata.json`; do not retain an additional wrapper directory such as `full/Image_Atlas/`.

The batch preparation command automatically uses this directory after installation. It loads the exact fitted vocabulary and native species layouts from the clone's `assets/`, alongside `weights/`. You do not need to fit a new mapping or configure a separate mapping path.

See the [download and verification guide](../README.md) for the directory tree, validation command, partial-download selection and pretraining examples. If the full dataset is not installed, batch preparation uses the separate [24-cell example](../example/).
