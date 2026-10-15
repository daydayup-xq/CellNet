# Full pretraining data: reserved location

The complete training samples are **not included**. This directory currently contains only an empty manifest, a CSV header and an empty `shards/` directory.

Place future real NPZ shards under `shards/`, fill `cells.csv`, and update `metadata.json` with actual cell counts, hashes, paths and `status: available`. Follow [FORMAT.md](../FORMAT.md). Use `../example/` for the included real examples. No download address or full-data availability is asserted here.
