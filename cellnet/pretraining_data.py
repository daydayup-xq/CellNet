"""Load rendered pretraining shards and make paired masked views."""

from pathlib import Path
from collections import OrderedDict
import csv
import gzip
import hashlib
import io
import json
import re
import tarfile
import numpy as np


def _inside(directory, relative):
    """Resolve a dataset-relative file without following paths outside it."""
    relative = Path(relative)
    path = (directory / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(directory):
        raise ValueError(f"Dataset path must stay inside its directory: {relative}")
    return path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_directory(release):
    root = Path(release).resolve() if release is not None else Path(__file__).resolve().parents[1]
    if not (root / "assets/species_layouts.npz").is_file() or not (root / "assets/gene_vocab.json").is_file():
        raise FileNotFoundError(
            "Fitted CellNet assets are missing. Keep the cloned release's assets/ "
            "directory and pass release='/path/to/CellNet' explicitly."
        )
    with (root / "assets/species_layouts.npz").open("rb") as handle:
        if handle.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise FileNotFoundError(
                "Fitted CellNet layouts are Git LFS pointers. Run git lfs pull "
                "in the release directory to download the actual assets."
            )
    return root


def default_pretraining_directory(release=None):
    """Use an installed full atlas first, otherwise the bundled NPZ example.

    A full atlas with invalid or incomplete assets is reported by the reader;
    it never silently falls back to the example after encountering an error.
    """
    root = _release_directory(release)
    full = root / "datasets/pretraining/full"
    manifest = full / "metadata.json"
    if manifest.is_file():
        metadata = json.loads(manifest.read_text())
        if metadata.get("status") != "not_included":
            return full
    return root / "datasets/pretraining/example"


def iter_pretraining_cells(directory, *, release=None, storage_classes=None, shards=None):
    """Yield image [2,H,W], gene_ids [H,W] and descriptive metadata as NumPy arrays.

    Images already contain normalized expression. Do not normalize
    them again. Species and cell labels are descriptive; they are not losses.

    Native Image Atlas tar shards are streamed using the release's exact fitted
    species layouts. Masks and pixel maps are memory mapped. Explicit classes
    or dataset-relative shard paths allow a partial download; otherwise the
    directory is treated as the whole downloaded atlas.
    """
    directory = Path(directory).resolve()
    metadata = json.loads(_inside(directory, "metadata.json").read_text())
    if metadata.get("status") == "not_included":
        raise FileNotFoundError(
            "Full pretraining samples are not included; use example/"
        )
    if metadata.get("format") == "cellnet-pretraining-npz-v1":
        if storage_classes is not None or shards is not None:
            raise ValueError("Class/shard selection applies to the native Image Atlas only")
        yield from _iter_npz_cells(directory, metadata)
    elif metadata.get("format") == "cellnet-image-atlas-native-v1" or (
        metadata.get("mapping_mode") == "compact_species_injective"
        and metadata.get("sample_storage", {}).get("format") == "dense_npy"
    ):
        yield from _iter_native_cells(directory, metadata, release, storage_classes, shards)
    else:
        raise ValueError("Unsupported pretraining data format")


def _iter_npz_cells(directory, metadata):
    with _inside(directory, metadata["cells"]).open(newline="") as handle:
        records = list(csv.DictReader(handle))
    by_id = {row["sample_id"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate sample IDs in cells.csv")
    seen = set()
    for shard in metadata["shards"]:
        path = _inside(directory, shard["path"])
        if _sha256(path) != shard["sha256"]:
            raise ValueError(f"Shard checksum mismatch: {shard['path']}")
        with np.load(path, allow_pickle=False) as data:
            expression = data["expression"]
            masks = data["measurement_mask"]
            gene_ids = data["gene_ids"]
            layout = data["layout_index"]
            names = data["sample_ids"]
        shape = tuple(metadata["image_size"])
        n = len(expression)
        if (
            expression.dtype != np.float16
            or expression.shape != (n, *shape)
            or masks.dtype != np.bool_
            or masks.ndim != 3
            or masks.shape[1:] != shape
            or gene_ids.shape != masks.shape
            or gene_ids.dtype != np.int32
            or layout.dtype != np.int32
            or layout.shape != (n,)
            or names.shape != (n,)
            or names.dtype.kind != "U"
            or n != shard["samples"]
        ):
            raise ValueError("Invalid shard array shapes or dtypes")
        if (
            not np.isfinite(expression).all()
            or (expression < 0).any()
            or (expression > 1).any()
            or (layout < 0).any()
            or (layout >= len(masks)).any()
            or (gene_ids < 0).any()
            or not np.array_equal(gene_ids > 0, masks)
        ):
            raise ValueError("Invalid expression, layout, mask or gene ID values")
        for i, sample_id in enumerate(names.tolist()):
            if sample_id in seen or sample_id not in by_id:
                raise ValueError("Missing or duplicate sample metadata")
            row = by_id[sample_id]
            index = int(layout[i])
            if int(row["layout_index"]) != index:
                raise ValueError("CSV and shard layout indices differ")
            visible = masks[index]
            if np.any(expression[i][~visible]):
                raise ValueError("Expression is nonzero at an unmeasured pixel")
            seen.add(sample_id)
            yield {
                "image": np.stack([expression[i].astype(np.float32), visible]),
                "gene_ids": gene_ids[index].astype(np.int64),
                "metadata": row,
            }
    if len(seen) != metadata["samples"] or seen != set(by_id):
        raise ValueError("Dataset sample count and cells.csv do not match")


class _NativeAssets:
    """Read-only fitted layouts plus a bounded dataset/species mask cache."""

    def __init__(self, directory, metadata, release):
        root = _release_directory(release)
        config = json.loads((root / "weights/config.json").read_text())
        self.shape = tuple(config["image_size"])
        n_genes = config["vocabulary_size"]
        if tuple(metadata["image_size"]) != self.shape:
            raise ValueError("Image Atlas dimensions differ from the fitted CellNet release")
        storage = metadata.get("sample_storage", {})
        if storage.get("format") != "dense_npy" or storage.get("dtype") != "float16":
            raise ValueError("Native Image Atlas requires dense_npy float16 expression")
        if metadata.get("mapping_mode") != "compact_species_injective":
            raise ValueError("Native Image Atlas requires fitted compact_species_injective layouts")
        source_vocab = _inside(directory, "gene_vocab.json")
        release_vocab = root / "assets/gene_vocab.json"
        if _sha256(source_vocab) != _sha256(release_vocab):
            if json.loads(source_vocab.read_text()) != json.loads(release_vocab.read_text()):
                raise ValueError("Image Atlas and CellNet release gene vocabularies differ; use matching fitted assets")
        self.masks = np.load(_inside(directory, "masks/gene_masks.npy"), mmap_mode="r", allow_pickle=False)
        self.maps = np.load(_inside(directory, "pixel_maps/gene_to_flat_pixel.npy"), mmap_mode="r", allow_pickle=False)
        datasets = json.loads(_inside(directory, "masks/dataset_index.json").read_text())
        self.datasets = {int(value): str(key) for key, value in datasets.items()}
        unresolved = metadata.get("unresolved_dataset_ids", [])
        if (
            not isinstance(unresolved, list)
            or any(type(value) is not int or value not in self.datasets for value in unresolved)
            or len(set(unresolved)) != len(unresolved)
        ):
            raise ValueError("Invalid unresolved_dataset_ids in Image Atlas metadata")
        self.unresolved_datasets = set(unresolved)
        self.species_rows = json.loads(_inside(directory, "pixel_maps/dataset_index.json").read_text())
        if (
            self.masks.ndim != 2 or self.masks.shape[1] != n_genes
            or self.maps.ndim != 2 or self.maps.shape[1] != n_genes
            or self.maps.dtype.kind not in "iu" or self.masks.dtype.kind not in "bui"
            or len(self.datasets) != len(datasets)
            or len(self.datasets) != len(self.masks)
            or set(self.datasets) != set(range(len(self.masks)))
            or set(self.species_rows.values()) != set(range(len(self.maps)))
            or len(self.species_rows) != len(self.maps)
        ):
            raise ValueError("Invalid native mask/map dimensions or dataset/species index")
        self.layouts = {}
        with np.load(root / "assets/species_layouts.npz", allow_pickle=False) as data:
            species = data["species"].tolist()
            if set(species) != set(self.species_rows):
                raise ValueError("Image Atlas species differ from the fitted CellNet release")
            for i, name in enumerate(species):
                ids, pixels = data[f"ids_{i}"].copy(), data[f"pixels_{i}"].copy()
                row = self.maps[self.species_rows[name]]
                if (
                    ids.ndim != 1 or pixels.shape != ids.shape
                    or (ids <= 0).any() or (ids > n_genes).any()
                    or (pixels < 0).any() or (pixels >= np.prod(self.shape)).any()
                    or len(np.unique(pixels)) != len(pixels)
                    or np.count_nonzero(row >= 0) != len(ids)
                    or not np.array_equal(row[ids - 1], pixels)
                ):
                    raise ValueError(f"Image Atlas {name} layout differs from the fitted CellNet release; use matching fitted assets")
                self.layouts[name] = (ids, pixels)
        names = json.loads(_inside(directory, "cell_type_vocab.json").read_text())["names"]
        self.labels = {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))
        self.cache = OrderedDict()
        self.ownership = {}
        self.ownership_sources = set(metadata.get("ownership_source_ids", []))
        ownership_file = metadata.get("ownership_file")
        if ownership_file:
            if not self.ownership_sources or not self.ownership_sources <= self.datasets.keys():
                raise ValueError("Invalid ownership_source_ids in Image Atlas metadata")
            path = _inside(directory, ownership_file)
            manifest = json.loads(_inside(directory, "release_manifest.json").read_text())
            entries = [entry for entry in manifest["files"] if entry["path"] == ownership_file]
            if len(entries) != 1 or not re.fullmatch(r"[0-9a-f]{64}", entries[0].get("sha256", "")):
                raise ValueError("Image Atlas ownership index requires its published SHA256")
            if _sha256(path) != entries[0]["sha256"]:
                raise ValueError("Image Atlas ownership index SHA256 mismatch")
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8") as handle:
                ownership = json.load(handle)
            if (
                ownership.get("format") != "cellnet-image-atlas-ownership-v1"
                or set(ownership.get("source_dataset_ids", [])) != self.ownership_sources
                or ownership.get("key_stride") != 10_000_000
            ):
                raise ValueError("Invalid Image Atlas ownership index contract")
            self.key_stride = ownership["key_stride"]
            total = 0
            for bucket, entry in ownership["buckets"].items():
                if not re.fullmatch(r"class_\d+", bucket):
                    raise ValueError("Invalid ownership storage class")
                intervals = entry["intervals"]
                previous = 0
                stops, rows = [], []
                for start, stop, row, dataset_uuid in intervals:
                    if (
                        any(type(value) is not int for value in (start, stop, row))
                        or start != previous or not start < stop <= self.key_stride
                        or self.datasets.get(row) != dataset_uuid
                    ):
                        raise ValueError(f"Invalid ownership interval in {bucket}")
                    previous = stop
                    stops.append(stop)
                    rows.append(row)
                if not intervals:
                    raise ValueError(f"Empty ownership intervals in {bucket}")
                self.ownership[bucket] = (np.asarray(stops), np.asarray(rows))
                total += previous
            if total != ownership.get("samples") or total == 0:
                raise ValueError("Image Atlas ownership index sample count mismatch")
        elif self.ownership_sources:
            raise FileNotFoundError("Missing ownership_file for declared aggregate Image Atlas sources")

    def source_dataset(self, dataset_id, sample_id, relative_shard):
        if dataset_id in self.unresolved_datasets:
            raise ValueError(
                f"Unresolved Image Atlas dataset ownership: dataset_id={dataset_id}, "
                f"sample={sample_id}, shard={relative_shard}. "
                "Obtain the per-cell ownership index mapping (relative shard path, "
                "sample key) to the true source dataset UUID/mask row. "
                "This aggregate source ID cannot select a dataset-specific measurement mask."
            )
        if dataset_id not in self.ownership_sources:
            return dataset_id
        bucket = relative_shard.parent.name
        if bucket not in self.ownership or not re.fullmatch(r"\d{1,20}", sample_id):
            raise ValueError(f"Missing Image Atlas ownership for {relative_shard}:{sample_id}")
        key = int(sample_id)
        if key // self.key_stride != int(bucket.split("_")[1]):
            raise ValueError(f"Image Atlas ownership key and storage class differ: {sample_id}")
        ordinal = key % self.key_stride
        stops, rows = self.ownership[bucket]
        index = int(np.searchsorted(stops, ordinal, side="right"))
        if index == len(rows):
            raise ValueError(f"Image Atlas ownership index does not cover {relative_shard}:{sample_id}")
        return int(rows[index])

    def layout(self, species, dataset_id):
        key = species, dataset_id
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if species not in self.layouts or dataset_id not in self.datasets:
            raise ValueError(f"Unknown native sample species/dataset: {species}/{dataset_id}")
        ids, pixels = self.layouts[species]
        measured = self.masks[dataset_id, ids - 1]
        if not np.isin(measured, [0, 1]).all():
            raise ValueError("Native measurement masks must contain only zero and one")
        selected = measured.astype(bool)
        mask = np.zeros(np.prod(self.shape), bool)
        gene_ids = np.zeros(mask.shape, np.int64)
        mask[pixels[selected]] = True
        gene_ids[pixels[selected]] = ids[selected]
        result = mask.reshape(self.shape), gene_ids.reshape(self.shape)
        self.cache[key] = result
        if len(self.cache) > 16:
            self.cache.popitem(last=False)
        return result


def _native_shards(directory, storage_classes=None, shards=None):
    if storage_classes is not None and shards is not None:
        raise ValueError("Choose storage_classes or shards, not both")
    buckets = _inside(directory, "class_shards")
    if not buckets.is_dir():
        raise FileNotFoundError("Missing Image Atlas class_shards/ directory")
    if shards is not None:
        if not shards or len(set(map(str, shards))) != len(shards):
            raise ValueError("Provide unique dataset-relative native shard paths")
        for relative in shards:
            path = _inside(directory, relative)
            if path.parent.parent != buckets or not re.fullmatch(r"class_\d+|unlabeled", path.parent.name):
                raise ValueError("Native shards must be under class_shards/class_*/ or class_shards/unlabeled/")
            if not path.is_file() or not re.fullmatch(r"shard-.*\.tar(?:\.gz)?", path.name):
                raise FileNotFoundError(f"Missing or unsupported native shard: {relative}")
            yield path
        return
    if storage_classes is not None:
        if not storage_classes or len(set(storage_classes)) != len(storage_classes):
            raise ValueError("Provide unique storage class directory names")
        classes = []
        for name in storage_classes:
            if not re.fullmatch(r"class_\d+|unlabeled", name):
                raise ValueError(f"Invalid native storage class: {name}")
            classes.append(_inside(directory, f"class_shards/{name}"))
    else:
        classes = sorted(path for path in buckets.iterdir() if re.fullmatch(r"class_\d+|unlabeled", path.name))
    found = False
    for bucket in classes:
        if not bucket.is_dir():
            raise FileNotFoundError(f"Missing native storage class: {bucket.name}")
        matched = False
        for path in sorted(bucket.iterdir()):
            if re.fullmatch(r"shard-.*\.tar(?:\.gz)?", path.name):
                path = _inside(directory, path.relative_to(directory))
                if not path.is_file():
                    raise FileNotFoundError(path)
                found = matched = True
                yield path
        if storage_classes is not None and not matched:
            raise FileNotFoundError(f"No native shards in requested class: {bucket.name}")
    if not found:
        raise FileNotFoundError("No native Image Atlas shards found; download class_shards/ or explicitly select downloaded classes/shards")


def _native_records(path):
    key, fields = None, {}
    with tarfile.open(path, "r|*") as archive:
        for member in archive:
            if member.isdir():
                continue
            if not member.isfile() or Path(member.name).name != member.name or "." not in member.name:
                raise ValueError(f"Invalid native tar member in {path.name}: {member.name}")
            current, field = member.name.rsplit(".", 1)
            if not current or member.size > 1024 * 1024:
                raise ValueError(f"Invalid native record in {path.name}: {member.name}")
            if key is not None and current != key:
                yield key, fields
                fields = {}
            key = current
            if field in fields:
                raise ValueError(f"Duplicate native record field: {member.name}")
            with archive.extractfile(member) as handle:
                fields[field] = handle.read()
        if key is not None:
            yield key, fields


def _iter_native_cells(directory, metadata, release, storage_classes, shards):
    assets = _NativeAssets(directory, metadata, release)
    count = 0
    selected = storage_classes is not None or shards is not None
    for path in _native_shards(directory, storage_classes, shards):
        for sample_id, fields in _native_records(path):
            if not {"npy", "species", "dataset_id", "cell_type"} <= fields.keys():
                raise ValueError(f"Incomplete native sample {sample_id} in {path.name}")
            species, stored_dataset_id = fields["species"].decode(), int(fields["dataset_id"])
            dataset_id = assets.source_dataset(stored_dataset_id, sample_id, path.relative_to(directory))
            mask, ids = assets.layout(species, dataset_id)
            image = np.load(io.BytesIO(fields["npy"]), allow_pickle=False)
            if (
                image.dtype != np.float16 or image.shape != assets.shape
                or not np.isfinite(image).all() or (image < 0).any() or (image > 1).any()
                or np.any(image[~mask])
            ):
                raise ValueError(f"Invalid native expression or measurement mask: {sample_id}")
            label = int(fields["cell_type"])
            if label not in assets.labels:
                raise ValueError(f"Unknown native cell type: {label}")
            row = {
                "sample_id": sample_id,
                "species": species,
                "cell_type": assets.labels[label].split("::", 1)[-1],
                "source_dataset": assets.datasets[dataset_id],
                "dataset_id": dataset_id,
                "stored_dataset_id": stored_dataset_id,
                "storage_class": path.parent.name,
                "source_shard": str(path.relative_to(directory)),
                "selection_scope": "selected_download" if selected else "all_downloaded_shards",
                "measured_genes": int(mask.sum()),
                "detected_genes": int(np.count_nonzero(image)),
            }
            count += 1
            yield {"image": np.stack([image.astype(np.float32), mask]), "gene_ids": ids.copy(), "metadata": row}
    expected = metadata.get("samples_verified")
    if not selected and expected is not None and count != expected:
        raise ValueError(f"Image Atlas has {count} readable cells, expected {expected}; verify the download or explicitly select downloaded classes/shards")


def _manifest_entries(directory, manifest):
    """Read root assets first, then the optional compressed shard inventory."""
    yield from manifest["files"]
    relative = manifest.get("shard_manifest")
    if relative is not None:
        if not any(entry["path"] == relative for entry in manifest["files"]):
            raise ValueError("Shard inventory must be included in the root file manifest")
        path = _inside(directory, relative)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            shard_manifest = json.load(handle)
        if shard_manifest.get("format") != "cellnet-image-atlas-shards-v1":
            raise ValueError("Unsupported Image Atlas shard manifest")
        for entry in shard_manifest["files"]:
            if not entry["path"].startswith("class_shards/"):
                raise ValueError("Shard inventory entries must be under class_shards/")
            yield entry


def verify_image_atlas(directory, *, release=None, samples=2, checksums=False, storage_classes=None, shards=None):
    """Check a remote-inventory manifest and bounded real records.

    File sizes are always checked. Only provided SHA256 or explicitly verified
    server MD5 values are trusted as expected digests. Unverified Baidu MD5
    values are never silently treated as independent integrity evidence.
    """
    from itertools import islice

    if samples < 1:
        raise ValueError("samples must be positive")
    directory = Path(directory).resolve()
    metadata = json.loads(_inside(directory, "metadata.json").read_text())
    manifest_path = _inside(directory, "release_manifest.json")
    report = {"format": metadata.get("format"), "files_checked": 0, "bytes_checked": 0,
              "checksums_verified": 0, "checksums_computed_without_reference": 0,
              "selection_scope": "selected_download" if storage_classes is not None or shards is not None else "all_manifest_files",
              "unresolved_dataset_ids": metadata.get("unresolved_dataset_ids", []),
              "source_ownership_status": (
                  "unresolved" if metadata.get("unresolved_dataset_ids") else
                  "resolved_with_index" if metadata.get("ownership_file") else
                  "no_unresolved_sources_declared"),
              "ownership_file": metadata.get("ownership_file")}
    checked_native_shards = set()
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != "cellnet-image-atlas-manifest-v1":
            raise ValueError("Unsupported Image Atlas release manifest")
        required = {
            "metadata.json", "gene_vocab.json", "cell_type_vocab.json",
            "masks/dataset_index.json", "masks/gene_masks.npy",
            "pixel_maps/dataset_index.json", "pixel_maps/gene_to_flat_pixel.npy",
        }
        selected_classes = set(storage_classes) if storage_classes is not None else None
        selected_shards = set(map(str, shards)) if shards is not None else None
        seen, hashes = set(), {}
        catalogued_classes = set()
        has_shards = False
        for entry in _manifest_entries(directory, manifest):
            relative = entry["path"]
            if relative in seen:
                raise ValueError(f"Duplicate release manifest path: {relative}")
            seen.add(relative)
            parts = Path(relative).parts
            if not parts or Path(relative).is_absolute() or ".." in parts:
                raise ValueError(f"Dataset path must stay inside its directory: {relative}")
            native_shard = parts[0] == "class_shards"
            if native_shard and len(parts) < 3:
                raise ValueError("Invalid native shard inventory path")
            has_shards = has_shards or native_shard
            if native_shard:
                catalogued_classes.add(parts[1])
            if native_shard and (
                (selected_classes is not None and parts[1] not in selected_classes)
                or (selected_shards is not None and relative not in selected_shards)
            ):
                continue
            path = _inside(directory, relative)
            if not path.is_file() or path.stat().st_size != entry["bytes"]:
                raise ValueError(f"Missing or incomplete Image Atlas file: {relative}")
            report["files_checked"] += 1
            report["bytes_checked"] += entry["bytes"]
            if native_shard:
                checked_native_shards.add(relative)
            if checksums:
                sha = _sha256(path)
                hashes[relative] = sha
                if entry.get("sha256"):
                    if sha != entry["sha256"]:
                        raise ValueError(f"Image Atlas SHA256 mismatch: {relative}")
                    report["checksums_verified"] += 1
                elif entry.get("server_md5_verified") is True and entry.get("server_md5"):
                    digest = hashlib.md5()
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                            digest.update(chunk)
                    if digest.hexdigest() != entry["server_md5"]:
                        raise ValueError(f"Image Atlas MD5 mismatch: {relative}")
                    report["checksums_verified"] += 1
                else:
                    report["checksums_computed_without_reference"] += 1
        if metadata.get("format") != "cellnet-pretraining-npz-v1":
            if required - seen or not has_shards:
                raise ValueError("Image Atlas manifest omits required assets or native shards")
            if selected_shards is not None and not selected_shards <= seen:
                raise ValueError("Requested native shard is absent from the release manifest")
            if selected_classes is not None and not selected_classes <= catalogued_classes:
                raise ValueError("Requested storage class is absent from the release manifest")
        report["files_catalogued"] = len(seen)
        if manifest.get("shard_manifest"):
            report["shard_manifest"] = manifest["shard_manifest"]
        if checksums:
            report["computed_sha256"] = hashes
        report["inventory_status"] = "listed file sizes verified"
        report["population_samples_verified"] = manifest.get("samples_verified")
    else:
        if metadata.get("format") != "cellnet-pretraining-npz-v1":
            raise FileNotFoundError("Missing release_manifest.json; download the complete Image Atlas metadata")
        report["inventory_status"] = "release_manifest.json absent; file inventory not verified"
        report["population_samples_verified"] = None
    cells = list(islice(iter_pretraining_cells(directory, release=release, storage_classes=storage_classes, shards=shards), samples))
    if not cells:
        raise ValueError("No Image Atlas records were read")
    if metadata.get("format") != "cellnet-pretraining-npz-v1":
        if any(cell["metadata"]["source_shard"] not in checked_native_shards for cell in cells):
            raise ValueError("Sampled native shard is absent from the checked release inventory")
    report.update(records_checked=len(cells), species_checked=sorted({cell["metadata"]["species"] for cell in cells}),
                  fitted_mapping="exact release vocabulary and all species layouts checked" if metadata.get("format") != "cellnet-pretraining-npz-v1" else "embedded NPZ masks and gene IDs checked",
                  records_scope="bounded sample, not an exhaustive cell scan")
    return report


def make_pretraining_views(cells, views=2, keep_fraction=0.75, seed=0):
    """Create independent measured-gene masks while keeping each cell paired.

    Each view independently masks measured genes. Hidden expression is never
    returned as input.
    """
    if len(cells) < 2 or views < 2 or not 0 < keep_fraction <= 1:
        raise ValueError("Need >=2 cells, >=2 views and keep_fraction in (0,1]")
    rng = np.random.default_rng(seed)
    images, ids = [], []
    for cell in cells:
        measured = cell["image"][1].astype(bool)
        cell_images, cell_ids = [], []
        for _ in range(views):
            visible = measured & (rng.random(measured.shape) < keep_fraction)
            cell_images.append(np.stack([cell["image"][0] * visible, visible]))
            cell_ids.append(np.where(visible, cell["gene_ids"], 0))
        images.append(cell_images)
        ids.append(cell_ids)
    return {
        "image": np.asarray(images, np.float32),
        "gene_ids": np.asarray(ids, np.int64),
    }
