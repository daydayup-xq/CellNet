"""Native atlas checks: fixed mapping, measurement masks and download integrity."""

import hashlib
import gzip
import io
import json
import shutil
from pathlib import Path
import tarfile
import tempfile
import unittest
import numpy as np
from cellnet.pretraining_data import (
    default_pretraining_directory, iter_pretraining_cells, verify_image_atlas,
)


class NativeAtlasTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.release = Path(self.temp.name)
        self.data = self.release / "datasets/pretraining/full"
        for path in ("assets", "weights", "datasets/pretraining/full/masks",
                     "datasets/pretraining/full/pixel_maps", "datasets/pretraining/full/class_shards/class_0"):
            (self.release / path).mkdir(parents=True, exist_ok=True)
        self.write(self.release / "weights/config.json", {"image_size": [2, 2], "vocabulary_size": 3})
        self.write(self.release / "assets/gene_vocab.json", {"Human::g1": 0, "Human::g2": 1, "Human::g3": 2})
        self.write(self.data / "gene_vocab.json", {"Human::g1": 0, "Human::g2": 1, "Human::g3": 2})
        np.savez(self.release / "assets/species_layouts.npz", species=np.array(["Human"]),
                 ids_0=np.array([1, 2, 3]), pixels_0=np.array([0, 1, 2]))
        self.write(self.data / "metadata.json", {
            "format": "cellnet-image-atlas-native-v1", "image_size": [2, 2],
            "mapping_mode": "compact_species_injective",
            "sample_storage": {"format": "dense_npy", "dtype": "float16"},
            "samples_verified": 2,
        })
        self.write(self.data / "cell_type_vocab.json", {"names": {"0": "Human::cell"}})
        self.write(self.data / "masks/dataset_index.json", {"source": 0})
        self.write(self.data / "pixel_maps/dataset_index.json", {"Human": 0})
        np.save(self.data / "masks/gene_masks.npy", np.array([[1, 1, 0]], np.uint8))
        np.save(self.data / "pixel_maps/gene_to_flat_pixel.npy", np.array([[0, 1, 2]], np.int32))
        self.shard = self.data / "class_shards/class_0/shard-00000.tar.gz"
        self.shard_name = str(self.shard.relative_to(self.data))
        with tarfile.open(self.shard, "w:gz") as tar:
            for i in range(2):
                image = io.BytesIO()
                np.save(image, np.array([[0.5, 0], [0, 0]], np.float16), allow_pickle=False)
                fields = {"npy": image.getvalue(), "species": b"Human", "dataset_id": b"0", "cell_type": b"0"}
                for key, payload in fields.items():
                    member = tarfile.TarInfo(f"cell{i}.{key}")
                    member.size = len(payload)
                    tar.addfile(member, io.BytesIO(payload))
        self.manifest()

    @staticmethod
    def write(path, data):
        path.write_text(json.dumps(data))

    def manifest(self):
        entries = [{"path": str(p.relative_to(self.data)), "bytes": p.stat().st_size,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                   for p in sorted(self.data.rglob("*")) if p.is_file() and p.name != "release_manifest.json"]
        self.write(self.data / "release_manifest.json", {
            "format": "cellnet-image-atlas-manifest-v1", "files": entries, "samples_verified": 2,
        })

    def cells(self, **kwargs):
        return list(iter_pretraining_cells(self.data, release=self.release, **kwargs))

    def test_reconstructs_exact_mapping_and_measured_zero(self):
        cells = self.cells()
        self.assertEqual(len(cells), 2)
        np.testing.assert_array_equal(cells[0]["gene_ids"], [[1, 2], [0, 0]])
        np.testing.assert_array_equal(cells[0]["image"][1], [[1, 1], [0, 0]])
        np.testing.assert_array_equal(cells[0]["image"][0], [[0.5, 0], [0, 0]])
        self.assertEqual(cells[0]["metadata"]["source_dataset"], "source")

    def test_unresolved_source_is_rejected_even_when_nonzero_pixels_fit_the_mask(self):
        path = self.data / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["unresolved_dataset_ids"] = [0]
        self.write(path, metadata)
        self.manifest()
        with self.assertRaisesRegex(ValueError, "per-cell ownership index"):
            self.cells()
        with self.assertRaisesRegex(ValueError, "dataset_id=0, sample=cell0"):
            verify_image_atlas(self.data, release=self.release)

    def ownership_fixture(self):
        metadata_path = self.data / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update(ownership_file="cell_sources.json.gz", ownership_source_ids=[0])
        self.write(metadata_path, metadata)
        self.write(self.data / "masks/dataset_index.json", {"aggregate": 0, "true_source": 1})
        np.save(self.data / "masks/gene_masks.npy",
                np.array([[1, 0, 0], [1, 1, 0]], np.uint8))
        with tarfile.open(self.shard, "w:gz") as tar:
            for i in range(2):
                payload = io.BytesIO()
                np.save(payload, np.array([[0.5, 0], [0, 0]], np.float16))
                for field, value in {"npy": payload.getvalue(), "species": b"Human",
                                     "dataset_id": b"0", "cell_type": b"0"}.items():
                    member = tarfile.TarInfo(f"{i:012d}.{field}")
                    member.size = len(value)
                    tar.addfile(member, io.BytesIO(value))
        ownership = {
            "format": "cellnet-image-atlas-ownership-v1",
            "source_dataset_ids": [0], "key_stride": 10_000_000, "samples": 2,
            "buckets": {"class_0": {"intervals": [[0, 2, 1, "true_source"]]}},
        }
        self.write_ownership(ownership)
        return ownership

    def write_ownership(self, ownership):
        with gzip.open(self.data / "cell_sources.json.gz", "wt") as handle:
            json.dump(ownership, handle)
        self.manifest()

    def test_ownership_index_restores_source_specific_measured_zero(self):
        self.ownership_fixture()
        cells = self.cells()
        self.assertEqual(cells[0]["metadata"]["stored_dataset_id"], 0)
        self.assertEqual(cells[0]["metadata"]["dataset_id"], 1)
        self.assertEqual(cells[0]["metadata"]["source_dataset"], "true_source")
        np.testing.assert_array_equal(cells[0]["gene_ids"], [[1, 2], [0, 0]])
        report = verify_image_atlas(self.data, release=self.release)
        self.assertEqual(report["source_ownership_status"], "resolved_with_index")

    def test_ownership_index_requires_matching_hash_uuid_and_covered_ranges(self):
        ownership = self.ownership_fixture()
        path = self.data / "cell_sources.json.gz"
        with path.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "ownership index SHA256"):
            self.cells()
        ownership["buckets"]["class_0"]["intervals"][0][3] = "wrong_source"
        self.write_ownership(ownership)
        with self.assertRaisesRegex(ValueError, "Invalid ownership interval"):
            self.cells()
        ownership["buckets"]["class_0"]["intervals"] = [[1, 2, 1, "true_source"]]
        self.write_ownership(ownership)
        with self.assertRaisesRegex(ValueError, "Invalid ownership interval"):
            self.cells()
        ownership["buckets"]["class_0"]["intervals"] = [[0, 1, 1, "true_source"]]
        ownership["samples"] = 1
        self.write_ownership(ownership)
        with self.assertRaisesRegex(ValueError, "does not cover"):
            self.cells()
        ownership["buckets"]["class_0"]["intervals"] = [[0, 10_000_001, 1, "true_source"]]
        ownership["samples"] = 10_000_001
        self.write_ownership(ownership)
        with self.assertRaisesRegex(ValueError, "Invalid ownership interval"):
            self.cells()

    def test_installed_atlas_is_selected_without_fallback_on_error(self):
        self.assertEqual(default_pretraining_directory(self.release), self.data)
        (self.data / "metadata.json").write_text("invalid JSON")
        with self.assertRaises(json.JSONDecodeError):
            default_pretraining_directory(self.release)
        self.write(self.data / "metadata.json", {"status": "not_included"})
        self.assertEqual(default_pretraining_directory(self.release), self.release / "datasets/pretraining/example")

    def test_missing_lfs_payload_has_an_actionable_error(self):
        (self.release / "assets/species_layouts.npz").write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64 + "\nsize 100\n")
        with self.assertRaisesRegex(FileNotFoundError, "git lfs pull"):
            default_pretraining_directory(self.release)

    def test_vocab_and_pixel_map_mismatches_are_rejected(self):
        np.save(self.data / "pixel_maps/gene_to_flat_pixel.npy", np.array([[1, 0, 2]], np.int32))
        with self.assertRaisesRegex(ValueError, "layout differs"):
            self.cells()
        np.save(self.data / "pixel_maps/gene_to_flat_pixel.npy", np.array([[0, 1, 2]], np.int32))
        self.write(self.data / "gene_vocab.json", {"Human::g2": 0, "Human::g1": 1, "Human::g3": 2})
        with self.assertRaisesRegex(ValueError, "vocabularies differ"):
            self.cells()

    def test_full_count_and_explicit_partial_selection(self):
        p = self.data / "metadata.json"
        value = json.loads(p.read_text())
        value["samples_verified"] = 5
        self.write(p, value)
        with self.assertRaisesRegex(ValueError, "2 readable cells, expected 5"):
            self.cells()
        self.assertEqual(len(self.cells(storage_classes=["class_0"])), 2)
        self.assertEqual(len(self.cells(shards=[self.shard_name])), 2)
        with self.assertRaisesRegex(ValueError, "inside"):
            self.cells(shards=["../outside.tar.gz"])

    def test_manifest_sizes_and_trusted_hashes(self):
        report = verify_image_atlas(self.data, release=self.release, checksums=True)
        self.assertEqual(report["records_checked"], 2)
        self.assertEqual(report["checksums_verified"], report["files_checked"])
        with self.shard.open("ab") as handle:
            handle.write(b"unexpected bytes")
        with self.assertRaisesRegex(ValueError, "Missing or incomplete"):
            verify_image_atlas(self.data, release=self.release)

    def test_unlisted_storage_class_is_rejected(self):
        bucket = self.data / "class_shards/class_1"
        bucket.mkdir()
        shutil.copyfile(self.shard, bucket / self.shard.name)
        with self.assertRaisesRegex(ValueError, "storage class is absent"):
            verify_image_atlas(self.data, release=self.release, storage_classes=["class_1"])

    def test_sampled_unlisted_shard_is_rejected(self):
        shutil.copyfile(self.shard, self.shard.parent / "shard-00000-extra.tar.gz")
        with self.assertRaisesRegex(ValueError, "Sampled native shard is absent"):
            verify_image_atlas(self.data, release=self.release, storage_classes=["class_0"])

    def test_unverified_server_md5_is_not_reported_as_verified(self):
        p = self.data / "release_manifest.json"
        value = json.loads(p.read_text())
        for entry in value["files"]:
            entry.pop("sha256")
            entry["server_md5"] = "0" * 32
        self.write(p, value)
        report = verify_image_atlas(self.data, release=self.release, checksums=True)
        self.assertEqual(report["checksums_verified"], 0)
        self.assertEqual(report["checksums_computed_without_reference"], report["files_checked"])

    def test_compressed_shard_inventory_and_partial_selection(self):
        path = self.data / "release_manifest.json"
        manifest = json.loads(path.read_text())
        native = [entry for entry in manifest["files"] if entry["path"].startswith("class_shards/")]
        native.append({"path": "class_shards/class_1/shard-missing.tar.gz", "bytes": 100})
        index = self.data / "shards_manifest.json.gz"
        with gzip.open(index, "wt") as handle:
            json.dump({"format": "cellnet-image-atlas-shards-v1", "files": native}, handle)
        manifest["files"] = [entry for entry in manifest["files"] if not entry["path"].startswith("class_shards/")]
        manifest["files"].append({"path": index.name, "bytes": index.stat().st_size,
                                   "sha256": hashlib.sha256(index.read_bytes()).hexdigest()})
        manifest["shard_manifest"] = index.name
        self.write(path, manifest)
        report = verify_image_atlas(self.data, release=self.release, storage_classes=["class_0"], checksums=True)
        self.assertEqual(report["records_checked"], 2)
        self.assertEqual(report["files_catalogued"], report["files_checked"] + 1)
        with self.assertRaisesRegex(ValueError, "Missing or incomplete"):
            verify_image_atlas(self.data, release=self.release)

    def test_missing_or_incomplete_manifest_is_rejected(self):
        p = self.data / "release_manifest.json"
        self.write(p, {"format": "cellnet-image-atlas-manifest-v1", "files": []})
        with self.assertRaisesRegex(ValueError, "omits required"):
            verify_image_atlas(self.data, release=self.release)
        p.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "release_manifest"):
            verify_image_atlas(self.data, release=self.release)


if __name__ == "__main__":
    unittest.main()
