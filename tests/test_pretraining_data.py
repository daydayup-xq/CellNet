"""Real-data contracts for pretraining loading, placeholders and masked views."""

from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
from cellnet.pretraining_data import iter_pretraining_cells, make_pretraining_views

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/pretraining/example"


class PretrainingDataTests(unittest.TestCase):
    def test_real_cells_preserve_measured_zeros_and_layouts(self):
        cells = list(iter_pretraining_cells(DATA))
        self.assertEqual(len(cells), 24)
        self.assertEqual(
            {c["metadata"]["species"] for c in cells},
            {"Mouse", "Zebrafish", "Fruit_fly"},
        )
        for cell in cells:
            image, ids = cell["image"], cell["gene_ids"]
            self.assertEqual(image.shape, (2, 384, 384))
            self.assertEqual(image.dtype, np.float32)
            self.assertTrue(np.array_equal(image[1] > 0, ids > 0))
            self.assertFalse(np.any(image[0][ids == 0]))
            self.assertTrue(np.any((image[0] == 0) & (ids > 0)))

    def test_views_are_independent_reproducible_and_hide_identity(self):
        cells = list(iter_pretraining_cells(DATA))[:2]
        batch = make_pretraining_views(cells, views=2, seed=17)
        repeat = make_pretraining_views(cells, views=2, seed=17)
        np.testing.assert_array_equal(batch["image"], repeat["image"])
        np.testing.assert_array_equal(batch["gene_ids"], repeat["gene_ids"])
        self.assertFalse(
            np.array_equal(batch["image"][:, 0, 1], batch["image"][:, 1, 1])
        )
        for i, cell in enumerate(cells):
            for view in range(2):
                visible = batch["image"][i, view, 1].astype(bool)
                ids = batch["gene_ids"][i, view]
                self.assertTrue(np.array_equal(visible, ids > 0))
                self.assertTrue(np.any((cell["gene_ids"] > 0) & ~visible))
                np.testing.assert_array_equal(
                    batch["image"][i, view, 0][visible], cell["image"][0][visible]
                )
                self.assertFalse(np.any(batch["image"][i, view, 0][~visible]))
                np.testing.assert_array_equal(ids[visible], cell["gene_ids"][visible])

    def test_full_placeholder_fails_explicitly(self):
        with self.assertRaisesRegex(FileNotFoundError, "not included"):
            list(iter_pretraining_cells(ROOT / "datasets/pretraining/full"))

    def test_corrupt_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "example"
            shutil.copytree(DATA, target)
            path = target / "shards/shard-00000.npz"
            with path.open("r+b") as handle:
                value = handle.read(1)
                handle.seek(0)
                handle.write(bytes([value[0] ^ 1]))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                list(iter_pretraining_cells(target))


if __name__ == "__main__":
    unittest.main()
