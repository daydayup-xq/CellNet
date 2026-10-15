"""Standalone contract tests; run from the CellNet directory."""

from pathlib import Path
import unittest
import numpy as np
import torch
from cellnet import CellNet, CellNetImageConverter
from cellnet.pretraining import CellNetPretrainer

ROOT = Path(__file__).resolve().parents[1]


class CellNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.converter = CellNetImageConverter(ROOT, "Human")
        cls.genes = [
            k.split("::", 1)[1]
            for k in cls.converter.gene_to_id
            if k.startswith("Human::")
        ][:12]
        cls.model = CellNet.from_pretrained(ROOT)

    def test_converter_measurement_semantics(self):
        x = np.array([[0, 2, 4] + [0] * 9], np.float32)
        measured = np.ones(12, bool)
        measured[2] = False
        batch = self.converter.transform(
            x, self.genes, input_scale="log1p", measured=measured
        )
        ids = self.converter.resolve_gene_ids(self.genes)
        p0 = self.converter.id_to_pixel[int(ids[0])]
        p2 = self.converter.id_to_pixel[int(ids[2])]
        self.assertEqual(batch["image"][0, 0].flatten()[p0], 0)
        self.assertEqual(batch["image"][0, 1].flatten()[p0], 1)
        self.assertEqual(batch["gene_ids"][0].flatten()[p0], ids[0])
        self.assertEqual(batch["image"][0, :, p2 // 384, p2 % 384].sum(), 0)
        self.assertEqual(batch["gene_ids"][0].flatten()[p2], 0)
        with self.assertRaises(ValueError):
            self.converter.transform(x, ["unknown"] * 12, input_scale="log1p")
        with self.assertRaises(ValueError):
            self.converter.transform(x, [self.genes[0]] * 12, input_scale="log1p")

    def test_counts_and_sparse_conversion(self):
        from scipy.sparse import csr_matrix

        x = np.arange(24, dtype=np.float32).reshape(2, 12)
        a = self.converter.transform(csr_matrix(x), self.genes, input_scale="counts")
        y = np.log1p(x * 10000 / x.sum(1, keepdims=True))
        b = self.converter.transform(y, self.genes, input_scale="log1p")
        torch.testing.assert_close(a["image"], b["image"])
        torch.testing.assert_close(a["gene_ids"], b["gene_ids"])

    def test_production_expression_storage_roundtrip(self):
        x = np.linspace(0, 6, 12, dtype=np.float32)[None]
        production = self.converter.transform(x, self.genes, input_scale="log1p")
        full_precision = self.converter.transform(
            x, self.genes, input_scale="log1p", storage_dtype="float32"
        )
        ids = self.converter.resolve_gene_ids(self.genes)
        pixels = [self.converter.id_to_pixel[int(g)] for g in ids]
        expected = (x / 6).astype(np.float16).astype(np.float32)
        np.testing.assert_array_equal(
            production["image"][0, 0].numpy().ravel()[pixels], expected[0]
        )
        self.assertGreater(
            float((production["image"] - full_precision["image"]).abs().max()), 0
        )

    def test_prediction_hides_target_values(self):
        x = np.arange(24, dtype=np.float32).reshape(2, 12) / 5
        b = self.converter.transform(x, self.genes, input_scale="log1p")
        ids = self.converter.resolve_gene_ids(self.genes[:3])
        alternate = b["image"].clone()
        target_pixels = torch.isin(b["gene_ids"], torch.as_tensor(ids))
        alternate[:, 0][target_pixels] = 0.97
        with torch.inference_mode():
            a = self.model.predict(**b, target_gene_ids=ids)
            z = self.model.predict(alternate, b["gene_ids"], ids)
        torch.testing.assert_close(
            a["expression_log1p"], z["expression_log1p"], rtol=0, atol=0
        )
        self.assertTrue(torch.isfinite(a["expression_log1p"]).all())
        self.assertEqual(a["expression_log1p"].shape, (2, 3))

    def test_empty_cells_and_readouts(self):
        batch = self.converter.transform(
            np.zeros((1, 12)),
            self.genes,
            input_scale="log1p",
            measured=np.zeros(12, bool),
        )
        with torch.inference_mode():
            e = self.model.encode(**batch)
            p = self.model.decode(e, self.converter.resolve_gene_ids(self.genes[:2]))
        self.assertEqual(e["cls_layers"].shape, (8, 1, 1, 768))
        self.assertTrue(torch.isfinite(e["cls_deep"]).all())
        self.assertTrue(torch.isfinite(p["expression_log1p"]).all())

    def test_pretraining_gradients(self):
        config = dict(
            self.model.config,
            image_size=[16, 16],
            patch_size=16,
            embedding_dim=32,
            depth=2,
            attention_heads=4,
            vocabulary_size=16,
            readout_layers=2,
            decoder_hidden_dim=32,
            decoder_depth=1,
            decoder_heads=4,
        )
        torch.manual_seed(7)
        tiny = CellNet(config)
        trainer = CellNetPretrainer(tiny).train()
        image = torch.rand(2, 2, 2, 16, 16)
        image[:, :, 1] = 1
        ids = torch.ones(2, 2, 16, 16, dtype=torch.long)
        targets = {
            "gene_ids": torch.full((2, 2), 2, dtype=torch.long),
            "mask": torch.ones(2, 2, dtype=torch.bool),
            "expression_normalized": torch.tensor([[0.2, 0], [0.1, 0]]),
            "positive_pool_count": torch.ones(2),
            "zero_pool_count": torch.ones(2),
        }
        result = trainer(image, ids, n_global=1, targets=targets)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertIsNotNone(tiny.backbone.patch_embed.gene_projector.weight.grad)
        self.assertIsNotNone(tiny.decoder.out.weight.grad)
        self.assertIsNone(tiny.backbone.patch_embed.gene_identity_embed.weight.grad)

    def test_projector_weights_load(self):
        trainer = CellNetPretrainer.from_pretrained(ROOT)
        trainer.eval()
        with torch.inference_mode():
            y = trainer.projector(torch.zeros(2, 768))
        self.assertEqual(y.shape, (2, 256))
        self.assertTrue(torch.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
