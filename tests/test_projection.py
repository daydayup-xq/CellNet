"""Optional fitted-projection and new-gene inference test (large index)."""

from pathlib import Path
import os
import unittest
import numpy as np
import torch
from cellnet import CellNet, CellNetImageConverter, CellNetProjection

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    os.getenv("CELLNET_TEST_PROJECTION") == "1",
    "Set CELLNET_TEST_PROJECTION=1 to test the fitted million-gene index",
)
class ProjectionTests(unittest.TestCase):
    def test_fitted_projection_and_new_gene_inference(self):
        torch.set_num_threads(4)
        fitter = CellNetProjection(ROOT)
        reference = np.load(ROOT / "tests/data/projection_reference.npz")
        vectors = reference["vectors"]
        np.testing.assert_allclose(
            fitter.transform_features(vectors),
            reference["features"],
            atol=1e-6,
            rtol=1e-6,
        )
        projected = fitter.project(vectors)
        np.testing.assert_allclose(projected, reference["scaled"], atol=1e-4, rtol=1e-5)
        model = CellNet.from_pretrained(ROOT)
        original_size = model.config["vocabulary_size"]
        genes = ["test_gene_A", "test_gene_B", "test_gene_C", "test_gene_D"]
        converter = CellNetImageConverter.from_embeddings(
            ROOT, model, "ProjectionTestSpecies", genes, vectors, projection=fitter
        )
        self.assertEqual(model.config["vocabulary_size"], original_size + 4)
        self.assertEqual(len(np.unique(converter.layout_pixels)), 4)
        batch = converter.transform(
            np.array([[0.0, 1.0, 2.0, 3.0]], np.float32), genes, input_scale="log1p"
        )
        with torch.inference_mode():
            embeddings = model(**batch)
            predictions = model.predict(
                **batch, target_gene_ids=converter.resolve_gene_ids(genes[:1])
            )
        self.assertEqual(embeddings.shape, (1, 768))
        self.assertTrue(torch.isfinite(embeddings).all())
        self.assertTrue(torch.isfinite(predictions["expression_log1p"]).all())


if __name__ == "__main__":
    unittest.main()
