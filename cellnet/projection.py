"""Frozen out-of-sample gene projection; fitted arrays, no Python pickle."""

from pathlib import Path
import json
import numpy as np
from scipy import sparse


class CellNetProjection:
    def __init__(self, directory, *, threads=4):
        from openTSNE import TSNEEmbedding
        from openTSNE.affinity import MultiscaleMixture
        from openTSNE.nearest_neighbors import Annoy
        from openTSNE.dependencies.annoy import AnnoyIndex

        root = Path(directory)
        self.params = json.loads((root / "assets/layout_projection.json").read_text())
        with np.load(root / "assets/layout_projection.npz", allow_pickle=False) as z:
            self.mean = z["pca_mean"].copy()
            self.components = z["pca_components"].copy()
            coords = z["reference_coords"].copy()
            self.minimum = z["coordinate_min"].copy()
            self.maximum = z["coordinate_max"].copy()
        self.pixel_size = tuple(self.params["pixel_size"])
        # Restore the fitted search index, not its training neighbor/distance caches.
        index = Annoy.__new__(Annoy)
        index.n_samples = self.params["reference_genes"]
        index.n_jobs = threads
        index.verbose = False
        metric = (
            "angular"
            if self.params["neighbor_metric"] == "cosine"
            else self.params["neighbor_metric"]
        )
        index.index = AnnoyIndex(self.params["neighbor_dimensions"], metric)
        if not index.index.load(str(root / "assets/layout_neighbors.ann")):
            raise RuntimeError("Cannot load fitted neighbor index")
        affinity = MultiscaleMixture.__new__(MultiscaleMixture)
        affinity.knn_index = index
        affinity.perplexities = self.params["perplexities"]
        affinity.n_jobs = threads
        affinity.verbose = False
        # openTSNE checks reference sample count at construction; no training edges are needed.
        affinity.P = sparse.csr_matrix((len(coords), len(coords)))
        gradient = dict(self.params["gradient_descent"], n_jobs=threads)
        self.embedding = TSNEEmbedding(coords, affinity, random_state=0, **gradient)
        affinity.P = None

    def transform_features(self, vectors):
        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.mean.size or not np.isfinite(x).all():
            raise ValueError(
                f"Expected finite [genes, {self.mean.size}] raw gene embeddings"
            )
        if self.params["embedding_l2_normalize"]:
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        # Same operation order as the fitted PCA transform, without estimator pickles.
        return np.asarray(
            x @ self.components.T - self.mean[None] @ self.components.T,
            dtype=np.float32,
        )

    def project(self, vectors):
        """Project compatible 1,920-D gene embeddings into the frozen image canvas.

        No fitting is performed. Coordinates may collide; use the original
        balanced assignment algorithm before rendering a new gene panel.
        """
        features = self.transform_features(vectors)
        if not len(features):
            return np.empty((0, 2), dtype=np.float64)
        coordinates = np.asarray(self.embedding.transform(features))
        span = np.where(self.maximum > self.minimum, self.maximum - self.minimum, 1)
        return (coordinates - self.minimum) / span * (np.asarray(self.pixel_size) - 1)
