"""Data leakage and fixed voting edge cases for representative downstream examples."""
from pathlib import Path
import sys
import json
import unittest
import numpy as np
import anndata as ad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples'))
from reference_mapping import prior_corrected_vote


class TaskProtocolTests(unittest.TestCase):
    def test_exact_matches_exclude_nonzero_distance_votes(self):
        labels = np.array(['a', 'a', 'a', 'b'])
        # The only exact match is b; arbitrarily close a neighbors must not vote.
        prediction = prior_corrected_vote(np.array([[1e-12, 1e-12, 1e-12, 0.]]),
                                          np.array([[0, 1, 2, 3]]), labels)
        self.assertEqual(prediction.tolist(), ['b'])

    def test_prior_correction_and_tie_order(self):
        labels = np.array(['a', 'a', 'a', 'b'])
        distances = np.ones((1, 2))
        neighbors = np.array([[0, 3]])
        self.assertEqual(prior_corrected_vote(distances, neighbors, labels).tolist(), ['b'])
        self.assertEqual(prior_corrected_vote(distances, neighbors, labels, gamma=0).tolist(), ['a'])

    def test_whole_fold_absence_and_gene_axis_partition(self):
        parent = ad.read_h5ad(ROOT / 'examples/data/totalx_pbmc_96.h5ad')
        universe = set(parent.var_names)
        folder = ROOT / 'examples/data/completion_fivefold'
        hidden = set()
        for fold in range(1, 6):
            q = ad.read_h5ad(folder / f'fold{fold}_query.h5ad')
            genes = json.loads((folder / f'fold{fold}_targets.json').read_text())
            self.assertFalse(set(q.var_names) & set(genes))
            self.assertFalse(hidden & set(genes))
            self.assertEqual(set(q.var_names) | set(genes), universe)
            self.assertEqual(q.uns['cellnet_example']['input_scale'], 'counts')
            self.assertFalse(q.layers)
            self.assertIsNone(q.raw)
            with np.load(folder / f'fold{fold}_truth.npz', allow_pickle=False) as truth:
                np.testing.assert_array_equal(truth['cell_ids'], q.obs_names.to_numpy(str))
                np.testing.assert_array_equal(truth['gene_ids'], genes)
            hidden.update(genes)
        self.assertEqual(hidden, universe)

    def test_tabula_donor_folds_have_shared_labels_and_nonempty_inputs(self):
        a = ad.read_h5ad(ROOT / 'examples/data/tabula_sapiens_192.h5ad')
        self.assertGreater(a.n_vars, 10000)
        self.assertTrue(np.all(np.asarray(a.X.sum(1)).ravel() > 0))
        self.assertEqual(a.n_obs, 192)
        for tissue in a.obs.tissue.unique():
            obs = a.obs[a.obs.tissue == tissue]
            self.assertEqual(obs.donor_id.nunique(), 3)
            for donor in obs.donor_id.unique():
                ref, query = obs[obs.donor_id != donor], obs[obs.donor_id == donor]
                self.assertFalse(set(ref.donor_id) & set(query.donor_id))
                self.assertEqual(set(ref.cell_type), set(query.cell_type))


if __name__ == '__main__':
    unittest.main()
