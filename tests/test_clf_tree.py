"""
    Testing all features for the classification tree, which include:
        - Tree building process based on classifier
        - Making the predictions for both an entire dataset and one single sample
        - Computing feature importances
        - Plotting the tree
        - Extracting a textual explanation for one single prediction
"""

from decode import ClassificationTree, ClassificationTreeNumba
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import make_classification
import unittest

class TestClassificationTree(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_classification(
            n_samples=300, n_features=8, n_informative=4, n_classes=3, random_state=42
        )
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X, self.y)

    def _run_workflow(self, TreeCls):
        tree = TreeCls(
            model=self.model,
            scalerX=None,
            scalerY=None,
            dataset=self.X,
            targets=self.y,
        )
        tree.fit(
            max_depth=4,
            imp_threshold=0.0,
            min_samples_split=2,
            min_impurity_decrease=0.0,
            class_weights='balanced',
            criterium='soft_log_loss',
        )

        # Predictions (batch + single sample)
        preds, paths = tree.predict(self.X, return_paths=True)
        preds_only = tree.predict(self.X)
        np.testing.assert_array_equal(preds, preds_only)
        one_pred, one_path = tree.predict(self.X[0], return_paths=True)
        self.assertEqual(preds[0], one_pred[0])

        # Feature importances
        fi = tree.compute_feature_importance()
        self.assertEqual(len(fi), self.X.shape[1])
        self.assertAlmostEqual(float(np.sum(fi)), 1.0, places=6)

        # Pruning
        pruned = tree.prune_tree(by='depth', threshold=2)
        self.assertTrue(max(pruned.DEPTH) <= 2)

        # Textual explanation
        text = tree.extract_prediction_path(self.X[0])
        self.assertIsInstance(text, str)
        return tree, preds, fi

    def test_clf_tree_features(self):
        self._run_workflow(ClassificationTree)

    def test_clf_tree_numba_equivalence(self):
        for crit in ['gini', 'log_loss', 'soft_gini', 'soft_log_loss', 'brier', 'mutual_info', 'kl', ('gini', 'soft_log_loss')]:
            for adaptive in [False, True]:
                t_base = ClassificationTree(model=self.model, dataset=self.X, targets=self.y, seed=42)
                t_numba = ClassificationTreeNumba(model=self.model, dataset=self.X, targets=self.y, seed=42)

                kw = dict(max_depth=4, n_steps='auto', adaptive=adaptive, criterium=crit, class_weights='balanced')
                t_base.fit(**kw)
                t_numba.fit(**kw)

                self.assertEqual(t_base.FEATURE, t_numba.FEATURE, f"FEATURE mismatch on {crit}, adaptive={adaptive}")
                self.assertEqual(t_base.VALUE, t_numba.VALUE, f"VALUE mismatch on {crit}, adaptive={adaptive}")
                self.assertEqual(t_base.CLASS, t_numba.CLASS, f"CLASS mismatch on {crit}, adaptive={adaptive}")
                np.testing.assert_allclose(t_base.IMPURITY, t_numba.IMPURITY, atol=1e-12)
                np.testing.assert_array_equal(t_base.predict(self.X), t_numba.predict(self.X))
                np.testing.assert_allclose(
                    t_base.compute_feature_importance(),
                    t_numba.compute_feature_importance(),
                    atol=1e-12,
                )


if __name__ == '__main__':
    unittest.main()