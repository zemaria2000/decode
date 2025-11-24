"""
    Testing all features for the regression tree
"""

from decode import ClassificationTree
from sklearn.dummy import DummyClassifier
from sklearn.datasets import make_classification
import unittest

class TestClassificationTree(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_classification(n_samples = 200, n_features = 10, n_informative = 4, n_classes = 3)
        self.model = DummyClassifier(strategy = 'uniform')
        self.model.fit(self.X, self.y)

    def test_reg_tree_features(self):
        # Tree building
        tree = ClassificationTree(model = self.model,
                              scalerX = None,
                              scalerY = None,
                              dataset = self.X,
                              targets = self.y)
        tree.build_tree(max_depth = 5,
                        imp_threshold = 0.05,
                        min_samples_split = 2,
                        min_impurity_decrease = 0)
        
        # Tree predictions
        # Entire dataset
        preds, paths = tree.make_predictions(self.X)
        # One sample
        one_pred, one_path = tree.make_predictions(self.X[0])

        # Feature importances
        tree.compute_feature_importance()

        # Plot tree
        tree.plot_tree()
        # Textual explanation of one prediction
        text = tree.extract_prediction_path(self.X[0])
        print(text)


if __name__ == '__main__':
    unittest.main()

