"""
    Testing all features for the classification tree, which include:
        - Tree building process based on classifier
        - Making the predictions for both an entire dataset and one single sample
        - Computing feature importances
        - Plotting the tree
        - Extracting a textual explanation for one single prediction
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
        # Tree initialisation
        tree = ClassificationTree(model = self.model,
                              scalerX = None,
                              scalerY = None,
                              dataset = self.X,
                              targets = self.y)
        # Tree building process
        tree.fit(max_depth = 5,
                 imp_threshold = 0.05,
                 min_samples_split = 2,
                 min_impurity_decrease = 0,
                 class_weights = 'balanced')
        
        # Tree predictions for entire dataset + one sample
        preds, paths = tree.predict(self.X, return_paths = True)
        preds = tree.predict(self.X)
        one_pred, one_path = tree.predict(self.X[0], return_paths = True)

        # Computing feature importances
        tree.compute_feature_importance()

        # Plotting tree
        tree.plot_tree()
        
        # Textual explanation of one prediction
        text = tree.extract_prediction_path(self.X[0])
        print(text)


if __name__ == '__main__':
    unittest.main()

