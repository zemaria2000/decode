"""
    Testing all features for the regression tree
"""

from decode import RegressionTree
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import unittest


class TestRegressionTree(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_regression(n_samples = 200, n_features = 5, n_targets = 1)
        self.model = LinearRegression()
        self.model.fit(self.X, self.y)

    def test_reg_tree_features(self):
        # Tree building
        tree = RegressionTree(model = self.model,
                              scalerX = None,
                              scalerY = None,
                              dataset = self.X,
                              targets = self.y)
        tree.build_tree(max_depth = 3,
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

        # Plot without boxplots Vs. with boxplots
        tree.plot_tree(plot_boxplots = False, save_fig = False)
        tree.plot_tree(plot_boxplots = True, save_fig = False)
        # Textual explanation of one prediction
        text = tree.extract_prediction_path(self.X[0])
        print(text)


if __name__ == '__main__':
    unittest.main()

