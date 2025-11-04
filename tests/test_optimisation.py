import optuna
from decode import ClassificationTree, RegressionTree, Optimise
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_classification, make_regression
import unittest


class OptimisationTests(unittest.TestCase):

    def test_reg_optimisation(self):

        X, y = make_regression(n_samples = 500, n_features = 5, n_targets = 1)
        model = LinearRegression()
        model.fit(X, y)
        # Base tree
        base_tree = RegressionTree(model = model,
                                    dataset = X,
                                    targets = y,
                                    auto_create_folders = False)
        # Optimisation object
        opt = base_tree.instantiate_opt_object(model_type = 'reg')
        opt.run_optimisation(n_trials = 10)
        base_tree.store_opt_results(opt_name = 'unittests_optimisation_reg', store_params = True)

    
    def test_clf_optimisation(self):

        X, y = make_classification(n_samples = 200, n_features = 10, n_informative = 4, n_classes = 3)
        model = DummyClassifier(strategy = 'uniform')
        model.fit(X, y)
        # Base tree
        base_tree = ClassificationTree(model = model,
                                    dataset = X,
                                    targets = y,
                                    auto_create_folders = False)
        # Optimisation object
        opt = base_tree.instantiate_opt_object(model_type = 'clf')
        opt.run_optimisation(n_trials = 10)
        base_tree.store_opt_results(opt_name = 'unittests_optimisation_clf', store_params = True)


if __name__ == '__main__':
    unittest.main()
