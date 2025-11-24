from decode import utils
from sklearn.dummy import DummyClassifier
from sklearn.datasets import make_classification
import unittest
import numpy as np


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.reg_y = np.random.uniform(0, 1, size = (500, 3))
        self.clf_y = np.random.randint(0, 4, size = (500, 1))

    ################################################
    ############ REGRESSION FUNCTIONS ##############
    ################################################

    def test_mse(self):
        mse = utils.mse(self.reg_y)
        self.assertIsInstance(mse, float)

    def test_msle(self):
        msle = utils.msle(self.reg_y)
        self.assertIsInstance(msle, float)
    
    def test_mae(self):
        mae = utils.mae(self.reg_y)
        self.assertIsInstance(mae, float)
    
    def test_huber(self):
        huber = utils.huber(self.reg_y)
        self.assertIsInstance(huber, float)

    ################################################
    ########## CLASSIFICATION FUNCTIONS ############
    ################################################
    
    def test_gini(self):
        dist = np.bincount(self.clf_y.flatten()) / len(self.clf_y)
        gini = utils.gini(dist)
        self.assertIsInstance(gini, float)

    def test_log_loss(self):
        dist = np.bincount(self.clf_y.flatten()) / len(self.clf_y)
        log_loss = utils.log_loss(dist)
        self.assertIsInstance(log_loss, float)

if __name__ == '__main__':
    unittest.main()