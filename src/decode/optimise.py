"""
    New optimisation file compatible with the DECODE class and methods. It should be as abstract as possible to allow for optimisation of any possible model + it is fully compatible with optuna
"""

from .classification import ClassificationTree
from .regression import RegressionTree
from .utils import make_serializable
from .logger import logger
import logging

import numpy as np
import os
import pandas as pd
from datetime import datetime
import yaml
from munch import Munch
from sklearn import metrics
from sklearn.model_selection import train_test_split
import optuna


class Optimise:

    def __init__(self, model, model_type, dataset, targets, obj_function: str = None, direction: str = 'maximize', feature_types = None, scalerX = None, scalerY = None, seed = 17, **kwargs):
        """
            Args:
                - model: mandatory to provide a pre-trained model, either a classifier or a regressor
                - model_type: we should provide info on whether we have a classifier ('clf') or a regressor ('reg') model
                - dataset: also mandatory to provide the input dataset (normally a trainX or something like that)
                - targets: outputs for the model. An array with either classification or regression outputs (for now regression only compatible with a single output)
                - obj_function (str): objective function name compatible with those available in sklearn.metrics 
                - direction (str): the direction for optimisation. Depending on the metric, we should be careful in defining it (minimize vs maximize)
                - feature_types: array with 'N' for numerical features and 'C' for categorical features. Useful for proper scaling of the models
                - scalerX: scaler for the inputs (if used by the classifier/regressor)
                - scalerY: scaler for the outputs (if used for the classifier/regressor)
                - seed: for reproducibility
                - kwargs: additional kwargs for specific functions
        """
        self._logging_options()

        self.kwargs = kwargs
        self.model = model
        self.model_type = model_type
        self.direction = direction
        if self.model_type not in ['clf', 'reg']:
            raise KeyError("Model type should either be 'clf' for classification models or 'reg' for regression models!")
        self.feature_types = feature_types
        if self.feature_types is None: # We just assume they are all numerical. NOTE: this is not the final solution, as the scaler would have previously been fit already according to this feature tyopolgy...
            self.feature_types = ['N' for _ in range(dataset.shape[1])]                
        self.scalerX = scalerX
        self.scalerY = scalerY
        self.seed = seed

        # Reducing dataset size for datasets that are too large (> 2k samples). We just want to find a combination of hyperparameters that is promising through a fast and efficient process...
        if len(dataset) < 10000:
            self.dataset = dataset
            self.targets = targets
        else:
            rng = np.random.default_rng(self.seed)
            target_idx = rng.choice(len(dataset), size = 10000, replace = False)
            self.dataset, self.targets = dataset[target_idx], targets[target_idx]

        # Creating a simple, stratified, 80-20 split for fitting the DECODE tree and evaluating it, in case no KFold is wanted
        if self.model_type == 'clf':
            self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.dataset, self.targets, test_size = 0.2, stratify = self.targets, random_state = self.seed)
        elif self.model_type == 'reg':
            self.trainX, self.testX, self.trainY, self.testY = self._created_stratified_split_regression()
        
        # Obtaining the objective function to conduct evaluation
        if obj_function is not None:
            self.metric_name = obj_function.strip().lower().replace(" ", "_")
        else:
            if self.model_type == 'clf':
                self.metric_name = 'matthews_corrcoef'
            else:
                self.metric_name = 'r2_score'
        self.metric_fn = getattr(metrics, self.metric_name)

    def _logging_options(self):
        optuna.logging.enable_propagation()
        optuna_logger = logging.getLogger("optuna")
        optuna_logger.handlers = logger.handlers
        optuna_logger.setLevel(logging.INFO)
    
    def _created_stratified_split_regression(self):
        """
        Create a stratified train/test split for regression targets.
        Automatically merges sparse bins to avoid ValueError.
        """
        # Step 1: Bin continuous target into discrete strata
        bins = np.histogram_bin_edges(self.targets, bins='auto')
        groups = np.digitize(self.targets, bins)

        # Step 2: Iteratively merge sparse bins until all bins have ≥ 2 samples
        while True:
            unique, counts = np.unique(groups, return_counts=True)
            sparse_mask = counts < 2
            if not np.any(sparse_mask):
                break  # all bins ok

            # If all bins are sparse → cannot stratify
            if np.all(sparse_mask):
                logger.debug("[Warning] All bins are sparse — using random split.")
                return train_test_split(
                    self.dataset, self.targets, test_size=0.2, random_state=self.seed
                )

            # Merge each sparse bin into its nearest dense bin
            for sparse_bin in unique[sparse_mask]:
                dense_bins = unique[~sparse_mask]
                nearest_bin = dense_bins[np.argmin(np.abs(dense_bins - sparse_bin))]
                groups[groups == sparse_bin] = nearest_bin

        # Step 3: Final validation
        unique, counts = np.unique(groups, return_counts=True)
        if len(unique) < 2 or np.any(counts < 2):
            logger.debug("[Warning] Could not safely stratify regression targets — using random split.")
            stratify = None
        else:
            stratify = groups

        # Step 4: Perform split
        return train_test_split(
            self.dataset, self.targets, test_size=0.2, stratify=stratify, random_state=self.seed
        )
    
    def _generate_hyperparams(self, trial):
        """
            Function that for each trial will generate all tree hyperparams 
            Returns:
                - hyperparams (dict): dictionary with the new tree hyperparams to be tested
        """
        
        # 1. New optimisation attempt with new hyperparameters
        max_depth = trial.suggest_int('max_depth', low = 2, high = 5, step = 1)
        imp_threshold = trial.suggest_float('imp_threshold', 0.0, 0.1, step = 0.05)
        n_steps = 'auto'
        adaptive = False
        min_samples_split = int(trial.suggest_categorical('min_samples_split', [2, 5, 10]))
        min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 0.0, 0.1, step = 0.05)
        if self.model_type == 'reg':
            criterium = trial.suggest_categorical('criterium', ['mse', 'mae'])
        elif self.model_type == 'clf':
            criterium = trial.suggest_categorical('criterium', ['gini', 'log_loss'])
        perturb_dataset = False        
        
        # Storing some of the "fixed" parameters
        trial.set_user_attr("perturb_dataset", perturb_dataset)
        trial.set_user_attr("n_steps", n_steps)
        trial.set_user_attr("adaptive", adaptive)

        # 2. Storing those in a dictionary
        hyperparams = {
            'max_depth': max_depth,
            'imp_threshold': imp_threshold,
            'n_steps': n_steps,
            'adaptive': adaptive,
            'min_samples_split': min_samples_split,
            'min_impurity_decrease': min_impurity_decrease,
            'criterium': criterium,
            'perturb_dataset': perturb_dataset
        }

        return hyperparams

    def objective(self, trial):
        """
            Objective function to evaluate our optimisation attempt
        """

        # 1. Retrieving the new hyperparams
        decode_hyperparams = self._generate_hyperparams(trial)

        # 2. Tree building and objective function evaluation
        # Instantiate tree
        if self.model_type == 'clf':
            decode = ClassificationTree(model = self.model, scalerX = self.scalerX, scalerY = self.scalerY, dataset = self.trainX, targets = self.trainY, feature_names = None, target_names = None, feature_types = self.feature_types, seed = self.seed)
        elif self.model_type == 'reg':
            decode = RegressionTree(model = self.model, scalerX = self.scalerX, scalerY = self.scalerY, dataset = self.trainX, targets = self.trainY, feature_names = None, target_names = None, feature_types = self.feature_types, seed = self.seed)
        # Build tree with optimisation hyperparams
        try:
            decode.build_tree(**decode_hyperparams)
        except Exception as e:
            trial.set_user_attr("error", str(e))
            return -1e-8 if self.direction == "maximize" else 1e8
        # Conduct predictions with tree (train + test)
        train_preds, _ = decode.make_predictions(self.trainX)
        test_preds, _ = decode.make_predictions(self.testX)
        # Evaluate the predictions
        train_score = self.metric_fn(self.trainY, train_preds, **self.kwargs)
        test_score = self.metric_fn(self.testY, test_preds, **self.kwargs)

        # Storing this auxiliary optimisation attribute
        trial.set_user_attr("train_score", train_score)

        return test_score
        
    def run_optimisation(self, sampler = None, n_trials: int = 100):
        """
            Running the optimisation through optuna
            Args:
                - sampler: None for now, meaning that I always run the same sampler/optimisation approach
                - n_trials (int): number of optimisation attempts
        """
        logger.info("Initialising an optimisation session...")
        self.study = optuna.create_study(direction = self.direction, sampler = optuna.samplers.GPSampler(seed = self.seed))
        logger.info(f"New optimisation starting: n_trials = {n_trials}")
        self.study.optimize(self.objective, n_trials = n_trials)

