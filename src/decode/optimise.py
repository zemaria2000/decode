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
        logger.info("Initialising an optimisation session...")

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
    
        # Storage folder for optimisations
        self._storage_settings()

        logger.debug(f"Important optimisation parameters: objective = {self.metric_name}, direction = {self.direction}")

    def _logging_options(self):
        optuna.logging.enable_propagation()
        optuna_logger = logging.getLogger("optuna")
        optuna_logger.handlers = logger.handlers
        optuna_logger.setLevel(logging.INFO)

    def _storage_settings(self):
        # NOTE: just a copy from the base tree class...
        """
            Simple internal function just to define some basic storage folders
        """
        ###
        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cwd = os.getcwd()  # current working directory

        # Heuristic: if we're running from site-packages, use cwd as project root
        if "site-packages" in package_root or "dist-packages" in package_root:
            project_root = cwd
        else:
            project_root = package_root  # local dev case
        ### 

        self.storage = Munch()
        self.storage.optimisation = os.path.join(project_root, "optimisations")
        os.makedirs(self.storage.optimisation, exist_ok=True)
    
    def _created_stratified_split_regression(self):
        """
        Create a stratified train/test split for regression targets.
        Automatically merges sparse bins to avoid ValueError.
        """
        bins = np.histogram_bin_edges(self.targets, bins='auto')
        groups = np.digitize(self.targets, bins)

        ## Need to check+ensure if all bins have more than one sample...
        # Count samples per bin
        unique, counts = np.unique(groups, return_counts=True)
        # find sparse bins (fewer than 2 samples)
        sparse_bins = unique[counts < 2]
        if len(sparse_bins) > 0:
            # merge sparse bins into nearest non-sparse bin
            for sparse_bin in sparse_bins:
                # find nearest bin index that is not sparse
                diffs = np.abs(unique - sparse_bin)
                nearest_bin = unique[np.argmin(diffs[counts >= 2])]
                groups[groups == sparse_bin] = nearest_bin
            # verify after merging
            unique, counts = np.unique(groups, return_counts=True)
        # final check: if still only one group, fall back to random split
        if len(unique) < 2:
            logger.debug("[Warning] Could not stratify regression targets — using random split.")
            stratify = None
        else:
            stratify = groups

        return train_test_split(self.dataset, self.targets, test_size = 0.2, stratify = stratify, random_state = self.seed)
    
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
        self.study = optuna.create_study(direction = self.direction, sampler = optuna.samplers.GPSampler(seed = self.seed))
        logger.info(f"New optimisation starting: n_trials = {n_trials}")
        self.study.optimize(self.objective, n_trials = n_trials)

    def store_opt_results(self, opt_name: str = None, store_params: bool = True):
        """
            Store the results from the optimisation process, based on the optuna study
            Args:
                - opt_name: name of the folder used to store information regarding the optimisation. It is recommended, for instance, that we give the name of the dataset to the folder, for instance
                - store_params: Flag that enables storage, in a yaml file, of the optimised hyperparams for the DECODE model + additional info
        """

        # Creating a dataframe with results
        df = pd.DataFrame()
        # Other important params
        other_params = {'metric': self.metric_name, 'opt_direction': self.direction}
        for i, trial in enumerate(self.study.trials):
            df = pd.concat([df, pd.concat([pd.DataFrame(trial.params, index = [i]), pd.DataFrame(trial.user_attrs, index = [i]), pd.DataFrame(other_params, index = [i]), pd.DataFrame({'test_score': trial.value}, index = [i])], axis = 1)], axis = 0)
        df = df.sort_values(by = 'test_score', ascending = False if self.direction == 'maximize' else True)
        df.reset_index(drop = False, inplace = True)
        df.rename(columns = {'index': 'trial_no'}, inplace = True)

        # Storing the df in the specific folder
        target_folder_name = os.path.join(self.storage.optimisation, opt_name if opt_name is not None else datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
        os.makedirs(target_folder_name, exist_ok = True)
        df.to_csv(os.path.join(target_folder_name, 'opt_results.csv'))

        # Storing the yaml with main information
        if store_params:
            # Filtering to get the simplest tree (lower depth) with the highest possible score
            performance_mask = df['test_score'] == df['test_score'].max()
            perf_df = df[performance_mask]
            perf_df.sort_values(by = 'max_depth', ascending = True, inplace = True) 
            perf_df = perf_df.reset_index(drop = True)
            # Storing the hyperparameters
            store_hyperparams = Munch()
            store_hyperparams.model_name = type(self.model).__name__
            store_hyperparams.final_res = np.round(df.loc[0, 'test_score'], 3)
            store_hyperparams.opt_metric = str(self.metric_name)
            store_hyperparams.decode_params = Munch(
                max_depth = df.loc[0, 'max_depth'],
                imp_threshold = df.loc[0, 'imp_threshold'],
                n_steps = df.loc[0, 'n_steps'],
                adaptive = df.loc[0, 'adaptive'],
                min_samples_split = df.loc[0, 'min_samples_split'],
                min_impurity_decrease = df.loc[0, 'min_impurity_decrease'],
                criterium = df.loc[0, 'criterium'],
                perturb_dataset = df.loc[0, 'perturb_dataset']
            )
            store_hyperparams_dict = store_hyperparams.toDict()
            store_hyperparams_dict = make_serializable(store_hyperparams_dict)
            with open(os.path.join(target_folder_name, "hyperparams.yaml"), "w") as f:
                yaml.safe_dump(store_hyperparams_dict, f, sort_keys = False)