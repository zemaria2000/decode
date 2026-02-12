"""
    Base tree class 
"""

import numpy as np
from pathlib import Path
from .utils import *
from .tools import *
from abc import ABC, abstractmethod
from typing import Union
from .logger import logger
from munch import Munch
import os, yaml
import pandas as pd
from joblib import Parallel, delayed
import logging 
from scipy.stats.qmc import Sobol, LatinHypercube, scale
from sklearn.model_selection import train_test_split 
from copy import deepcopy

class BaseTree(ABC):

    def __init__(self, 
                 model, 
                 dataset: np.ndarray, 
                 targets: np.ndarray, 
                 scalerX = None, 
                 scalerY = None, 
                 feature_names: list = None, 
                 target_names: list = None, 
                 feature_types: list = None,
                 use_synth_ds: bool = False,
                 synth_ds_info: dict = None,
                 seed: int = 42, 
                 root_path: str = None, 
                 use_logging: bool = True,
                 log_level = logging.INFO):
        """
            Args:
                - model: model instance for the classifier/regressor
                - scalerX: scaler for the inputs
                - scalerY: scaler for the outputs (only relevant for regression problems)
                - dataset (np.ndarray): input dataset with features - not-normalised
                - targets (np.ndarray): output dataset with targets - not-normalised
                - feature_names (list): list with all feature names for plotting purposes
                - target_names (list): list with all target names for plotting purposes
                - feature_types (list): list that defines numerical features - 'N' - and categorical features - 'C'. Categorical features will not be considered for normalisation
                - use_synth_ds (bool): flag to enable or disable the synthetic dataset generation process
                - synth_ds_info (dict): dictionary with information about the synthetic dataset generation process. If None, no synthetic dataset will be generated. It must contain info regarding (1) sampler to be used, (2) n_samples to be generated, and (3) strategy for the final dataset creation, and (4) sampler kwargs. More specific info in the 
                - seed (int): seed for random number generation
                - root_path (str): specific path to store results. If not given, it stores in the current working directory, inly when required
                - logging (bool): flag to enable or disable logging
        """
        
        self.model = model
        self.scalerX = scalerX
        self.scalerY = scalerY
        self.dataset = dataset
        self.targets = targets
        self.feature_names = feature_names
        self.target_names = target_names
        self.feature_types = feature_types
        self.seed = seed
        self.root_path = root_path

        np.random.seed(self.seed)

        if self.model is None:
            raise KeyError("Model instance has not been passed")
        if self.dataset is None:
            raise KeyError("No input dataset has been provided")
        if self.targets is None:
            raise KeyError("No output targets have been provided")
        if self.target_names is None:
            logger.warning("No target names have been provided. Replaced by an ordered list of integers")
            self.target_names = [f'Target {i}' for i in range(len(np.unique(targets)))]
        if self.feature_names is None:
            logger.warning("No feature names have been provided. Replaced by an ordered list of integers")
            self.feature_names = [f'Feature {i}' for i in range(dataset.shape[1])]
        if self.feature_types is None:
            logger.warning("No feature types have been provided. Considering all features as numerical (may not be aligned with model training...)")
            self.feature_types = ['N' for _ in range(dataset.shape[1])]
        
        self.numerical_idx = [i for i, feat in enumerate(self.feature_types) if feat == 'N']
        self.categorical_idx = [i for i, feat in enumerate(self.feature_types) if feat == 'C']

        # Reshaping targets so that they have always 2 dimensions
        if len(self.targets.shape) == 1:
            self.targets = self.targets.reshape(-1, 1)

        # Handling the synthetic dataset generation process
        if not use_synth_ds:   # if not, default dataset will be used
            self.decode_dataset = self.dataset
            self.decode_targets = self.targets
            self.decode_normalised_dataset = self._scale_inputs(self.decode_dataset)
        else:
            # Getting a series of info regarding the synthetic dataset generation
            sampler = synth_ds_info.get('sampler', 'sobol')
            n_samples = synth_ds_info.get('n_samples', 'auto')
            strategy = synth_ds_info.get('strategy', 'synth')
            sampler_kwargs = synth_ds_info.get('sampler_kwargs', {})
            self._generate_synthetic_dataset(sampler = sampler, n_samples = n_samples, strategy = strategy, **sampler_kwargs)

        # Conducting the predictions with the model        
        self.model_preds = self._postprocess_prediction(self.model.predict(self.decode_normalised_dataset))
        self.model_probs = self.model.predict_proba(self.decode_normalised_dataset)

        # Resetting the tree
        self._reset_tree()

        # Defining storage settings
        self._storage_settings()

        # Internal flag that states if the tree is for multi-output or single_output regression
        self.flag_multi_output = True if self.decode_targets.shape[1] > 1 else False

        if use_logging:
            logger.setLevel(log_level)
        else:
            logger.info("User disabled logging for the tree")
            logger.setLevel(logging.CRITICAL)

        logger.info("Initialised tree object")




    #########################################################
    #################### STORAGE FOLDERS ####################
    #########################################################
    def _storage_settings(self):
        """
            Simple internal function just to define some basic storage folders
        """

        # Resolving the paths
        if self.root_path is not None:
            root = Path(self.root_path).expanduser()
            # If relative, resolve against cwd
            if not root.is_absolute():
                self.root = (cwd / root).resolve()
        else:
            cwd = os.getcwd()  # current working directory
            default_root = cwd
            self.root = default_root
        
        # Base folders for storage
        self.storage = Munch()
        self.storage.folders = Munch(
            figs = os.path.join(self.root, "figures"),
            results = os.path.join(self.root, "results"),
            data = os.path.join(self.root, "datasets"),
            logs = os.path.join(self.root, "logs")
        )

        # Creating the logs folders
        os.makedirs(os.path.join(self.root, self.storage.folders.logs), exist_ok = True)


    #########################################################
    ################## TREE INITIALISATION ##################
    #########################################################
    
    def _reset_tree(self):
        """
            Initialising all possible parameters for the tree building process 
        """
        self.N_NODE_SAMPLES = []    # Number of samples covered in a node's rule
        self.IMPURITY = []          # Gini score for each node's rule (classification) / MSE score for each node's rule (regression)
        self.VALUE = []             # Value of the feature that is incorporated in the rule
        self.FEATURE = []           # Index of the feature of the target rule for a node
        self.LEFT = []              # Whether a particular node is a left node (1) or a right node (0)
        self.DEPTH = []             # Depth of a specific node in the overall tree
        self.LEAF = []              # Whether a specific node is a leaf node
        self.PARENT = []            # Indicator of the parent for a specific node
        self.NODE_DS = []           # Placeholder for the original dataset indexes to be used    
        # Variables specific for classification
        self.CLASS_DIST = []        # Class distribution of the resulting rule 
        self.CLASS = []             # Class of the target rule for a node
        # Variables specific for regression
        self.AVG_VALUE = []         # Average output value for a specific node


    #########################################################
    ################### SCALING UTILITIES ###################
    #########################################################
    
    def _scale_inputs(self, X: np.ndarray):
        """
            Given a specific input dataset, and if a scalerX is defined, will return the scaled inputs
            Args:
                - X (np.ndarray): dataset to be normalised
            Returns:
                - X_scaled/X (np.ndarray): normalised dataset/original dataset if no scaler exists
        """
        if hasattr(self, "scalerX") and self.scalerX is not None:
            X_scaled = X.copy()
            X_scaled[:, self.numerical_idx] = self.scalerX.transform(
                X_scaled[:, self.numerical_idx]
            )
            return X_scaled
        else:
            return X
    
    def _inverse_scale_inputs(self, X_scaled: np.ndarray):
        """
            Given a specific input dataset, and if a scalerX is defined, will return the inverse scaled inputs
            Args:
                - X_scaled (np.ndarray): normalised dataset
            Returns:
                - X (np.ndarray): original dataset/original dataset if no scaler exists
        """
        if hasattr(self, "scalerX") and self.scalerX is not None:
            X = X_scaled.copy()
            X[:, self.numerical_idx] = self.scalerX.inverse_transform(
                X_scaled[:, self.numerical_idx]
            )
            return X
        else:
            return X_scaled
        
    def _inverse_scale_outputs(self, y: np.ndarray):
        """
            Given a specific set of predictions y, inverse scaling them to the original format (relevant for regression problems)
            Args:
                - y (np.ndarray): outputs from the model, scaled
            Returns:
                - inverse_y/ y (np.ndarray): either the inverse transformed y, or the original if no scaler is provided
        """
        if hasattr(self, "scalerY") and self.scalerY is not None:
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            inverse_y = self.scalerY.inverse_transform(y)
            return inverse_y
        return y        
    
    def _postprocess_prediction(self, y_raw):

        if self.task == "classification":
            return self._postprocess_classification(y_raw)

        elif self.task == "regression":
            return self._postprocess_regression(y_raw)

        else:
            raise ValueError("BaseTree.task not set")
        

    ###############################################################
    ################# SYNTHETIC DATASET GENERATION ################
    ###############################################################

    def _generate_synthetic_dataset(self, sampler: str = None, n_samples: Union[str, int] = 'auto', strategy: str = None, **sampler_kwargs):

        # Generating the synthetic dataset
        if len(self.numerical_idx) == 0:   # Categorical dataset
            x_synth, y_synth = generate_categorical_synthetic_dataset(self.dataset, self.model)
            if x_synth.shape[0] > 1e6:
                logging.warning("Generated synthetic dataset has more than 1 million samples. This may lead to memory issues. As such, we will be defaulting to the original dataset only")
                self.decode_dataset = self.dataset
                self.decode_targets = self.targets
                self.decode_normalised_dataset = self._scale_inputs(self.decode_dataset)
            else:
                self.decode_dataset = x_synth
                self.decode_targets = y_synth.reshape(-1, 1)
                self.decode_normalised_dataset = self._scale_inputs(self.decode_dataset)
        else:   # Continuous or mixed dataset
            x_synth, y_synth = generate_continuous_synth_dataset(dataset = self.dataset, 
                                                                    scaler = self.scalerX,
                                                                    model = self.model,
                                                                    numerical_idx = self.numerical_idx,
                                                                    categorical_idx = self.categorical_idx,
                                                                    sampler_name = sampler,
                                                                    n_samples = n_samples,
                                                                    seed = self.seed, 
                                                                    **sampler_kwargs)
            if strategy == 'synth':
                self.decode_dataset = x_synth
                self.decode_targets = y_synth.reshape(-1, 1)
                self.decode_normalised_dataset = self._scale_inputs(self.decode_dataset)
            elif strategy == 'both':
                self.decode_dataset = np.concatenate((self.dataset, x_synth), axis = 0)
                self.decode_targets = np.concatenate((self.targets, y_synth.reshape(-1, 1)), axis = 0)
                self.decode_normalised_dataset = self._scale_inputs(self.decode_dataset)

        

    ###############################################################
    ################### VARIABLE DISCRETIZATION ###################
    ###############################################################

    @abstractmethod
    def _refine_step_bins(self, feature_data, preds, bin_idx, edges):
        """ Subclass decides which bins to refine """
        pass

    def _create_auto_steps(self,
                            dataset_idx: np.ndarray,
                            n_steps: Union[str, int] = None,
                            adaptive: bool = False,
                            refine_factor: int = 2,
                            n_refine: int = 2,
                            criterium: str = 'brier',
                            alpha: float = 0.5
                            ):
        """
            Generates discretization steps per feature.
            - Numeric features:
                * non-adaptive: histogram 'auto' bins or quantile bins
                * adaptive: iteratively refines bins with high within-bin impurity, using model soft predictions
            - Categorical features:
                * uses midpoints between sorted unique integer-coded categories
            Args:
                - dataset_idx (np.ndarray): index, relative to the original dataset (self.decode_dataset), of our current dataset to be assessed
                - n_steps (str/int): 'auto' or int number of initial steps for numeric features
                - adaptive (bool): whether to use adaptive bin refinement for numeric features
                - refine_factor (int): factor by which to refine each selected bin
                - n_refine (int): number of refinement iterations
                - criterium (str): criterium used to score bins for refinement
                
            Returns
                - step_info : dict
                    step_info[feature] = {'min','max','median','steps'} for numeric
                    step_info[feature] = {'steps'} for categorical
        """
        step_info = {}

        for j, feature in enumerate(self.feature_names):
            x = self.decode_dataset[dataset_idx, j]

            # HANDLING NUMERICAL FEATURES
            if self.feature_types[j] == "N":
                
                # Filter invalid values (keeps alignment with preds if adaptive)
                finite_mask = np.isfinite(x)
                x_f = x[finite_mask]

                # Basic stats (handle empty/constant cases)
                if x_f.size == 0:
                    step_info[feature] = {"min": np.nan, "max": np.nan, "median": np.nan, "steps": np.array([])}
                    continue

                # min, max, and median values
                x_min = float(np.min(x_f))
                x_max = float(np.max(x_f))
                x_med = float(np.median(x_f))
                step_info[feature] = {"min": x_min, "max": x_max, "median": x_med}

                # Constant feature => no thresholds
                if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
                    step_info[feature]["steps"] = np.array([])
                    continue

                # Generating the initial edges for feature binning
                if isinstance(n_steps, int):
                    # Quantile-based initial edges
                    qs = np.linspace(0.0, 1.0, n_steps + 1)
                    edges = np.unique(np.quantile(x_f, qs))[1:-1]
                else:
                    # Histogram auto bins by default
                    _, bin_edges = np.histogram(x_f, bins = "auto")
                    edges = bin_edges[1:-1]

                # Reassuring that edges are larger than min and lower than max
                edges = np.unique(np.sort(edges))
                edges = edges[(edges > x_min) & (edges < x_max)]

                # Non-adaptive case: just store edges
                if not adaptive:
                    step_info[feature]["steps"] = edges
                    continue

                # ADAPTIVE CASE: iteratively refine bins with high impurity
                preds_all = self.model_probs[dataset_idx]
                preds_f = preds_all[finite_mask]

                # If no valid edges, nothing to refine (but keep empty)
                if edges.size == 0:
                    step_info[feature]["steps"] = edges
                    continue

                # Minimum number of samples per bin to consider for refinement
                min_bin = max(2, getattr(self, "min_samples_split", 2))

                for _ in range(n_refine):
                    # Assign each sample to a current bin
                    bin_idx = np.digitize(x_f, edges, right = True)

                    # Score bins and choose which to refine
                    refine_bins = self._refine_step_bins(
                        preds = preds_f,          
                        bin_idx = bin_idx,
                        edges = edges,
                        min_bin = min_bin,
                        criterium = criterium,
                        alpha = alpha
                    )

                    if len(refine_bins) == 0:
                        break

                    new_edges = edges.tolist()

                    for b in refine_bins:
                        # Refine exactly this bin (avoid boundary logic bugs)
                        mask_b = (bin_idx == b)
                        if int(np.sum(mask_b)) < 2:
                            continue

                        # Split this bin locally using quantiles
                        sub_qs = np.linspace(0.0, 1.0, refine_factor + 1)
                        sub_edges = np.quantile(x_f[mask_b], sub_qs)[1:-1]
                        new_edges.extend(sub_edges.tolist())

                    # Update edges: unique, sorted, inside (min,max)
                    edges = np.unique(np.sort(np.array(new_edges, dtype=float)))
                    edges = edges[(edges > x_min) & (edges < x_max)]

                    # If refinement adds nothing new, stop
                    if edges.size == 0:
                        break

                step_info[feature]["steps"] = edges


            # In case we are dealing with a categorical feature
            elif self.feature_types[j] == 'C':
                unique_vals = np.unique(x).astype(int)
                step_info[feature] = {}
                step_info[feature]['steps'] = (unique_vals[:-1] + unique_vals[1:]) / 2

            else:
                raise ValueError(f"Unknown feature type '{self.feature_types[j]}' for feature '{feature}'")

        return step_info


    ##################################################################
    ################### SPLIT EVALUATION + SCORING ###################
    ##################################################################

    @abstractmethod
    def _impurity_calculation(self, preds: np.ndarray, alpha: float, criterium: str = 'default'):
        """ Impurity calculation handled by the subclass """
        pass

    def _candidate_score(self, dataset_idx: np.ndarray, feature_idx: int, feature_values: np.ndarray, criterium: str = 'default', alpha: float = 0.5):
        """
            Auxiliary function that automatically calculates the candidate score for a particular split of the dataset
            Args:
                - dataset_idx (np.ndarray): index, relative to the original dataset (self.decode_dataset), of our current dataset to be assessed
                - preds (np.ndarray): predictions for our current dataset
                - feature_idx (int): feature index of the target rule
                - feature_values (np.ndarray): all possible values for the target rule
                - criterium (str): criterium used to compute the loss
            Returns:
                - scores (np.ndarray): score for all candidate splits of a particular feature
            See https://scikit-learn.org/stable/modules/tree.html#tree-mathematical-formulation for more details
        """
        # Current dataset
        current_ds = self.decode_dataset[dataset_idx]

        # Current preds
        # preds = self.model_preds[dataset_idx] if criterium in ['gini', 'log_loss'] else self.model_probs[dataset_idx]
        preds = self.model_probs[dataset_idx]

        # Conduct all splits
        n_samples = current_ds.shape[0]
        feature_column = current_ds[:, feature_idx]
        
        # Left and right datasets
        masks = feature_column[None, :] <= feature_values[:, None]
        n_left_samples = masks.sum(axis = 1)
        n_right_samples = n_samples - n_left_samples

        # Filtering any mask that does not comply with the threshold for min_samples_split
        valid_mask = (n_left_samples >= self.min_samples_split) & (n_right_samples >= self.min_samples_split)
        if not np.any(valid_mask):
            return np.full(len(feature_values), np.nan)
        
        # Impurity calculations
        original_impurity = self._impurity_calculation(preds, criterium, alpha)
        left_impurities = np.array([self._impurity_calculation(preds[mask], criterium, alpha) for mask in masks[valid_mask]])
        right_impurities = np.array([self._impurity_calculation(preds[~mask], criterium, alpha) for mask in masks[valid_mask]])
        
        # Candidate scores
        candidate_scores = original_impurity - ((n_left_samples[valid_mask]/n_samples)*left_impurities + (n_right_samples[valid_mask]/n_samples)*right_impurities)

        # Global impurity decreases + checking their compliance with the min_impurity_decrease criterium
        global_impurity_decreases = (n_samples/len(self.decode_dataset))*candidate_scores
        candidate_scores[global_impurity_decreases <= self.min_impurity_decrease] = np.nan

        # Returning the scores for all possible feature values
        scores = np.full(len(feature_values), np.nan)
        scores[valid_mask] = candidate_scores 

        return scores 

    def execute_trial(self, dataset_idx: np.ndarray, step_info: dict, criterium: str = 'default', alpha: float = 0.5):
        """
            Executes a new trial for each feature and its steps, based on the providade dataset
            Args:
                - dataset_idx (np.ndarray): index, relative to the original dataset (self.decode_dataset), of our current dataset to be assessed
                - step_info (dict): dictionary with the feature steps information
                - criterium (str): str that defines the choosing criterium - "default" by default 
            Returns:
                - score_list (list): list of tuples that contemplates (feature_idx, feature_value, candidate_score) for each individual step in each feature considered
        """         
        score_list = None 

        for feature_idx, feature in enumerate(self.feature_names):
            feature_values = np.array(step_info[feature]['steps'])
            candidate_scores = self._candidate_score(dataset_idx = dataset_idx, 
                                                     feature_idx = feature_idx, 
                                                     feature_values = feature_values, 
                                                     criterium = criterium,
                                                     alpha = alpha)
            scores_aux = np.concatenate((np.repeat(feature_idx, len(feature_values))[:, None], feature_values[:, None], candidate_scores[:, None]), axis = 1)
            score_list = np.concatenate((score_list, scores_aux), axis = 0) if score_list is not None else scores_aux
        
        # Remove rows with NaN values and check if we have more than 1 admissible split
        score_list = score_list[~np.isnan(score_list).any(axis = 1)]
        if len(score_list) == 0:
            logger.warning(f"No admissible splits were found for node {len(self.FEATURE) + 1}")
            return None
        
        # Sort by candidate score (last column)
        sorted_list = score_list[score_list[:, -1].argsort()[::-1]]
        return sorted_list[0]
    

    ###################################################################
    ##################### TREE GROWTH/DEVELOPMENT #####################
    ###################################################################  

    @abstractmethod
    def _initialise_root_node(self, final_res: tuple, original_preds: np.ndarray):
        """ Function handled by the subclass - creating the root node """
        pass

    @abstractmethod
    def _expand_node(self, prev_node_idx: int, current_depth: int, dataset: np.ndarray, imp_threshold: float, n_steps: Union[str, int], adaptive: bool, criterium: str, perturb_dataset: bool, left: bool, alpha: float = 0.5):
        """ Function handled by the subclass - expanding a particular node """

    def _compute_class_weights(self, class_weights: Union[str, dict] = None):
        """
            Internal function, only actually relevant for classification problems, that computes the class weights based on the dataset
            Args:
                - class_weights (str/dict): only relevant for classification problems. If 'str' and 'balanced', it will automatically compute the class weights based on the dataset. If 'dict', it will use the provided dictionary to weight each class accordingly           
        """
        if isinstance(class_weights, str) and class_weights == 'balanced':
            # Computing the class weights based on the dataset + pre-computed formula (based on sklearn's one)
            n_samples, n_classes = len(self.decode_targets), len(np.unique(self.decode_targets))
            class_counts = np.bincount(self.decode_targets.flatten().astype(int))
            self.class_weights = {i: n_samples/(n_classes*class_counts[i]) for i in range(n_classes)}
            self.class_weights = np.array([self.class_weights[i] for i in range(n_classes)])
        elif isinstance(class_weights, dict):
            self.class_weights = class_weights
        else:
            self.class_weights = None


    def fit(self, 
            max_depth: int = 5,
            imp_threshold: float = 0.0,
            n_steps: Union[str, int] = 'auto',
            adaptive: bool = False,
            min_samples_split: int = 2,
            min_impurity_decrease: float = 0.0,
            criterium: str = 'default',
            class_weights: Union[None, str, dict] = None,
            alpha: float = 0.5):
        """
            Generates the decision tree based on the provided dataset and model
            Args:
                - max_depth (int): maximum depth of the tree
                - impurity_threshold (float): impurity threshold to stop the tree growth
                - n_steps (str/int): numerical variable discretization approach. If 'str' and 'auto', it will automatically create the steps for each numerical feature. If 'int', it will create that number of quantile-based steps for each numerical feature
                - adaptive (bool): whether to create adaptive quantile-based steps for numerical features
                - min_samples_split (int): minimum samples required to attempt a split
                - min_impurity_decrease (float): minimum impurity decrease required to accept a split
                - criterium (str): the criterium to be used to evaluate candidate nodes of the tree
                - class_weights (str/dict/None): only relevant for classification problems. If 'str' and 'balanced', it will automatically compute the class weights based on the dataset. If 'dict', it will use the provided dictionary to weight each class accordingly
        """

        logger.info(f"Starting the tree generation process")
        logger.debug(f"Parameters: max_depth={max_depth}, imp_threshold={imp_threshold}, n_steps={n_steps}, adaptive = {adaptive}, min_samples_split={min_samples_split}, min_impurity_decrease={min_impurity_decrease}, criterium={criterium}, alpha={alpha}")

        # Just some sanity checking regarding the adaptive steps
        if adaptive and criterium in ['gini', 'log_loss']:
            logger.warning("Adaptive step generation is not compatible with 'gini' or 'log_loss' criteria. Disabling adaptive step generation.")
            adaptive = False
        
        # Instantiating by storing the stopping criteria
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_depth = max_depth
        self.imp_threshold = imp_threshold

        # Handling class weights for classification problems
        self._compute_class_weights(class_weights)

        # Clearing, for safety, all tree parameters
        self._reset_tree()

        ### 1. INITIAL NODE TRIAL
        original_idx = np.arange(len(self.decode_dataset))
        step_info = self._create_auto_steps(original_idx, n_steps = n_steps, adaptive = adaptive, criterium = criterium, alpha = alpha)
        final_res = self.execute_trial(original_idx, step_info, criterium, alpha = alpha)
        if final_res is None:
            raise RuntimeError("No valid split found for root node. Try reducing constraints.")    
        # Storing the root info
        self._initialise_root_node(final_res, criterium)
        logger.info(f"Root node created. Rule is {self.feature_names[self.FEATURE[0]]} <= {self.VALUE[0]}")
        logger.debug(f"Root node impurity = {self.IMPURITY[0]}")

        ### 2. TREE GROWTH LOOP

        for i in range(1, max_depth + 1):

            # Gathering the previous depth indices - parents
            prev_depth_indices = [index for index, value in enumerate(self.DEPTH) if value == (i - 1)]

            # Iterating for all possible parent nodes
            for prev_node_index in prev_depth_indices:

                # Skip if already leaf
                if self.LEAF[prev_node_index] == 1:
                    continue

                ## 2.1. SELECT DATASET IDXS FOR THE NEXT NODE
                current_ds = self.decode_dataset[self.NODE_DS[prev_node_index]]    
                left_mask = current_ds[:, self.FEATURE[prev_node_index]] <= self.VALUE[prev_node_index]
                left_ds_idx = self.NODE_DS[prev_node_index][left_mask]
                right_ds_idx = self.NODE_DS[prev_node_index][~left_mask]
                
                ## 2.2. PROCESS LEFT NODE
                if len(left_ds_idx) > 0:
                    self._expand_node(prev_node_idx = prev_node_index, current_depth = i, dataset_idx = left_ds_idx, n_steps = n_steps, adaptive = adaptive, criterium = criterium, left = True, alpha = alpha)

                ## 2.3. PROCESS RIGHT NODE
                if len(right_ds_idx) > 0:
                    self._expand_node(prev_node_idx = prev_node_index, current_depth = i, dataset_idx = right_ds_idx, n_steps = n_steps, adaptive = adaptive, criterium = criterium, left = False, alpha = alpha)

        logger.info(f"Tree building process completed: total nodes = {len(self.N_NODE_SAMPLES)}, leaf nodes = {np.sum(self.LEAF)}")

    ###################################################################
    ###################### CONDUCTING PREDICTIONS #####################
    ###################################################################  

    @abstractmethod
    def _return_leaf_output(self, node_idx: int):
        """ Handled by each subclass to return the adequate predictions """

    def predict(self, samples: np.ndarray, return_paths: bool = False):
        """
            Make predictions for one or multiple samples based on the decision tree rules defined by the parameters.
            Args:
                - samples (np.ndarray): One or multiple samples to predict (must be in the shape of (n_samples, n_features))
                - return_paths (bool): whether to return the prediction paths for each sample
            Returns:
                - predictions (np.ndarray): Array with the predictions for each sample
                - prediction_paths (list): list of lists, where each sublist contains the nodes traversed for each sample

        """
        # 1. Initial check to see if we have just one sample or multiple samples
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        # 2. Now, iterate for each sample
        predictions, prediction_paths = [], []
        for sample in samples:

            current_node = 0 # Start in the root node
            pred_path = [0]  # Keep track of the path taken

            while True:   # This allows us to iterate until we reach a leaf node
                
                # In case we are in the leaf node, we'll just return the prediction (most likely for now)
                if (self.LEAF[current_node] == 1 or self.FEATURE[current_node] is None or self.VALUE[current_node] is None):
                    pred = self._return_leaf_output(node_idx = current_node)
                    predictions.append(pred)
                    if return_paths:
                        prediction_paths.append(pred_path)
                    break

                # Otherwise...
                condition = (sample[self.FEATURE[current_node]] <= self.VALUE[current_node])
                if condition:
                    # Iterate to find the left child node
                    for index, parent in enumerate(self.PARENT):
                        if parent == current_node and self.LEFT[index] == 1:  # it only happens if the parent of the next node is the current node and its LEFT array says it is indeed a left child (that node)
                            current_node = index
                            if return_paths:
                                pred_path.append(current_node)
                            break
                        else:
                            continue
                else:   # If we have a right node...
                    # Iterate to find the right child node
                    for index, parent in enumerate(self.PARENT):
                        if parent == current_node and self.LEFT[index] == 0:  # it only happens if the parent of the next node is the current node and its RIGHT array says it is indeed a right child (that node)
                            current_node = index
                            if return_paths:
                                pred_path.append(current_node)
                            break
                        else:
                            continue

        if not return_paths:
            return np.array(predictions)
        else:
            return np.array(predictions), prediction_paths

 
    ###################################################################
    ######################## FEATURE IMPORTANCE #######################
    ################################################################### 

    def compute_feature_importance(self):
        """
            Function that will compute feature importance for each individual feature in the tree
            Returns:
                - norm_feat_importance (np.ndarrray): a normalised feature importance ranking (sums up to 1) based on the criterium used (usually mse) approach for each individual input feature in the dataset
        """
        n_nodes = len(self.FEATURE)
        # Creating an empty array of feature importances for all variables
        feat_importance = np.zeros(len(self.feature_names)) 
        
        for node in range(n_nodes):
            
            # Only applicable to non-leaf nodes
            if self.FEATURE[node] is not None:
                # Retrieve child indices for the node
                child_indices = [j for j, val in enumerate(self.PARENT) if val == node]
                for idx in child_indices:
                    if self.LEFT[idx] == 0:
                        right_child = idx 
                    elif self.LEFT[idx] == 1:
                        left_child = idx
            
                # Gathering the parent and child impurities
                parent_imp = self.IMPURITY[node]
                left_child_imp = self.IMPURITY[left_child] if left_child is not None else 0
                right_child_imp = self.IMPURITY[right_child] if right_child is not None else 0
                # Other important variables - numer of left, right and total samples 
                N_t = self.N_NODE_SAMPLES[node]
                N_l = self.N_NODE_SAMPLES[left_child]
                N_r = self.N_NODE_SAMPLES[right_child]
                p_l = N_l/N_t
                p_r= N_r/N_t
                # Getting the importante "increment"
                imp_increment = N_t*(parent_imp - p_l*left_child_imp - p_r*right_child_imp)
                # Adding it to the adequate variable
                feat_importance[self.FEATURE[node]] += imp_increment

        # Getting the denominator for the equation
        # Retrieve all non-leaf nodes
        non_leaf_idx = [idx for idx, val in enumerate(self.LEAF) if val == 0] 
        denominator = 0
        for idx in non_leaf_idx:
            denominator += self.N_NODE_SAMPLES[idx]*self.IMPURITY[idx]
            
        # Normalising so that importances sum up to 1
        feat_importance = feat_importance/denominator
        norm_feat_importance = feat_importance/np.sum(feat_importance)

        return norm_feat_importance    


    #######################################################################
    ######################## EXTRACT PRED PATH TEXT #######################
    #######################################################################

    @abstractmethod
    def extract_prediction_path(self, sample: np.ndarray):
        """
            Function that will extract the prediction path for each sample in the test set
            Args:
                - sample (np.array): sample to be analysed
            Returns:
                - text (str): textual explanation of the prediction path for the sample

            Code inspiration:"
            "https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html""
        """        
        pass
        


    
    #######################################################################
    ########################### PRUNING THE TREE ##########################
    #######################################################################

    def prune_tree(self, by: str = 'depth', threshold: Union[int, float] = 5):
        """
            Function that will prune the tree based on a particular criterium
            Args:
                - by (str): criterium to be used to prune the tree. Options are 'depth' or 'impurity'
                - threshold (Union[int, float]): threshold to be used when pruning by the chosen criterium. For 'depth', this is an int. For 'impurity', this is a float.
        """

        if by not in ['depth', 'impurity']:
            raise ValueError("Pruning criterium not recognised. Please choose one of: 'depth', 'impurity'.")
       
        # Deepcopying the tree to avoid any issues
        pruned_tree = deepcopy(self)
        # Turning all characteristics of the nodes into arrays for easier handling and indexing
        # NOTE: node_ds is difficult to turn into a numpy array
        class_, class_dist, depth, feature, impurity, leaf, left, n_node_samples, parent, value = np.array(pruned_tree.CLASS), np.array(pruned_tree.CLASS_DIST), np.array(pruned_tree.DEPTH), np.array(pruned_tree.FEATURE), np.array(pruned_tree.IMPURITY), np.array(pruned_tree.LEAF), np.array(pruned_tree.LEFT), np.array(pruned_tree.N_NODE_SAMPLES), np.array(pruned_tree.PARENT), np.array(pruned_tree.VALUE)

        # By depth
        if by == 'depth':
            # Identifying target nodes
            target_idx = np.where(depth <= threshold)[0]
        
        elif by == 'impurity':
            # The first one node is always included
            target_idx = [0]
            for idx in range(1, len(impurity)):
                c1 = impurity[idx] >= threshold                     # Current node impurity above threshold
                c2 = parent[idx] in target_idx                  # Parent node is included (previously complied with all criteria)
                c3 = np.all(impurity[parent == parent[idx]] >= threshold)   # The sibling nodes also comply with the impurity threshold
                if c1 and c2 and c3:
                    target_idx.append(idx)
            target_idx = np.array(target_idx)

        # Updating all tree characteristics
        pruned_tree.CLASS = list(class_[target_idx])
        pruned_tree.CLASS_DIST = list(class_dist[target_idx])
        pruned_tree.DEPTH = list(depth[target_idx])
        pruned_tree.FEATURE = list(feature[target_idx])
        pruned_tree.IMPURITY = list(impurity[target_idx])
        pruned_tree.LEAF = list(leaf[target_idx])
        pruned_tree.LEFT = list(left[target_idx])
        pruned_tree.N_NODE_SAMPLES = list(n_node_samples[target_idx])
        pruned_tree.PARENT = list(parent[target_idx])
        pruned_tree.VALUE = list(value[target_idx])
        # Updating the NODE_DS
        new_node_ds = []
        for idx in target_idx:
            new_node_ds.append(pruned_tree.NODE_DS[idx])
        pruned_tree.NODE_DS = new_node_ds

        # NOTE: This is still not perfect, and only adapted to depth pruning:
        final_depth_nodes = np.where(np.array(pruned_tree.DEPTH) == np.max(pruned_tree.DEPTH))[0]
        for node in final_depth_nodes:
            if pruned_tree.LEAF[node] == 0:
                pruned_tree.LEAF[node] = 1   # Turning into leaf nodes
                pruned_tree.FEATURE[node] = None
                pruned_tree.VALUE[node] = None

        return pruned_tree


