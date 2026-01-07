"""
    Base tree class 
"""

import numpy as np
from pathlib import Path
from .utils import *
from abc import ABC, abstractmethod
from typing import Union
from .logger import logger
from munch import Munch
import os, yaml
import pandas as pd
from joblib import Parallel, delayed
import logging 
from scipy.stats.qmc import Sobol, LatinHypercube

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
                 use_logging: bool = True):
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
            self._generate_synthetic_dataset(sampler = sampler,
                                             n_samples = n_samples,
                                             strategy = strategy,
                                             **sampler_kwargs)

        # Resetting the tree
        self._reset_tree()

        # Defining storage settings
        self._storage_settings()

        # Internal flag that states if the tree is for multi-output or single_output regression
        self.flag_multi_output = True if self.decode_targets.shape[1] > 1 else False

        if use_logging:
            logger.setLevel(logging.DEBUG)
        else:
            logger.info("User disbaled logging for the tree")
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

    def _generate_synthetic_dataset(self, sampler: str = None, 
                                    n_samples: Union[str, int] = 'auto',
                                    strategy: str = None,
                                    **sampler_kwargs):
        """
            Auxiliary function to generate a synthetic dataset, based on a particular sampler, that covers the entire feature space adequately
            Args:
                - smapler (str): sampler to be used. Valid samplers are 'sobol', 'lhs', or 'random'
                - n_samples (Union[str, int]): number of samples to be generated. If 'auto', it will generate 10x the original dataset size, and if int, it will generate n_samples samples
                - strategy (str): strategy to be used to build the overall tree (provided as a base argument for the tree building process). Valid strategies are 'synth' and 'both'
            Returns:
                - x_synthetic (np.ndarray): synthetic dataset generated
                - y_synthetic (np.ndarray): synthetic labels generated
        """
        # NOTE: only works, for now, for datasets which contain either all numerical variables or a mix of numerical and categorical variables
        
        # Auxiliary function to find the min and max values for each continuous feature
        def _auxiliary_scaling(x_scaled, samples):
            """
                Auxiliary function that will scale the sampler samples (from [0, 1]) to the original SCALED feature space. This way, when conducting the inverse transform, we will be aligned with the original feature distributions
                Args:
                    - x_scaled (np.ndarray): scaled original dataset
                    - samples (np.ndarray): samples generated between [0, 1]
            """
            min_vals = x_scaled[:, self.numerical_idx].min(axis = 0)
            max_vals = x_scaled[:, self.numerical_idx].max(axis = 0)
            cont_scaled_samples = min_vals + samples[:, self.numerical_idx] * (max_vals - min_vals)
            return cont_scaled_samples

        if strategy is None:
            logging.warning("No strategy for synthetic dataset generation has been provided. Using only the original dataset'")
            self.decode_dataset = self.dataset
            self.decode_targets = self.targets
            self.decode_normalised_dataset = self._scale_inputs(self.decode_dataset)
            return 
        
        else:
            ## 1. Instantiate sampler and generate a set of samples between [0, 1]
            if n_samples == 'auto':
                n_samples = self.dataset.shape[0] * 10
            n_dims = self.dataset.shape[1]
    
            if sampler not in ['sobol', 'lhs', 'random']:
                logging.warning(f"Sampler {sampler} not recognised. Defaulting to 'random uniform'")
                sampler = 'random'
            if sampler == 'sobol':
                samples = Sobol(d = n_dims, scramble = sampler_kwargs.get('scramble', True), optimization = sampler_kwargs.get('optimization', 'random-cd')).random(n = n_samples)
            elif sampler == 'lhs':
                samples = LatinHypercube(d = n_dims, scramble = sampler_kwargs.get('scramble', True), optimization = sampler_kwargs.get('optimization', 'random-cd')).random(n = n_samples)
            elif sampler == 'random':
                samples = np.random.uniform(low = 0.0, high = 1.0, size = (n_samples, n_dims))
            # NOTE: for the optimization process of the samples, although lloyd seems to better evenly space the points, it takes much more time to do so

            ## 2. Normalising the original dataset 

            # 2.1. Handling continuous features 
            x_scaled = self._scale_inputs(self.dataset)
            cont_scaled_samples = _auxiliary_scaling(x_scaled, samples)

            # 2.2 Handling categorical features
            cat_samples = np.zeros_like(samples[:, self.categorical_idx])
            for i, ft in enumerate(self.categorical_idx):
                n_categories = len(np.unique(self.dataset[:, ft]))
                bins = np.linspace(0, 1, n_categories + 1)
                cat_values = np.digitize(samples[:, ft], bins) - 1
                cat_samples[:, i] = cat_values       

            # 2.3. Joining them together
            synthetic_dataset = np.zeros_like(samples)
            synthetic_dataset[:, self.numerical_idx] = cont_scaled_samples
            synthetic_dataset[:, self.categorical_idx] = cat_samples

            # 2.4. Inverse transforming the continuous features so that we have the dataset under its original distribution
            x_synthetic = self._inverse_scale_inputs(synthetic_dataset)

            ## 3. Generating the synthetic labels
            y_synthetic = self.model.predict(self._scale_inputs(x_synthetic))
            # Reshaping targets so that they have always 2 dimensions
            if len(y_synthetic.shape) == 1:
                y_synthetic = y_synthetic.reshape(-1, 1)

            ## 4. Handling/overlapping the overall dataset based on the strategy
            if strategy == 'synth':
                self.decode_dataset = x_synthetic
                self.decode_targets = y_synthetic
            elif strategy == 'both':
                self.decode_dataset = np.concatenate((self.dataset, x_synthetic), axis = 0)
                self.decode_targets = np.concatenate((self.targets, y_synthetic), axis = 0)
            self.decode_normalised_dataset = self._scale_inputs(self.decode_dataset)
            return
        

    ###############################################################
    ################### VARIABLE DISCRETIZATION ###################
    ###############################################################

    @abstractmethod
    def _refine_step_bins(self, feature_data, preds, bin_idx, edges):
        """ Subclass decides which bins to refine """
        pass

    def _create_auto_steps(self, dataset_idx: np.ndarray, n_steps: Union[str, int] = None, adaptive: bool = False, refine_factor: int = 2, n_refine: int = 2):
        """
            Generates the steps automatically, leveraging the numpy.histogram function to create optimal bins for each feature. Additionally, it contemplates the possibility to create adaptive quantile-based steps. Depending on the method that is deployed in the Regressor/Classifier classes, the refinement will be conducted accordingly
            Args:
                - dataset_idx (np.ndarray): indices, relative to the original dataset, that should be taken into account to conduct the variable discretization
                - n_steps (str): Whether to automatically create the steps for each feature or to use a fixed number of steps for the quantiles
                - adaptive (bool): Whether to create adaptive quantile-based steps
                - refine_factor (int): Number of new bins to create from a candidate split
                - n_refine (int): number of possible refinement iterations
            Returns:
                - step_info (dict): Dictionary with the feature names as keys and their min, max and steps as values
        """
        step_info = {}
        for i, feature in enumerate(self.feature_names):
            feature_data = self.decode_dataset[dataset_idx, i]
            # In case we are dealing with a numerical feature
            if self.feature_types[i] == 'N':            
                step_info[feature] = {
                    'min': np.min(feature_data),
                    'max': np.max(feature_data),
                    'median': np.median(feature_data)
                }
                if not adaptive:
                    # Creating an array with the steps
                    if isinstance(n_steps, str) or n_steps is None:
                        _, bin_edges = np.histogram(feature_data, bins = 'auto')
                        step_info[feature]['steps'] = bin_edges[1:-1]   # First value is the minimum, last value is the maximum - they generate degenerate splits
                    elif isinstance(n_steps, int):
                        quantiles = np.linspace(0, 1, n_steps + 1)
                        steps = np.unique(np.quantile(feature_data, quantiles))
                        # Creating an array with n-steps between min and max
                        step_info[feature]['steps'] = steps[1:-1]       # First value is the minimum, last value is the maximum - they generate degenerate splits
                
                else:   # We will create adaptive quantile-based steps
                    
                    # Creating an array with the initial steps
                    if isinstance(n_steps, str) or n_steps is None:
                        _, edges = np.histogram(feature_data, bins = 'auto')
                        edges = edges[1:-1]
                    elif isinstance(n_steps, int):
                        quantiles = np.linspace(0, 1, n_steps + 1)
                        edges = np.unique(np.quantile(feature_data, quantiles))
                        edges = edges[1:-1]

                    # Make predictions with the model
                    inp_dataset = self.decode_normalised_dataset[dataset_idx, :]
                    preds = self.model.predict(inp_dataset)       
                    preds = self._postprocess_prediction(preds)             
                    
                    # Iterate to find possible refinment of bins
                    for _ in range(n_refine):
                        bin_idx = np.digitize(feature_data, edges, right=True)
                        # Getting the bins to refine based on the tree class
                        refine_bins = self._refine_step_bins(feature_data, preds, bin_idx, edges)
                        if len(refine_bins) == 0:
                            break   # No more bins to refine
                        new_edges = list(edges)
                        for b in refine_bins:
                            # re-split the bin locally
                            mask = (feature_data > edges[b]) & (feature_data <= edges[b+1])
                            if np.sum(mask) < 2:  # skip tiny bins
                                continue
                            sub_qs = np.linspace(0, 1, refine_factor + 1)
                            sub_edges = np.quantile(feature_data[mask], sub_qs)[1:-1]
                            new_edges.extend(sub_edges)
                        
                        edges = np.unique(np.sort(new_edges))
                    
                    step_info[feature]['steps'] = edges

            # In case we are dealing with a categorical feature
            elif self.feature_types[i] == 'C':
                unique_vals = np.unique(feature_data).astype(int)
                step_info[feature] = {}
                step_info[feature]['steps'] = (unique_vals[:-1] + unique_vals[1:]) / 2
        
        return step_info


    ##################################################################
    ################### SPLIT EVALUATION + SCORING ###################
    ##################################################################

    @abstractmethod
    def _impurity_calculation(self, preds: np.ndarray, criterium: str = 'default'):
        """ Impurity calculation handled by the subclass """
        pass

    def _candidate_score(self, dataset_idx: np.ndarray, preds: np.ndarray, feature_idx: int, feature_values: np.ndarray, criterium: str = 'default'):
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
        original_impurity = self._impurity_calculation(preds, criterium)
        left_impurities = np.array([self._impurity_calculation(preds[mask], criterium) for mask in masks[valid_mask]])
        right_impurities = np.array([self._impurity_calculation(preds[~mask], criterium) for mask in masks[valid_mask]])
        
        # Candidate scores
        candidate_scores = original_impurity - ((n_left_samples[valid_mask]/n_samples)*left_impurities + (n_right_samples[valid_mask]/n_samples)*right_impurities)

        # Global impurity decreases + checking their compliance with the min_impurity_decrease criterium
        global_impurity_decreases = (n_samples/len(self.decode_dataset))*candidate_scores
        candidate_scores[global_impurity_decreases < self.min_impurity_decrease] = np.nan

        # Returning the scores for all possible feature values
        scores = np.full(len(feature_values), np.nan)
        scores[valid_mask] = candidate_scores 

        return scores 

    def execute_trial(self, dataset_idx: np.ndarray, preds: np.ndarray, step_info: dict, criterium: str = 'default'):
        """
            Executes a new trial for each feature and its steps, based on the providade dataset
            Args:
                - dataset_idx (np.ndarray): index, relative to the original dataset (self.decode_dataset), of our current dataset to be assessed
                - preds (np.ndarray): predictions for the current dataset (self.model.predict(self.decode_normalised_dataset[dataset_idx]))
                - step_info (dict): dictionary with the feature steps information
                - criterium (str): str that defines the choosing criterium - "default" by default 
            Returns:
                - score_list (list): list of tuples that contemplates (feature_idx, feature_value, candidate_score) for each individual step in each feature considered
        """         
        score_list = None 

        for feature_idx, feature in enumerate(self.feature_names):
            feature_values = np.array(step_info[feature]['steps'])
            candidate_scores = self._candidate_score(dataset_idx = dataset_idx, preds = preds, feature_idx = feature_idx, feature_values = feature_values, criterium = criterium)
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
    def _expand_node(self, prev_node_idx: int, current_depth: int, dataset: np.ndarray, imp_threshold: float, n_steps: Union[str, int], adaptive: bool, criterium: str, perturb_dataset: bool, left: bool):
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
        elif isinstance(class_weights, dict):
            self.class_weights = class_weights
        else:
            self.class_weights = None


    def build_tree(self, 
                   max_depth: int = 5,
                   imp_threshold: float = 0.0,
                   n_steps: Union[str, int] = 'auto',
                   adaptive: bool = False,
                   min_samples_split: int = 2,
                   min_impurity_decrease: float = 0.0,
                   criterium: str = 'default',
                   class_weights: Union[None, str, dict] = None):
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
        logger.debug(f"Parameters: max_depth={max_depth}, imp_threshold={imp_threshold}, n_steps={n_steps}, adaptive = {adaptive}, min_samples_split={min_samples_split}, min_impurity_decrease={min_impurity_decrease}, criterium={criterium}")

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
        original_preds = self.model.predict(self.decode_normalised_dataset)
        original_preds = self._postprocess_prediction(original_preds)            
        step_info = self._create_auto_steps(original_idx, n_steps = n_steps, adaptive = adaptive)
        final_res = self.execute_trial(original_idx, original_preds, step_info, criterium = criterium)
        if final_res is None:
            raise RuntimeError("No valid split found for root node. Try reducing constraints.")    
        # Storing the root info
        self._initialise_root_node(final_res, original_preds)
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
                left_preds = self.model.predict(self.decode_normalised_dataset[left_ds_idx])
                right_preds = self.model.predict(self.decode_normalised_dataset[right_ds_idx])
                left_preds = self._postprocess_prediction(left_preds)
                right_preds = self._postprocess_prediction(right_preds)

                ## 2.2. PROCESS LEFT NODE
                if len(left_ds_idx) > 0:
                    self._expand_node(prev_node_idx = prev_node_index, current_depth = i, dataset_idx = left_ds_idx, preds = left_preds, n_steps = n_steps, adaptive = adaptive, criterium = criterium, left = True)

                ## 2.3. PROCESS RIGHT NODE
                if len(right_ds_idx) > 0:
                    self._expand_node(prev_node_idx = prev_node_index, current_depth = i, dataset_idx = right_ds_idx, preds = right_preds, n_steps = n_steps, adaptive = adaptive, criterium = criterium, left = False)

        logger.info(f"Tree building process completed: total nodes = {len(self.N_NODE_SAMPLES)}, leaf nodes = {np.sum(self.LEAF)}")

    ###################################################################
    ###################### CONDUCTING PREDICTIONS #####################
    ###################################################################  

    @abstractmethod
    def _return_leaf_output(self, node_idx: int):
        """ Handled by each subclass to return the adequate predictions """

    def make_predictions(self, samples: np.ndarray):
        """
            Make predictions for one or multiple samples based on the decision tree rules defined by the parameters.
            Args:
                - samples (np.ndarray): One or multiple samples to predict (must be in the shape of (n_samples, n_features))
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
                    prediction_paths.append(pred_path)
                    break

                # Otherwise...
                condition = (sample[self.FEATURE[current_node]] <= self.VALUE[current_node])
                if condition:
                    # Iterate to find the left child node
                    for index, parent in enumerate(self.PARENT):
                        if parent == current_node and self.LEFT[index] == 1:  # it only happens if the parent of the next node is the current node and its LEFT array says it is indeed a left child (that node)
                            current_node = index
                            pred_path.append(current_node)
                            break
                        else:
                            continue
                else:   # If we have a right node...
                    # Iterate to find the right child node
                    for index, parent in enumerate(self.PARENT):
                        if parent == current_node and self.LEFT[index] == 0:  # it only happens if the parent of the next node is the current node and its RIGHT array says it is indeed a right child (that node)
                            current_node = index
                            pred_path.append(current_node)
                            break
                        else:
                            continue

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
        