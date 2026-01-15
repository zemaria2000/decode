import numpy as np
import os
from scipy.stats.qmc import Sobol, LatinHypercube, scale
from ess.nn import FaissFlatL2NN, FaissHNSWFlatNN
from ess.ess import ess, esa
from typing import Union
from itertools import product


######################################################
######### SYNTHETIC DATASET GENERATION TOOLS #########
######################################################

def generate_continuous_synth_dataset(dataset: np.ndarray,
                                      scaler: np.ndarray,
                                      model,
                                      numerical_idx: Union[np.ndarray, list],
                                      categorical_idx: Union[np.ndarray, list],
                                      sampler_name: str = 'auto', # random
                                      n_samples: Union[str, int] = 'auto', # equal to the size of the dataset
                                    #   strategy: str = 'auto', # 
                                      seed: int = 42,
                                      **sampler_kwargs):
    """
        Function that will genreate a synthetic dataset through a trained model and a sampling strategy.
        Args:
            - dataset (np.ndarray): The original dataset used for training the model.
            - scaler (np.ndarray): The scaler used to scale the numerical features of the dataset before training the model.
            - model (np.ndarray): The trained model that will be used to generate the synthetic labels.
            - numerical_idx (Union[np.ndarray, list]): The indices of the numerical features in the dataset.
            - categorical_idx (Union[np.ndarray, list]): The indices of the categorical features in the dataset.
            - sampler (str): The sampling strategy to use. Options are: 'auto', 'random', 'sobol', 'lhs', 'ess'.
            - n_samples (Union[str, int]): The number of samples to generate. If 'auto', it will be equal to the size of the dataset.
            - seed (int): The random seed to use for reproducibility.
            - sampler_kwargs: Additional keyword arguments to pass to the sampler.
        Returns:
            - x_synthetic (np.ndarray): The generated synthetic features.
            - y_synthetic (np.ndarray): The generated synthetic labels.
    """

    # 1. Instantiate a sampler and generate a set of samples

    # Some initialisation vars
    feature_bounds = np.vstack((np.min(dataset, axis = 0), np.max(dataset, axis = 0))).T
    if n_samples == 'auto':
        n_samples = len(dataset)
    n_dims = dataset.shape[1]
    rng = np.random.default_rng(seed)

    # Instantiating samplers + generating samples
    if sampler_name not in ['auto', 'random', 'sobol', 'lhs', 'ess']:
        raise ValueError(f"Sampler '{sampler_name}' not recognised. Please choose one of: 'auto', 'random', 'sobol', 'lhs', 'ess'.")

    # NOTE: for all samplers except the ess, a unit hypercube will be sampled and then scaled to the feature bounds. For ess, the feature bounds are directly used.
    if sampler_name == 'auto' or sampler_name == 'random':
        samples = rng.uniform(low = 0.0, high = 1.0, size = (n_samples, n_dims))
    elif sampler_name == 'sobol':
        samples = Sobol(d = n_dims, scramble = sampler_kwargs.get('scramble', True), optimization = sampler_kwargs.get('optimization', None), rng = rng).random(n = n_samples)
    elif sampler_name == 'lhs':
        samples = LatinHypercube(d = n_dims, scramble = sampler_kwargs.get('scramble', True), optimization = sampler_kwargs.get('optimization', None), rng = rng).random(n = n_samples)
    elif sampler_name == 'ess':
        sampler = FaissFlatL2NN(dimension = n_dims, seed = seed)
        # Choose initial samples randomly from the training set
        init_indices = rng.choice(dataset.shape[0], size = int(0.05*dataset.shape[0]), replace = False)
        init_samples = dataset[init_indices]
        samples = esa(
                    samples = init_samples,
                    bounds = feature_bounds,
                    n = n_samples,
                    nn_instance = sampler)
    
    # 2. Scaling samples to feature bounds (when needed) + taking into account different feature types
    
    # Handling continuous features
    if sampler_name != 'ess':
        cont_samples = scale(samples[:, numerical_idx], l_bounds = feature_bounds[numerical_idx, 0], u_bounds = feature_bounds[numerical_idx, 1] + 1e-8)
    else:
        cont_samples = samples[:, numerical_idx]    
    
    # Handling categorical features
    cat_samples = np.zeros_like(samples[:, categorical_idx])
    for i, ft in enumerate(categorical_idx):
        n_categories = len(np.unique(dataset[:, ft]))
        if sampler_name != 'ess':
            bins = np.linspace(0, 1, n_categories + 1)   # Because we are in the unit hypercube
        else:
            bins = np.linspace(feature_bounds[ft, 0], feature_bounds[ft, 1], n_categories + 1)  # Because we are technically under the "original feature bounds" (lowest cat - highest cat)
        cat_values = np.digitize(samples[:, ft], bins) - 1
        cat_samples[:, i] = cat_values        

    # Combining both types of features
    x_synthetic = np.zeros_like(samples)
    x_synthetic[:, numerical_idx] = cont_samples
    x_synthetic[:, categorical_idx] = cat_samples    

    # 3. Generating the synthetic labels 
    x_synthetic_scaled = x_synthetic.copy()
    x_synthetic_scaled[:, numerical_idx] = scaler.transform(x_synthetic[:, numerical_idx])
    y_synthetic = model.predict(x_synthetic_scaled)

    return x_synthetic, y_synthetic


def generate_categorical_synthetic_dataset(dataset: np.ndarray,
                                           model):
    """
        Generating a synthetic datasets using only purely categorical features. In the end, we have a dataset with 1 single sample per feature combination.
        Warning: This function can generate very large datasets if the number of features and/or their cardinality is high...
        Args:
            - dataset (np.ndarray): The original dataset used for training the model.
            - model (np.ndarray): The trained model that will be used to generate the synthetic labels.
    """

    # Retrieving all possible feature combinations
    all_combos = []
    for i in range(dataset.shape[1]):
        feat_vals = np.unique(dataset[:, i]).tolist()
        all_combos.append(feat_vals)
    
    # Generate all possible combinations
    x_synthetic = np.array(list(product(*all_combos)))
    # Generate synthetic labels
    y_synthetic = model.predict(x_synthetic)

    # Through a warning if the dataset is too large
    if x_synthetic.shape[0] > 1e6:
        print(f"Warning: The generated synthetic dataset has {x_synthetic.shape[0]} samples. This may lead to high memory usage.")

    return x_synthetic, y_synthetic
