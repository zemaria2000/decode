"""
    Simple function calculations used by the trees
"""

import numpy as np
import pickle


# MSE - regression
def mse(y: np.ndarray):
    """
        Computing the mse for a particular set of predictions
        Args:
            - y (np.ndarray): Array of output predictions
        Returns:
            - mse_score (float): the mean squared error for that set of predictions
    """
    mean_y = np.mean(y, axis=0)
    sq_dist = np.sum((y - mean_y) ** 2, axis = 1)
    return np.mean(sq_dist)

# MAE - regression
def mae(y: np.ndarray):
    """
        Computing the mae for a particular set of predictions
        Args:
            - y (np.ndarray): Array of output predictions
        Returns:
            - mae_score (float): the mean absolute error for that set of predictions
    """
    median_y = np.median(y, axis = 0)
    abs_dev = np.abs(y - median_y)
    l1_dist_per_sample = np.sum(abs_dev, axis = 1)
    return np.mean(l1_dist_per_sample)

# Possion deviance - regression
def poisson_deviance(y: np.ndarray):
    """
        Computing the mean poisson deviance for a particular set of predictions. Note that it can only be used with y > 0
        Args:
            - y (np.ndarray): Array of output predictions
        Returns:
            - poission (float): the mean poisson deviance for that set of predictions
    """        
    avg_y = np.mean(y)
    poisson = (2/len(y))*np.sum(y*np.log(y/avg_y) - y + avg_y)
    return poisson

# Gini score - classification
def gini(dist: np.ndarray):
    """
        Computing the gini impurity score for a given class distribution
        Args:
            - dist (np.ndarray): Array of class distributions
        Returns:
            - gini (float): gini impurity score
    """
    gini = 1 - np.sum((dist/np.sum(dist))**2)
    return gini

# Log loss - classification
def log_loss(dist: np.ndarray):
    """
        Computing the log loss score for a given class distribution
        Args:
            - dist (np.ndarray): Array of class distributions
        Returns:
            - log_loss (float): log loss score
    """
    probs = dist/max(1, np.sum(dist))
    probs = np.clip(probs, 1e-12, 1.0) # To prevent divisions by zero
    log_loss = - np.sum(probs*np.log(probs))/np.log(2)
    return log_loss


# Auxiliary function to store numpy objects in a yaml
def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(v) for v in obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj
    
# Check if a function is pickable (useful for parallelization)
def is_picklable(obj):
    try:
        pickle.dumps(obj)
    except TypeError:
        return False
    except AttributeError:
        return True
    return True