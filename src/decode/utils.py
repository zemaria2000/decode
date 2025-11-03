"""
    Simple function calculations used by the trees
"""

import numpy as np


# MSE - regression
def mse(y: np.ndarray):
    """
        Computing the mse for a particular set of predictions
        Args:
            - y (np.ndarray): Array of output predictions
        Returns:
            - mse_score (float): the mean squared error for that set of predictions
    """
    avg_y = np.mean(y)
    mse_score = (1/len(y))*np.sum((y - avg_y)**2)
    return mse_score

# MAE - regression
def mae(y: np.ndarray):
    """
        Computing the mae for a particular set of predictions
        Args:
            - y (np.ndarray): Array of output predictions
        Returns:
            - mae_score (float): the mean absolute error for that set of predictions
    """
    median_y = np.median(y)
    mae_score = (1/len(y))*np.sum(np.abs(y-median_y))
    return mae_score

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
    poisson = (2/len(y))*np.sum(y*np.log(y/avg_y)-y+avg_y)
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