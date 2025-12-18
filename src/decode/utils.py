"""
    Simple function calculations used by the trees
"""

import numpy as np
import pickle
from keras import Model as KerasModel


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

# MSLE - mean squared log error
# NOTE: Only works for targets that are larger than -1!!
def msle(y: np.ndarray):
    """
        Computing the mean squared log error for a particular set of predictions
        Args:
            - y (np.ndarray): Array of output predictions
        Returns:
            - msle_score (float): the mean squared logarithmic error for that set of predictions
    """
    mean_y = np.mean(y, axis=0)
    sq_dist = np.sum((np.log(1+y) - np.log(1+mean_y)) ** 2, axis = 1)
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

# Huber loss
def huber(y: np.ndarray):
    """
        Computing the huber loss for a particular set of predictions
        Args:
            - y (np.ndarray): Array of output predictions
        Returns:
            - huber_score (float): the mean absolute error for that set of predictions
    """
    # delta value
    delta = np.std(y, axis = 0)
    # huber loss
    mean_y = np.mean(y, axis = 0)
    diff_y = y - mean_y
    abs_diff = np.abs(diff_y)  # Useful for determining the huber loss criterium
    mask = abs_diff <= delta
    loss_per_pred_dim = np.where(mask, 0.5*(diff_y**2), delta*(abs_diff - 0.5*delta))

    return np.mean(loss_per_pred_dim)


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
def gini(probs: np.ndarray):
    """
        Computing the gini impurity score for a given class distribution
        Args:
            - probs (np.ndarray): Array of (weighted) class probabilities
        Returns:
            - gini (float): gini impurity score
    """
    gini = np.sum(probs * (1 - probs))
    return gini

# Log loss - classification
def log_loss(probs: np.ndarray):
    """
        Computing the log loss score for a given class distribution
        Args:
            - probs (np.ndarray): Array of (weighted) class probabilities
        Returns:
            - log_loss (float): log loss score
    """
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

# Auxiliary function to conduct model predictions
def model_preds(model, X: np.ndarray, model_type: str = 'reg'):
    """
        Conducting predictions with a given model in a unified way (because, for instance, keras methods do not output the same as sklearn models)
        Args:
            - model: the model to use for predictions
            - X (np.ndarray): input data for predictions
            - model_type (str): type of model ('reg' for regression, 'clf' for classification)
    """
    # NOTE: currently, the function is only adapted to sklearn and keras models
    if model_type == 'reg':
        return model.predict(X)
    else:
        if isinstance(model, KerasModel):
            # 2 cases: (1) single output (sigmoid), or (2) one-hot encoded output (softmax)
            preds = model.predict(X)
            if preds.shape[1] == 1:
                return (preds > 0.5).astype(int).flatten()
            else:
                return np.argmax(preds, axis = 1)
        else:
            return model.predict(X)

