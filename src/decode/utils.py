"""
    Simple function calculations used by the trees
"""

import numpy as np
import pickle
from keras import Model as KerasModel

########################################################
################ METRICS FOR REGRESSION ################
########################################################

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


#########################################################
######### HARD LABEL METRICS FOR CLASSIFICATION #########
#########################################################

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

#########################################################
######### SOFT LABEL METRICS FOR CLASSIFICATION #########
#########################################################

def safe_normalize_probs(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
        Ensure probs are valid distributions: clip to [eps, 1], renormalize rows.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2:
        raise ValueError(f"preds must be 2D (n_samples, n_classes), got shape={P.shape}")

    P = np.clip(P, eps, 1.0)
    row_sums = P.sum(axis=1, keepdims=True)
    # avoid division by 0
    row_sums = np.maximum(row_sums, eps)
    return P / row_sums

# KL divergence
def kl(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
        Row-wise KL(P || Q). P and Q are (n, K).
        Returns shape (n,).
    """
    P = safe_normalize_probs(P, eps=eps)
    Q = safe_normalize_probs(Q, eps=eps)
    return np.sum(P * (np.log(P) - np.log(Q)), axis = 1)

def kl_divergence(P, Q, eps=1e-12):
    # KL(P||Q) per-sample
    P = safe_normalize_probs(P, eps)
    Q = safe_normalize_probs(Q, eps)
    return np.sum(P * (np.log(P) - np.log(Q)), axis=1)

def entropy_vec(pi, eps=1e-12):
    # pi shape: (..., K) or (K,)
    pi = np.clip(pi, eps, 1.0)
    pi = pi / np.sum(pi, axis=-1, keepdims=True)
    return -np.sum(pi * np.log(pi), axis=-1)

# Jensen-Shannon divergence
def js_divergence(P, Q, eps=1e-12):
    # JS(P,Q) per-sample (symmetric, bounded)
    P = safe_normalize_probs(P, eps)
    Q = safe_normalize_probs(Q, eps)
    M = 0.5 * (P + Q)
    return 0.5 * kl_divergence(P, M, eps) + 0.5 * kl_divergence(Q, M, eps)

# Hellinger distance
def hellinger2_to_mean(P: np.ndarray, meanP: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
        Row-wise squared Hellinger distance H^2(P_i, meanP).
        H^2 = (1/2) * sum_k (sqrt(p_k) - sqrt(q_k))^2
        Returns shape (n,).
    """
    P = safe_normalize_probs(P, eps=eps)
    meanP = np.asarray(meanP, dtype=float)
    meanP = np.clip(meanP, eps, 1.0)
    meanP = meanP / np.maximum(meanP.sum(), eps)

    sP = np.sqrt(P)
    sM = np.sqrt(meanP[None, :])
    return 0.5 * np.sum((sP - sM) ** 2, axis=1)

# Total Variation distance
def tv_to_mean(P: np.ndarray, meanP: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
        Row-wise Total Variation distance TV(P_i, meanP) = 0.5*sum|p - q|
        Returns shape (n,).
    """
    P = safe_normalize_probs(P, eps=eps)
    meanP = np.asarray(meanP, dtype=float)
    meanP = np.clip(meanP, eps, 1.0)
    meanP = meanP / np.maximum(meanP.sum(), eps)
    return 0.5 * np.sum(np.abs(P - meanP[None, :]), axis=1)

# Brier dispersion
def brier_dispersion_to_mean(P: np.ndarray, meanP: np.ndarray, class_weights: np.ndarray | None = None,
                              eps: float = 1e-12) -> np.ndarray:
    """
        Row-wise squared L2 dispersion around mean, optionally component-weighted:
        d_i = sum_k alpha_k (p_ik - mean_k)^2
        Returns shape (n,).
    """
    P = safe_normalize_probs(P, eps=eps)
    meanP = np.asarray(meanP, dtype=float)
    meanP = np.clip(meanP, eps, 1.0)
    meanP = meanP / np.maximum(meanP.sum(), eps)

    D = (P - meanP[None, :]) ** 2
    if class_weights is not None:
        alpha = np.asarray(class_weights, dtype=float)
        if alpha.shape != (P.shape[1],):
            raise ValueError(f"class_weights must have shape (n_classes,), got {alpha.shape}")
        # Normalize alpha to keep scale stable (optional but recommended)
        alpha = alpha * (P.shape[1] / np.maximum(alpha.sum(), eps))
        D = D * alpha[None, :]
    return np.sum(D, axis=1)


###########################################################

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


