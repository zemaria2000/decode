"""
    Sub-class for classification problems
"""

# xai_tree/regression.py
import logging
import numpy as np
from .base_tree import BaseTree
from .utils import *
from typing import Union
from .plotting import plot_classification_tree
import matplotlib.pyplot as plt
from .logger import logger
import os

class ClassificationTree(BaseTree):

    def __init__(self, model, 
                 scalerX = None, 
                 scalerY = None, 
                 dataset: np.ndarray = None, 
                 targets: np.ndarray = None, 
                 target_probs: np.ndarray = None,
                 feature_names: list = None, 
                 target_names: list = None, 
                 feature_types: list = None, 
                 seed: int = 42, 
                 root_path: str = None, 
                 use_logging: bool = True,
                 log_level = logging.INFO):
        
        self.task = 'classification'
        
        super().__init__(
            model = model,
            scalerX = scalerX,
            scalerY = scalerY,
            dataset = dataset,
            targets = targets,
            target_probs = target_probs,
            feature_names = feature_names,
            target_names = target_names,
            feature_types = feature_types,
            seed = seed,
            root_path = root_path,
            use_logging = use_logging,
            log_level = log_level
        )
        

    # Auxiliar method to postprocess classification outputs
    def _postprocess_classification(self, y_raw):
        # Sigmoid output
        if y_raw.ndim == 2 and y_raw.shape[1] == 1:
            p = y_raw.squeeze()
            return (p >= 0.5).astype(int)
        # Softmax/one-hot output
        elif y_raw.ndim == 2:
            proba = y_raw
            return np.argmax(proba, axis=1)
        # Hard labels only
        else:
            return y_raw.astype(int)

    def _refine_step_bins(self,
                        preds: np.ndarray,
                        bin_idx: np.ndarray,
                        edges: np.ndarray,
                        criterium: str,
                        min_bin: int = 10,
                        score_mult: float = 1.0,
                        support: str = "log",      # "none" | "log" | "sqrt"
                        alpha: float = 0.5,
                        eps: float = 1e-12,
                        normalized: bool = True
    ):
        n_bins = len(edges) + 1
        n_classes = len(self.target_names)
        counts = np.bincount(bin_idx, minlength=n_bins)
        ok = counts >= min_bin
        if not np.any(ok):
            return []

        ok_counts = counts[ok]
        is_tuple = isinstance(criterium, tuple)
        hard_crit = criterium[0] if is_tuple else (criterium if criterium in ('gini', 'log_loss') else None)
        soft_crit = criterium[1] if is_tuple else (None if hard_crit is not None else criterium)

        # 1. Vectorised hard-label bin impurities (when needed)
        if hard_crit is not None:
            hard_preds = preds.ravel().astype(np.int64) if preds.ndim == 1 else np.argmax(preds, axis=1)
            hist = np.bincount(
                bin_idx * n_classes + hard_preds,
                minlength=n_bins * n_classes
            ).reshape(n_bins, n_classes)
            probs = hist[ok] / ok_counts[:, None]
            if hard_crit == 'gini':
                hard_imp = np.sum(probs * (1.0 - probs), axis=1)
            elif hard_crit == 'log_loss':
                p_clip = np.clip(probs, eps, 1.0)
                hard_imp = -np.sum(p_clip * np.log(p_clip), axis=1) / np.log(2.0)

        # 2. Vectorised soft-label bin impurities (when needed)
        if soft_crit is not None:
            P = preds if normalized else safe_normalize_probs(preds, eps = eps)
            if soft_crit in ('soft_gini', 'soft_log_loss', 'mutual_info', 'kl', 'brier'):
                sum_P = np.zeros((n_bins, n_classes), dtype=float)
                np.add.at(sum_P, bin_idx, P)
                meanP = sum_P[ok] / ok_counts[:, None]

                if soft_crit == 'soft_gini':
                    soft_imp = np.sum(meanP * (1.0 - meanP), axis=1)
                elif soft_crit == 'soft_log_loss':
                    m_clip = np.clip(meanP, eps, 1.0)
                    soft_imp = -np.sum(m_clip * np.log(m_clip), axis=1) / np.log(2.0)
                elif soft_crit in ('mutual_info', 'kl'):
                    # (1/n_b) * sum_{i in b} KL(P_i || meanP_b) == H(meanP_b) - mean_{i in b} H(P_i)
                    h_sample = entropy_vec(P, eps=eps)
                    mean_h = np.bincount(bin_idx, weights=h_sample, minlength=n_bins)[ok] / ok_counts
                    soft_imp = entropy_vec(meanP, eps=eps) - mean_h
                elif soft_crit == 'brier':
                    # (1/n_b) * sum_{i in b} ||P_i - meanP_b||^2 == mean_{i in b} ||P_i||^2 - ||meanP_b||^2
                    q_sample = np.sum(P ** 2, axis=1)
                    mean_q = np.bincount(bin_idx, weights=q_sample, minlength=n_bins)[ok] / ok_counts
                    soft_imp = mean_q - np.sum(meanP ** 2, axis=1)

            elif soft_crit in ('soft_label_distance_perfect', 'brier_perfect', 'kl_perfect', 'js_perfect'):
                labels = np.argmax(preds, axis=1)
                one_hot = np.zeros((labels.shape[0], n_classes), dtype=float)
                one_hot[np.arange(labels.shape[0]), labels] = 1.0
                if soft_crit == 'soft_label_distance_perfect':
                    d_sample = np.linalg.norm(P - one_hot, axis=1)
                elif soft_crit == 'brier_perfect':
                    d_sample = np.sum((P - one_hot) ** 2, axis=1)
                elif soft_crit == 'kl_perfect':
                    d_sample = kl(P, one_hot, eps=eps)
                elif soft_crit == 'js_perfect':
                    d_sample = js_divergence(P, one_hot, eps=eps)
                soft_imp = np.bincount(bin_idx, weights=d_sample, minlength=n_bins)[ok] / ok_counts

            else:
                # Non-linear mean criteria (e.g. 'js'): group once by sorting bin_idx instead of O(B*N) masks
                order = np.argsort(bin_idx, kind="stable")
                P_sorted = P[order]
                ends = np.cumsum(counts)
                starts = ends - counts
                ok_bins = np.where(ok)[0]
                soft_imp = np.empty(ok_bins.size, dtype=float)
                for idx, b in enumerate(ok_bins):
                    pb = P_sorted[starts[b]:ends[b]]
                    mean_pb = pb.mean(axis=0)[None, :]
                    soft_imp[idx] = float(np.mean(js_divergence(pb, mean_pb, eps)))

        # 3. Combine scores (single criterion or hybrid tuple)
        if not is_tuple:
            scores_ok = hard_imp if hard_crit is not None else soft_imp
        else:
            norm_hard = (1.0 - 1.0 / n_classes) if hard_crit == 'gini' else np.log(n_classes)
            if soft_crit == 'soft_gini':
                norm_soft = 1.0 - 1.0 / n_classes
            elif soft_crit in ('soft_log_loss', 'mutual_info'):
                norm_soft = np.log(n_classes)
            elif soft_crit == 'js':
                norm_soft = np.log(2.0)
            elif soft_crit == 'brier':
                norm_soft = 2.0 * (1.0 - 1.0 / n_classes)
            scores_ok = alpha * (hard_imp / norm_hard) + (1.0 - alpha) * (soft_imp / norm_soft)

        # 4. Apply support factor across all valid bins at once
        if support == "log":
            scores_ok = scores_ok * np.log1p(ok_counts)
        elif support == "sqrt":
            scores_ok = scores_ok * np.sqrt(ok_counts)
        elif support != "none":
            raise ValueError(f"Unknown support='{support}'")

        scores = np.full(n_bins, np.nan, dtype=float)
        scores[ok] = scores_ok

        valid = np.where(~np.isnan(scores))[0]
        if valid.size == 0:
            return []
        med = np.nanmedian(scores[valid])
        refine_bins = valid[scores[valid] > (med * score_mult)]
        return refine_bins.tolist()  

    def _hard_probs(self, preds: np.ndarray, n_classes: int) -> np.ndarray:
        """Compute (optionally class-weighted) hard-label distribution only when needed."""
        hard_preds = preds.ravel().astype(np.int64) if preds.ndim == 1 else np.argmax(preds, axis=1)
        class_counts = np.bincount(hard_preds, minlength=n_classes)
        if getattr(self, "class_weights", None) is not None:
            weights = (
                np.asarray(list(self.class_weights.values()), dtype=float)
                if isinstance(self.class_weights, dict)
                else np.asarray(self.class_weights, dtype=float)
            )
            wc = weights * class_counts
            return wc / wc.sum()
        return class_counts / class_counts.sum()

    def _soft_impurity(self, P: np.ndarray, soft_crit: str, eps: float = 1e-12) -> float:
        """Evaluate a soft-label criterion on pre-normalized probability matrix P."""
        if soft_crit in ('soft_gini', 'soft_log_loss', 'mutual_info', 'kl', 'brier', 'js'):
            meanP = P.mean(axis=0)
            if soft_crit == 'soft_gini':
                return float(gini(meanP))
            elif soft_crit == 'soft_log_loss':
                return float(log_loss(meanP))
            elif soft_crit in ('mutual_info', 'kl'):
                return float(entropy_vec(meanP, eps=eps, normalized=True) - np.mean(entropy_vec(P, eps=eps, normalized=True)))
            elif soft_crit == 'brier':
                return float(np.mean(np.sum(P ** 2, axis=1)) - np.sum(meanP ** 2))
            elif soft_crit == 'js':
                return float(np.mean(js_divergence(P, meanP[None, :], eps=eps, normalized=True)))

        # Criteria based on distance to sample-wise one-hot ("perfect") distribution
        p_max = np.max(P, axis=1)
        if soft_crit == 'brier_perfect':
            return float(np.mean(np.sum(P ** 2, axis=1) - 2.0 * p_max + 1.0))
        elif soft_crit == 'soft_label_distance_perfect':
            sq_dist = np.maximum(np.sum(P ** 2, axis=1) - 2.0 * p_max + 1.0, 0.0)
            return float(np.mean(np.sqrt(sq_dist)))
        elif soft_crit in ('kl_perfect', 'js_perfect'):
            labels = np.argmax(P, axis=1)
            one_hot = np.zeros((P.shape[0], len(self.target_names)), dtype=float)
            one_hot[np.arange(P.shape[0]), labels] = 1.0
            one_hot = safe_normalize_probs(one_hot, eps=eps)
            if soft_crit == 'kl_perfect':
                return float(np.mean(kl(P, one_hot, eps=eps, normalized=True)))
            return float(np.mean(js_divergence(P, one_hot, eps=eps, normalized=True)))

        raise ValueError(f"Unknown soft criterium '{soft_crit}'")

    def _impurity_calculation(self, preds: np.ndarray, 
                              criterium = 'soft_log_loss', 
                              alpha: float = 0.5,
                              eps: float = 1e-12, 
                              normalized: bool = True) -> float:
        """
            Computing the impurity of a node based on the selected criterium
            Args:
                - preds (np.ndarray): model predictions for the samples in the current node (can either be hard or soft labels, depending on the chosen criterium)
                - criterium (str): criterium for impurity calculation
                - alpha (float): weight for hybrid criterium (if applicable)
                - eps (float): small value to ensure numerical stability
                - normalized (bool): whether to normalize the predictions to probabilities (for soft-label criteria)
            Returns:
                - impurity (float): impurity score for the current node
        """

        if preds.shape[0] == 0:
            return 0.0  # convention; should not happen due to min_samples_split

        n_classes = len(self.target_names)
        if criterium == 'default':
            criterium = 'soft_log_loss'

        if isinstance(criterium, str):
            ## 1. CRITERIA BASED ON HARD LABELS (skip P normalization & meanP) ##
            if criterium in ('gini', 'log_loss'):
                probs = self._hard_probs(preds, n_classes)
                return float(gini(probs) if criterium == 'gini' else log_loss(probs))

            ## 2 & 3. CRITERIA BASED ON SOFT LABELS (skip hard bincount) ##
            P = preds if normalized else safe_normalize_probs(preds, eps=eps)
            return self._soft_impurity(P, criterium, eps=eps)

        ## 4. HARD LABEL + SOFT LABEL HYBRID CRITERIA ##
        elif isinstance(criterium, tuple):
            probs = self._hard_probs(preds, n_classes)
            if criterium[0] == 'gini':
                hard_imp = gini(probs) / (1.0 - 1.0 / n_classes)
            elif criterium[0] == 'log_loss':
                hard_imp = log_loss(probs) / np.log(n_classes)
            else:
                raise ValueError(f"Unknown hard criterium '{criterium[0]}'")

            P = preds if normalized else safe_normalize_probs(preds, eps=eps)
            raw_soft = self._soft_impurity(P, criterium[1], eps=eps)
            if criterium[1] == 'soft_gini':
                soft_imp = raw_soft / (1.0 - 1.0 / n_classes)
            elif criterium[1] in ('soft_log_loss', 'mutual_info'):
                soft_imp = raw_soft / np.log(n_classes)
            elif criterium[1] == 'js':
                soft_imp = raw_soft / np.log(2.0)
            elif criterium[1] == 'brier':
                soft_imp = raw_soft / (2.0 * (1.0 - 1.0 / n_classes))
            else:
                raise ValueError(f"Unknown soft criterium '{criterium[1]}'")

            return float(alpha * hard_imp + (1.0 - alpha) * soft_imp)

    def _initialise_root_node(self, final_res, criterium: str):
        """
            Initialising the root node information
            Args:
                - final_res (tuple): tuple that comprises the best candidate score, in the form of (feature_idx, feature_value, candidate_score)
                - criterium (str): criterium for impurity calculation
        """
        # Auxiliary variable - class distribution for the original targets
        target_dist = np.unique(self.targets, return_counts = True)[1]

        ## Appending the necessary information
        self.N_NODE_SAMPLES.append(len(self.dataset))
        # self.IMPURITY.append(self._impurity_calculation(self.targets if criterium in ['gini', 'log_loss'] else self.target_probs, criterium = criterium))
        self.IMPURITY.append(self._impurity_calculation(self.target_probs, criterium = 'gini'))
        self.VALUE.append(final_res[1])
        self.FEATURE.append(int(final_res[0]))
        self.LEFT.append(0)                      
        self.DEPTH.append(0)           
        self.LEAF.append(0)            
        self.PARENT.append(-1)
        self.CLASS.append(np.argmax(target_dist))
        self.CLASS_DIST.append(target_dist)
        self.NODE_DS.append(np.arange(len(self.dataset)))    


    def _expand_node(self, 
                     prev_node_idx: int, 
                     current_depth: int, 
                     dataset_idx: np.ndarray, 
                     n_steps: Union[str, int], 
                     adaptive: bool, 
                     criterium: str, 
                     left: bool,
                     alpha: float = 0.5):
        """
            Creating a new node based on the previous
            Args:  
                - prev_node_idx (int): parent node index
                - current_depth (int): current depth in the tree
                - dataset_idx (np.ndarray): indexes, relative to the original dataset, of the current dataset being assessed
                - n_steps (Union[str, int]): steps approach for discretization
                - adaptive (bool): whether to implement adaptive discretization
                - criterium (str): criterium for loss assessment in the tree
                - left (bool): if we are in a left (True) or a right (False) node
        """
        logger.debug(f"Expanding node {prev_node_idx} (depth={self.DEPTH[prev_node_idx]}) | "
             f"feature={self.feature_names[self.FEATURE[prev_node_idx]]} | threshold={self.VALUE[prev_node_idx]:.3f}")

        # 0. Some initial base information regarding the node
        self.LEFT.append(1 if left else 0)
        self.PARENT.append(prev_node_idx)
        self.DEPTH.append(current_depth)
        self.NODE_DS.append(dataset_idx)

        # 1. Initial information for the tree
        # node_imp = self._impurity_calculation(self.targets[dataset_idx] if criterium in ['gini', 'log_loss'] else self.target_probs[dataset_idx], criterium = criterium)
        node_imp = self._impurity_calculation(self.target_probs[dataset_idx], criterium = 'gini')

        preds_dist = np.bincount(self.targets[dataset_idx].ravel(), minlength = len(self.target_names))
        self.N_NODE_SAMPLES.append(len(dataset_idx))
        self.IMPURITY.append(node_imp)
        self.CLASS.append(np.argmax(preds_dist))
        self.CLASS_DIST.append(preds_dist)

        # 1st verification - impurity threshold or max depth
        if node_imp <= self.imp_threshold or current_depth == self.max_depth:
            self.LEAF.append(1)
            self.VALUE.append(None)
            self.FEATURE.append(None)
            if current_depth == self.max_depth:
                logger.debug(f"Max depth reached at node {len(self.FEATURE)-1} -> leaf node")
            if node_imp <= self.imp_threshold:
                logger.debug(f"Stopping split: impurity {node_imp:.4f} <= threshold {self.imp_threshold}")
            # Seeing if we already have a perfect class distribution
        else:
            # Conduct the optimisation process for the rule
            step_info = self._create_auto_steps(dataset_idx, n_steps = n_steps, adaptive = adaptive, criterium = criterium, alpha = alpha)
            final_res = self.execute_trial(dataset_idx, step_info, criterium = criterium, alpha = alpha)
            # In case no target rule was found
            if final_res is None:
                logger.warning(f"No acceptable candidate split was found for node {len(self.FEATURE)}. Marking it as a leaf node")
                self.LEAF.append(1)
                self.VALUE.append(None)
                self.FEATURE.append(None)            
            else:
                logger.debug(f"Candidate split found for node {len(self.FEATURE)-1}: Rule is {int(final_res[0])} <= {final_res[1]}")
                self.LEAF.append(0)
                self.VALUE.append(final_res[1])
                self.FEATURE.append(int(final_res[0]))
        logger.debug(f"Details: number of samples = {self.N_NODE_SAMPLES[-1]} | impurity = {self.IMPURITY[-1]} | output class = {self.CLASS[-1]} | class distribution = {self.CLASS_DIST[-1]}")
    
    def _return_leaf_output(self, node_idx: int):
        """
            Returns the output class for the current prediction (current node)
            Args:
                - node_idx (int): current index
        """
        return self.CLASS[node_idx]
    

    def plot_tree(self, save_fig: bool = False, file_name: str = None):
        """ 
            Plotting the classification tree
            Args:
                - save_fig (bool): whether the user wants, or not, to save the figure
                - file_name (str): name of the file to be stored (no need for the extension, I define pdf as default)
        """
        plot_classification_tree(self)
                
        if save_fig:
            if file_name is None:
                logger.warning("[WARNING] You have chosen to save the figure, but no file name was given. Using 'classification_tree' as default.")
                file_name = 'classification_tree'
            # Creating the folder if it does not exist
            os.makedirs(self.storage.folders.figs, exist_ok = True)
            plt.savefig(os.path.join(self.storage.folders.figs, f'{file_name}.pdf'), dpi = 300)

    def extract_prediction_path(self, sample: np.array):
        """
            Function that will extract the prediction path for each sample in the test set
            Args:
                - sample (np.array): sample to be analysed
            Returns:
                - text (str): textual explanation of the prediction path for the sample

            Code inspiration:"
            "https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html""
        """

        # 1. Making a prediction for our sample
        pred, path = self.predict(sample, return_path = True) 

        # Auxiliary step
        sample = np.array(sample).reshape(1, -1)

        # 2. Extracting relevant information from the tree to generate the text 
        nodes = np.array(path).squeeze()
        feature_indices = np.array(self.FEATURE).squeeze()[nodes]
        feature_values = np.array(self.VALUE).squeeze()[nodes]
        left_right = np.array(self.LEFT).squeeze()[nodes[1:]] 
        left_right = ['left' if val == 1 else 'right' for val in left_right]
        final_pred_str = self.target_names[pred[0]]
        final_class_dist = np.array(self.CLASS_DIST).squeeze()[nodes[-1]]
        certainty = final_class_dist[pred[0]] / np.sum(final_class_dist)

        # 3. Generating a textual explanation for the prediction path

        if len(nodes) <= 2:
            # Initial rule == only rule in fact
            text = f"\n\nThe decision tree starts at the root node, with the initial and only rule for this node being: \033[3mif {self.feature_names[feature_indices[0]]} \u2264 {np.round(feature_values[0], 3)}, move to the left child; otherwise, move to the right\033[0m.\n    The test sample has value {self.feature_names[feature_indices[0]]} = {sample[0, feature_indices[0]]}, so it reaches the leaf node {nodes[1]}.\n\n"
            # Final text for the prediction
            text += f"\033[1mThe final prediction is {final_pred_str}, with a certainty of {np.round(100*certainty, 2)}%.\033[0m\n"

        else:

            # Initial text for the initial rule
            text = f"\n\nThe decision tree starts at the root node, with the initial rule being: \033[3mif {self.feature_names[feature_indices[0]]} \u2264 {np.round(feature_values[0], 3)}, move to the left child; otherwise, move to the right\033[0m.\n    The test sample has value {self.feature_names[feature_indices[0]]} = {sample[0, feature_indices[0]]}, so it moves to the {left_right[0]} child (node {nodes[1]}).\n"

            # Cycle for the middle rules
            for i, (index, feature_index, rule_feature_value) in enumerate(zip(nodes[1:-2], feature_indices[1:-2], feature_values[1:-2])):
                text += f"At node {index}, the rule is: \033[3mif {self.feature_names[feature_index]} \u2264 {np.round(rule_feature_value, 3)}, move to the left child; otherwise, move to the right\033[0m.\n    The test sample has value {self.feature_names[feature_index]} = {sample[0, feature_index]}, so it moves to the {left_right[i+1]} child (node {nodes[i+2]}).\n"

            # Final text for the last rule
            text += f"At node {nodes[-2]}, the last decision node for this prediction, the final rule is: \033[3mif {self.feature_names[feature_indices[-2]]} \u2264 {np.round(feature_values[-2], 3)}, move to the left child; otherwise, move to the right\033[0m.\n     The test sample has value {self.feature_names[feature_indices[-2]]} = {sample[0, feature_indices[-2]]}, so it reaches the leaf node {nodes[-1]}.\n\n"

            # Final prediction and sentence
            final_pred_str = self.target_names[pred[0]]
            text += f"\033[1mThe final prediction is {final_pred_str}, with a certainty of {np.round(100*certainty, 2)}%.\033[0m\n"

        return text


    ######################################################
    #### CURRENTLY IN DEVELOPMENT!!!!! ###################
    ######################################################

    def extract_base_paths(self):
        """
            Function that will extract all base paths from the classification tree
        """
        leaf_indexes = np.where(np.array(self.LEAF) == 1)[0]
        all_paths = []
        for leaf_idx in leaf_indexes:
            parent_node_idx = None
            new_path = []
            new_path.append(leaf_idx)
            current_idx = leaf_idx
            while parent_node_idx != 0:
                parent_node_idx = self.PARENT[current_idx]
                new_path.append(parent_node_idx)
                current_idx = parent_node_idx
            # new_path.append(0)
            all_paths.append(sorted(new_path))

        return all_paths
    

    def extract_domains(self, all_paths):
        """
            Function in development still
        """
        # paths = np.array(paths)

        LEAF = np.array(self.LEAF)
        FEATURE = np.array(self.FEATURE)
        VALUE = np.array(self.VALUE)
        LEFT = np.array(self.LEFT)

        start_domain = np.concatenate((np.min(self.dataset, axis = 0).reshape(1, -1), np.max(self.dataset, axis = 0).reshape(1, -1)), axis = 0).T        
        all_rule_info = {}

        for i, path in enumerate(all_paths):
        
            # STEP 1 - Rule domain extraction
            path = np.array(path)
            specific_domain = start_domain.copy()
            for node, next_node in zip(path[:-1], path[1:]):
                if LEFT[next_node] == 1 and LEAF[node] == 0:
                    specific_domain[FEATURE[node]][1] = VALUE[node]
                elif LEFT[next_node] == 0 and LEAF[node] == 0:
                    specific_domain[FEATURE[node]][0] = VALUE[node]
                else:
                    pass
            
            # STEP 2 - Identifying all the variables that were used in the path, and only sticking with those
            # NOTE: by subtracting the original domain with the filtered one, any row where one of the values is unchanged will yield a 0 in that position
            aux_subtract = start_domain - specific_domain
            used_variables_domain = specific_domain[~np.all(aux_subtract == 0, axis = 1)]
            used_variables = np.unique(np.delete(FEATURE[path], np.where(FEATURE[path] == None))).astype(int)
            info = {}
            info['domain'] = {}
            info['limits'] = {}
            for j, feat in enumerate(used_variables):
                info['domain'][self.feature_names[feat]] = np.round(used_variables_domain[j], 3)
                info['limits'][self.feature_names[feat]] = np.array([np.round(np.min(self.dataset[:, feat]), 3), np.round(np.max(self.dataset[:, feat]), 3)])

            # STEP 3 - Extracting rule coverage information + output probability
            class_dist = self.CLASS_DIST[path[-1]]
            rule_samples, total_samples = np.sum(class_dist), self.dataset.shape[0]
            info['coverage'] = np.round(rule_samples/total_samples, 3)
            info['output_probability'] = np.round(class_dist[np.argmax(class_dist)]/rule_samples, 3)
            info['target_class'] = self.target_names[self.CLASS[path[-1]]]
            
            all_rule_info[f'Rule {i}'] = info
        
        return all_rule_info
