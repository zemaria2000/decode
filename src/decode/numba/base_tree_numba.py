"""
    Numba-accelerated BaseTree subclass
"""

from copy import copy
from typing import Union
import numpy as np
import numba as nb

from ..base_tree import BaseTree as _BaseTree
from ..logger import logger


@nb.njit(cache=True, parallel=True)
def _predict_nb(X, feature, value, left_child, right_child, is_leaf, leaf_out):
    """Parallel O(depth) tree traversal per sample using precompiled child arrays."""
    n = X.shape[0]
    out = np.empty(n, dtype=leaf_out.dtype)
    for i in nb.prange(n):
        node = 0
        while not is_leaf[node]:
            if X[i, feature[node]] <= value[node]:
                node = left_child[node]
            else:
                node = right_child[node]
        out[i] = leaf_out[node]
    return out


class BaseTreeNumba(_BaseTree):

    def _reset_tree(self):
        super()._reset_tree()
        self._compiled_arrays = None

    def _compile_tree_arrays(self):
        """Precompute O(1) child index arrays and contiguous node vectors."""
        n = len(self.FEATURE)
        left_child = np.full(n, -1, dtype=np.int64)
        right_child = np.full(n, -1, dtype=np.int64)
        for idx in range(1, n):
            parent_idx = self.PARENT[idx]
            if parent_idx >= 0:
                if self.LEFT[idx] == 1:
                    left_child[parent_idx] = idx
                else:
                    right_child[parent_idx] = idx

        is_leaf = np.array(
            [self.LEAF[i] == 1 or self.FEATURE[i] is None or self.VALUE[i] is None for i in range(n)],
            dtype=np.bool_,
        )
        feature = np.array([0 if f is None else int(f) for f in self.FEATURE], dtype=np.int64)
        value = np.array([0.0 if v is None else float(v) for v in self.VALUE], dtype=np.float64)
        leaf_out = (
            np.asarray(self.CLASS, dtype=np.int64)
            if self.task == "classification"
            else np.asarray(self.AVG_VALUE, dtype=np.float64)
        )
        self._compiled_arrays = (feature, value, left_child, right_child, is_leaf, leaf_out)
        return self._compiled_arrays

    def _get_compiled_arrays(self):
        if getattr(self, "_compiled_arrays", None) is None or len(self._compiled_arrays[0]) != len(self.FEATURE):
            return self._compile_tree_arrays()
        return self._compiled_arrays

    def _select_best_split(self, thr_list: list, scores: np.ndarray):
        """Select the (feature_idx, threshold, score) row with the highest valid score."""
        if scores.size == 0 or np.all(np.isnan(scores)):
            logger.warning(f"No admissible splits were found for node {len(self.FEATURE) + 1}")
            return None
        feat_col = np.concatenate([np.repeat(f, len(t)) for f, t in enumerate(thr_list)]).astype(float)
        score_list = np.stack((feat_col, np.concatenate(thr_list).astype(float), scores), axis=1)
        score_list = score_list[~np.isnan(score_list).any(axis=1)]
        if len(score_list) == 0:
            logger.warning(f"No admissible splits were found for node {len(self.FEATURE) + 1}")
            return None
        best_idx = int(np.argmax(np.round(score_list[:, -1], 12)))
        return score_list[best_idx]

    def _candidate_score(
        self,
        dataset_idx: np.ndarray,
        feature_idx: int,
        feature_values: np.ndarray,
        criterium: str = 'soft_log_loss',
        alpha: float = 0.5,
        current_ds: np.ndarray = None,
        preds: np.ndarray = None,
        original_impurity: float = None
    ):
        if len(feature_values) == 0:
            return np.empty(0, dtype=float)

        if current_ds is None:
            current_ds = self.dataset[dataset_idx]
        if preds is None:
            preds_source = self.target_probs if self.task == "classification" else self.targets
            preds = preds_source[dataset_idx]

        n_samples = current_ds.shape[0]
        feature_column = current_ds[:, feature_idx]

        masks = feature_column[None, :] <= feature_values[:, None]
        n_left_samples = masks.sum(axis=1)
        n_right_samples = n_samples - n_left_samples

        valid_mask = (n_left_samples >= self.min_samples_split) & (n_right_samples >= self.min_samples_split)
        if not np.any(valid_mask):
            return np.full(len(feature_values), np.nan)

        if original_impurity is None:
            original_impurity = self._impurity_calculation(preds, criterium, alpha)
        left_impurities = np.array([self._impurity_calculation(preds[mask], criterium, alpha) for mask in masks[valid_mask]])
        right_impurities = np.array([self._impurity_calculation(preds[~mask], criterium, alpha) for mask in masks[valid_mask]])

        candidate_scores = original_impurity - (
            (n_left_samples[valid_mask] / n_samples) * left_impurities
            + (n_right_samples[valid_mask] / n_samples) * right_impurities
        )

        global_impurity_decreases = (n_samples / len(self.dataset)) * candidate_scores
        candidate_scores[global_impurity_decreases < self.min_impurity_decrease] = np.nan

        scores = np.full(len(feature_values), np.nan)
        scores[valid_mask] = candidate_scores
        return scores

    def execute_trial(self, dataset_idx: np.ndarray, step_info: dict, criterium: str = 'soft_log_loss', alpha: float = 0.5):
        current_ds = self.dataset[dataset_idx]
        preds_source = self.target_probs if self.task == "classification" else self.targets
        preds = preds_source[dataset_idx]
        original_impurity = self._impurity_calculation(preds, criterium, alpha)

        thr_list = [np.asarray(step_info[f]['steps'], dtype=float) for f in self.feature_names]
        scores_parts = [
            self._candidate_score(
                dataset_idx=dataset_idx,
                feature_idx=feature_idx,
                feature_values=feature_values,
                criterium=criterium,
                alpha=alpha,
                current_ds=current_ds,
                preds=preds,
                original_impurity=original_impurity
            )
            for feature_idx, feature_values in enumerate(thr_list)
        ]
        scores = np.concatenate(scores_parts) if len(scores_parts) > 0 else np.empty(0, dtype=float)
        return self._select_best_split(thr_list, scores)

    def fit(
        self,
        max_depth: int = 5,
        imp_threshold: float = 0.0,
        n_steps: Union[str, int] = 'auto',
        adaptive: bool = False,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        criterium: str = 'soft_log_loss',
        class_weights: Union[None, str, dict] = None,
        alpha: float = 0.5
    ):
        logger.info("Starting the tree generation process")
        logger.debug(
            f"Parameters: max_depth={max_depth}, imp_threshold={imp_threshold}, n_steps={n_steps}, "
            f"adaptive={adaptive}, min_samples_split={min_samples_split}, "
            f"min_impurity_decrease={min_impurity_decrease}, criterium={criterium}, alpha={alpha}"
        )

        if adaptive and criterium in ['gini', 'log_loss']:
            logger.warning("Adaptive step generation is not compatible with 'gini' or 'log_loss' criteria. Disabling adaptive step generation.")
            adaptive = False

        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_depth = max_depth
        self.imp_threshold = imp_threshold

        self._compute_class_weights(class_weights)
        self._reset_tree()

        ### 1. INITIAL NODE TRIAL
        original_idx = np.arange(len(self.dataset))
        step_info = self._create_auto_steps(original_idx, n_steps=n_steps, adaptive=adaptive, criterium=criterium, alpha=alpha)
        final_res = self.execute_trial(original_idx, step_info, criterium, alpha=alpha)
        if final_res is None:
            raise RuntimeError("No valid split found for root node. Try reducing constraints.")
        self._initialise_root_node(final_res, criterium)
        logger.info(f"Root node created. Rule is {self.feature_names[self.FEATURE[0]]} <= {self.VALUE[0]}")
        logger.debug(f"Root node impurity = {self.IMPURITY[0]}")

        ### 2. TREE GROWTH LOOP
        for i in range(1, max_depth + 1):
            prev_depth_indices = [index for index, value in enumerate(self.DEPTH) if value == (i - 1)]

            for prev_node_index in prev_depth_indices:
                if self.LEAF[prev_node_index] == 1:
                    continue

                ## 2.1. SINGLE-COLUMN SLICE (avoids copying all F columns of the parent node)
                node_ds = self.NODE_DS[prev_node_index]
                left_mask = self.dataset[node_ds, self.FEATURE[prev_node_index]] <= self.VALUE[prev_node_index]
                left_ds_idx = node_ds[left_mask]
                right_ds_idx = node_ds[~left_mask]

                ## 2.2. PROCESS LEFT NODE
                if len(left_ds_idx) > 0:
                    self._expand_node(
                        prev_node_idx=prev_node_index,
                        current_depth=i,
                        dataset_idx=left_ds_idx,
                        n_steps=n_steps,
                        adaptive=adaptive,
                        criterium=criterium,
                        left=True,
                        alpha=alpha
                    )

                ## 2.3. PROCESS RIGHT NODE
                if len(right_ds_idx) > 0:
                    self._expand_node(
                        prev_node_idx=prev_node_index,
                        current_depth=i,
                        dataset_idx=right_ds_idx,
                        n_steps=n_steps,
                        adaptive=adaptive,
                        criterium=criterium,
                        left=False,
                        alpha=alpha
                    )

        # Precompile O(1) child index arrays for predict() and compute_feature_importance()
        self._compile_tree_arrays()
        logger.info(f"Tree building process completed: total nodes = {len(self.N_NODE_SAMPLES)}, leaf nodes = {np.sum(self.LEAF)}")

    def predict(self, samples: np.ndarray, return_paths: bool = False):
        """
        Fast O(depth) prediction using precompiled child arrays (parallel Numba
        kernel when return_paths=False; O(1) child stepping when return_paths=True).
        """
        X = np.atleast_2d(np.asarray(samples, dtype=np.float64))
        feature, value, left_child, right_child, is_leaf, leaf_out = self._get_compiled_arrays()

        if not return_paths and leaf_out.ndim == 1:
            return _predict_nb(X, feature, value, left_child, right_child, is_leaf, leaf_out)

        predictions, prediction_paths = [], []
        for sample in X:
            current_node = 0
            pred_path = [0]
            while not is_leaf[current_node]:
                if sample[feature[current_node]] <= value[current_node]:
                    current_node = int(left_child[current_node])
                else:
                    current_node = int(right_child[current_node])
                if return_paths:
                    pred_path.append(current_node)
            predictions.append(self._return_leaf_output(node_idx=current_node))
            if return_paths:
                prediction_paths.append(pred_path)

        if not return_paths:
            return np.array(predictions)
        return np.array(predictions), prediction_paths

    def compute_feature_importance(self):
        """Vectorised O(N_nodes) feature importance using precompiled child arrays."""
        feature, _, left_child, right_child, is_leaf, _ = self._get_compiled_arrays()
        n_samples = np.asarray(self.N_NODE_SAMPLES, dtype=np.float64)
        impurity = np.asarray(self.IMPURITY, dtype=np.float64)

        internal = ~is_leaf
        feat_importance = np.zeros(len(self.feature_names), dtype=np.float64)
        if not np.any(internal):
            return feat_importance

        left = left_child[internal]
        right = right_child[internal]
        n_t = n_samples[internal]
        imp_t = impurity[internal]

        imp_l = np.where(left >= 0, impurity[np.maximum(left, 0)], 0.0)
        imp_r = np.where(right >= 0, impurity[np.maximum(right, 0)], 0.0)
        n_l = np.where(left >= 0, n_samples[np.maximum(left, 0)], 0.0)
        n_r = np.where(right >= 0, n_samples[np.maximum(right, 0)], 0.0)

        imp_increment = n_t * (imp_t - (n_l / n_t) * imp_l - (n_r / n_t) * imp_r)
        np.add.at(feat_importance, feature[internal], imp_increment)

        non_leaf_mask = np.asarray(self.LEAF) == 0
        denominator = np.sum(n_samples[non_leaf_mask] * impurity[non_leaf_mask])
        if denominator > 0:
            feat_importance = feat_importance / denominator
        total = np.sum(feat_importance)
        if total > 0:
            feat_importance = feat_importance / total
        return feat_importance

    def prune_tree(self, by: str = 'depth', threshold: Union[int, float] = 5):
        """
        Lightweight O(N_nodes) tree pruning that avoids deepcopying dataset/model
        and remaps parent indices cleanly.
        """
        if by not in ['depth', 'impurity']:
            raise ValueError("Pruning criterium not recognised. Please choose one of: 'depth', 'impurity'.")

        _, _, left_child, right_child, _, _ = self._get_compiled_arrays()
        depth = np.asarray(self.DEPTH)
        impurity = np.asarray(self.IMPURITY, dtype=float)
        parent = np.asarray(self.PARENT, dtype=np.int64)
        n_nodes = len(self.FEATURE)

        if by == 'depth':
            target_idx = np.where(depth <= threshold)[0]
        else:
            keep = np.zeros(n_nodes, dtype=bool)
            keep[0] = True
            for idx in range(1, n_nodes):
                p = parent[idx]
                if keep[p]:
                    l_sib, r_sib = left_child[p], right_child[p]
                    l_ok = (l_sib < 0) or (impurity[l_sib] >= threshold)
                    r_ok = (r_sib < 0) or (impurity[r_sib] >= threshold)
                    if impurity[idx] >= threshold and l_ok and r_ok:
                        keep[idx] = True
            target_idx = np.where(keep)[0]

        # Map old node indices -> new node indices (-1 if pruned)
        idx_map = np.full(n_nodes, -1, dtype=np.int64)
        idx_map[target_idx] = np.arange(len(target_idx), dtype=np.int64)

        # Shallow-copy tree (shares read-only dataset, targets, target_probs, model)
        pruned_tree = copy(self)
        if self.task == "classification":
            pruned_tree.CLASS = [self.CLASS[i] for i in target_idx]
            pruned_tree.CLASS_DIST = [self.CLASS_DIST[i] for i in target_idx]
        else:
            pruned_tree.AVG_VALUE = [self.AVG_VALUE[i] for i in target_idx]

        pruned_tree.DEPTH = [self.DEPTH[i] for i in target_idx]
        pruned_tree.FEATURE = [self.FEATURE[i] for i in target_idx]
        pruned_tree.IMPURITY = [self.IMPURITY[i] for i in target_idx]
        pruned_tree.LEAF = [self.LEAF[i] for i in target_idx]
        pruned_tree.LEFT = [self.LEFT[i] for i in target_idx]
        pruned_tree.N_NODE_SAMPLES = [self.N_NODE_SAMPLES[i] for i in target_idx]
        pruned_tree.PARENT = [-1 if self.PARENT[i] < 0 else int(idx_map[self.PARENT[i]]) for i in target_idx]
        pruned_tree.VALUE = [self.VALUE[i] for i in target_idx]
        pruned_tree.NODE_DS = [self.NODE_DS[i] for i in target_idx]

        # Any surviving node whose children were pruned becomes a leaf node
        has_kept_child = np.zeros(len(target_idx), dtype=bool)
        for new_idx in range(1, len(target_idx)):
            p_new = pruned_tree.PARENT[new_idx]
            if p_new >= 0:
                has_kept_child[p_new] = True

        for new_idx in range(len(target_idx)):
            if not has_kept_child[new_idx] and pruned_tree.LEAF[new_idx] == 0:
                pruned_tree.LEAF[new_idx] = 1
                pruned_tree.FEATURE[new_idx] = None
                pruned_tree.VALUE[new_idx] = None

        pruned_tree._compile_tree_arrays()
        return pruned_tree