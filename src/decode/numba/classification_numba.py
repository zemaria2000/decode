"""
    Numba-accelerated ClassificationTree subclass (redefines only split-search methods)
"""

import numpy as np
import numba as nb

from ..classification import ClassificationTree as _ClassificationTree
from .base_tree_numba import BaseTreeNumba 
from ..utils import safe_normalize_probs, entropy_vec, kl, js_divergence


###############################################################################
############## NUMBA HISTOGRAM + PREFIX-SUM SPLIT KERNELS #####################
###############################################################################

HARD_NONE = -1
HARD_LOGLOSS = 0
HARD_GINI = 1

SOFT_NONE = -1
SOFT_LOGLOSS = 0
SOFT_GINI = 1
SOFT_MI_KL = 2      # mutual_info and kl: H(meanP) - mean(H(P_i))
SOFT_BRIER = 3      # brier: mean(||P_i||^2) - ||meanP||^2
SOFT_PERFECT = 4    # *_perfect: sample-wise distance to 1-hot

_LOG2 = np.log(2.0)


def _parse_criterium(criterium, n_classes: int):
    if criterium == 'default':
        criterium = 'soft_log_loss'

    if isinstance(criterium, str):
        if criterium == 'log_loss':
            return HARD_LOGLOSS, SOFT_NONE, 1.0, 1.0, False
        if criterium == 'gini':
            return HARD_GINI, SOFT_NONE, 1.0, 1.0, False
        if criterium == 'soft_log_loss':
            return HARD_NONE, SOFT_LOGLOSS, 1.0, 1.0, False
        if criterium == 'soft_gini':
            return HARD_NONE, SOFT_GINI, 1.0, 1.0, False
        if criterium in ('mutual_info', 'kl'):
            return HARD_NONE, SOFT_MI_KL, 1.0, 1.0, False
        if criterium == 'brier':
            return HARD_NONE, SOFT_BRIER, 1.0, 1.0, False
        if criterium in ('soft_label_distance_perfect', 'brier_perfect', 'kl_perfect', 'js_perfect'):
            return HARD_NONE, SOFT_PERFECT, 1.0, 1.0, False
        return None

    if isinstance(criterium, tuple) and len(criterium) == 2:
        h_name, s_name = criterium
        if h_name == 'gini':
            h_code, h_norm = HARD_GINI, (1.0 - 1.0 / n_classes)
        elif h_name == 'log_loss':
            h_code, h_norm = HARD_LOGLOSS, np.log(n_classes)
        else:
            return None

        if s_name == 'soft_gini':
            s_code, s_norm = SOFT_GINI, (1.0 - 1.0 / n_classes)
        elif s_name == 'soft_log_loss':
            s_code, s_norm = SOFT_LOGLOSS, np.log(n_classes)
        elif s_name == 'mutual_info':
            s_code, s_norm = SOFT_MI_KL, np.log(n_classes)
        elif s_name == 'brier':
            s_code, s_norm = SOFT_BRIER, 2.0 * (1.0 - 1.0 / n_classes)
        else:
            return None
        return h_code, s_code, float(h_norm), float(s_norm), True

    return None


@nb.njit(cache=True)
def _pw_block(a, lo, n):
    if n < 8:
        res = 0.0
        for i in range(lo, lo + n):
            res += a[i]
        return res
    r0 = a[lo]; r1 = a[lo + 1]; r2 = a[lo + 2]; r3 = a[lo + 3]
    r4 = a[lo + 4]; r5 = a[lo + 5]; r6 = a[lo + 6]; r7 = a[lo + 7]
    m = n - (n % 8)
    i = 8
    while i < m:
        r0 += a[lo + i]; r1 += a[lo + i + 1]; r2 += a[lo + i + 2]; r3 += a[lo + i + 3]
        r4 += a[lo + i + 4]; r5 += a[lo + i + 5]; r6 += a[lo + i + 6]; r7 += a[lo + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    for k in range(m, n):
        res += a[lo + k]
    return res


@nb.njit(cache=True)
def _np_pairwise_sum(a, lo, n):
    if n <= 128:
        return _pw_block(a, lo, n)
    los = np.empty(64, np.int64)
    ns = np.empty(64, np.int64)
    st = np.zeros(64, np.int64)
    acc = np.empty(64, np.float64)
    sp = 0
    los[0] = lo; ns[0] = n; st[0] = 0
    ret = 0.0
    have_ret = False
    while True:
        if not have_ret:
            m = ns[sp]
            if m <= 128:
                ret = _pw_block(a, los[sp], m)
                have_ret = True
                sp -= 1
            else:
                n2 = m // 2
                n2 -= n2 % 8
                st[sp] = 1
                los[sp + 1] = los[sp]; ns[sp + 1] = n2; st[sp + 1] = 0
                sp += 1
        else:
            if sp < 0:
                return ret
            if st[sp] == 1:
                acc[sp] = ret
                st[sp] = 2
                n2 = ns[sp] // 2
                n2 -= n2 % 8
                los[sp + 1] = los[sp] + n2; ns[sp + 1] = ns[sp] - n2; st[sp + 1] = 0
                sp += 1
                have_ret = False
            else:
                ret = acc[sp] + ret
                sp -= 1


@nb.njit(cache=True, inline='always')
def _bin_of(thr, t0, t1, x):
    lo, hi = t0, t1
    while lo < hi:
        mid = (lo + hi) >> 1
        if thr[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo - t0


@nb.njit(cache=True)
def _hard_impurity_nb(cnt, weights, use_w, h_code, tmp):
    C = cnt.shape[0]
    if use_w:
        for c in range(C):
            tmp[c] = weights[c] * cnt[c]
        s = _np_pairwise_sum(tmp, 0, C)
    else:
        tot = 0
        for c in range(C):
            tot += cnt[c]
            tmp[c] = cnt[c]
        s = float(tot)
    if h_code == HARD_LOGLOSS:
        for c in range(C):
            p = tmp[c] / s
            if p < 1e-12:
                p = 1e-12
            elif p > 1.0:
                p = 1.0
            tmp[c] = p * np.log(p)
        return -_np_pairwise_sum(tmp, 0, C) / _LOG2
    else:
        for c in range(C):
            p = tmp[c] / s
            tmp[c] = p * (1.0 - p)
        return _np_pairwise_sum(tmp, 0, C)


@nb.njit(cache=True)
def _soft_impurity_nb(sum_p, sum_s, n_sub, s_code, tmp):
    if s_code == SOFT_PERFECT:
        return sum_s / n_sub
    C = sum_p.shape[0]
    inv_n = 1.0 / n_sub
    if s_code == SOFT_GINI:
        for c in range(C):
            p = sum_p[c] * inv_n
            tmp[c] = p * (1.0 - p)
        return _np_pairwise_sum(tmp, 0, C)
    elif s_code == SOFT_LOGLOSS:
        for c in range(C):
            p = sum_p[c] * inv_n
            if p < 1e-12:
                p = 1e-12
            elif p > 1.0:
                p = 1.0
            tmp[c] = p * np.log(p)
        return -_np_pairwise_sum(tmp, 0, C) / _LOG2
    elif s_code == SOFT_MI_KL:
        for c in range(C):
            p = sum_p[c] * inv_n
            if p < 1e-12:
                p = 1e-12
            elif p > 1.0:
                p = 1.0
            tmp[c] = p * np.log(p)
        h_mean = -_np_pairwise_sum(tmp, 0, C)
        return h_mean - (sum_s * inv_n)
    else:  # SOFT_BRIER
        for c in range(C):
            p = sum_p[c] * inv_n
            tmp[c] = p * p
        return (sum_s * inv_n) - _np_pairwise_sum(tmp, 0, C)


@nb.njit(cache=True)
def _feature_scores_nb(
    xcol, hard_preds, P, sample_s, thr, t0, t1, n_classes,
    weights, use_w, h_code, s_code, h_norm, s_norm, is_tuple, alpha,
    parent_imp, min_samples_split, min_imp_dec, n_total, out
):
    T = t1 - t0
    n = xcol.shape[0]
    need_hard = h_code != HARD_NONE
    need_soft_vec = s_code in (SOFT_GINI, SOFT_LOGLOSS, SOFT_MI_KL, SOFT_BRIER)
    need_soft_scalar = s_code in (SOFT_MI_KL, SOFT_BRIER, SOFT_PERFECT)

    bin_cnt = np.zeros(T + 1, dtype=np.int64)
    hist_hard = np.zeros((T + 1, n_classes), dtype=np.int64) if need_hard else np.zeros((1, 1), dtype=np.int64)
    hist_P = np.zeros((T + 1, n_classes), dtype=np.float64) if need_soft_vec else np.zeros((1, 1), dtype=np.float64)
    hist_s = np.zeros(T + 1, dtype=np.float64) if need_soft_scalar else np.zeros(1, dtype=np.float64)

    for i in range(n):
        x = xcol[i]
        if x != x:
            b = T
        else:
            b = _bin_of(thr, t0, t1, x)
        bin_cnt[b] += 1
        if need_hard:
            hist_hard[b, hard_preds[i]] += 1
        if need_soft_vec:
            for c in range(n_classes):
                hist_P[b, c] += P[i, c]
        if need_soft_scalar:
            hist_s[b] += sample_s[i]

    tot_hard = np.zeros(n_classes, dtype=np.int64)
    left_hard = np.zeros(n_classes, dtype=np.int64)
    right_hard = np.zeros(n_classes, dtype=np.int64)
    if need_hard:
        for b in range(T + 1):
            for c in range(n_classes):
                tot_hard[c] += hist_hard[b, c]

    tot_P = np.zeros(n_classes, dtype=np.float64)
    left_P = np.zeros(n_classes, dtype=np.float64)
    right_P = np.zeros(n_classes, dtype=np.float64)
    if need_soft_vec:
        for b in range(T + 1):
            for c in range(n_classes):
                tot_P[c] += hist_P[b, c]

    tot_s = 0.0
    left_s = 0.0
    if need_soft_scalar:
        for b in range(T + 1):
            tot_s += hist_s[b]

    tmp = np.empty(n_classes, dtype=np.float64)
    nl = 0
    for j in range(T):
        nl += bin_cnt[j]
        nr = n - nl
        if need_hard:
            for c in range(n_classes):
                left_hard[c] += hist_hard[j, c]
                right_hard[c] = tot_hard[c] - left_hard[c]
        if need_soft_vec:
            for c in range(n_classes):
                left_P[c] += hist_P[j, c]
                right_P[c] = tot_P[c] - left_P[c]
        if need_soft_scalar:
            left_s += hist_s[j]
        right_s = tot_s - left_s

        if nl < min_samples_split or nr < min_samples_split:
            out[t0 + j] = np.nan
            continue

        if not is_tuple:
            if need_hard:
                li = _hard_impurity_nb(left_hard, weights, use_w, h_code, tmp)
                ri = _hard_impurity_nb(right_hard, weights, use_w, h_code, tmp)
            else:
                li = _soft_impurity_nb(left_P, left_s, nl, s_code, tmp)
                ri = _soft_impurity_nb(right_P, right_s, nr, s_code, tmp)
        else:
            li_h = _hard_impurity_nb(left_hard, weights, use_w, h_code, tmp) / h_norm
            ri_h = _hard_impurity_nb(right_hard, weights, use_w, h_code, tmp) / h_norm
            li_s = _soft_impurity_nb(left_P, left_s, nl, s_code, tmp) / s_norm
            ri_s = _soft_impurity_nb(right_P, right_s, nr, s_code, tmp) / s_norm
            li = alpha * li_h + (1.0 - alpha) * li_s
            ri = alpha * ri_h + (1.0 - alpha) * ri_s

        cs = parent_imp - ((nl / n) * li + (nr / n) * ri)
        if (n / n_total) * cs < min_imp_dec:
            out[t0 + j] = np.nan
        else:
            out[t0 + j] = cs


@nb.njit(cache=True)
def node_scores_serial(
    XT, hard_preds, P, sample_s, thr, offsets, n_classes,
    weights, use_w, h_code, s_code, h_norm, s_norm, is_tuple, alpha,
    parent_imp, min_samples_split, min_imp_dec, n_total
):
    out = np.empty(thr.shape[0], dtype=np.float64)
    for f in range(XT.shape[0]):
        t0, t1 = offsets[f], offsets[f + 1]
        if t1 > t0:
            _feature_scores_nb(
                XT[f], hard_preds, P, sample_s, thr, t0, t1, n_classes,
                weights, use_w, h_code, s_code, h_norm, s_norm, is_tuple, alpha,
                parent_imp, min_samples_split, min_imp_dec, n_total, out
            )
    return out


@nb.njit(cache=True, parallel=True)
def node_scores_parallel(
    XT, hard_preds, P, sample_s, thr, offsets, n_classes,
    weights, use_w, h_code, s_code, h_norm, s_norm, is_tuple, alpha,
    parent_imp, min_samples_split, min_imp_dec, n_total
):
    out = np.empty(thr.shape[0], dtype=np.float64)
    for f in nb.prange(XT.shape[0]):
        t0, t1 = offsets[f], offsets[f + 1]
        if t1 > t0:
            _feature_scores_nb(
                XT[f], hard_preds, P, sample_s, thr, t0, t1, n_classes,
                weights, use_w, h_code, s_code, h_norm, s_norm, is_tuple, alpha,
                parent_imp, min_samples_split, min_imp_dec, n_total, out
            )
    return out


###############################################################################
############## SUBCLASS REDEFINING ONLY SPLIT-SEARCH METHODS ##################
###############################################################################

class ClassificationTreeNumba(_ClassificationTree, BaseTreeNumba):

    def _sample_scalar_invariants(self, P: np.ndarray, soft_crit: str, eps: float = 1e-12) -> np.ndarray:
        """Compute per-sample scalar sufficient statistics for additive soft-label criteria."""
        if soft_crit in ('mutual_info', 'kl'):
            return np.ascontiguousarray(entropy_vec(P, eps=eps, normalized=True), dtype=np.float64)
        if soft_crit == 'brier':
            return np.ascontiguousarray(np.sum(P ** 2, axis=1), dtype=np.float64)
        if soft_crit == 'brier_perfect':
            p_max = np.max(P, axis=1)
            return np.ascontiguousarray(np.sum(P ** 2, axis=1) - 2.0 * p_max + 1.0, dtype=np.float64)
        if soft_crit == 'soft_label_distance_perfect':
            p_max = np.max(P, axis=1)
            sq_dist = np.maximum(np.sum(P ** 2, axis=1) - 2.0 * p_max + 1.0, 0.0)
            return np.ascontiguousarray(np.sqrt(sq_dist), dtype=np.float64)
        if soft_crit in ('kl_perfect', 'js_perfect'):
            labels = np.argmax(P, axis=1)
            one_hot = np.zeros((P.shape[0], len(self.target_names)), dtype=float)
            one_hot[np.arange(P.shape[0]), labels] = 1.0
            one_hot = safe_normalize_probs(one_hot, eps=eps)
            d = kl(P, one_hot, eps=eps, normalized=True) if soft_crit == 'kl_perfect' else js_divergence(P, one_hot, eps=eps, normalized=True)
            return np.ascontiguousarray(d, dtype=np.float64)
        return np.empty(1, dtype=np.float64)

    def execute_trial(self, dataset_idx: np.ndarray, step_info: dict, criterium: str = 'soft_log_loss', alpha: float = 0.5):
        n_classes = len(self.target_names)
        parsed = _parse_criterium(criterium, n_classes)
        if parsed is None:
            return super().execute_trial(dataset_idx, step_info, criterium=criterium, alpha=alpha)

        h_code, s_code, h_norm, s_norm, is_tuple = parsed

        # 1. Gather node data once in cache-friendly transposed layout
        XT = np.ascontiguousarray(self.dataset[dataset_idx].T, dtype=np.float64)
        P = np.ascontiguousarray(self.target_probs[dataset_idx], dtype=np.float64)
        hard_preds = (
            np.ascontiguousarray(np.argmax(P, axis=1), dtype=np.int64)
            if h_code != HARD_NONE
            else np.empty(1, dtype=np.int64)
        )

        # 2. Prepare class weights & sample-level scalars (when required by criterium)
        if getattr(self, "class_weights", None) is not None:
            use_w = True
            weights = (
                np.ascontiguousarray(list(self.class_weights.values()), dtype=np.float64)
                if isinstance(self.class_weights, dict)
                else np.ascontiguousarray(self.class_weights, dtype=np.float64)
            )
        else:
            use_w = False
            weights = np.ones(n_classes, dtype=np.float64)

        soft_name = criterium[1] if is_tuple else (criterium if criterium != 'default' else 'soft_log_loss')
        sample_s = (
            self._sample_scalar_invariants(P, soft_name)
            if s_code in (SOFT_MI_KL, SOFT_BRIER, SOFT_PERFECT)
            else np.empty(1, dtype=np.float64)
        )

        # 3. Pack all candidate thresholds into a flat vector with feature offsets
        thr_list = []
        for f in self.feature_names:
            t = np.asarray(step_info[f]['steps'], dtype=np.float64)
            if len(t) > 1 and np.any(np.diff(t) < 0):
                t = np.sort(t)
            thr_list.append(t)
        offsets = np.zeros(len(thr_list) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(t) for t in thr_list])
        if offsets[-1] == 0:
            return None
        thr = np.concatenate(thr_list)

        # 4. Evaluate all (feature, threshold) candidates in a single Numba call
        parent_imp = float(self._impurity_calculation(P, criterium=criterium, alpha=alpha))
        kernel = node_scores_parallel if len(dataset_idx) > 20_000 else node_scores_serial
        scores = kernel(
            XT, hard_preds, P, sample_s, thr, offsets, n_classes,
            weights, use_w, h_code, s_code, h_norm, s_norm, is_tuple, float(alpha),
            parent_imp, int(self.min_samples_split), float(self.min_impurity_decrease),
            len(self.dataset)
        )
        return self._select_best_split(thr_list, scores)

    def extract_domains(self, all_paths):
        """Reuses precomputed per-feature bounds from start_domain instead of rescanning self.dataset."""
        LEAF = np.array(self.LEAF)
        FEATURE = np.array(self.FEATURE)
        VALUE = np.array(self.VALUE)
        LEFT = np.array(self.LEFT)

        start_domain = np.column_stack((np.min(self.dataset, axis=0), np.max(self.dataset, axis=0)))
        rounded_limits = np.round(start_domain, 3)
        all_rule_info = {}

        for i, path in enumerate(all_paths):
            path = np.array(path)
            specific_domain = start_domain.copy()
            for node, next_node in zip(path[:-1], path[1:]):
                if LEFT[next_node] == 1 and LEAF[node] == 0:
                    specific_domain[FEATURE[node]][1] = VALUE[node]
                elif LEFT[next_node] == 0 and LEAF[node] == 0:
                    specific_domain[FEATURE[node]][0] = VALUE[node]

            aux_subtract = start_domain - specific_domain
            used_variables_domain = specific_domain[~np.all(aux_subtract == 0, axis=1)]
            used_variables = np.unique(np.delete(FEATURE[path], np.where(FEATURE[path] == None))).astype(int)
            info = {'domain': {}, 'limits': {}}
            for j, feat in enumerate(used_variables):
                feat_name = self.feature_names[feat]
                info['domain'][feat_name] = np.round(used_variables_domain[j], 3)
                info['limits'][feat_name] = rounded_limits[feat].copy()

            class_dist = self.CLASS_DIST[path[-1]]
            rule_samples, total_samples = np.sum(class_dist), self.dataset.shape[0]
            info['coverage'] = np.round(rule_samples / total_samples, 3)
            info['output_probability'] = np.round(class_dist[np.argmax(class_dist)] / rule_samples, 3)
            info['target_class'] = self.target_names[self.CLASS[path[-1]]]

            all_rule_info[f'Rule {i}'] = info

        return all_rule_info