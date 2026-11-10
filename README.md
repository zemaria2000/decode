# DECODE: DEcision tree Capturing Opaque DEcisions

DECODE constitutes a comprehensive user-friendly framework aimed at **decoding the inner logic and underlying decision-making systems of complex ML models**. Its objective is to **transform any black-box model into a tree-based decision structure**, interpretable by any user.

Whether you are explaining a neural network, auditing a random forest model, or deploying explainable models in real-time systems, DECODE provides a unified framework for **interpretable modelling, visualisation, and performance assessment**. Overall, from lab experiments to production pipelines, *DECODE helps you see what the model sees*.

This library is currently compatible with any supervised ML model, and able to perform the following tasks:
1. Binary classification
2. Multi-label classification
3. Univariate regression
4. Multi-output regression


## Numba-Accelerated Classification Tree

For medium-to-large datasets or hyperparameter sweeps, DECODE provides `ClassificationTreeNumba`, a drop-in subclass of `ClassificationTree` powered by Numba JIT kernels that produces **bit-for-bit
identical trees** at **10x–25x faster** fit and prediction speeds:

```python
from decode import ClassificationTree, ClassificationTreeNumba

# Drop-in replacement for ClassificationTree
tree = ClassificationTreeNumba(
    model=black_box_model,
    dataset=X_train,
    targets=y_train,
    feature_names=feature_names,
    target_names=target_names,
)

tree.fit(
    max_depth=6,
    n_steps='auto',
    adaptive=True,
    criterium='soft_log_loss',  # or 'gini', 'log_loss', 'soft_gini', 'brier', 'kl', 'mutual_info', ('gini', 'soft_log_loss')
    class_weights='balanced',
)

preds = tree.predict(X_test)
importances = tree.compute_feature_importance()
pruned_tree = tree.prune_tree(by='depth', threshold=4)
```

### Verification Checklist

Once you apply these changes, you can verify everything end-to-end with:
1. `python others/benchmark_trees.py` — confirms `identical = yes` across all criteria and dataset scales.
2. `python -m unittest discover -s tests` — runs the unit and equivalence test suite.


## Running unit tests

Currently, two unit tests are developed that allow for testing the robustness of a classification and a regression tree, and to test all methods associated with those. These unit tests were written in [unittest](https://docs.python.org/3/library/unittest.html).
Please run the following commands to execute the unit tests

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install .
python -m unittest
```

## Installation
 
To install the library locally, simple execute the following commands:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install .
```

## Original paper

DECODE is part of my Ph.D. work, and its core idea and principle can be analysed in its [original paper](https://ieeexplore.ieee.org/document/11659693)
