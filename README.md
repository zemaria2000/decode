# DECODE: DEcision tree Capturing Opaque DEcisions

DECODE constitutes a comprehensive user-friendly framework aimed at **decoding the inner logic and underlying decision-making systems of complex ML models**. Its objective is to **transform any black-box model into a tree-based decision structure**, interpretable by any user.

Whether you are explaining a neural network, auditing a random forest model, or deploying explainable models in real-time systems, DECODE provides a unified framework for **interpretable modelling, visualisation, and performance assessment**. Overall, from lab experiments to production pipelines, *DECODE helps you see what the model sees*.

This library is currently compatible with any supervised ML model, and able to perform the following tasks:
1. Binary classification
2. Multi-label classification
3. Univariate regression
4. Multi-output regression


## Running unit tests

Currently, two unit tests are developed that allow for testing the robustness of a classification and a regression tree, and to test all methods associated with those. These unit tests were written in [unittest](https://docs.python.org/3/library/unittest.html).
Please run the folloing commands to execute the unit tests

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