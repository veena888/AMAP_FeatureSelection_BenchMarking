"""
Method 1 of 6 -- BASELINE, all features, no selection.

The control for the selection methods: any technique that cannot beat this is
not earning its complexity. Note this is a different control from
00_length_baseline.py, which controls for the dataset's length artifact.

Run:  python 01_baseline.py
"""

from common import load_data, run_method

X_train, y_train, X_test, y_test = load_data()

run_method(
    "Baseline",
    lambda X, y, k: list(X.columns),      # no selection
    X_train, y_train, X_test, y_test,
)
