"""
Method 0 of 6 -- LENGTH-ONLY CONTROL.

This is not a feature selection method. It is the number every other result in
the study has to be read against.

In this dataset the negatives have a minimum length of 11 residues while the
positives go down to 1. Roughly half the positives (48/110 train, 14/28 test)
therefore sit in a length region that contains no negatives at all. A single
threshold exploits that:

    predict positive if Length < 11   ->   test accuracy 0.9692, MCC 0.6958

with no model, no composition features and no training. Any pipeline reporting
around that level has learned a length threshold and nothing more.

Two versions are reported:

  Rule       -- the hard threshold above, tuned on the TRAINING set only.
  Length-only model -- the full SMOTE + XGBoost pipeline restricted to the
                       single Length column, so it passes through exactly the
                       same evaluation as the real methods.

Run:  python 00_length_baseline.py
"""

import json

import numpy as np
import pandas as pd

from common import (METRIC_ORDER, RESULTS_DIR, compute_metrics, load_data,
                    run_method)

X_train, y_train, X_test, y_test = load_data()

# ---------------------------------------------------------------------------
# 1. Hard threshold, chosen on the training set by maximising MCC
# ---------------------------------------------------------------------------
from sklearn.metrics import matthews_corrcoef

candidates = sorted(X_train["Length"].unique())
best_t, best_mcc = None, -1.0
for t in candidates:
    pred = (X_train["Length"] < t).astype(int)
    mcc = matthews_corrcoef(y_train, pred)
    if mcc > best_mcc:
        best_t, best_mcc = t, mcc

print(f"\nThreshold tuned on TRAIN: Length < {best_t:.0f}  (train MCC {best_mcc:.4f})")

pred_test = (X_test["Length"] < best_t).astype(int).to_numpy()
# Score for AUC: shorter = more likely positive, so negate length.
rule_scores = compute_metrics(y_test, pred_test, -X_test["Length"].to_numpy())

print("\nTest set, threshold rule only (no model at all):")
for m in METRIC_ORDER:
    print(f"  {m:<12} {rule_scores[m]:.4f}")

RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "length_rule.json").write_text(json.dumps({
    "method": "Length rule (no model)",
    "n_features": 1,
    "features": ["Length"],
    "stable_features": ["Length"],
    "threshold": float(best_t),
    "cv_mean": {}, "cv_std": {},
    "test": rule_scores,
    "probabilities": (-X_test["Length"]).tolist(),
    "y_test": y_test.tolist(),
}, indent=2))

# ---------------------------------------------------------------------------
# 2. The same XGBoost pipeline, given only Length
# ---------------------------------------------------------------------------
run_method(
    "Length only",
    lambda X, y, k: ["Length"],
    X_train, y_train, X_test, y_test,
)
