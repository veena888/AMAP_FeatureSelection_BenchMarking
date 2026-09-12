"""
Method 2 of 5 -- mRMR (minimum Redundancy Maximum Relevance).

Greedy forward selection. At each step it picks the feature with the best ratio
of relevance to the label against redundancy with what has already been chosen:

    score(f) = F_statistic(f, y) / mean(|pearson_r(f, s)| for s in selected)

This is the FCQ variant, the same one the `mrmr_selection` package implements as
mrmr_classif. It is written out here so the script has no extra dependency, and
so the intermediate scores can be inspected -- useful when a reviewer asks why a
particular residue fraction survived and its correlate did not.

Relevant to peptide features specifically: AAC columns are compositional (they
sum to 1) and the derived fractions (K_Fraction, HydrophobicFraction, ...) are
literally sums of AAC columns. That builds in heavy redundancy, which is exactly
the structure mRMR is designed to cut through.

Run:  python 02_mrmr.py
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

from common import K_FEATURES, load_data, run_method


def mrmr_select(X, y, k, redundancy_floor=1e-3, verbose=True):
    k = min(k, X.shape[1])

    # Relevance: ANOVA F-statistic of each feature against the class label.
    F, _ = f_classif(X.to_numpy(), y)
    relevance = pd.Series(np.nan_to_num(F, nan=0.0), index=X.columns)

    # Redundancy: absolute Pearson correlation between features.
    corr = X.corr(method="pearson").abs().fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(corr, 0.0)
    corr = pd.DataFrame(corr, index=X.columns, columns=X.columns)

    selected = [relevance.idxmax()]          # most relevant feature seeds the set
    remaining = [c for c in X.columns if c != selected[0]]
    if verbose:
        print(f"  1. {selected[0]:<24} F={relevance[selected[0]]:.2f} (seed)")

    while len(selected) < k and remaining:
        redundancy = corr.loc[remaining, selected].mean(axis=1)
        # Floor the denominator: an almost-uncorrelated feature would otherwise
        # divide by ~0 and dominate the ranking on noise alone.
        score = relevance[remaining] / redundancy.clip(lower=redundancy_floor)
        best = score.idxmax()
        if verbose:
            print(f"  {len(selected)+1}. {best:<24} F={relevance[best]:.2f}  "
                  f"redundancy={redundancy[best]:.3f}  score={score[best]:.2f}")
        selected.append(best)
        remaining.remove(best)
    return selected


X_train, y_train, X_test, y_test = load_data()

# Selection is re-fitted inside every CV fold, so it is quiet there and verbose
# only for the final full-training-set subset.
run_method(
    "mRMR",
    lambda X, y, k: mrmr_select(X, y, k, verbose=False),
    X_train, y_train, X_test, y_test,
)

print(f"\nFinal mRMR trace (full training set, k={K_FEATURES}):")
mrmr_select(X_train, y_train, K_FEATURES, verbose=True)
