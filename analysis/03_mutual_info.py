"""
Method 3 of 6 -- MUTUAL INFORMATION (filter).

Ranks each feature by the mutual information it shares with the class label and
keeps the top k. MI measures any statistical dependence, not just a monotonic
one, so it catches relationships an F-test or a correlation would miss -- for
instance a net charge that is predictive when strongly positive OR strongly
negative but not in between.

The trade-off is that it scores every feature in isolation. Two features that
are individually informative but carry the same information both score highly
and both get selected. Contrast with mRMR, which explicitly penalises that.
Expect overlapping picks here (e.g. K_Fraction alongside the raw K column).

One practical note: sklearn's estimator is k-nearest-neighbour based and draws
random noise to break ties in continuous data, so scores shift slightly between
runs. random_state is fixed, and N_REPEATS averages over several draws to make
the ranking stable enough to report.

Run:  python 04_mutual_info.py
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from common import K_FEATURES, SEED, load_data, run_method

N_REPEATS = 10          # average MI over this many seeds for a stable ranking


def mutual_info_select(X, y, k, verbose=True, n_repeats=N_REPEATS):
    k = min(k, X.shape[1])

    runs = [mutual_info_classif(X.to_numpy(), y, random_state=SEED + i)
            for i in range(n_repeats)]
    mi = pd.DataFrame(runs, columns=X.columns)
    ranking = pd.DataFrame({"mi_mean": mi.mean(), "mi_std": mi.std()}) \
                .sort_values("mi_mean", ascending=False)

    if verbose:
        print(f"  MI averaged over {n_repeats} runs (top {k}):")
        for name, row in ranking.head(k).iterrows():
            print(f"    {name:<24} MI={row.mi_mean:.4f} +/- {row.mi_std:.4f}")
        cut = ranking.iloc[k - 1].mi_mean - ranking.iloc[k].mi_mean if k < len(ranking) else None
        if cut is not None and cut < 0.005:
            print(f"  NOTE: gap at the cut-off is only {cut:.4f} -- ranks {k} and "
                  f"{k+1} are effectively tied, so k is doing arbitrary work here")

    return list(ranking.head(k).index)


X_train, y_train, X_test, y_test = load_data()

# 3 repeats inside CV folds (25 folds x 3 = 75 MI estimations) keeps runtime sane;
# the final reported ranking uses the full N_REPEATS.
run_method(
    "Mutual Information",
    lambda X, y, k: mutual_info_select(X, y, k, verbose=False, n_repeats=3),
    X_train, y_train, X_test, y_test,
)

print("\nFinal MI ranking (full training set):")
mutual_info_select(X_train, y_train, K_FEATURES, verbose=True)
