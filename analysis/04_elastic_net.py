"""
Method 4 of 6 -- ELASTIC NET (embedded selection).

An L1+L2 penalised logistic regression is fitted and the features whose
coefficients survive shrinkage are kept. L1 drives coefficients to exactly zero
(that is the selection); L2 stops the model picking one member of a correlated
group arbitrarily and zeroing its neighbours -- which matters here, since AAC
columns and the derived fractions are correlated by construction.

Two things that are easy to get wrong, both handled below:

  1. Scaling. Penalised regression is not scale-invariant. Length runs to the
     hundreds while AAC fractions sit near 0.05; unscaled, the penalty would
     flatten the fractions purely because their raw magnitudes are small. The
     scaler is fitted on the training rows of each fold only.

  2. The C grid. On this dataset (27 features, 1818 rows) the regularisation
     path is narrow: at C >= 0.05 nothing is eliminated at all, and at
     C <= 0.001 everything is. All the interesting behaviour is in between:

         C = 0.001  ->  0 features    CV AUC 0.500
         C = 0.005  -> 11 features    CV AUC 0.849
         C = 0.010  -> 15 features    CV AUC 0.859
         C = 0.050  -> 27 features    CV AUC 0.858

     A default grid of [0.01, 0.1, 1, 10] would sit almost entirely in the
     saturated region and "Elastic Net" would silently become the baseline.

Unlike mRMR/MI/RFE this returns however many features survive, not exactly k.
Set MATCH_K = True if the paper needs subset size held constant across methods.

Run:  python 04_elastic_net.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import K_FEATURES, SEED, load_data, run_method

MATCH_K = False

# Grid concentrated on the sparse region of the path (see docstring).
C_GRID = [0.002, 0.005, 0.01, 0.02]
L1_GRID = [0.3, 0.5, 0.9]
INNER_CV = 3          # inner folds for choosing C; 3 keeps the nested run tractable
MAX_ITER = 2000


def elastic_net_select(X, y, k=K_FEATURES, verbose=True):
    scaler = StandardScaler().fit(X)                 # fitted on THIS fold's train rows
    Xs = scaler.transform(X)

    grid = GridSearchCV(
        LogisticRegression(penalty="elasticnet", solver="saga",
                           class_weight="balanced", max_iter=MAX_ITER,
                           random_state=SEED),
        {"C": C_GRID, "l1_ratio": L1_GRID},
        cv=StratifiedKFold(INNER_CV, shuffle=True, random_state=SEED),
        scoring="roc_auc", n_jobs=-1,
    ).fit(Xs, y)

    model = grid.best_estimator_
    coef = pd.Series(model.coef_.ravel(), index=X.columns)
    kept = coef[coef.abs() > 1e-8].abs().sort_values(ascending=False)

    if kept.empty:                                   # penalty too aggressive
        kept = coef.abs().sort_values(ascending=False).head(k)
    if MATCH_K:
        kept = kept.head(k)

    if verbose:
        print(f"  best C={grid.best_params_['C']}, "
              f"l1_ratio={grid.best_params_['l1_ratio']}, "
              f"inner CV AUC={grid.best_score_:.4f}")
        print(f"  {len(kept)} of {X.shape[1]} features survived shrinkage")
        for name in kept.index:
            print(f"    {name:<20} coef={coef[name]:+.4f}")
    return list(kept.index)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()

    run_method(
        "Elastic Net",
        lambda X, y, k: elastic_net_select(X, y, k, verbose=False),
        X_train, y_train, X_test, y_test,
    )

    print("\nFinal Elastic Net fit (full training set):")
    elastic_net_select(X_train, y_train, K_FEATURES, verbose=True)
