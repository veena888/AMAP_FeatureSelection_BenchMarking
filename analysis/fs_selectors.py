"""
The five selection functions in one importable module.

Why this exists: module names beginning with a digit (02_mrmr.py) cannot be
imported with a normal import statement, so the k-sweep and tier-sweep scripts
need the selectors somewhere importable. This is the single source of truth --
if you change a selector, change it here.

Every function has the same signature:

    select(X: DataFrame, y: array, k: int) -> list of column names

and must see TRAINING ROWS ONLY. run_method() in common.py calls them once per
CV fold on that fold's training rows, and once on the full training set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from common import SEED, XGB_PARAMS

# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
def select_baseline(X, y, k=None):
    return list(X.columns)


# ---------------------------------------------------------------------------
# mRMR (FCQ variant: F-statistic relevance / mean absolute correlation)
# ---------------------------------------------------------------------------
def select_mrmr(X, y, k, redundancy_floor=1e-3):
    k = min(k, X.shape[1])
    F, _ = f_classif(X.to_numpy(), y)
    relevance = pd.Series(np.nan_to_num(F, nan=0.0), index=X.columns)

    corr = X.corr(method="pearson").abs().fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(corr, 0.0)
    corr = pd.DataFrame(corr, index=X.columns, columns=X.columns)

    selected = [relevance.idxmax()]
    remaining = [c for c in X.columns if c != selected[0]]
    while len(selected) < k and remaining:
        redundancy = corr.loc[remaining, selected].mean(axis=1)
        score = relevance[remaining] / redundancy.clip(lower=redundancy_floor)
        best = score.idxmax()
        selected.append(best)
        remaining.remove(best)
    return selected


# ---------------------------------------------------------------------------
# Mutual information
# ---------------------------------------------------------------------------
def select_mutual_info(X, y, k, n_repeats=3):
    k = min(k, X.shape[1])
    runs = [mutual_info_classif(X.to_numpy(), y, random_state=SEED + i)
            for i in range(n_repeats)]
    mi = pd.DataFrame(runs, columns=X.columns).mean()
    return list(mi.sort_values(ascending=False).head(k).index)


# ---------------------------------------------------------------------------
# Elastic net
# ---------------------------------------------------------------------------
# The useful range of C depends on the number of features: with more columns the
# penalty is spread thinner, so sparsity appears at larger C. At 27 features
# nothing was eliminated above C = 0.05; at 500+ expect that ceiling to rise.
# The grid spans both regimes so it does not silently saturate on the wide tiers.
EN_C_GRID = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
EN_L1_GRID = [0.3, 0.5, 0.9]
EN_INNER_CV = 3
EN_MAX_ITER = 2000


def select_elastic_net(X, y, k, match_k=False):
    Xs = StandardScaler().fit_transform(X)          # fitted on this fold's train rows
    grid = GridSearchCV(
        LogisticRegression(penalty="elasticnet", solver="saga",
                           class_weight="balanced", max_iter=EN_MAX_ITER,
                           random_state=SEED),
        {"C": EN_C_GRID, "l1_ratio": EN_L1_GRID},
        cv=StratifiedKFold(EN_INNER_CV, shuffle=True, random_state=SEED),
        scoring="roc_auc", n_jobs=-1,
    ).fit(Xs, y)

    coef = pd.Series(np.abs(grid.best_estimator_.coef_.ravel()), index=X.columns)
    kept = coef[coef > 1e-8].sort_values(ascending=False)
    if kept.empty:
        kept = coef.sort_values(ascending=False).head(k)
    if match_k:
        kept = kept.head(k)
    return list(kept.index)


def select_elastic_net_k(X, y, k):
    """Elastic net truncated to exactly k, for the sweep where subset size is
    the independent variable and must be held equal across methods."""
    return select_elastic_net(X, y, k, match_k=True)


# ---------------------------------------------------------------------------
# RFE
# ---------------------------------------------------------------------------
RFE_TREES = 100        # ranking loop only; the reported model uses XGB_PARAMS


def select_rfe(X, y, k, step=None):
    k = min(k, X.shape[1])
    # Eliminating one feature at a time costs (p - k) fits. That is fine at 27
    # features and painful at 574, so drop a fraction per iteration on wide data.
    if step is None:
        step = 1 if X.shape[1] <= 60 else 0.1
    est = XGBClassifier(**{**XGB_PARAMS, "n_estimators": RFE_TREES, "n_jobs": -1})
    sel = RFE(estimator=est, n_features_to_select=k, step=step).fit(X.to_numpy(), y)
    return list(X.columns[sel.support_])


SELECTORS = {
    "Baseline": select_baseline,
    "mRMR": select_mrmr,
    "Mutual Information": select_mutual_info,
    "Elastic Net": select_elastic_net,
    "RFE": select_rfe,
}

# For the k-sweep, every method must return exactly k features or the x-axis
# is meaningless. Baseline is excluded (it has no k) and drawn as a flat line.
SWEEP_SELECTORS = {
    "mRMR": select_mrmr,
    "Mutual Information": select_mutual_info,
    "Elastic Net": select_elastic_net_k,
    "RFE": select_rfe,
}
