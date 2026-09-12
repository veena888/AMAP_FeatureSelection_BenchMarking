"""
Method 5 of 6 -- RFE (Recursive Feature Elimination, wrapper).

Fits XGBoost on all features, drops the least important one, refits, and repeats
until k remain. Unlike the filters it evaluates features in the context of the
model that will actually be used, so it can keep a feature that is weak alone but
useful in combination -- and drop one that looks strong in isolation but adds
nothing once its correlates are present.

The cost is compute: it fits the estimator (p - k) times. With ~33 peptide
features that is seconds. Do not naively scale this to thousands of features.

Two options below:

  MODE = "fixed"  -- eliminate down to exactly K_FEATURES. Use this for a
                     like-for-like comparison against mRMR and MI at the same k.
  MODE = "cv"     -- RFECV cross-validates every subset size and picks k for you.
                     More honest if you have no principled reason for k, but it
                     makes the subset size a fitted quantity rather than a
                     constant, so say so in your methods section.

Note the estimator inside RFE uses fewer trees than the final model. That is
deliberate: it is ranking features, not producing the reported scores, and 500
trees x 18 eliminations is wasted time.

Run:  python 05_rfe.py
"""

from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from common import K_FEATURES, SEED, XGB_PARAMS, load_data, run_method

MODE = "fixed"          # "fixed" or "cv"
STEP = 2                # features dropped per iteration
RANK_TREES = 100        # trees in the ranking loop only; the reported model uses XGB_PARAMS


def rfe_select(X, y, k, verbose=True):
    # Lighter estimator for the ranking loop only.
    est = XGBClassifier(**{**XGB_PARAMS, "n_estimators": RANK_TREES, "n_jobs": -1})

    if MODE == "cv":
        sel = RFECV(
            estimator=est, step=STEP, min_features_to_select=5,
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
            scoring="roc_auc", n_jobs=-1,
        ).fit(X.to_numpy(), y)
        if verbose:
            print(f"  RFECV chose k={sel.n_features_} "
                  f"(best CV AUC={sel.cv_results_['mean_test_score'].max():.4f})")
    else:
        k = min(k, X.shape[1])
        sel = RFE(estimator=est, n_features_to_select=k, step=STEP).fit(X.to_numpy(), y)

    chosen = list(X.columns[sel.support_])

    if verbose:
        # sel.ranking_ is 1 for every kept feature; 2, 3, ... show elimination order
        # in reverse, so the highest rank was the first feature discarded.
        dropped = sorted(
            [(r, c) for c, r in zip(X.columns, sel.ranking_) if r > 1], reverse=True
        )
        print(f"  kept {len(chosen)}: {chosen}")
        if dropped:
            print(f"  first eliminated: {', '.join(c for _, c in dropped[:5])}")
            print(f"  last to survive:  {', '.join(c for _, c in dropped[-3:])}")
    return chosen


X_train, y_train, X_test, y_test = load_data()

run_method(
    "RFE",
    lambda X, y, k: rfe_select(X, y, k, verbose=False),
    X_train, y_train, X_test, y_test,
)

print(f"\nFinal RFE trace (full training set, mode={MODE}):")
rfe_select(X_train, y_train, K_FEATURES, verbose=True)
