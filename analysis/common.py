"""
Shared setup for the iAMAP-SCM feature-selection comparison.

Every method script imports from here, so the data, the model, the resampling
and the metrics are identical across methods. The only thing that varies is
which columns reach XGBoost.

Two numbers are reported for each method:

  CV   -- repeated stratified k-fold on the TRAINING set, with feature
          selection re-fitted inside every fold. This is the honest estimate
          and the one to compare methods on. Reported as mean +/- std.

  TEST -- a single fit on all training data, scored on the held-out test set.
          With 28 positives in the test set, one flipped prediction moves
          sensitivity by 3.6 points. Treat this as a final confirmation, not
          as the basis for ranking methods.

In a notebook, run this once first:
    %run common.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
TRAIN_CSV = "../extract/tiers/tier3_dpc_train.csv"
TEST_CSV = "../extract/tiers/tier3_dpc_test.csv"
LABEL_COL = "Label"
ID_COL = "ID"

K_FEATURES = 140          # target subset size for mRMR / MI / RFE (of 27 available)
SEED = 42

CV_SPLITS = 5
CV_REPEATS = 5           # 25 fits per method; ~110 positives means 22 per fold

RESULTS_DIR = Path("results_tier3")

# Identical for every method.
XGB_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=SEED,
)

METRIC_ORDER = ["Accuracy", "Sensitivity", "Specificity", "Precision",
                "Recall", "F1", "MCC", "ROC-AUC"]


# ----------------------------------------------------------------------------
# LOADING
# ----------------------------------------------------------------------------
def load_data(train_csv=TRAIN_CSV, test_csv=TEST_CSV, verbose=True):
    """Returns X_train, y_train, X_test, y_test with matched numeric columns."""
    tr, te = pd.read_csv(train_csv), pd.read_csv(test_csv)

    y_train = tr[LABEL_COL].astype(int).to_numpy()
    y_test = te[LABEL_COL].astype(int).to_numpy()

    drop = [c for c in (ID_COL, LABEL_COL) if c in tr.columns]
    X_train, X_test = tr.drop(columns=drop), te.drop(columns=drop)

    shared = [c for c in X_train.columns if c in X_test.columns]
    missing = [c for c in X_train.columns if c not in X_test.columns]
    if missing:
        print(f"  WARNING: train columns absent from test, dropped: {missing}")
    X_train, X_test = X_train[shared], X_test[shared]

    # Imputation medians and constant-column decisions come from TRAIN only.
    medians = X_train.median(numeric_only=True)
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(medians)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(medians)

    const = X_train.columns[X_train.nunique() <= 1].tolist()
    if const:
        X_train, X_test = X_train.drop(columns=const), X_test.drop(columns=const)
        print(f"  dropped constant columns: {const}")

    if verbose:
        print(f"Train {X_train.shape[0]} x {X_train.shape[1]}  "
              f"(pos {int(y_train.sum())} / neg {int((y_train == 0).sum())}, "
              f"1:{(y_train == 0).sum() / y_train.sum():.1f})")
        print(f"Test  {X_test.shape[0]} x {X_test.shape[1]}  "
              f"(pos {int(y_test.sum())} / neg {int((y_test == 0).sum())})")
    return X_train, y_train, X_test, y_test


# ----------------------------------------------------------------------------
# MODEL + METRICS
# ----------------------------------------------------------------------------
def build_model(y_train):
    """SMOTE + XGBoost. SMOTE sits inside the pipeline so it only ever resamples
    training rows, never a validation fold.

    scale_pos_weight is deliberately NOT set: combined with SMOTE it corrects
    for the 1:15.5 imbalance twice and pushes sensitivity up at specificity's
    expense.
    """
    minority = int(min(np.bincount(y_train)))
    return ImbPipeline([
        ("smote", SMOTE(random_state=SEED, k_neighbors=max(1, min(5, minority - 1)))),
        ("clf", XGBClassifier(**XGB_PARAMS)),
    ])


def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,   # identical to Recall
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }


# ----------------------------------------------------------------------------
# EVALUATION
# ----------------------------------------------------------------------------
def run_method(name, select_fn, X_train, y_train, X_test, y_test,
               k=K_FEATURES, save=True, verbose_selection=True):
    """Nested CV on train + one final fit scored on test.

    select_fn(X, y, k) -> list of column names. It is called once per CV fold on
    that fold's training rows only, and once more on the full training set for
    the final model.
    """
    # ---- nested CV: selection re-fitted inside each fold --------------------
    cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS,
                                 random_state=SEED)
    fold_rows, fold_features = [], []
    for tr_idx, va_idx in cv.split(X_train, y_train):
        Xf, yf = X_train.iloc[tr_idx], y_train[tr_idx]
        feats = select_fn(Xf, yf, k)
        fold_features.append(feats)

        model = build_model(yf)
        model.fit(Xf[feats], yf)
        prob = model.predict_proba(X_train.iloc[va_idx][feats])[:, 1]
        fold_rows.append(compute_metrics(y_train[va_idx], (prob >= 0.5).astype(int), prob))

    cv_df = pd.DataFrame(fold_rows)

    # ---- how stable is the selection across folds? -------------------------
    counts = pd.Series([f for fs in fold_features for f in fs]).value_counts()
    n_folds = len(fold_features)
    always = counts[counts == n_folds].index.tolist()

    # ---- final model on the full training set ------------------------------
    final_feats = select_fn(X_train, y_train, k)
    model = build_model(y_train)
    model.fit(X_train[final_feats], y_train)
    prob = model.predict_proba(X_test[final_feats])[:, 1]
    test_scores = compute_metrics(y_test, (prob >= 0.5).astype(int), prob)

    # ---- report ------------------------------------------------------------
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    if verbose_selection:
        print(f"Final subset ({len(final_feats)}): {final_feats}")
        print(f"Selected in all {n_folds} CV folds ({len(always)}): {always}")
    print(f"\nCV ({CV_SPLITS}x{CV_REPEATS}, selection re-fitted per fold):")
    for m in METRIC_ORDER:
        print(f"  {m:<12} {cv_df[m].mean():.4f} +/- {cv_df[m].std():.4f}")
    print("\nTest set (single fit):")
    for m in METRIC_ORDER:
        print(f"  {m:<12} {test_scores[m]:.4f}")

    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        payload = {
            "method": name,
            "n_features": len(final_feats),
            "features": final_feats,
            "stable_features": always,
            "cv_mean": {m: float(cv_df[m].mean()) for m in METRIC_ORDER},
            "cv_std": {m: float(cv_df[m].std()) for m in METRIC_ORDER},
            # Per-fold scores are kept so methods can be compared with a PAIRED
            # test: every method sees the identical folds (RepeatedStratifiedKFold
            # with a fixed seed), so the fold is a matched unit and an unpaired
            # test would throw away that pairing and lose power.
            "cv_folds": {m: cv_df[m].tolist() for m in METRIC_ORDER},
            "test": test_scores,
            "probabilities": prob.tolist(),
            "y_test": y_test.tolist(),
        }
        out = RESULTS_DIR / f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.json"
        out.write_text(json.dumps(payload, indent=2))
        print(f"\n-> {out}")

    return cv_df, test_scores, final_feats
