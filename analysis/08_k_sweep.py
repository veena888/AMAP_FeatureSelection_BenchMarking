"""
Sweep the number of selected features and plot MCC against k.

This answers a sharper question than "which method wins": how aggressively can
each method prune before the model degrades? At a single k the methods may be
statistically indistinguishable while their degradation curves are not -- a
method that holds performance down to k=5 is more useful than one that collapses
below k=15, even if the two tie at k=15.

For every k and every method, feature selection is re-fitted inside each CV
fold, exactly as in run_method(). Nothing is fitted on the test set.

The baseline (all features, no selection) is drawn as a horizontal reference
line, since it has no k.

RUNTIME. Cost is len(K_VALUES) x n_methods x CV_SPLITS x CV_REPEATS model fits,
plus selection inside each fold. Elastic Net dominates because of its inner grid
search. Use CV_REPEATS = 2 in common.py for a fast first pass, then raise it for
the publication figure.

Changes in this version
-----------------------
1. TAG BUG FIXED. `tag` previously defaulted to the hardcoded string "tier1"
   whenever the script was run without arguments, so a tier 2 run wrote
   `k_sweep_tier1.*` into results_tier2/ and titled the figure "tier1". The
   argument branch was also wrong: "../extract/tiers/tier2_ctd_train.csv"
   .replace("_train.csv","") keeps the directory components. The tag is now
   derived with Path().stem from whichever file was actually loaded, in both
   branches.

2. Elastic Net's actual k is annotated. Shrinkage caps the number of surviving
   coefficients, so requested k and actual k diverge (tier 2: 63.9 actual for
   140 requested; tier 3: 228.8 for 500). Points where they differ are drawn
   hollow and the plateau is labelled, so the x-axis cannot be misread.
   The label is positioned in axes-fraction coordinates in the lower-middle
   region -- under the plateau and left of the legend -- because a fixed point
   offset from the data point collided with the baseline's shaded band.

3. The "within 1 SD" summary is printed alongside a caveat: at tier 1 the paired
   test disagreed with it (1 SD said k=10, the paired test said k=15). Use the
   paired test for claims; this figure is for shape.

Usage:
    python 08_k_sweep.py
    python 08_k_sweep.py ../extract/tiers/tier3_dpc_train.csv ../extract/tiers/tier3_dpc_test.csv
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from common import (CV_REPEATS, CV_SPLITS, RESULTS_DIR, SEED, TEST_CSV,
                    TRAIN_CSV, build_model, compute_metrics, load_data)
from fs_selectors import SWEEP_SELECTORS, select_baseline

METRIC = "MCC"

# Must be edited per tier -- every value has to be below the tier's feature count.
#   tier 1 (27)  : [3, 5, 7, 10, 15, 20, 25]
#   tier 2 (162) : [5, 10, 20, 40, 60, 100, 140]
#   tier 3 (562) : [10, 25, 50, 100, 200, 350, 500]
K_VALUES = [10, 25, 50, 100, 200, 350, 500]

# One fixed colour per method, matching 06_compare.py so figures agree.
COLOURS = {
    "mRMR": "#2a78d6",
    "Mutual Information": "#eb6834",
    "Elastic Net": "#1baf7a",
    "RFE": "#9b5de5",
}
MARKERS = {"mRMR": "o", "Mutual Information": "s",
           "Elastic Net": "^", "RFE": "D"}

# Flag a point as mislabelled when actual k falls this far below the request.
ACTUAL_K_TOLERANCE = 0.5


def derive_tag(train_csv):
    """'../extract/tiers/tier2_ctd_train.csv' -> 'tier2_ctd'.

    Path().stem drops both the directory and the extension, so the result is
    always a bare filename component safe to use in an output filename.
    """
    return Path(train_csv).stem.replace("_train", "")


def cv_score(X_train, y_train, select_fn, k):
    """Nested CV at one k: selection re-fitted per fold. Returns per-fold scores
    and the mean number of features the selector actually returned."""
    cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS,
                                 random_state=SEED)
    scores, sizes = [], []
    for tr_idx, va_idx in cv.split(X_train, y_train):
        Xf, yf = X_train.iloc[tr_idx], y_train[tr_idx]
        feats = select_fn(Xf, yf, k)
        sizes.append(len(feats))

        model = build_model(yf)
        model.fit(Xf[feats], yf)
        prob = model.predict_proba(X_train.iloc[va_idx][feats])[:, 1]
        scores.append(compute_metrics(y_train[va_idx],
                                      (prob >= 0.5).astype(int), prob)[METRIC])
    return np.array(scores), float(np.mean(sizes))


def main():
    if len(sys.argv) >= 3:
        train_csv, test_csv = sys.argv[1], sys.argv[2]
    else:
        train_csv, test_csv = TRAIN_CSV, TEST_CSV

    X_train, y_train, X_test, y_test = load_data(train_csv, test_csv)
    tag = derive_tag(train_csv)

    n_features = X_train.shape[1]
    ks = [k for k in K_VALUES if k < n_features]
    if not ks:
        raise SystemExit(
            f"No K_VALUES below the feature count ({n_features}). "
            f"Edit K_VALUES at the top of this script for this tier."
        )
    if max(ks) < n_features * 0.5:
        print(f"NOTE: largest k ({max(ks)}) is under half of {n_features} features. "
              f"The curve may not reach the baseline. Consider larger K_VALUES.")

    print(f"\nTag: {tag}   sweeping k over {ks} on {n_features} features\n")

    # Reference line: no selection at all.
    base_scores, _ = cv_score(X_train, y_train, select_baseline, n_features)
    base_mu, base_sd = base_scores.mean(), base_scores.std()
    print(f"Baseline (all {n_features}): {METRIC} {base_mu:.4f} +/- {base_sd:.4f}\n")

    rows = []
    for name, fn in SWEEP_SELECTORS.items():
        print(f"--- {name} ---")
        for k in ks:
            scores, actual_k = cv_score(X_train, y_train, fn, k)
            diverged = (k - actual_k) > ACTUAL_K_TOLERANCE
            rows.append({"method": name, "k": k, "actual_k": actual_k,
                         "diverged": diverged,
                         "mean": scores.mean(), "std": scores.std(),
                         "folds": scores.tolist()})
            flag = f"  (actual mean k={actual_k:.1f})" if diverged else ""
            print(f"  k={k:<4} {METRIC} {scores.mean():.4f} +/- {scores.std():.4f}{flag}")
        print()

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.drop(columns="folds").to_csv(RESULTS_DIR / f"k_sweep_{tag}.csv", index=False)
    (RESULTS_DIR / f"k_sweep_{tag}_folds.json").write_text(
        json.dumps({"tag": tag, "train_csv": str(train_csv),
                    "baseline": {"mean": base_mu, "std": base_sd,
                                 "n_features": int(n_features),
                                 "folds": base_scores.tolist()},
                    "sweep": rows}, indent=2))

    # ---- where does each method stop tracking the baseline? ----------------
    print("=" * 74)
    print("Smallest k at which each method stays within 1 SD of the baseline")
    print(f"(baseline {base_mu:.4f}, 1 SD = {base_sd:.4f})")
    print("=" * 74)
    for name in SWEEP_SELECTORS:
        sub = df[df.method == name].sort_values("k")
        ok = sub[sub["mean"] >= base_mu - base_sd]
        if ok.empty:
            print(f"  {name:<22} never within 1 SD")
        else:
            kmin = int(ok.k.min())
            row = ok[ok.k == kmin].iloc[0]
            note = f"  [actual k={row.actual_k:.1f}]" if row.diverged else ""
            print(f"  {name:<22} k={kmin:<4} "
                  f"({METRIC} {row['mean']:.4f}, "
                  f"{100 * kmin / n_features:.0f}% of features){note}")
    print("\n  CAVEAT: this 1 SD heuristic is descriptive only. At tier 1 it said")
    print("  k=10 while the paired Wilcoxon against the baseline said k=15. Use")
    print("  07_significance.py for any claim; use this figure for shape.")

    # ---- figure -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.axhline(base_mu, color="#52514e", ls="--", lw=1.5,
               label=f"Baseline, all {n_features} ({base_mu:.3f})")
    ax.axhspan(base_mu - base_sd, base_mu + base_sd, color="#52514e", alpha=0.10)

    annotated = False
    for name in SWEEP_SELECTORS:
        sub = df[df.method == name].sort_values("k")
        colour = COLOURS.get(name, "#333333")
        ax.errorbar(sub.k, sub["mean"], yerr=sub["std"],
                    marker=MARKERS.get(name, "o"), color=colour,
                    lw=2, capsize=3, markersize=6, label=name)

        # Hollow markers where the selector could not reach the requested k.
        bad = sub[sub["diverged"]]
        if not bad.empty:
            ax.plot(bad.k, bad["mean"], MARKERS.get(name, "o"),
                    markerfacecolor="white", markeredgecolor=colour,
                    markeredgewidth=1.8, markersize=7, linestyle="none")
            if not annotated:
                # These curves rise steeply then plateau, so the reliably empty
                # region is the lower middle -- under the plateau and left of the
                # legend. Axes-fraction coordinates keep the label there whatever
                # the data range, instead of colliding with the shaded band as a
                # fixed point-offset did.
                last = bad.iloc[-1]
                ax.annotate(f"{name}: capped at ~{last.actual_k:.0f} features "
                            f"(requested {int(last.k)})",
                            xy=(last.k, last["mean"]), xycoords="data",
                            xytext=(0.42, 0.16), textcoords="axes fraction",
                            fontsize=8, color=colour, ha="center", va="center",
                            arrowprops=dict(arrowstyle="->", color=colour, lw=1,
                                            alpha=0.8, shrinkB=5,
                                            connectionstyle="arc3,rad=0.15"))
                annotated = True

    ax.set_xlabel("number of selected features (k)")
    ax.set_ylabel(f"cross-validated {METRIC}")
    ax.set_title(f"Degradation with feature count — {tag} ({n_features} features)")
    ax.set_xticks(ks)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)

    if annotated:
        fig.text(0.01, 0.01,
                 "Hollow markers: the selector returned fewer features than the "
                 "requested k (see actual_k in the CSV).",
                 fontsize=7, color="#898781")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"k_sweep_{tag}.png", dpi=200)
    print(f"\nWritten to {RESULTS_DIR.resolve()}  (files tagged '{tag}')")


if __name__ == "__main__":
    main()
