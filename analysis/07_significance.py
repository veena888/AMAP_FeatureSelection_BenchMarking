"""
Paired significance testing across the shared CV folds.

Every method was evaluated on identical folds (RepeatedStratifiedKFold with a
fixed seed), so folds are matched units and the comparison must be PAIRED. An
unpaired test would discard the pairing and lose most of its power.

Two tests are reported per pair:

  Wilcoxon signed-rank -- non-parametric, no normality assumption. This is the
      one to quote; per-fold MCC on a 1:15.5 imbalanced set is not Gaussian.

  Paired t-test -- reported alongside for readers who expect it. Note that with
      5x5 repeated CV the 25 folds are NOT independent (each repeat reuses every
      sample), so nominal p-values are anti-conservative. Nadeau & Bengio's
      corrected resampled t-test is applied as well, which inflates the variance
      estimate to account for the train/test overlap between folds.

Holm-Bonferroni correction is applied across the pairwise family.

Run after 00-05:  python 07_significance.py
"""

import itertools
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import CV_REPEATS, CV_SPLITS, RESULTS_DIR

METRIC = "MCC"
ALPHA = 0.05

runs = {}
for p in RESULTS_DIR.glob("*.json"):
    d = json.loads(p.read_text())
    if d.get("cv_folds"):
        runs[d["method"]] = np.array(d["cv_folds"][METRIC])

if len(runs) < 2:
    raise SystemExit("Need at least two methods with cv_folds. Re-run 00-05.")

names = sorted(runs, key=lambda n: -runs[n].mean())
n_folds = len(next(iter(runs.values())))
print(f"Paired comparison on per-fold {METRIC}, {n_folds} folds "
      f"({CV_SPLITS}x{CV_REPEATS})\n")

for n in names:
    print(f"  {n:<24} {runs[n].mean():.4f} ± {runs[n].std():.4f}")


def corrected_t(diff, n_splits=CV_SPLITS):
    """Nadeau & Bengio corrected resampled t-test.

    Variance is inflated by (1/n + ratio) where ratio = test_size/train_size,
    because repeated CV folds share training data and the naive t-test treats
    them as independent.
    """
    n = len(diff)
    ratio = 1.0 / (n_splits - 1)
    denom = np.sqrt((1.0 / n + ratio) * diff.var(ddof=1))
    if denom == 0:
        return 0.0, 1.0
    t = diff.mean() / denom
    return t, 2 * stats.t.sf(abs(t), df=n - 1)


rows = []
for a, b in itertools.combinations(names, 2):
    d = runs[a] - runs[b]
    if np.allclose(d, 0):
        w_p = 1.0
    else:
        w_p = stats.wilcoxon(runs[a], runs[b]).pvalue
    t_stat, t_p = stats.ttest_rel(runs[a], runs[b])
    _, ct_p = corrected_t(d)
    rows.append({"A": a, "B": b, "mean_diff": d.mean(),
                 "wilcoxon_p": w_p, "paired_t_p": t_p, "corrected_t_p": ct_p})

res = pd.DataFrame(rows)

# Holm-Bonferroni across the family of pairwise tests.
for col in ["wilcoxon_p", "corrected_t_p"]:
    order = res[col].argsort().to_numpy()
    m = len(res)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * res[col].to_numpy()[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    res[col.replace("_p", "_holm")] = adj

res = res.sort_values("mean_diff", key=abs, ascending=False)
res.to_csv(RESULTS_DIR / "significance.csv", index=False)

print(f"\nPairwise, Holm-corrected (alpha={ALPHA}):\n")
print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

sig = res[res["wilcoxon_holm"] < ALPHA]
print(f"\n{len(sig)} of {len(res)} pairs differ significantly (Wilcoxon, Holm):")
if sig.empty:
    print("  none")
else:
    for _, r in sig.iterrows():
        print(f"  {r.A} vs {r.B}: diff={r.mean_diff:+.4f}, p={r.wilcoxon_holm:.4f}")

# The comparison the paper actually turns on.
fs_methods = [n for n in names if n not in ("Baseline", "Length only")]
print("\nEach selection method vs the no-selection baseline:")
for n in fs_methods:
    if "Baseline" not in runs:
        break
    d = runs[n] - runs["Baseline"]
    p = 1.0 if np.allclose(d, 0) else stats.wilcoxon(runs[n], runs["Baseline"]).pvalue
    verdict = "differs" if p < ALPHA else "no significant difference"
    print(f"  {n:<24} diff={d.mean():+.4f}  p={p:.4f}  -> {verdict}")
