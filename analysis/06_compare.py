"""
Aggregates the JSON written by scripts 00-05 into publication-ready output.

Produces:
    <RESULTS_DIR>/table_cv.csv          mean +/- std over the nested CV -- the
                                        table to rank methods on
    <RESULTS_DIR>/table_test.csv        independent test set, single fit
    <RESULTS_DIR>/table_latex.txt       both tables as LaTeX, ready to paste
    <RESULTS_DIR>/feature_overlap.csv   which features each method selected
    <RESULTS_DIR>/error_counts.csv      raw TP/FN/FP/TN per method
    <RESULTS_DIR>/comparison.png        ROC curves + metric bars

Changes in this version
-----------------------
1. COLOUR BUG FIXED. Colours now come from a fixed per-method map, so a method
   is the same colour in BOTH panels. Previously the left panel sorted by AUC
   while the right panel used insertion order, and matplotlib's default cycle
   assigned colours by draw position -- blue was Elastic Net on the left and the
   length rule on the right. Each legend was internally correct, but comparing
   across panels was misleading.

2. Controls (the two length rows) sort last, draw dashed on the ROC, are hatched
   in the bar panel, and sit below a rule in the LaTeX tables. They are
   reference points, not competitors.

3. Raw confusion-matrix counts are reported. On this dataset every method
   recovers the same number of true positives, so the whole spread in test MCC
   comes from a handful of false positives. Better to state that than to leave a
   reader inferring it from three decimal places of specificity.

4. Consensus is reported twice: features in every final subset, and features
   stable in every CV fold. The second is the stricter, more honest claim.

Run after 00-05:  python 06_compare.py
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from common import METRIC_ORDER, RESULTS_DIR

# Methods under comparison first, controls last.
METHOD_ORDER = ["Baseline", "mRMR", "Elastic Net", "Mutual Information", "RFE"]
CONTROL_ORDER = ["Length only", "Length rule (no model)"]
ORDER = METHOD_ORDER + CONTROL_ORDER
CONTROLS = set(CONTROL_ORDER)

# One fixed colour per method, used by every panel and every future figure.
COLOURS = {
    "Baseline":               "#52514e",
    "mRMR":                   "#2a78d6",
    "Elastic Net":            "#1baf7a",
    "Mutual Information":     "#eb6834",
    "RFE":                    "#9b5de5",
    "Length only":            "#c2bfb6",
    "Length rule (no model)": "#898781",
}
SHORT = {"Length rule (no model)": "Length rule"}


def label_of(r):
    return SHORT.get(r["method"], r["method"])


runs = [json.loads(p.read_text()) for p in RESULTS_DIR.glob("*.json")]
if not runs:
    raise SystemExit("No results. Run 00_length_baseline.py ... 05_rfe.py first.")
runs.sort(key=lambda d: ORDER.index(d["method"]) if d["method"] in ORDER else 99)

unknown = [r["method"] for r in runs if r["method"] not in ORDER]
if unknown:
    print(f"NOTE: methods not listed in ORDER, appended last: {unknown}")

# ---------------------------------------------------------------------------
# Table 1 -- nested CV on the training set (the main result)
# ---------------------------------------------------------------------------
cv_rows = []
for r in runs:
    if not r.get("cv_mean"):                    # the threshold rule has no CV
        continue
    row = {"Method": r["method"], "n": r["n_features"]}
    for m in METRIC_ORDER:
        row[m] = f"{r['cv_mean'][m]:.3f} ± {r['cv_std'][m]:.3f}"
    cv_rows.append(row)
cv_table = pd.DataFrame(cv_rows)
cv_table.to_csv(RESULTS_DIR / "table_cv.csv", index=False)

print("=" * 100)
print("TABLE 1 — nested cross-validation on the training set")
print("           (feature selection re-fitted inside every fold)")
print("=" * 100)
print(cv_table.to_string(index=False))

# ---------------------------------------------------------------------------
# Table 2 -- independent test set
# ---------------------------------------------------------------------------
test_table = pd.DataFrame([
    {"Method": r["method"], "n": r["n_features"],
     **{m: round(r["test"][m], 4) for m in METRIC_ORDER}}
    for r in runs
])
test_table.to_csv(RESULTS_DIR / "table_test.csv", index=False)

y_ref = np.array(runs[0]["y_test"])
n_pos, n_neg = int(y_ref.sum()), int((y_ref == 0).sum())

print("\n" + "=" * 100)
print(f"TABLE 2 — independent test set ({n_pos + n_neg} peptides: "
      f"{n_pos} positive, {n_neg} negative)")
print("=" * 100)
print(test_table.to_string(index=False))

# ---------------------------------------------------------------------------
# Raw error counts -- what the metric differences actually rest on
# ---------------------------------------------------------------------------
err_rows = []
for r in runs:
    y = np.array(r["y_test"])
    prob = np.array(r["probabilities"])
    if prob.min() >= 0.0 and prob.max() <= 1.0:
        pred = (prob >= 0.5).astype(int)        # a real probability
    else:
        # The threshold rule stores negated length as its score.
        pred = (prob > -r.get("threshold", 11.0)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    err_rows.append({"Method": r["method"], "n_features": r["n_features"],
                     "TP": tp, "FN": fn, "FP": fp, "TN": tn})
err = pd.DataFrame(err_rows)
err.to_csv(RESULTS_DIR / "error_counts.csv", index=False)

print("\n" + "=" * 100)
print("TEST-SET ERROR COUNTS")
print("=" * 100)
print(err.to_string(index=False))

methods_only = err[~err.Method.isin(CONTROLS)]
if len(methods_only) > 1 and methods_only["TP"].nunique() == 1:
    tp = int(methods_only["TP"].iloc[0])
    fn = int(methods_only["FN"].iloc[0])
    lo, hi = int(methods_only["FP"].min()), int(methods_only["FP"].max())
    print(f"\n  Every method recovers the SAME {tp} of {n_pos} positives.")
    print(f"  The entire spread in test MCC therefore comes from false positives,")
    print(f"  which range from {lo} to {hi} out of {n_neg} negatives — a difference of")
    print(f"  {hi - lo} peptides. This is why methods are ranked on cross-validation")
    print(f"  rather than on the test set.")
    print(f"\n  {fn} positives are missed by EVERY method regardless of feature subset.")
    print(f"  Those reflect a limit of the feature space, not of selection.")

# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
ranked = [(r["method"], r["cv_mean"]["MCC"], r["cv_std"]["MCC"])
          for r in runs if r.get("cv_mean") and r["method"] not in CONTROLS]
ranked.sort(key=lambda t: -t[1])

if ranked:
    best, best_mcc, best_sd = ranked[0]
    print("\n" + "=" * 100)
    print("RANKING by CV MCC (controls excluded)")
    print("=" * 100)
    for name, mu, sd in ranked:
        gap = best_mcc - mu
        flag = "" if name == best else ("  (within 1 SD of best)" if gap < best_sd else "")
        print(f"  {name:<24} {mu:.4f} ± {sd:.4f}{flag}")

    tied = [n for n, mu, _ in ranked if best_mcc - mu < best_sd]
    if len(tied) > 1:
        print(f"\n  NOTE: {len(tied)} methods sit within one standard deviation of the")
        print(f"  best ({best_sd:.3f}). A single winner is not supported by this")
        print(f"  spread — see 07_significance.py for the paired test.")

# ---------------------------------------------------------------------------
# Feature overlap -- selection methods only, baseline excluded (it takes all)
# ---------------------------------------------------------------------------
sel_runs = [r for r in runs if r["method"] not in CONTROLS and r["method"] != "Baseline"]
if sel_runs:
    all_feats = sorted({f for r in sel_runs for f in r["features"]})
    overlap = pd.DataFrame(
        {r["method"]: [f in r["features"] for f in all_feats] for r in sel_runs},
        index=all_feats,
    )
    overlap["n_methods"] = overlap[[r["method"] for r in sel_runs]].sum(axis=1)
    overlap["n_stable"] = [
        sum(f in r.get("stable_features", []) for r in sel_runs) for f in all_feats
    ]
    overlap = overlap.sort_values(["n_methods", "n_stable"], ascending=False)
    overlap.to_csv(RESULTS_DIR / "feature_overlap.csv")

    n_sel = len(sel_runs)
    consensus = overlap.index[overlap["n_methods"] == n_sel].tolist()
    stable_consensus = overlap.index[overlap["n_stable"] == n_sel].tolist()

    print("\n" + "=" * 100)
    print(f"CONSENSUS across {n_sel} selection methods")
    print("=" * 100)
    print(f"  in every final subset ({len(consensus)}): {consensus}")
    print(f"  AND stable in every CV fold ({len(stable_consensus)}): {stable_consensus}")
    print("\n  The second list is the stricter claim. Report both — the gap between")
    print("  them shows how much the selection churns from fold to fold.")

    baseline_run = next((r for r in runs if r["method"] == "Baseline"), None)
    if baseline_run:
        never = sorted(set(baseline_run["features"]) - set(all_feats))
        print(f"\n  Never selected by any method ({len(never)}): {never}")
        print("  Caution: absence here can mean redundancy rather than irrelevance —")
        print("  a feature correlated with a selected one is dropped by design.")

# ---------------------------------------------------------------------------
# LaTeX -- controls below a rule
# ---------------------------------------------------------------------------
SHORT_METRICS = ["Accuracy", "Sensitivity", "Specificity", "MCC", "ROC-AUC"]


def latex_block(caption, cells_fn):
    lines = [f"% {caption}",
             "\\begin{tabular}{l r " + "r" * len(SHORT_METRICS) + "}", "\\hline",
             "Method & $n$ & " + " & ".join(SHORT_METRICS) + " \\\\", "\\hline"]
    body, ctrl = [], []
    for r in runs:
        cells = cells_fn(r)
        if cells is None:
            continue
        line = f"{label_of(r)} & {r['n_features']} & " + " & ".join(cells) + " \\\\"
        (ctrl if r["method"] in CONTROLS else body).append(line)
    lines += body
    if ctrl:
        lines.append("\\hline")
        lines += ctrl
    lines += ["\\hline", "\\end{tabular}", ""]
    return lines


latex = []
latex += latex_block(
    "Table 1: nested cross-validation on the training set (controls below rule)",
    lambda r: ([f"{r['cv_mean'][m]:.3f} $\\pm$ {r['cv_std'][m]:.3f}"
                for m in SHORT_METRICS] if r.get("cv_mean") else None))
latex += latex_block(
    "Table 2: independent test set (controls below rule)",
    lambda r: [f"{r['test'][m]:.3f}" for m in SHORT_METRICS])
(RESULTS_DIR / "table_latex.txt").write_text("\n".join(latex))

# ---------------------------------------------------------------------------
# Figure -- one colour per method across BOTH panels
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Left: ROC, drawn in AUC order for a readable legend but coloured by name.
for r in sorted(runs, key=lambda d: -d["test"]["ROC-AUC"]):
    name = r["method"]
    fpr, tpr, _ = roc_curve(np.array(r["y_test"]), np.array(r["probabilities"]))
    ax[0].plot(fpr, tpr,
               ls="--" if name in CONTROLS else "-",
               lw=2, color=COLOURS.get(name, "#333333"),
               label=f"{label_of(r)} ({r['test']['ROC-AUC']:.3f})")
ax[0].plot([0, 1], [0, 1], "k:", lw=1)
ax[0].set(xlabel="False positive rate", ylabel="True positive rate",
          title="ROC — independent test set")
ax[0].legend(loc="lower right", fontsize=8)

# Right: bars in ORDER, same colours, controls hatched.
idx, width = np.arange(len(SHORT_METRICS)), 0.8 / len(runs)
for i, r in enumerate(runs):
    name = r["method"]
    ax[1].bar(idx + i * width, [r["test"][m] for m in SHORT_METRICS], width,
              color=COLOURS.get(name, "#333333"),
              hatch="//" if name in CONTROLS else None,
              edgecolor="white", linewidth=0.5, label=label_of(r))
ax[1].set_xticks(idx + width * (len(runs) - 1) / 2)
ax[1].set_xticklabels(SHORT_METRICS, fontsize=9)
ax[1].set(ylim=(0, 1.05), title="Test-set metrics by method")
ax[1].legend(fontsize=8, ncol=2)

fig.tight_layout()
fig.savefig(RESULTS_DIR / "comparison.png", dpi=200)
print(f"\nWritten to {RESULTS_DIR.resolve()}")