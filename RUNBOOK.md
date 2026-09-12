# Execution runbook — iAMAP-SCM feature selection study

Follow phases in order. Each step lists the exact command, what it writes, and
what to check before moving on.

---

## Phase 0 — Setup (once)

```bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn biopython scipy matplotlib
```

Directory layout:

```
project/
├── data/                          the four FASTA files
├── extract/                       extraction scripts + outputs
│   └── blocks/  tiers/            created below
└── analysis/                      selection + evaluation scripts
```

Place files:

| Into `extract/` | Into `analysis/` |
|---|---|
| `aac_feature_fixed.py` | `common.py` |
| `core_bio_feature_fixed.py` | `selectors.py` |
| `ctd_feature.py` | `00_length_baseline.py` |
| `dpc_feature.py` | `01_baseline.py` |
| `pseaac_feature.py` | `02_mrmr.py` |
| `build_tiers.py` | `03_mutual_info.py` |
| | `04_elastic_net.py` |
| | `05_rfe.py` |
| | `06_compare.py` |
| | `07_significance.py` |
| | `08_k_sweep.py` |

`merge_features_fixed.py` is **not needed** — `build_tiers.py` replaces it.

Everything in `analysis/` must stay in one folder; `selectors.py` and scripts
`00`–`08` all import from `common.py`.

---

## Phase A — Extract feature blocks

```bash
cd extract
mkdir -p blocks tiers

for s in TR_pos TR_neg TS_pos TS_neg; do
  python aac_feature_fixed.py      ../data/${s}_iAMAPSCM.txt  blocks/${s}_aac.csv
  python core_bio_feature_fixed.py ../data/${s}_iAMAPSCM.txt  blocks/${s}_core.csv
  python ctd_feature.py            ../data/${s}_iAMAPSCM.txt  blocks/${s}_ctd.csv
  python dpc_feature.py            ../data/${s}_iAMAPSCM.txt  blocks/${s}_dpc.csv
  python pseaac_feature.py         ../data/${s}_iAMAPSCM.txt  blocks/${s}_pseaac.csv
done
```

Windows (no bash): run the five commands four times, substituting `TR_pos`,
`TR_neg`, `TS_pos`, `TS_neg`.

**Writes:** 20 files in `blocks/`.

**Check before continuing**

- Row counts must be 110 / 1708 / 28 / 427 for TR_pos / TR_neg / TS_pos / TS_neg.
- Every file has an `ID` column and IDs match across blocks for the same split.
- Expected warnings — these are informational, not errors:
  - `core`: sequences shorter than 5 residues (23 in TR_pos, 8 in TS_pos).
  - `dpc`: sequences shorter than 2 residues produce all-zero rows.
  - `pseaac`: sequences ≤ λ have undefined correlation factors.

**Decision point.** If the `pseaac` warning reports more than 25% of TR_pos
affected, drop tier 4 — with λ = 5 and a median positive length of 13.5, that
block may carry almost nothing. Either lower λ (`python pseaac_feature.py ...
../data/X.txt blocks/X_pseaac.csv 3`) or skip it and ignore `tier4_pse` below.

---

## Phase B — Build tiers

```bash
python build_tiers.py blocks/ tiers/
```

**Writes:** `tiers/tierN_*_train.csv`, `tiers/tierN_*_test.csv`,
`tiers/tierN_*_collinearity.csv`, `tiers/tier_summary.csv`.

**Check**

- `tier1_27_train.csv` has 27 features and 1818 rows. This must match the
  earlier `train_fixed.csv` — it is the same feature set.
- Tier 3 will report many all-zero DPC columns dropped. Expect roughly 250–350
  surviving features rather than the nominal 574; that is correct behaviour on
  short peptides, not a bug.
- Skim `tier1_27_collinearity.csv`. It should be nearly empty — the known
  duplicates were already removed. If `GRAVY ~ BomanIndex` appears at r > 0.999,
  you are still running the old `core_bio_feature.py`.

Keep the `collinearity.csv` files. They are your audit trail if a reviewer asks
how duplicates were handled.

---

## Phase C — Main results, tier 1

```bash
cd ../analysis
cp ../extract/tiers/tier1_27_train.csv .
cp ../extract/tiers/tier1_27_test.csv .
```

Edit `common.py`:

```python
TRAIN_CSV = "tier1_27_train.csv"
TEST_CSV  = "tier1_27_test.csv"
RESULTS_DIR = Path("results_tier1")     # per-tier, so runs do not overwrite
K_FEATURES = 15
```

Then, in order:

```bash
python 00_length_baseline.py     # length control — run first
python 01_baseline.py            # no selection
python 02_mrmr.py
python 03_mutual_info.py
python 04_elastic_net.py         # slowest, ~5 min
python 05_rfe.py
python 06_compare.py             # tables + LaTeX + ROC figure
python 07_significance.py        # paired Wilcoxon, Holm-corrected
```

Scripts `00`–`05` are independent and order does not matter; `06` and `07` must
come after them because they read the JSON the others write.

**Writes:** one JSON per method, then `table_cv.csv`, `table_test.csv`,
`table_latex.txt`, `feature_overlap.csv`, `comparison.png`, `significance.csv`.

**Check**

- `06_compare.py` prints a NOTE when methods sit within one standard deviation.
  Expect that note at tier 1.
- `07_significance.py` reports each method against the baseline. This is the
  paragraph your results section is built on.

---

## Phase D — k-sweep, tier 1

First pass, fast. In `common.py` set `CV_REPEATS = 2`, then:

```bash
python 08_k_sweep.py tier1_27_train.csv tier1_27_test.csv
```

Look at `results_tier1/k_sweep_tier1_27.png`. **This is the decision point for
the whole study:**

- **Curves separate at small k** — you have a real finding at 27 features. The
  paper can be built on the degradation curves alone.
- **Curves overlap everywhere** — selection genuinely has nothing to work with
  here. Proceed to Phase E; the wide tiers are where your contribution lies.

Then restore `CV_REPEATS = 5` and rerun for the publication figure.

---

## Phase E — Wider tiers

For each of `tier2_ctd`, `tier3_dpc`, and optionally `tier4_pse`:

1. Copy the two CSVs into `analysis/`.
2. In `common.py` update `TRAIN_CSV`, `TEST_CSV`, and `RESULTS_DIR`
   (e.g. `results_tier2`).
3. Set `K_FEATURES` to roughly 25% of that tier's feature count.
4. Rerun `00`–`08`.

**RESULTS_DIR must change per tier.** The scripts write fixed filenames, so
leaving it as `results/` silently overwrites the previous tier.

Runtime grows with feature count. Tier 3 will take substantially longer —
Elastic Net's inner grid search and RFE dominate. Run tier 2 first to calibrate.

---

## Phase F — Assemble the paper

Four results across tiers:

1. **Main table** — `table_cv.csv` per tier. Rank on CV MCC, not the test set.
2. **Significance** — `significance.csv`. Report each method vs. the baseline.
3. **Degradation curves** — `k_sweep_*.png`. The most informative figure.
4. **Cross-tier comparison** — best method per tier vs. the baseline at that
   tier. This is the study's actual claim: *when* does selection start to pay?

Points to state explicitly in the write-up:

- Selection was fitted on training folds only; the test set was never involved.
- Sensitivity and Recall are the same quantity; report one, labelled
  "Sensitivity/Recall".
- The length-only control, and why it is reported as a method rather than a
  footnote (negatives are ≥ 11 residues, positives go to 1).
- The corrected Boman index, with the reference validation value.
- Identical XGBoost hyperparameters across every method and tier.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: common` | script not run from `analysis/` |
| `Missing blocks/TR_pos_ctd.csv` | Phase A incomplete for that block |
| `ID mismatch` in `build_tiers.py` | blocks generated from different FASTA versions |
| Tier 3 far fewer features than 574 | expected — all-zero DPC columns dropped |
| `04_elastic_net.py` selects everything | C grid saturated; raise `EN_C_GRID` upper values |
| Results differ from an earlier run | check `SEED = 42` and library versions |
| Tier N overwrote tier N−1 | `RESULTS_DIR` not changed in `common.py` |
