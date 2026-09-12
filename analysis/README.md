# Feature selection comparison — iAMAP-SCM

Comparison of five feature-selection strategies on a fixed SMOTE + XGBoost
classifier, using the iAMAP-SCM antimalarial peptide benchmark
(Charoenkwan et al., *ACS Omega* 2022, 7(45):41082).

## Running

```bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn biopython scipy matplotlib

python 00_length_baseline.py     # control: length threshold + length-only model
python 01_baseline.py            # all 27 features, no selection
python 02_mrmr.py
python 03_mutual_info.py
python 04_elastic_net.py         # slowest, ~5 min
python 05_rfe.py
python 06_compare.py             # tables, LaTeX, figure
python 07_significance.py        # paired Wilcoxon + corrected t-test
```

Configuration lives in `common.py`: `TRAIN_CSV`, `TEST_CSV`, `K_FEATURES`,
`XGB_PARAMS`, `CV_SPLITS`, `CV_REPEATS`. Scripts are independent and can be run
in any order; each writes `results/<method>.json`.

## Design

**Selection is fitted on training rows only.** Each `select_fn(X, y, k)` is
called once per CV fold on that fold's training rows, and once on the full
training set for the final model. The test set is only ever indexed by the
resulting column names. Scaling (Elastic Net) and imputation medians are
likewise train-only.

**Two numbers per method.** Nested CV (5×5, selection re-fitted per fold) is the
estimate to rank methods on. The independent test set is a single confirmation —
with 28 positives, one flipped prediction moves sensitivity by 3.6 points, so it
cannot separate methods that differ by hundredths of an MCC.

**SMOTE is inside an imblearn Pipeline**, so it resamples training folds only and
never a validation fold. `scale_pos_weight` is deliberately unset: combined with
SMOTE it corrects the 1:15.5 imbalance twice and inflates sensitivity at
specificity's expense.

**Hyperparameters are identical across all methods.** The only thing that varies
is the feature subset.

## Results

Dataset: 1818 training peptides (110 positive, 1708 negative, 1:15.5);
455 test peptides (28 positive, 427 negative). 27 features after removing
duplicated columns.

### Nested CV on the training set (5×5, selection re-fitted per fold)

| Method | n | Accuracy | Sensitivity | Specificity | MCC | ROC-AUC |
|---|---|---|---|---|---|---|
| Elastic Net | 14 | 0.970 ± 0.007 | 0.693 ± 0.101 | 0.988 ± 0.004 | **0.720 ± 0.069** | 0.894 ± 0.047 |
| Baseline (no FS) | 27 | 0.970 ± 0.007 | 0.687 ± 0.096 | 0.988 ± 0.005 | 0.720 ± 0.068 | 0.897 ± 0.040 |
| mRMR | 15 | 0.967 ± 0.007 | 0.698 ± 0.099 | 0.985 ± 0.005 | 0.705 ± 0.070 | 0.901 ± 0.041 |
| Mutual Information | 15 | 0.967 ± 0.008 | 0.689 ± 0.103 | 0.985 ± 0.006 | 0.698 ± 0.075 | 0.898 ± 0.046 |
| RFE | 15 | 0.966 ± 0.007 | 0.689 ± 0.096 | 0.984 ± 0.006 | 0.692 ± 0.070 | 0.894 ± 0.042 |
| *Length only (control)* | 1 | 0.876 ± 0.023 | 0.587 ± 0.107 | 0.895 ± 0.026 | 0.341 ± 0.067 | 0.792 ± 0.060 |

### Independent test set

| Method | n | Accuracy | Sensitivity | Specificity | MCC | ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline (no FS) | 27 | 0.978 | 0.821 | 0.988 | **0.810** | 0.905 |
| RFE | 15 | 0.974 | 0.821 | 0.984 | 0.780 | 0.911 |
| mRMR | 15 | 0.974 | 0.786 | 0.986 | 0.772 | 0.911 |
| Mutual Information | 15 | 0.974 | 0.786 | 0.986 | 0.772 | 0.911 |
| Elastic Net | 14 | 0.965 | 0.786 | 0.977 | 0.716 | 0.912 |
| *Length rule, no model* | 1 | 0.969 | 0.500 | 1.000 | 0.696 | 0.772 |
| *Length only (model)* | 1 | 0.897 | 0.679 | 0.911 | 0.428 | 0.837 |

### Significance (paired Wilcoxon over the 25 shared folds, Holm-corrected)

Against the no-selection baseline:

| Method | ΔMCC | p | Verdict |
|---|---|---|---|
| Elastic Net | +0.0001 | 0.961 | no significant difference |
| mRMR | −0.0148 | 0.081 | no significant difference |
| Mutual Information | −0.0219 | 0.072 | no significant difference |
| RFE | −0.0275 | 0.008 | significantly **worse** |

Of 15 pairwise comparisons, 6 reach significance after Holm correction — and
five of those six are a selection method beating the length-only control.
The single remaining one is Elastic Net over RFE (Δ = +0.028, p = 0.045), which
does not survive the Nadeau–Bengio corrected t-test (p = 1.00 after correction).

## What this supports

1. **No selection method significantly outperforms using all 27 features.** With
   only 27 features to begin with, there is too little redundancy for selection
   to recover. This is a legitimate negative result and worth stating plainly.
2. **RFE is significantly worse than the baseline**, the only method-vs-baseline
   difference that reaches significance.
3. **Elastic Net matches the baseline with 14 features instead of 27** — a
   roughly 50% reduction at no measurable cost. If the argument is parsimony
   rather than accuracy, this is the result to lead with.
4. **Nine features are selected by all five methods:** `AAC_A`, `AAC_F`,
   `AAC_G`, `AAC_S`, `AAC_I`, `AAC_V`, `AAC_T`, `pI`, `Length`. Consensus across
   methods is a more defensible biological claim than any single ranking.

## Caveats to state in the paper

**Length confound.** The negatives have a minimum length of 11 residues; the
positives go down to 1. Roughly half of all positives therefore fall in a length
region containing no negatives. A bare threshold (`Length < 11`, tuned on train)
reaches test accuracy 0.969 and MCC 0.696 with no model at all — above the
accuracy reported for iAMAP-SCM itself (0.957), though below its MCC (0.834).
This is a property of the benchmark, not of the pipeline, and it affects all
methods equally, so the *relative* comparison stands. But absolute performance
figures should be read against the length-only row, which is why it is reported
as a method rather than a footnote.

**Sensitivity and Recall are the same quantity** (TP/(TP+FN)). Both are computed
because the original protocol listed both; report one, labelled
"Sensitivity/Recall".

**Test set size.** 28 positives. Test-set differences between methods
(0.716–0.810 MCC) are within the noise of a handful of predictions and should
not be used to rank methods.

## Upstream fixes

Three bugs in the original extraction were corrected before this comparison
(see `../extraction_fixed/`):

1. `BomanIndex` summed `ProtParamData.kd`, the Kyte–Doolittle scale — which is
   the definition of GRAVY. The two columns were identical (r = 1.000000). Now
   uses the Radzicka & Wolfenden (1988) transfer free energies; validated
   against the R `Peptides` package (`boman("FLPVLAGLTPSIVPKLVCLLTKKC")` =
   −1.235833).
2. `K_frac`, `R_frac`, `H_frac`, `D_frac`, `E_frac` duplicated `AAC_9`, `AAC_15`,
   `AAC_7`, `AAC_3`, `AAC_4` exactly. Five collinear pairs removed; 32 → 27
   features.
3. AAC columns renamed from `AAC_1..AAC_20` to residue letters, and merges now
   join on sequence ID rather than row position.
