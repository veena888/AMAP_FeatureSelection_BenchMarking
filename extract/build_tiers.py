"""
Assemble the four feature tiers, with an automatic collinearity audit.

Tiers
    tier1   27       existing AAC + core biophysical  (low-dimensional anchor)
    tier2   ~174     + CTD                            (moderate)
    tier3   ~574     + DPC                            (high-dimensional, sparse)
    tier4   ~579     + PseAAC lambda block            (optional)

The point of the tiers is not that tier 4 is "best" -- it is that comparing
feature selection across them answers WHEN selection starts to matter, which is
a stronger claim than any single-tier result. Tier 1 is the control: selection
provides no benefit there. Keep it.

Expected inputs, produced by the extractor scripts, one per split and class:

    TR_pos_aac.csv    TR_pos_core.csv    TR_pos_ctd.csv    TR_pos_dpc.csv    TR_pos_pseaac.csv
    TR_neg_*.csv      TS_pos_*.csv       TS_neg_*.csv

Every file must carry an ID column. Blocks are joined on ID, never on row order.

The audit
    Columns are dropped when they are:
      - all-zero across the training set (no information; unavoidable with DPC
        on short peptides)
      - constant across the training set
      - correlated at |r| > 0.999 with an earlier column (exact duplication)

    Columns correlated at |r| > 0.95 are REPORTED BUT KEPT. That distinction
    matters for this study: redundancy is the phenomenon under investigation and
    mRMR exists precisely to handle it. Only exact duplication has to go, because
    it makes Elastic Net's coefficient split arbitrary between the pair and
    distorts mRMR's redundancy term.

    Drop decisions are made on the TRAINING set only and then applied to test.

Usage:
    python build_tiers.py <input_dir> <output_dir>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BLOCKS = {"aac": "aac", "core": "core", "ctd": "ctd", "dpc": "dpc", "pseaac": "pseaac"}
TIERS = {
    "tier1_27":   ["aac", "core"],
    "tier2_ctd":  ["aac", "core", "ctd"],
    "tier3_dpc":  ["aac", "core", "ctd", "dpc"],
    "tier4_pse":  ["aac", "core", "ctd", "dpc", "pseaac"],
}

# AAC_K == K_frac, AAC_R == R_frac, and so on. Keep the core-feature version.
REDUNDANT_AAC = ["AAC_K", "AAC_R", "AAC_H", "AAC_D", "AAC_E"]

DUPLICATE_THRESHOLD = 0.999      # drop
REDUNDANT_THRESHOLD = 0.95       # report only


def load_block(indir, split, cls, block):
    path = Path(indir) / f"{split}_{cls}_{block}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    if "ID" not in df.columns:
        raise ValueError(f"{path} has no ID column. Regenerate with the fixed extractors.")
    return df


def assemble(indir, split, blocks):
    """Join all blocks for both classes of one split, on ID."""
    frames = []
    for cls, label in (("pos", 1), ("neg", 0)):
        merged = None
        for block in blocks:
            df = load_block(indir, split, cls, block)
            merged = df if merged is None else merged.merge(
                df, on="ID", how="inner", validate="one_to_one")
        if merged is None:
            raise ValueError("No blocks requested")
        merged.insert(1, "Label", label)
        frames.append(merged)
    out = pd.concat(frames, axis=0, ignore_index=True)

    drop = [c for c in REDUNDANT_AAC if c in out.columns]
    if drop:
        out = out.drop(columns=drop)
    return out


def audit(train_df, test_df, verbose=True):
    """Decide drops on TRAIN, apply to both. Returns cleaned frames and a report."""
    X = train_df.drop(columns=["ID", "Label"])

    all_zero = X.columns[(X == 0).all()].tolist()
    X = X.drop(columns=all_zero)

    constant = X.columns[X.nunique() <= 1].tolist()
    X = X.drop(columns=constant)

    # Exact duplication: walk the upper triangle once, keep the first of any pair.
    corr = X.corr().abs().to_numpy()
    cols = list(X.columns)
    duplicates, redundant_pairs = [], []
    dropped = set()
    for i in range(len(cols)):
        if cols[i] in dropped:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] in dropped:
                continue
            r = corr[i, j]
            if np.isnan(r):
                continue
            if r > DUPLICATE_THRESHOLD:
                duplicates.append((cols[i], cols[j], r))
                dropped.add(cols[j])
            elif r > REDUNDANT_THRESHOLD:
                redundant_pairs.append((cols[i], cols[j], r))

    keep = [c for c in cols if c not in dropped]

    if verbose:
        print(f"    all-zero dropped : {len(all_zero)}")
        print(f"    constant dropped : {len(constant)}")
        print(f"    duplicates dropped (|r| > {DUPLICATE_THRESHOLD}): {len(duplicates)}")
        for a, b, r in duplicates[:10]:
            print(f"        {b} == {a}  (r={r:.6f})")
        if len(duplicates) > 10:
            print(f"        ... and {len(duplicates) - 10} more")
        print(f"    redundant pairs KEPT (|r| > {REDUNDANT_THRESHOLD}): "
              f"{len(redundant_pairs)}")
        for a, b, r in redundant_pairs[:10]:
            print(f"        {a} ~ {b}  (r={r:.4f})")
        if len(redundant_pairs) > 10:
            print(f"        ... and {len(redundant_pairs) - 10} more")

    train_out = train_df[["ID", "Label"] + keep]
    missing = [c for c in keep if c not in test_df.columns]
    if missing:
        raise ValueError(f"Test set missing columns kept in train: {missing[:5]}")
    test_out = test_df[["ID", "Label"] + keep]

    report = pd.DataFrame(
        [{"kind": "duplicate_dropped", "a": a, "b": b, "r": r} for a, b, r in duplicates]
        + [{"kind": "redundant_kept", "a": a, "b": b, "r": r} for a, b, r in redundant_pairs]
    )
    return train_out, test_out, report


def main(indir, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = []
    for tier, blocks in TIERS.items():
        print(f"\n=== {tier}  ({' + '.join(blocks)}) ===")
        try:
            train = assemble(indir, "TR", blocks)
            test = assemble(indir, "TS", blocks)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue

        print(f"  raw: train {train.shape[0]} x {train.shape[1] - 2} features")
        train_c, test_c, report = audit(train, test)
        n_feat = train_c.shape[1] - 2
        print(f"  final: {n_feat} features")

        train_c.to_csv(outdir / f"{tier}_train.csv", index=False)
        test_c.to_csv(outdir / f"{tier}_test.csv", index=False)
        if not report.empty:
            report.to_csv(outdir / f"{tier}_collinearity.csv", index=False)

        summary.append({"tier": tier, "blocks": " + ".join(blocks),
                        "n_features": n_feat,
                        "train_rows": train_c.shape[0], "test_rows": test_c.shape[0]})

    if summary:
        s = pd.DataFrame(summary)
        s.to_csv(outdir / "tier_summary.csv", index=False)
        print("\n" + "=" * 70)
        print(s.to_string(index=False))
        print(f"\nWritten to {outdir.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python build_tiers.py <input_dir> <output_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
