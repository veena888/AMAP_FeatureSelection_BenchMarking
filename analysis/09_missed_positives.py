"""
Identify the test positives that EVERY method misses at EVERY tier.

Across all three tiers, roughly six of the 28 test positives were never
recovered by any method regardless of feature subset. Those are a limit of the
feature space itself, not of feature selection -- but "six peptides" is a vague
limitation. This script names them, measures them, and checks whether they are
the ultra-short ones, which converts it into a concrete, stateable finding.

How the row alignment works
---------------------------
`run_method()` stores `probabilities` and `y_test` in each results JSON, in the
order the test CSV was read. `build_tiers.py` writes positives first, then
negatives, and `load_data()` preserves row order throughout, so index i in the
probability array is row i in the test CSV. The ID column comes along with it.

Controls are skipped: `length_rule.json` stores negated peptide lengths rather
than probabilities, and `length_only.json` is a one-feature control, not a
method under comparison.

Usage:
    python 09_missed_positives.py
    python 09_missed_positives.py --fasta ../data/TS_pos_iAMAPSCM.txt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONTROLS = {"Length rule (no model)", "Length only"}
DEFAULT_TIERS = {
    "tier1": ("results_tier1", "../extract/tiers/tier1_27_test.csv"),
    "tier2": ("results_tier2", "../extract/tiers/tier2_ctd_test.csv"),
    "tier3": ("results_tier3", "../extract/tiers/tier3_dpc_test.csv"),
}


def read_fasta(path):
    """Returns {id: sequence}."""
    seqs, name, buf = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name, buf = line[1:], []
            else:
                buf.append(line.upper())
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def load_tier(results_dir, test_csv):
    """Returns (ids, y_test, {method: predictions}) for one tier."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return None

    test = pd.read_csv(test_csv)
    ids = test["ID"].to_numpy()
    y_csv = test["Label"].astype(int).to_numpy()

    preds = {}
    y_ref = None
    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        method = payload.get("method")
        if method in CONTROLS or "probabilities" not in payload:
            continue

        prob = np.array(payload["probabilities"])
        y = np.array(payload["y_test"])

        if len(prob) != len(ids):
            print(f"  WARNING: {path.name} has {len(prob)} rows, "
                  f"test CSV has {len(ids)} — skipped")
            continue
        if not np.array_equal(y, y_csv):
            print(f"  WARNING: {path.name} label order differs from {test_csv} "
                  f"— skipped (row alignment cannot be trusted)")
            continue

        preds[method] = (prob >= 0.5).astype(int)
        y_ref = y

    return (ids, y_ref, preds) if preds else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default="../data/TS_pos_iAMAPSCM.txt",
                    help="positive test FASTA, for sequences")
    ap.add_argument("--out", default="missed_positives.csv")
    args = ap.parse_args()

    # ---- gather predictions from every tier --------------------------------
    tiers, ids_ref, y_ref = {}, None, None
    for tier, (results_dir, test_csv) in DEFAULT_TIERS.items():
        print(f"Loading {tier} from {results_dir}/ ...")
        loaded = load_tier(results_dir, test_csv)
        if loaded is None:
            print(f"  no usable results — skipped")
            continue
        ids, y, preds = loaded
        print(f"  {len(preds)} methods: {sorted(preds)}")

        if ids_ref is None:
            ids_ref, y_ref = ids, y
        elif not np.array_equal(ids, ids_ref):
            print(f"  WARNING: {tier} test IDs differ from the first tier — skipped")
            continue
        tiers[tier] = preds

    if not tiers:
        raise SystemExit("No tier results found. Run 00–05 for at least one tier.")

    pos_idx = np.where(y_ref == 1)[0]
    print(f"\n{len(pos_idx)} test positives, "
          f"{sum(len(p) for p in tiers.values())} method-tier combinations\n")

    # ---- how many methods found each positive? -----------------------------
    rows = []
    for i in pos_idx:
        row = {"ID": ids_ref[i]}
        total_found = 0
        total_methods = 0
        for tier, preds in tiers.items():
            found = sum(int(p[i]) for p in preds.values())
            row[f"{tier}_found_by"] = f"{found}/{len(preds)}"
            total_found += found
            total_methods += len(preds)
        row["found_total"] = total_found
        row["of"] = total_methods
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("found_total")

    # ---- attach sequence and length ----------------------------------------
    fasta_path = Path(args.fasta)
    if fasta_path.exists():
        seqs = read_fasta(fasta_path)
        df["Length"] = df["ID"].map(lambda x: len(seqs.get(x, "")) or np.nan)
        df["Sequence"] = df["ID"].map(lambda x: seqs.get(x, ""))
    else:
        print(f"NOTE: {fasta_path} not found — pass --fasta for sequences.\n")

    df.to_csv(args.out, index=False)

    # ---- report -------------------------------------------------------------
    never = df[df["found_total"] == 0]
    always = df[df["found_total"] == df["of"]]

    print("=" * 78)
    print("ALL TEST POSITIVES, ordered by how often they were recovered")
    print("=" * 78)
    print(df.to_string(index=False))

    print("\n" + "=" * 78)
    print(f"NEVER recovered by any method at any tier: {len(never)}")
    print("=" * 78)
    if never.empty:
        print("  none")
    else:
        for _, r in never.iterrows():
            extra = ""
            if "Length" in df.columns and not pd.isna(r.get("Length")):
                extra = f"  length={int(r['Length'])}  {r['Sequence']}"
            print(f"  {r['ID']}{extra}")

    if "Length" in df.columns and not df["Length"].isna().all():
        print("\n" + "=" * 78)
        print("LENGTH vs RECOVERABILITY — the question this script exists to answer")
        print("=" * 78)
        print(f"  never recovered : median length "
              f"{never['Length'].median():.1f}  (n={len(never)}, "
              f"range {never['Length'].min():.0f}–{never['Length'].max():.0f})")
        rest = df[df["found_total"] > 0]
        print(f"  recovered ≥ once: median length "
              f"{rest['Length'].median():.1f}  (n={len(rest)}, "
              f"range {rest['Length'].min():.0f}–{rest['Length'].max():.0f})")
        print(f"  always recovered: n={len(always)}")

        short = never[never["Length"] < 5]
        print(f"\n  Of the {len(never)} never recovered, {len(short)} are under 5 "
              f"residues.")
        if len(never) and len(short) / len(never) >= 0.5:
            print("  => The limitation is largely a LENGTH limitation. State it as:")
            print("     'the peptides no feature space could represent are")
            print("      predominantly those too short to have meaningful")
            print("      composition or sequence-order descriptors.'")
        else:
            print("  => NOT primarily a length effect. These are genuinely hard")
            print("     peptides — inspect the sequences before claiming a cause.")

    print(f"\nWritten to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
