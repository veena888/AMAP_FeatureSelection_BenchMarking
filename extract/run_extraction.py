"""
Phase A runner -- extracts all five feature blocks for all four FASTA files.

Replaces the bash for-loop, which does not run in Windows Command Prompt or
PowerShell. Works identically on Windows, macOS and Linux.

Place this in extract/ alongside the extractor scripts, then:

    python run_extraction.py

Options:
    python run_extraction.py --data ../data --out blocks
    python run_extraction.py --skip pseaac          # drop a block
    python run_extraction.py --lambda 3             # smaller PseAAC lambda
"""

import argparse
import subprocess
import sys
from pathlib import Path

SPLITS = ["TR_pos", "TR_neg", "TS_pos", "TS_neg"]

# block name -> (script, extra args)
BLOCKS = {
    "aac":    ("aac_feature_fixed.py", []),
    "core":   ("core_bio_feature_fixed.py", []),
    "ctd":    ("ctd_feature.py", []),
    "dpc":    ("dpc_feature.py", []),
    "pseaac": ("pseaac_feature.py", []),
}

EXPECTED_ROWS = {"TR_pos": 110, "TR_neg": 1708, "TS_pos": 28, "TS_neg": 427}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data", help="folder holding the FASTA files")
    ap.add_argument("--out", default="blocks", help="folder for the extracted CSVs")
    ap.add_argument("--suffix", default="_iAMAPSCM.txt", help="FASTA filename suffix")
    ap.add_argument("--skip", nargs="*", default=[], help="block names to skip")
    ap.add_argument("--lambda", dest="lam", type=int, default=None,
                    help="PseAAC lambda (default 5)")
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("tiers").mkdir(exist_ok=True)

    # Fail early on anything missing, rather than halfway through.
    missing_fasta = [s for s in SPLITS if not (data_dir / f"{s}{args.suffix}").exists()]
    if missing_fasta:
        print(f"ERROR: FASTA files not found in {data_dir.resolve()}:")
        for s in missing_fasta:
            print(f"  {s}{args.suffix}")
        print("\nPass the right folder with --data, e.g. --data ../data")
        sys.exit(1)

    blocks = {b: v for b, v in BLOCKS.items() if b not in args.skip}
    missing_scripts = [s for s, _ in blocks.values() if not Path(s).exists()]
    if missing_scripts:
        print(f"ERROR: extractor scripts not found in {Path.cwd()}:")
        for s in missing_scripts:
            print(f"  {s}")
        print("\nRun this from the extract/ folder.")
        sys.exit(1)

    total = len(SPLITS) * len(blocks)
    step, failures = 0, []

    for split in SPLITS:
        fasta = data_dir / f"{split}{args.suffix}"
        for block, (script, extra) in blocks.items():
            step += 1
            out_csv = out_dir / f"{split}_{block}.csv"
            cmd = [sys.executable, script, str(fasta), str(out_csv)] + list(extra)
            if block == "pseaac" and args.lam is not None:
                cmd.append(str(args.lam))

            print(f"\n[{step}/{total}] {split} -> {block}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            for line in result.stdout.splitlines():
                print(f"    {line}")
            if result.returncode != 0:
                failures.append((split, block))
                print(f"    FAILED (exit {result.returncode})")
                for line in result.stderr.strip().splitlines()[-6:]:
                    print(f"    {line}")

    # ---- verification ------------------------------------------------------
    print("\n" + "=" * 66)
    print("VERIFICATION")
    print("=" * 66)

    ok = True
    for split in SPLITS:
        counts = {}
        for block in blocks:
            path = out_dir / f"{split}_{block}.csv"
            if not path.exists():
                counts[block] = None
                continue
            with open(path) as f:
                counts[block] = sum(1 for _ in f) - 1        # minus header

        expected = EXPECTED_ROWS.get(split)
        values = [c for c in counts.values() if c is not None]
        consistent = len(set(values)) <= 1
        matches = (expected is None) or (values and values[0] == expected)

        status = "OK" if (consistent and matches) else "MISMATCH"
        if status != "OK":
            ok = False
        detail = ", ".join(f"{b}={c}" for b, c in counts.items())
        print(f"  {split:<8} {status:<9} rows: {detail}"
              + (f"   (expected {expected})" if expected else ""))

    if failures:
        ok = False
        print(f"\n  {len(failures)} extraction(s) failed: "
              + ", ".join(f"{s}/{b}" for s, b in failures))

    print()
    if ok:
        print("All blocks extracted and row counts consistent.")
        print("Next:  python build_tiers.py blocks/ tiers/")
    else:
        print("Fix the issues above before running build_tiers.py.")
        sys.exit(1)


if __name__ == "__main__":
    main()