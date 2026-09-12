"""
Dipeptide composition (DPC) -- 400 features.

For each of the 400 ordered residue pairs (AA, AC, ... YY), the fraction of the
peptide's adjacent pairs that it accounts for:

    DPC(i,j) = count(residue i followed by residue j) / (L - 1)

Where AAC records what residues are present, DPC records what follows what --
local sequence order. KWKW and WKWK have identical AAC but different DPC.

SPARSITY WARNING for this dataset. The positive peptides have a median length
of 13.5 residues, and a 13-mer contains only 12 dipeptides out of 400 possible.
The resulting matrix is roughly 97% zeros, and many columns will be zero for
every peptide in the training set. Those all-zero columns are dropped by
build_tiers.py (they carry no information and break correlation calculations),
so expect substantially fewer than 400 surviving features.

This sparsity is the point: it creates the high-dimensional, redundant, noisy
regime where mRMR / Elastic Net / RFE are designed to operate, and where
XGBoost on the full feature set should start to overfit.

Peptides of length < 2 have no dipeptides at all and yield an all-zero row.

Usage:
    python dpc_feature.py <input_fasta> <output_csv>
"""

import sys
from itertools import product

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
STANDARD = set(AMINO_ACIDS)
DIPEPTIDES = ["".join(p) for p in product(AMINO_ACIDS, repeat=2)]   # 400, fixed order


def read_fasta(file_path):
    """Returns [(id, sequence), ...] preserving file order."""
    records, name, buf = [], None, []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(buf)))
                name, buf = line[1:], []
            else:
                buf.append(line.upper())
    if name is not None:
        records.append((name, "".join(buf)))
    return records


def calc_dpc(seq):
    counts = dict.fromkeys(DIPEPTIDES, 0)
    n_pairs = 0
    for a, b in zip(seq, seq[1:]):
        if a in STANDARD and b in STANDARD:
            counts[a + b] += 1
            n_pairs += 1
    if n_pairs == 0:
        return [0.0] * len(DIPEPTIDES)
    return [counts[d] / n_pairs for d in DIPEPTIDES]


def extract_dpc(fasta_file, out_csv):
    records = read_fasta(fasta_file)

    bad = {c for _, s in records for c in s} - STANDARD
    if bad:
        print(f"WARNING: non-standard residues {sorted(bad)} -- pairs containing "
              f"them are excluded from both numerator and denominator")

    too_short = [n for n, s in records if len(s) < 2]
    if too_short:
        print(f"WARNING: {len(too_short)} sequences shorter than 2 residues have no "
              f"dipeptides and produce all-zero rows: {too_short[:5]}")

    with open(out_csv, "w") as f:
        f.write("ID," + ",".join(f"DPC_{d}" for d in DIPEPTIDES) + "\n")
        for name, seq in records:
            f.write(f"{name}," + ",".join(f"{x:.6f}" for x in calc_dpc(seq)) + "\n")

    print(f"Saved {len(records)} sequences x {len(DIPEPTIDES)} DPC features -> {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python dpc_feature.py <input_fasta> <output_csv>")
        sys.exit(1)
    extract_dpc(sys.argv[1], sys.argv[2])
