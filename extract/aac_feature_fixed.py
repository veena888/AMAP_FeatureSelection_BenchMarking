"""
Amino acid composition (AAC) -- corrected version.

Changes from the original aac_feature.py:

  1. Columns are named by the residue they represent (AAC_A ... AAC_Y) rather
     than AAC_1 ... AAC_20. The original numbering meant that "AAC_16 is the
     strongest single predictor" told you nothing until you counted letters in
     the alphabet string. It is serine.

  2. Sequence IDs are carried through, so a downstream merge cannot silently
     misalign the AAC block against the core-features block.

  3. Non-standard residues are reported. AAC denominators use full sequence
     length, so an unreported X or B quietly deflates every fraction.

Usage:
    python aac_feature_fixed.py <input_fasta> <output_csv>
"""

import sys

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
STANDARD = set(AMINO_ACIDS)


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


def calc_aac(seq):
    L = len(seq)
    if L == 0:
        return [0.0] * len(AMINO_ACIDS)
    return [seq.count(aa) / L for aa in AMINO_ACIDS]


def extract_aac(fasta_file, out_csv):
    records = read_fasta(fasta_file)

    bad = {c for _, s in records for c in s} - STANDARD
    if bad:
        print(f"WARNING: non-standard residues present {sorted(bad)} -- "
              f"counted in the length denominator but not in any numerator")

    with open(out_csv, "w") as f:
        f.write("ID," + ",".join(f"AAC_{aa}" for aa in AMINO_ACIDS) + "\n")
        for name, seq in records:
            f.write(f"{name}," + ",".join(f"{x:.6f}" for x in calc_aac(seq)) + "\n")

    print(f"Saved {len(records)} sequences x {len(AMINO_ACIDS)} AAC features -> {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python aac_feature_fixed.py <input_fasta> <output_csv>")
        sys.exit(1)
    extract_aac(sys.argv[1], sys.argv[2])
