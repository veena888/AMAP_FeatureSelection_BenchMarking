"""
Core biophysical features for peptide sequences -- corrected version.

Changes from the original core_bio_feature.py:

  1. BomanIndex is now the actual Boman index. The original summed
     ProtParamData.kd (Kyte-Doolittle hydropathy), which is the definition of
     GRAVY -- so the two columns came out numerically identical (r = 1.000000)
     and one of the 13 features was a duplicate. The real index uses the
     Radzicka & Wolfenden (1988) cyclohexane-to-water transfer free energies,
     as proposed by Boman (2003). Validated against the R Peptides package:
     boman("FLPVLAGLTPSIVPKLVCLLTKKC") = -1.235833.

  2. Sequences are validated: non-standard residues (B, J, O, U, X, Z, '-')
     are reported rather than silently distorting the composition.

  3. Output carries a header row and the sequence ID, so downstream merges
     cannot silently misalign rows.

Usage:
    python core_bio_feature_fixed.py <input_fasta> <output_csv> [pH]
"""

import sys

from Bio.SeqUtils.ProtParam import ProteinAnalysis

HYDROPHOBIC = set("AVILMFWY")
STANDARD = set("ACDEFGHIKLMNPQRSTVWY")

# Radzicka & Wolfenden (1988) side-chain transfer free energies, kcal/mol.
# Boman index = sum over residues / length. Values above 2.48 indicate high
# protein-binding potential; low or negative values indicate membrane
# selectivity, which is the AMP-typical regime.
BOMAN_SCALE = {
    "L": -4.92, "I": -4.92, "V": -4.04, "F": -2.98, "M": -2.35, "W": -2.33,
    "A": -1.81, "C": -1.28, "G": -0.94, "Y": 0.14, "P": 0.00, "T": 2.57,
    "S": 3.40, "H": 4.66, "Q": 5.54, "K": 5.55, "N": 6.64, "E": 6.81,
    "D": 8.72, "R": 14.92,
}

COLUMNS = [
    "Length", "NetCharge", "pI",
    "K_frac", "R_frac", "H_frac", "D_frac", "E_frac",
    "GRAVY", "HydrophobicFrac", "AromaticFrac", "BomanIndex",
]


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


def boman_index(seq):
    """Boman (2003) potential protein interaction index, kcal/mol per residue."""
    if not seq:
        return 0.0
    return sum(BOMAN_SCALE[aa] for aa in seq if aa in BOMAN_SCALE) / len(seq)


def calc_core_features(seq, ph=7.4):
    if len(seq) == 0:
        return [0.0] * len(COLUMNS)

    analysis = ProteinAnalysis(seq)
    length = len(seq)

    return [
        length,
        analysis.charge_at_pH(ph),
        analysis.isoelectric_point(),
        seq.count("K") / length,
        seq.count("R") / length,
        seq.count("H") / length,
        seq.count("D") / length,
        seq.count("E") / length,
        analysis.gravy(),
        sum(seq.count(aa) for aa in HYDROPHOBIC) / length,
        analysis.aromaticity(),
        boman_index(seq),                      # no longer a copy of GRAVY
    ]


def extract_core_features(fasta_file, out_csv, ph=7.4):
    records = read_fasta(fasta_file)

    bad = {c for _, s in records for c in s} - STANDARD
    if bad:
        print(f"WARNING: non-standard residues present {sorted(bad)} -- "
              f"these are excluded from composition sums and will skew fractions")

    short = [n for n, s in records if len(s) < 5]
    if short:
        print(f"WARNING: {len(short)} sequences shorter than 5 residues. "
              f"pI and net charge are dominated by terminal groups at this length "
              f"and are not meaningful. First few: {short[:5]}")

    with open(out_csv, "w") as f:
        f.write("ID," + ",".join(COLUMNS) + "\n")
        for name, seq in records:
            feats = calc_core_features(seq, ph)
            f.write(f"{name}," + ",".join(f"{x:.6f}" for x in feats) + "\n")

    print(f"Saved {len(records)} sequences x {len(COLUMNS)} core features -> {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python core_bio_feature_fixed.py <input_fasta> <output_csv> [pH]")
        sys.exit(1)
    ph = float(sys.argv[3]) if len(sys.argv) > 3 else 7.4
    extract_core_features(sys.argv[1], sys.argv[2], ph)
