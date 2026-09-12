"""
Pseudo amino acid composition (PseAAC), Chou 2001 -- LAMBDA BLOCK ONLY.

IMPORTANT: this script deliberately does NOT emit the standard first 20
components. Type-1 PseAAC is defined as 20 + lambda components, where the first
20 ARE the amino acid composition. Since your existing feature set already
contains the full AAC (15 AAC_* columns plus K_frac, R_frac, H_frac, D_frac and
E_frac, which are the AAC values for those five residues under different names),
emitting them again would create 20 exact duplicate columns -- and would
reintroduce the five collinear pairs that merge_features_fixed.py removes.

So only the lambda sequence-order correlation factors are written here. They are
the part of PseAAC that carries information AAC does not.

Each factor theta_k measures the average physicochemical dissimilarity between
residues k positions apart:

    theta_k = (1 / (L - k)) * sum over i of Theta(R_i, R_{i+k})

    Theta(R_i, R_j) = mean of the squared differences in three normalised
                      properties: hydrophobicity, hydrophilicity, side-chain mass

and is then weighted and normalised the standard way, with the sum of the 20
composition fractions equal to 1 in the denominator:

    component_k = w * theta_k / (1 + w * sum(theta))

LENGTH CONSTRAINT. theta_k is undefined when L <= k. Your dataset contains
peptides as short as 1 residue, so with the default LAMBDA = 5 many rows would
be undefined. Undefined factors are set to 0 and the affected sequences are
reported. If most of your positives are shorter than LAMBDA + 1, this block is
carrying very little information and is worth dropping from the study.

Usage:
    python pseaac_feature.py <input_fasta> <output_csv> [lambda] [weight]
"""

import sys

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
STANDARD = set(AMINO_ACIDS)

LAMBDA = 5          # number of correlation tiers
WEIGHT = 0.05       # Chou's w, conventionally 0.05

# Hydrophobicity (Tanford), hydrophilicity (Hopp-Woods), side-chain mass (Da).
_H1 = {"A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85,
       "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50,
       "M": 0.64, "F": 1.19, "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81,
       "Y": 0.26, "V": 1.08}
_H2 = {"A": -0.5, "R": 3.0, "N": 0.2, "D": 3.0, "C": -1.0, "Q": 0.2, "E": 3.0,
       "G": 0.0, "H": -0.5, "I": -1.8, "L": -1.8, "K": 3.0, "M": -1.3, "F": -2.5,
       "P": 0.0, "S": 0.3, "T": -0.4, "W": -3.4, "Y": -2.3, "V": -1.5}
_M = {"A": 15.0, "R": 101.0, "N": 58.0, "D": 59.0, "C": 47.0, "Q": 72.0,
      "E": 73.0, "G": 1.0, "H": 82.0, "I": 57.0, "L": 57.0, "K": 73.0,
      "M": 75.0, "F": 91.0, "P": 42.0, "S": 31.0, "T": 45.0, "W": 130.0,
      "Y": 107.0, "V": 43.0}


def _standardise(d):
    """Zero mean, unit population standard deviation, as Chou specifies."""
    vals = list(d.values())
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    sd = var ** 0.5
    return {k: (v - mean) / sd for k, v in d.items()}


PROPS = [_standardise(_H1), _standardise(_H2), _standardise(_M)]


def read_fasta(file_path):
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


def _theta(a, b):
    """Mean squared difference across the three normalised properties."""
    return sum((p[a] - p[b]) ** 2 for p in PROPS) / len(PROPS)


def calc_pseaac_lambda(seq, lam=LAMBDA, w=WEIGHT):
    clean = [aa for aa in seq if aa in STANDARD]
    L = len(clean)

    thetas = []
    for k in range(1, lam + 1):
        if L <= k:
            thetas.append(0.0)                 # undefined for this length
            continue
        vals = [_theta(clean[i], clean[i + k]) for i in range(L - k)]
        thetas.append(sum(vals) / len(vals))

    denom = 1.0 + w * sum(thetas)              # the 20 AAC fractions sum to 1
    if denom == 0:
        return [0.0] * lam
    return [w * t / denom for t in thetas]


def extract_pseaac(fasta_file, out_csv, lam=LAMBDA, w=WEIGHT):
    records = read_fasta(fasta_file)

    undefined = [n for n, s in records if len(s) <= lam]
    if undefined:
        pct = 100.0 * len(undefined) / len(records)
        print(f"WARNING: {len(undefined)} of {len(records)} sequences ({pct:.1f}%) "
              f"are <= lambda ({lam}) residues, so some correlation factors are "
              f"undefined and set to 0: {undefined[:5]}")
        if pct > 25:
            print("         More than a quarter of this file is affected. Consider "
                  "a smaller lambda, or dropping the PseAAC tier entirely.")

    with open(out_csv, "w") as f:
        f.write("ID," + ",".join(f"PseAAC_lambda{k}" for k in range(1, lam + 1)) + "\n")
        for name, seq in records:
            row = calc_pseaac_lambda(seq, lam, w)
            f.write(f"{name}," + ",".join(f"{x:.6f}" for x in row) + "\n")

    print(f"Saved {len(records)} sequences x {lam} PseAAC lambda features "
          f"(first 20 AAC components deliberately omitted) -> {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python pseaac_feature.py <input_fasta> <output_csv> "
              "[lambda] [weight]")
        sys.exit(1)
    lam = int(sys.argv[3]) if len(sys.argv) > 3 else LAMBDA
    w = float(sys.argv[4]) if len(sys.argv) > 4 else WEIGHT
    extract_pseaac(sys.argv[1], sys.argv[2], lam, w)
