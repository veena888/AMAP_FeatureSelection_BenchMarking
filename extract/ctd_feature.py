"""
CTD -- Composition, Transition, Distribution (Dubchak et al. 1995) -- 147 features.

Seven physicochemical attributes each partition the 20 amino acids into 3 groups.
The peptide is rewritten as a 3-letter string per attribute, and three blocks are
extracted from it:

  Composition  (3 per attribute, 21 total)
      fraction of residues in each group.

  Transition   (3 per attribute, 21 total)
      how often adjacent residues switch between group pairs (1-2, 1-3, 2-3),
      counted in either direction, divided by (L-1). This is the block that
      matters most for membrane-active peptides: alternation between polar and
      hydrophobic groups is a direct proxy for amphipathicity.

  Distribution (15 per attribute, 105 total)
      for each group, the sequence position (as a percentage of length) at which
      the 1st, 25%, 50%, 75% and 100% of that group's residues have occurred.
      This is what makes CTD positional rather than purely compositional.

21 + 21 + 105 = 147.

Unlike DPC, CTD stays dense on short peptides -- collapsing 20 residues into 3
classes means even a 5-mer populates most composition bins. That makes it a
genuine complement to DPC rather than a substitute.

OVERLAP with the existing 27 features (handled in build_tiers.py):
  - charge group 1 (K,R) == K_frac + R_frac exactly
  - charge group 3 (D,E) == D_frac + E_frac exactly
  Both are linear combinations of columns you already have. Correlated but not
  duplicated, so both are kept; the collinearity report will flag them.
  - hydrophobicity groups use CLVIMFW, whereas HydrophobicFrac uses AVILMFWY --
  similar, not identical.

Usage:
    python ctd_feature.py <input_fasta> <output_csv>
"""

import sys

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
STANDARD = set(AMINO_ACIDS)

# Each attribute maps to three disjoint groups covering all 20 residues.
ATTRIBUTES = {
    "hydrophobicity":       ["RKEDQN", "GASTPHY", "CLVIMFW"],
    "vdw_volume":           ["GASTPDC", "NVEQIL", "MHKFRYW"],
    "polarity":             ["LIFWCMVY", "PATGS", "HQRKNED"],
    "polarizability":       ["GASDT", "CPNVEQIL", "KMHFRYW"],
    "charge":               ["KR", "ANCQGHILMFPSTWYV", "DE"],
    "secondary_structure":  ["EALMQKRH", "VIYCWFT", "GNPSD"],
    "solvent_accessibility": ["ALFCGIVW", "RKQEND", "MSPTHY"],
}

DISTRIBUTION_POINTS = [0, 25, 50, 75, 100]      # percentiles of each group's residues


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


def _encode(seq, groups):
    """Rewrite the peptide as a string of group indices 1/2/3. Non-standard
    residues become 0 and are ignored by every downstream block."""
    lookup = {}
    for idx, members in enumerate(groups, start=1):
        for aa in members:
            lookup[aa] = idx
    return [lookup.get(aa, 0) for aa in seq]


def _composition(encoded):
    n = sum(1 for g in encoded if g > 0)
    if n == 0:
        return [0.0, 0.0, 0.0]
    return [sum(1 for g in encoded if g == k) / n for k in (1, 2, 3)]


def _transition(encoded):
    """Fraction of adjacent pairs that switch between a given pair of groups.
    Order-insensitive: 1->2 and 2->1 both count toward the (1,2) transition."""
    pairs = [(a, b) for a, b in zip(encoded, encoded[1:]) if a > 0 and b > 0]
    if not pairs:
        return [0.0, 0.0, 0.0]
    out = []
    for i, j in ((1, 2), (1, 3), (2, 3)):
        n = sum(1 for a, b in pairs if (a == i and b == j) or (a == j and b == i))
        out.append(n / len(pairs))
    return out


def _distribution(encoded):
    """For each group, the position (percent of sequence length) at which the
    1st, 25th, 50th, 75th and 100th percentile of its residues have appeared."""
    total = len(encoded)
    out = []
    for k in (1, 2, 3):
        positions = [i + 1 for i, g in enumerate(encoded) if g == k]
        n = len(positions)
        if n == 0 or total == 0:
            out.extend([0.0] * len(DISTRIBUTION_POINTS))
            continue
        for pct in DISTRIBUTION_POINTS:
            if pct == 0:
                rank = 1                       # first occurrence
            else:
                # ceil(pct/100 * n), clamped into [1, n]
                rank = int((pct * n + 99) // 100)
                rank = max(1, min(rank, n))
            out.append(100.0 * positions[rank - 1] / total)
    return out


def ctd_column_names():
    names = []
    for attr in ATTRIBUTES:
        names += [f"CTD_C_{attr}_g{k}" for k in (1, 2, 3)]
    for attr in ATTRIBUTES:
        names += [f"CTD_T_{attr}_{i}{j}" for i, j in ((1, 2), (1, 3), (2, 3))]
    for attr in ATTRIBUTES:
        for k in (1, 2, 3):
            names += [f"CTD_D_{attr}_g{k}_{p}" for p in DISTRIBUTION_POINTS]
    return names


def calc_ctd(seq):
    if len(seq) == 0:
        return [0.0] * len(ctd_column_names())

    encoded = {attr: _encode(seq, groups) for attr, groups in ATTRIBUTES.items()}

    comp, trans, dist = [], [], []
    for attr in ATTRIBUTES:                    # dict order is insertion order
        comp += _composition(encoded[attr])
        trans += _transition(encoded[attr])
        dist += _distribution(encoded[attr])
    return comp + trans + dist                 # matches ctd_column_names() order


def extract_ctd(fasta_file, out_csv):
    records = read_fasta(fasta_file)
    columns = ctd_column_names()

    bad = {c for _, s in records for c in s} - STANDARD
    if bad:
        print(f"WARNING: non-standard residues {sorted(bad)} -- excluded from all "
              f"CTD blocks")

    with open(out_csv, "w") as f:
        f.write("ID," + ",".join(columns) + "\n")
        for name, seq in records:
            row = calc_ctd(seq)
            assert len(row) == len(columns), \
                f"{name}: produced {len(row)} values, expected {len(columns)}"
            f.write(f"{name}," + ",".join(f"{x:.6f}" for x in row) + "\n")

    print(f"Saved {len(records)} sequences x {len(columns)} CTD features -> {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ctd_feature.py <input_fasta> <output_csv>")
        sys.exit(1)
    extract_ctd(sys.argv[1], sys.argv[2])
