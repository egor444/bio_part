import os
import re
import itertools
import numpy as np
import pandas as pd
from scipy import sparse
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from joblib import Parallel, delayed

AAS = "ACDEFGHIKLMNPQRSTVWY"          # fixed, sorted order -> deterministic columns
K = 3
N_KMERS = len(AAS) ** K
KMERS = ["".join(p) for p in itertools.product(AAS, repeat=K)]
PROP_COLS = ["molecular_weight", "isoelectric_point", "hydrophobicity",
             "aromaticity", "instability_index", "entropy", "charge", "length"]

_INVALID = re.compile(f"[^{AAS}]")
_LUT = np.zeros(256, dtype=np.int32)
for i, aa in enumerate(AAS):
    _LUT[ord(aa)] = i


def _kmer_row(clean):
    """Sparse 3-mer frequencies: (column indices, float32 frequencies)."""
    if len(clean) < K:
        return np.empty(0, np.int32), np.empty(0, np.float32)
    idx = _LUT[np.frombuffer(clean.encode("ascii"), dtype=np.uint8)]
    codes = idx[:-2] * 400 + idx[1:-1] * 20 + idx[2:]
    uniq, cnt = np.unique(codes, return_counts=True)
    return uniq.astype(np.int32), (cnt / cnt.sum()).astype(np.float32)


def process_chunk(seqs, use_kmers=True):
    n = len(seqs)
    props = np.full((n, len(PROP_COLS)), np.nan)
    indptr, indices, data = [0], [], []

    for i, seq in enumerate(seqs):
        clean = _INVALID.sub("", str(seq).upper())
        cols, vals = np.empty(0, np.int32), np.empty(0, np.float32)

        if clean:
            try:
                pa = ProteinAnalysis(clean)
                counts = np.array(list(pa.count_amino_acids().values()), dtype=float)
                p = counts[counts > 0] / counts.sum()
                props[i] = [
                    pa.molecular_weight(), pa.isoelectric_point(), pa.gravy(),
                    pa.aromaticity(), pa.instability_index(),
                    -(p * np.log2(p)).sum(), pa.charge_at_pH(7.0), len(clean),
                ]
            except Exception:
                pass  # row stays NaN
            if use_kmers:
                cols, vals = _kmer_row(clean)

        indices.append(cols)
        data.append(vals)
        indptr.append(indptr[-1] + len(cols))

    X = None
    if use_kmers:
        X = sparse.csr_matrix(
            (np.concatenate(data), np.concatenate(indices), np.array(indptr)),
            shape=(n, N_KMERS), dtype=np.float32,
        )
    return props, X


def get_physchem_df(df, sequence_col="sequence", use_kmers=False,
                    n_jobs=None, chunk_size=2000):
    """Returns (df with physchem columns, sparse 3-mer matrix or None)."""
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) // 2)
    seqs = df[sequence_col].tolist()
    chunks = (seqs[i:i + chunk_size] for i in range(0, len(seqs), chunk_size))

    results = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(process_chunk)(c, use_kmers) for c in chunks
    )

    props_parts, kmer_parts = [], []
    for props, X in results:          
        props_parts.append(props)
        if X is not None:
            kmer_parts.append(X)

    props_df = pd.DataFrame(np.vstack(props_parts), columns=PROP_COLS, index=df.index)
    kmer_matrix = sparse.vstack(kmer_parts, format="csr") if use_kmers else None
    return pd.concat([df, props_df], axis=1), kmer_matrix