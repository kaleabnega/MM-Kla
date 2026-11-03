"""Feature encodings for a locked-split HybridKla adaptation.

Resource files are read from the authors' official code archive at runtime;
neither the archive nor its contents are redistributed in this repository.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
CKSAAP_AA = STANDARD_AA + "X"
OBC_AA = "KLAEVGSDITR*PQNFYMHCWU"
GPS_AA = "ARNDCQEGHILKMFPSTWYVBZX*"
FEATURE_DIMS = {
    "ACF": 510,
    "AAINDEX": 510,
    "CKSAAP": 441,
    "GPS": 300,
    "OBC": 1122,
    "PSEAAC": 20,
}


@dataclass(frozen=True)
class HybridKlaResources:
    aaindex: np.ndarray
    similarity: np.ndarray


def load_resources(top10_path: Path, blosum_path: Path) -> HybridKlaResources:
    with top10_path.open(newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if len(rows) != 11 or rows[0][0] != "name" or set(rows[0][1:]) != set(STANDARD_AA):
        raise ValueError("Unexpected HybridKla AAindex resource")
    aaindex = np.asarray([[float(row[rows[0].index(aa)]) for aa in STANDARD_AA]
                          for row in rows[1:]], dtype=np.float32)
    if aaindex.shape != (10, 20) or not np.isfinite(aaindex).all():
        raise ValueError("Invalid HybridKla AAindex values")

    with blosum_path.open(newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if len(rows) < 2 or rows[0][0] != "intercept" or len(rows[0]) != 301 or len(rows[1]) != 301:
        raise ValueError("Unexpected HybridKla BLOSUM resource")
    similarity = np.zeros((24, 24), dtype=np.float32)
    amino_index = {aa: index for index, aa in enumerate(GPS_AA)}
    seen_pairs = set()
    for pair, raw_value in zip(rows[0][1:], rows[1][1:]):
        if len(pair) != 2 or any(aa not in amino_index for aa in pair):
            raise ValueError(f"Invalid BLOSUM pair: {pair}")
        left, right = amino_index[pair[0]], amino_index[pair[1]]
        pair_key = tuple(sorted((left, right)))
        if pair_key in seen_pairs:
            raise ValueError(f"Repeated BLOSUM pair: {pair}")
        seen_pairs.add(pair_key)
        similarity[left, right] = similarity[right, left] = float(raw_value)
    if len(seen_pairs) != 300 or not np.isfinite(similarity).all():
        raise ValueError("Incomplete HybridKla BLOSUM matrix")
    return HybridKlaResources(aaindex=aaindex, similarity=similarity)


def _as_ids(sequences: list[str], alphabet: str) -> np.ndarray:
    if not sequences:
        raise ValueError("At least one sequence is required")
    mapping = {aa: index for index, aa in enumerate(alphabet)}
    if any(len(sequence) != 51 or sequence[25] != "K" or
           set(sequence) - set(STANDARD_AA) for sequence in sequences):
        raise ValueError("HybridKla features require canonical 51-residue K-centered windows")
    return np.asarray([[mapping[aa] for aa in sequence] for sequence in sequences], dtype=np.int16)


def fit_gps_reference(training_sequences: list[str]) -> np.ndarray:
    """Estimate positional amino-acid frequencies using training windows only."""
    ids = _as_ids(training_sequences, GPS_AA)
    frequencies = np.stack([
        np.bincount(ids[:, position], minlength=24) / len(ids)
        for position in range(51)
    ]).astype(np.float32)
    if frequencies.shape != (51, 24) or not np.allclose(frequencies.sum(axis=1), 1):
        raise ValueError("Invalid GPS reference frequencies")
    return frequencies


def encode_features(
    sequences: list[str], resources: HybridKlaResources, gps_reference: np.ndarray,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    if gps_reference.shape != (51, 24) or not np.isfinite(gps_reference).all():
        raise ValueError("Invalid GPS reference")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    standard_ids = _as_ids(sequences, STANDARD_AA)
    count = len(sequences)
    numeric = resources.aaindex[:, standard_ids].transpose(1, 0, 2)
    aaindex = numeric.reshape(count, 510).astype(np.float32)
    acf = np.stack([
        np.round((numeric[:, :, :51 - lag] * numeric[:, :, lag:]).mean(axis=2), 2)
        for lag in range(51)
    ], axis=2).reshape(count, 510).astype(np.float32)

    ck_ids = _as_ids(sequences, CKSAAP_AA)
    pairs = ck_ids[:, :-1].astype(np.int32) * 21 + ck_ids[:, 1:]
    pair_rows = np.broadcast_to(np.arange(count, dtype=np.int32)[:, None], pairs.shape)
    cksaap = np.bincount((pair_rows * 441 + pairs).ravel(), minlength=count * 441)
    cksaap = cksaap.reshape(count, 441).astype(np.float32)

    obc_ids = _as_ids(sequences, OBC_AA)
    obc = np.eye(22, dtype=np.float32)[obc_ids].reshape(count, 1122)
    pseaac = np.stack([(standard_ids == index).mean(axis=1) for index in range(20)], axis=1)
    pseaac = pseaac.astype(np.float32)

    gps_ids = _as_ids(sequences, GPS_AA)
    gps = np.empty((count, 300), dtype=np.float32)
    triangle = np.triu_indices(24)
    identity = np.eye(24, dtype=np.float32)
    diagonal = np.arange(24)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        indicators = identity[gps_ids[start:end]]
        product = np.einsum("bli,lj->bij", indicators, gps_reference, optimize=True)
        product *= resources.similarity
        symmetric = product + product.transpose(0, 2, 1)
        symmetric[:, diagonal, diagonal] *= 0.5
        gps[start:end] = symmetric[:, triangle[0], triangle[1]]

    encoded = {"ACF": acf, "AAINDEX": aaindex, "CKSAAP": cksaap,
               "GPS": gps, "OBC": obc, "PSEAAC": pseaac}
    if any(array.shape != (count, FEATURE_DIMS[name]) or not np.isfinite(array).all()
           for name, array in encoded.items()):
        raise ValueError("HybridKla feature encoding produced invalid values or dimensions")
    return encoded
