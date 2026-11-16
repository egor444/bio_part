import os

import pandas as pd
import numpy as np

def read_fasta(file_path: str) -> pd.DataFrame:
    """
    Read a FASTA file with the following format:

    >id target=target_value [set=set_value]
    sequence
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    df_dict = {"id": [], "sequence": [], "target": [], "set": []}
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                parts = line[1:].strip().split()
                df_dict["id"].append(parts[0].strip())
                for part in parts[1:]:
                    part = part.strip().lower()
                    if part.startswith("target="):
                        target = part[7:]
                        df_dict["target"].append(target)
                    if part.startswith("set="):
                        df_dict["set"].append(part[4:].strip())
            else:
                sequence = line.strip()
                df_dict["sequence"].append(sequence)
    if len(df_dict["target"]) != len(df_dict["sequence"]):
        raise ValueError(f"Mismatch between number of sequences ({len(df_dict['sequence'])}) and targets ({len(df_dict['target'])}) in the FASTA file.")
    if len(df_dict["set"]) == 0:
        df_dict.pop("set")
    elif len(df_dict["set"]) != len(df_dict["sequence"]):
        raise ValueError(f"Mismatch between number of sequences ({len(df_dict['sequence'])}) and sets ({len(df_dict['set'])}) in the FASTA file.")
    out_df = pd.DataFrame(df_dict)
    del df_dict 
    return out_df

def write_fasta(df: pd.DataFrame, file_path: str, include_set: bool = True) -> None:
    """
    Write a DataFrame with columns ['id', 'sequence', 'target'] and, optionally, 'set'
    out to a FASTA file in the format read by `read_fasta`:

    >id target=target_value [set=set_value]
    sequence
    """
    required = {"id", "sequence", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required column(s): {sorted(missing)}")

    write_set = include_set and "set" in df.columns

    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w") as f:
        for _, row in df.iterrows():
            header = f">{row['id']} target={row['target']}"
            if write_set and pd.notna(row["set"]):
                header += f" set={row['set']}"
            f.write(header + "\n")
            f.write(f"{row['sequence']}\n")


def get_task_type_by_target(df: pd.DataFrame) -> str:
    """
    Determine the task type based on the 'target' column (and, where available, the
    'sequence' column) of the DataFrame.

    Returns one of: 'sequence_to_class', 'sequence_to_value', 'residue_to_class', 'residue_to_value'.
    """
    if "target" not in df.columns:
        raise ValueError("DataFrame must contain a 'target' column to determine task type.")
    if df.empty:
        raise ValueError("DataFrame is empty; cannot determine task type from no rows.")

    target_str = df["target"].astype(str)

    if target_str.str.contains(";").any():
        return "residue_to_value"

    if "sequence" in df.columns:
        is_residue_level = (target_str.str.len() == df["sequence"].astype(str).str.len()).all()
    else:
        is_residue_level = target_str.str.len().max() > 1

    if is_residue_level:
        return "residue_to_class"

    if pd.to_numeric(df["target"], errors="coerce").notna().all():
        return "sequence_to_value"

    return "sequence_to_class"

def compute_feature_kl(train_vals, test_vals, n_bins=50, epsilon=1e-10):
    """
    Compute the (histogram-based) KL divergence between a single physiochemical feature's
    train and test distributions. NaNs are dropped before binning. Returns NaN if either
    side has no valid values, and 0.0 if both sides are constant at the same value.
    """
    from scipy.stats import entropy

    train_vals = np.asarray(train_vals, dtype=float)
    test_vals = np.asarray(test_vals, dtype=float)
    train_vals = train_vals[~np.isnan(train_vals)]
    test_vals = test_vals[~np.isnan(test_vals)]

    if len(train_vals) == 0 or len(test_vals) == 0:
        return float("nan")

    combined_min = min(train_vals.min(), test_vals.min())
    combined_max = max(train_vals.max(), test_vals.max())
    if combined_min == combined_max:
        return 0.0
    bin_edges = np.linspace(combined_min, combined_max, n_bins + 1)

    train_hist, _ = np.histogram(train_vals, bins=bin_edges, density=True)
    test_hist, _ = np.histogram(test_vals, bins=bin_edges, density=True)
    train_hist = train_hist + epsilon
    test_hist = test_hist + epsilon
    train_hist = train_hist / train_hist.sum()
    test_hist = test_hist / test_hist.sum()

    return float(entropy(train_hist, test_hist))


def _aggregate_from_raw_kl(raw_kl, kl_bounds=None, p=2, method="lp_norm", alpha=0.8):
    """Shared normalize + aggregate step used by both compute_aggregate_KL variants."""

    def normalize(raw_dict, bounds):
        if bounds is None:
            vmin, vmax = min(raw_dict.values()), max(raw_dict.values())
        else:
            vmin, vmax = bounds
        rng = (vmax - vmin) if vmax > vmin else 1.0
        return {k: (v - vmin) / rng for k, v in raw_dict.items()}

    feature_scores = normalize(raw_kl, kl_bounds)

    features = list(feature_scores.keys())
    values = np.array([feature_scores[f] for f in features])
    n = len(values)
    if n == 0:
        raise ValueError("feature_scores is empty.")

    max_idx = int(np.argmax(values))
    driving_feature = features[max_idx]
    driving_score = float(values[max_idx])

    if method == "lp_norm":
        aggregate = (np.sum(values ** p) / n) ** (1.0 / p)
    elif method == "max_mean_blend":
        aggregate = alpha * values.max() + (1 - alpha) * values.mean()
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "aggregate_score": float(aggregate),
        "driving_feature": driving_feature,
        "driving_feature_score": driving_score,
        "raw_kl": raw_kl,
    }


def compute_aggregate_KL_from_df(df, feature_columns=None, train_label="train", test_label="test",
                                  kl_bounds=None, p=2, method="lp_norm", alpha=0.8,
                                  n_bins=50, epsilon=1e-10):
    """
    Compute the aggregate KL divergence between the train and test distributions for each
    physiochemical feature already present as columns in `df`.
    """
    if "set" not in df.columns:
        raise ValueError("DataFrame must contain a 'set' column to compute train/test KL divergence.")

    train_df = df[df["set"] == train_label]
    test_df = df[df["set"] == test_label]

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"DataFrame must contain rows for both '{train_label}' and '{test_label}' "
            "sets to compute KL divergence."
        )

    if feature_columns is None:
        feature_columns = [
            c for c in df.columns
            if c not in ("id", "sequence", "target", "set") and pd.api.types.is_numeric_dtype(df[c])
        ]
    if not feature_columns:
        raise ValueError("No numeric physiochemical feature columns found to compare.")

    raw_kl = {
        feat: compute_feature_kl(train_df[feat].values, test_df[feat].values, n_bins=n_bins, epsilon=epsilon)
        for feat in feature_columns
    }

    return _aggregate_from_raw_kl(raw_kl, kl_bounds=kl_bounds, p=p, method=method, alpha=alpha)


def compute_aggregate_KL(experiment_path, kl_bounds=None,
                           p=2, method="lp_norm", alpha=0.8, n_bins=50, epsilon=1e-10):
    """
    Compute the aggregate KL divergence between the train and test distributions for each
    physiochemical feature, reading the split dataset from `<experiment_path>/out_split.fasta`
    and computing the physiochemical property columns before comparing them.
    Returns a dictionary with the aggregate score, driving feature, and raw KL values for each feature.
    """
    if not os.path.exists(experiment_path):
        raise ValueError(f"Experiment path {experiment_path} does not exist.")

    from .physchem import get_physchem_df

    df = read_fasta(os.path.join(experiment_path, "out_split.fasta"))
    df, _ = get_physchem_df(df, sequence_col="sequence")

    return compute_aggregate_KL_from_df(
        df, kl_bounds=kl_bounds, p=p, method=method, alpha=alpha, n_bins=n_bins, epsilon=epsilon
    )