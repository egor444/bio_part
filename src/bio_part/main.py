"""
Command-line interface for bio_part.

Splits a protein FASTA dataset using one of three strategies (or no split at all), and
optionally runs a physiochemical analysis (distribution plots + KL divergence between the
train/test sets) both before and after splitting.

Usage:
    python -m bio_parts.main INPUT.fasta OUTPUT_DIR --split-strategy random --physchem-analysis
"""

import argparse
import gc
import json
import os

import pandas as pd

from .utils import read_fasta, write_fasta, get_task_type_by_target, compute_aggregate_KL_from_df
from .physchem import get_physchem_df
from .plot_generator import plot_physchem_dists_paper
from .split_strategy import OriginalSplitStrategy, RandomSplitStrategy, MaxDistanceSplitStrategy

BASE_COLUMNS = ["id", "sequence", "target", "set"]

STRATEGIES = {
    "original": OriginalSplitStrategy,
    "random": RandomSplitStrategy,
    "physchem": MaxDistanceSplitStrategy,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bio_parts",
        description="Split a protein FASTA dataset and analyse its physiochemical properties.",
    )
    parser.add_argument("input", help="Path to the input FASTA file.")
    parser.add_argument("output", help="Output directory (created if it doesn't exist).")
    parser.add_argument(
        "--split-strategy", "-s", dest="split_strategy",
        choices=["none", "original", "random", "physchem"], default="none",
        help="Which split strategy to use. 'original' keeps the dataset's existing 'set' "
             "labels, 'random' does a stratified random split, 'physchem' maximizes the "
             "physiochemical distance between splits. 'none' (default) skips splitting.",
    )
    parser.add_argument(
        "--physchem-analysis", "--physchem", dest="physchem_analysis", action="store_true",
        help="Run the physiochemical analysis: a distribution plot before (and, if a split "
             "is applied, after) splitting, plus a train/test KL divergence report for "
             "each dataset that has both sets available.",
    )
    parser.add_argument(
        "--val-size", type=float, default=None,
        help="Validation set fraction. Defaults to the proportion of rows already labelled "
             "'val' in the input dataset's 'set' column, or 0.15 if the input has no 'set' "
             "column.",
    )
    parser.add_argument(
        "--test-size", type=float, default=None,
        help="Test set fraction. Defaults analogously to --val-size, using the 'test' label.",
    )
    parser.add_argument(
        "--keep-old-test", action="store_true",
        help="Keep the input dataset's existing test rows (from its 'set' column) instead "
             "of resampling a new test set; only train/val are re-split.",
    )
    parser.add_argument(
        "--data-type", dest="data_type", default=None,
        choices=["sequence_to_class", "sequence_to_value", "residue_to_class", "residue_to_value"],
        help="Task type used to decide how to stratify the split. Auto-detected from the "
             "'target' column if omitted.",
    )
    parser.add_argument(
        "--num-bins", type=int, default=10,
        help="Number of bins used to stratify continuous targets (random/physchem strategies).",
    )
    parser.add_argument(
        "--exclude-features", nargs="*", default=None,
        help="Physiochemical features to exclude from the 'physchem' strategy's PCA ordering.",
    )
    return parser


def _base_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any extra (e.g. physiochemical) columns, keeping only id/sequence/target/set."""
    return df[[c for c in BASE_COLUMNS if c in df.columns]].copy()


def _default_split_sizes(df: pd.DataFrame, val_size, test_size):
    """
    Fill in missing val/test sizes from the proportions already present in the input
    dataset's 'set' column (if any), falling back to 0.15 / 0.15.
    """
    if val_size is not None and test_size is not None:
        return val_size, test_size

    if "set" in df.columns and df["set"].notna().any():
        proportions = df["set"].value_counts(normalize=True)
        default_val = float(proportions.get("val", 0.15))
        default_test = float(proportions.get("test", 0.15))
    else:
        default_val, default_test = 0.15, 0.15

    return (
        val_size if val_size is not None else default_val,
        test_size if test_size is not None else default_test,
    )


def _with_physchem(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the physiochemical property columns for a base id/sequence/target/set frame.
    This is the single, expensive (per-sequence Biopython) computation in the pipeline --
    call it once per dataset and reuse the result, rather than letting the split, the plot
    and the KL report each recompute it independently.
    """
    physchem_df, _ = get_physchem_df(_base_columns(df), sequence_col="sequence")
    return physchem_df


def _save_physchem_plot(physchem_df: pd.DataFrame, title: str, save_path: str) -> None:
    """
    Save a physiochemical distribution plot for `physchem_df`, which must already carry the
    physiochemical property columns (see `_with_physchem`) -- this function does not compute
    them itself. `plot_physchem_dists_paper` compares the 'train'/'val'/'test' rows of 'set',
    so a dataset with no existing split (e.g. the raw input before it has been split) is
    plotted as a single series labelled 'train' instead.
    """
    plot_df = physchem_df
    if "set" not in plot_df.columns:
        plot_df = plot_df.copy()
        plot_df["set"] = "train"
        print(f"'{title}' has no existing train/val/test split; plotting it as a single series.")

    plot_physchem_dists_paper(plot_df, title=title, save_path=save_path)
    print(f"Saved physiochemical distribution plot to {save_path}")


def _save_kl_report(physchem_df: pd.DataFrame, save_path: str, label: str) -> None:
    """
    Write a train/test KL divergence report for `physchem_df`, which must already carry the
    physiochemical property columns (see `_with_physchem`) -- this function does not compute
    them itself.
    """
    try:
        report = compute_aggregate_KL_from_df(physchem_df)
    except ValueError as exc:
        print(f"Skipping KL divergence report for the {label} dataset: {exc}")
        return

    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved KL divergence report for the {label} dataset to {save_path}")


def run_split(input_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    task_type = args.data_type or get_task_type_by_target(input_df)

    strategy_cls = STRATEGIES[args.split_strategy]
    kwargs = {"data_type": task_type}
    if args.split_strategy == "random":
        kwargs["num_bins"] = args.num_bins
    elif args.split_strategy == "physchem":
        kwargs["num_bins"] = args.num_bins
        kwargs["exclude_features"] = args.exclude_features
    strategy = strategy_cls(**kwargs)

    val_size, test_size = _default_split_sizes(input_df, args.val_size, args.test_size)
    if val_size + test_size >= 1:
        raise ValueError(f"val_size + test_size must be < 1 (got {val_size} + {test_size}).")

    train, val, test = strategy.split(
        input_df, val_size=val_size, test_size=test_size, keep_old_test=args.keep_old_test
    )

    train = train.copy()
    train["set"] = "train"
    val = val.copy()
    val["set"] = "val"
    parts = [train, val]
    if test is not None and not test.empty:
        test = test.copy()
        test["set"] = "test"
        parts.append(test)

    return pd.concat(parts, ignore_index=True)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    input_df = read_fasta(args.input)
    print(f"Read {len(input_df)} sequences from {args.input}")
    os.makedirs(args.output, exist_ok=True)

    results_dir = None
    if args.physchem_analysis:
        results_dir = os.path.join(args.output, "results")
        os.makedirs(results_dir, exist_ok=True)

    needs_physchem = args.physchem_analysis or args.split_strategy == "physchem"
    input_physchem_df = _with_physchem(input_df) if needs_physchem else None

    if args.physchem_analysis:
        _save_physchem_plot(
            input_physchem_df, title="Input Dataset",
            save_path=os.path.join(results_dir, "physchem_dist_input.png"),
        )
        _save_kl_report(input_physchem_df, os.path.join(results_dir, "kl_divergence_input.json"), label="input")

    split_df = None
    if args.split_strategy != "none":
        split_df = run_split(input_physchem_df if input_physchem_df is not None else input_df, args)

        # input_df and input_physchem_df are fully consumed at this point (their data lives on,
        # copied, inside split_df) -- drop the references so the originals can be freed.
        del input_df, input_physchem_df
        gc.collect()

        out_fasta_path = os.path.join(args.output, "out_split.fasta")
        write_fasta(_base_columns(split_df), out_fasta_path)
        print(f"Saved split dataset to {out_fasta_path}")

        if args.physchem_analysis:
            _save_physchem_plot(
                split_df, title="Split Dataset",
                save_path=os.path.join(results_dir, "physchem_dist_output.png"),
            )
            _save_kl_report(split_df, os.path.join(results_dir, "kl_divergence_output.json"), label="output")

        del split_df
        gc.collect()
    else:
        del input_df, input_physchem_df
        gc.collect()


if __name__ == "__main__":
    main()
