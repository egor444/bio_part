# bio_part

A small toolkit for splitting protein sequence datasets into train/validation/test sets and
analysing how similar their physiochemical property distributions are. It's built around
three interchangeable split strategies and a command-line entry point that ties reading,
splitting, plotting and reporting together.

## What it does

1. **Reads** a protein dataset from a FASTA file (sequence + target label, optionally with an
   existing train/val/test assignment).
2. **Computes physiochemical properties** per sequence — molecular weight, isoelectric point,
   hydrophobicity, aromaticity, instability index, Shannon entropy, net charge, and length —
   using Biopython, and (optionally) sparse 3-mer frequency features.
3. **Splits** the dataset into train/val/test using one of three strategies:
   - **original** — keeps whatever `set` labels are already in the input file.
   - **random** — a stratified random split.
   - **physchem** — a split that maximises the physiochemical distance between train, val and
     test (so the sets are deliberately dissimilar, useful for stress-testing generalisation).
4. **Plots** the distribution of each physiochemical property across train/val/test, before
   and after splitting, formatted for inclusion in a paper.
5. **Reports** how different the train and test distributions are, per feature and in
   aggregate, using a histogram-based KL divergence.

## Installation

Install from PyPI:

```bash
pip install bio-part
```

Or, while it's still on TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --no-deps bio-part
```


All required dependencies (Biopython, scikit-learn, scipy, joblib, pandas, numpy, matplotlib)
are installed automatically — there's nothing extra to install by hand.

## Usage

Once installed, `bio-part` is available as a standalone command:

```bash
bio-part INPUT.fasta OUTPUT_DIR [options]
```

It can equally be run as a module, which is useful if the console script isn't on your `PATH`:

```bash
python -m bio_part.main INPUT.fasta OUTPUT_DIR [options]
```

### Example

```bash
bio-part data/dataset.fasta results/experiment_1 \
    --split-strategy physchem \
    --physchem-analysis \
    --val-size 0.15 --test-size 0.15
```

This reads `dataset.fasta`, creates `results/experiment_1/`, plots and reports on the input
dataset's physiochemical properties, performs a physchem-maximising split, writes the split
dataset, and plots and reports on the split result too.

### Using it from Python

Every piece of the pipeline is also importable, so you can use `bio_part` as a library instead
of (or alongside) the command line:

```python
from bio_part.utils import read_fasta, write_fasta
from bio_part.physchem import get_physchem_df
from bio_part.split_strategy import MaxDistanceSplitStrategy
from bio_part.plot_generator import plot_physchem_dists_paper

df = read_fasta("data/dataset.fasta")
physchem_df, _ = get_physchem_df(df, sequence_col="sequence")

strategy = MaxDistanceSplitStrategy(data_type="sequence_to_class")
train, val, test = strategy.split(physchem_df, val_size=0.15, test_size=0.15)

plot_physchem_dists_paper(train.assign(set="train"), title="Train set", save_path="train_dist.png")
```

### Arguments

| Argument | Description |
|---|---|
| `input` | Path to the input FASTA file. Required. |
| `output` | Output directory. Created automatically if it doesn't exist. Required. |
| `--split-strategy`, `-s` | `none` (default), `original`, `random`, or `physchem`. See [Split strategies](#split-strategies). |
| `--physchem-analysis`, `--physchem` | Flag. If set, generates distribution plots and KL-divergence reports (see [Outputs](#outputs)). |
| `--val-size` | Validation fraction (e.g. `0.15`). Defaults to the proportion of sequences already labelled `val` in the input's `set` property, or `0.15` if the input has no `set` property. |
| `--test-size` | Test fraction, defaulting the same way using the `test` label. |
| `--keep-old-test` | Flag. Keeps the input's existing `test` sequences (from its `set`) unchanged, and only re-splits train/val from the rest. Requires the input to already have a `set`. |
| `--data-type` | `sequence_to_class`, `sequence_to_value`, `residue_to_class`, or `residue_to_value`. Auto-detected from the `target` if omitted — see [Task type detection](#task-type-detection). |
| `--num-bins` | Number of bins used to stratify continuous targets, for the `random` and `physchem` strategies. Default `10`. |
| `--exclude-features` | Space-separated list of physiochemical features to leave out of the `physchem` strategy's distance calculation (e.g. `--exclude-features length charge`). |

### Split strategies

- **`original`** — uses the `set` already present in the input file as-is. Fails if the
  input has no `set`.
- **`random`** — a stratified `train_test_split`. The stratification group is the class label
  (classification tasks), a binned value (regression tasks), or the most common residue label
  / mean valid value (residue-level tasks) — see [Task type detection](#task-type-detection).
- **`physchem`** — stratifies the same way as `random`, but within each stratification group,
  orders sequences along their first principal component of physiochemical properties and
  slices train/val/test from that ordering, so the splits differ as much as possible in their
  physiochemical makeup.

### Task type detection

If `--data-type` isn't given, it's inferred from the `target`:

- Contains `;` anywhere → **`residue_to_value`** (a semicolon-separated value per residue).
- Same length as its row's `sequence`, for every row → **`residue_to_class`** (a per-residue
  label string, letters or digits).
- Otherwise, parses as a number → **`sequence_to_value`**; anything else → **`sequence_to_class`**.

### Outputs

```
OUTPUT_DIR/
├── out_split.fasta                     # only if --split-strategy is not "none"
└── results/                            # only if --physchem-analysis is set
    ├── physchem_dist_input.png         # distribution plot of the input dataset
    ├── physchem_dist_output.png        # distribution plot of the split dataset (only if split)
    ├── kl_divergence_input.json        # only if the input already has both train & test sequences
    └── kl_divergence_output.json       # only if a split was performed
```

Each `kl_divergence_*.json` looks like:

```json
{
  "aggregate_score": 0.70,
  "driving_feature": "hydrophobicity",
  "driving_feature_score": 1.0,
  "raw_kl": {
    "molecular_weight": 7.66,
    "isoelectric_point": 13.07,
    "hydrophobicity": 15.15,
    "...": "..."
  }
}
```

`raw_kl` is the KL divergence of each feature's train vs. test histogram; `aggregate_score`
combines them (default: an L2 norm of the min-max normalised scores) into a single number,
and `driving_feature` names whichever feature differs most between train and test.

## Input FASTA file structure

```
>id1 target=target_value [set=set_value]
SEQUENCE
>id2 target=target_value [set=set_value]
SEQUENCE
...
```

- Each record is a header line starting with `>`, followed by the sequence on the next line.
- The header's first token (after `>`) is used as the record's `id`.
- `target=...` is **required** on every record. Its meaning depends on the task — a single
  class or value for sequence-level tasks, or a same-length string / semicolon-separated
  series of values for residue-level tasks (see [Task type detection](#task-type-detection)).
- `set=...` is **optional**. If present on every record, it should be one of `train`, `val`,
  or `test` — this is what the `original` split strategy and `--keep-old-test` rely on, and
  what lets the "before splitting" physchem plot and KL report show an existing split. If
  omitted entirely, the dataset is treated as unsplit (and the `original` strategy and
  `--keep-old-test` can't be used).
- Sequences should consist of the 20 standard amino acids (`ACDEFGHIKLMNPQRSTVWY`); any other
  character is stripped out before physiochemical properties are computed.

### Example

```
>prot001 target=1 set=train
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWELVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL
>prot002 target=0 set=test
MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVYLLPRRGPRLGVRATRKTSERSQPRGRRQPIPKARRPEGRTWAQPGYPWPLYGNEGCGWAGWLLSPRGSRPSWGPTDPRRRSRNLGKVIDTLTCGFADLMGYIPLVGAPLGGAARALAHGVRVLEDGVNYATGNLPGCSFSIFLLALLSCLTVPASA
```
