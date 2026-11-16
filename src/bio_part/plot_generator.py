

def plot_physchem_dists_paper(df, title=None, save_path=None, dpi=300):
    """
    Plot the distributions of physiochemical properties comparing train, val and test sets
    for a given experiment, formatted for inclusion in a paper (large fonts, clean layout,
    no empty subplot, smoothed density curves).
    """
    import os
    import matplotlib.pyplot as plt

    features = ["molecular_weight", "isoelectric_point", "hydrophobicity",
                "aromaticity", "instability_index", "entropy", "charge", "length"]

    feature_labels = {
        "molecular_weight": "Molecular Weight (Da)",
        "isoelectric_point": "Isoelectric Point (pI)",
        "hydrophobicity": "Hydrophobicity (GRAVY)",
        "aromaticity": "Aromaticity",
        "instability_index": "Instability Index",
        "entropy": "Sequence Entropy",
        "charge": "Net Charge",
        "length": "Sequence Length (aa)",
    }

    if not set(features).issubset(df.columns):
        from .physchem import get_physchem_df
        df = get_physchem_df(df, sequence_col="sequence")[0]

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "figure.titlesize": 20,
    })

    set_colors = {
        "train": "#4C72B0",
        "val": "#DD8452",
        "test": "#55A868",
    }

    nrows, ncols = 2, 4

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, 10))
    axes = axes.flatten()

    if title:
        fig.suptitle(title, fontsize=20, y=1.03)
    else:
        fig.suptitle(f'Physiochemical Distributions',
                      fontsize=20, y=1.03)

    handles_labels = {}

    for ax, feature in zip(axes, features):
        for set_type in ['train', 'val', 'test']:
            subset = df[df['set'] == set_type]
            if subset.empty:
                continue

            color = set_colors.get(set_type, None)

            ax.hist(
                subset[feature], bins=40, density=True, alpha=0.25,
                color=color, edgecolor="none"
            )
            try:
                from scipy.stats import gaussian_kde
                import numpy as np
                kde = gaussian_kde(subset[feature].dropna())
                xs = np.linspace(subset[feature].min(), subset[feature].max(), 300)
                line, = ax.plot(xs, kde(xs), color=color, linewidth=2.2, label=set_type)
            except ImportError:
                _, _, patches = ax.hist(
                    subset[feature], bins=40, density=True, histtype="step",
                    linewidth=2.2, color=color, label=set_type
                )
                line = patches[0]

            handles_labels[set_type] = line

        ax.set_title(feature_labels.get(feature, feature), fontsize=16, pad=10)
        ax.set_xlabel(feature_labels.get(feature, feature), fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.tick_params(axis='both', labelsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    order = ['train', 'val', 'test']
    handles = [handles_labels[s] for s in order if s in handles_labels]
    labels = [s for s in order if s in handles_labels]
    fig.legend(
        handles, labels, loc='upper center', ncol=len(labels),
        bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=15
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    #plt.show()