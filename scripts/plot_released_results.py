"""Create verification plots from released summaries; never replace paper figures."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'archive/scripts'))
from plot_style import apply_arial_style, REFERENCE_PALETTE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/verification_plots')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.is_relative_to((ROOT / 'results').resolve()):
        raise ValueError('Output must not be inside the frozen results directory.')
    out.mkdir(parents=True, exist_ok=True)
    apply_arial_style()
    table = pd.read_csv(ROOT / 'results/3.3_generalization_ladder/homology_auc_summary_numeric.csv')
    h = table[table.evaluation.str.startswith('homology_')].copy()
    h['identity'] = h.evaluation.str.split('_').str[-1].astype(int)
    h = h.sort_values('identity')
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric, label, color in [('ROC-AUC', 'Macro ROC-AUC', REFERENCE_PALETTE[1]),
                                 ('micro_roc_auc', 'Micro ROC-AUC', REFERENCE_PALETTE[5])]:
        ax.errorbar(h.identity, h[metric+'_mean'], yerr=h[metric+'_sd'], marker='o', capsize=3, label=label, color=color)
    ax.set(xlabel='Homology clustering identity threshold (%)', ylabel='AUC (mean ± SD)', ylim=(.5, 1.02), xticks=range(10, 100, 10))
    ax.legend()
    fig.tight_layout()
    for ext in ['png', 'svg', 'pdf']:
        target = out / f'generalization_verification.{ext}'
        if target.exists():
            raise FileExistsError(target)
        fig.savefig(target, dpi=220)
    plt.close(fig)
    table = pd.read_csv(ROOT / 'results/3.4_multifunction_interpretability/feature_ablation_auc_ap_numeric.csv')
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, metric in zip(axes, ['ROC-AUC', 'AP']):
        x = np.arange(len(table))
        for i, row in enumerate(table.itertuples(index=False, name=None)):
            color = REFERENCE_PALETTE[i % len(REFERENCE_PALETTE)]
            ax.errorbar(i, table.iloc[i][metric+'_mean'], yerr=table.iloc[i][metric+'_sd'], fmt='o', capsize=4, color=color)
        ax.set(ylabel=f'Macro {metric} (mean ± SD)', ylim=(.45, 1.02))
    axes[-1].set_xticks(np.arange(len(table)), table.feature_set, rotation=25, ha='right')
    fig.tight_layout()
    for ext in ['png', 'svg', 'pdf']:
        target = out / f'ablation_verification.{ext}'
        if target.exists():
            raise FileExistsError(target)
        fig.savefig(target, dpi=220)
    plt.close(fig)
    print('Verification plots saved to', out)


if __name__ == '__main__':
    main()
