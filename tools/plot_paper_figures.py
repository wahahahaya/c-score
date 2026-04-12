#!/usr/bin/env python3
"""
Generate Figure 2 and Figure 3 for the C-Score paper.

Prerequisites:
    python tools/aggregate_results.py   (creates tools/paper_data/*.csv)

Outputs (saved to tools/plots/paper/):
    fig2_masking_ood_effect.pdf / .png   -- 2-panel: masking demo + OOD source effect
    fig3_alg_comparison.pdf / .png       -- algorithm comparison across r_ood (optional)

Usage:
    python tools/plot_paper_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

matplotlib.rcParams.update({
    'font.family':      'serif',
    'font.size':        8,
    'axes.labelsize':   8,
    'axes.titlesize':   8.5,
    'xtick.labelsize':  7,
    'ytick.labelsize':  7,
    'legend.fontsize':  6.5,
    'lines.linewidth':  1.4,
    'axes.linewidth':   0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'pdf.fonttype':     42,   # TrueType in PDF (for IEEE)
    'ps.fonttype':      42,
})

BASE      = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE / 'tools' / 'paper_data'
OUT_DIR   = BASE / 'tools' / 'plots' / 'paper'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette (colour-blind friendly) ───────────────────────────────────
C_ACC = '#2166AC'   # blue  — accuracy
C_CCI = '#D73027'   # red   — CCI
C_NEAR = '#E07B39'  # orange — near-OOD
C_FAR  = '#4A90D9'  # blue  — far-OOD

ALG_COLORS = {
    'fixmatch':  '#1f77b4',
    'flexmatch': '#ff7f0e',
    'softmatch': '#2ca02c',
    'ds3l':      '#d62728',
}
ALG_LABELS = {
    'fixmatch':  'FixMatch',
    'flexmatch': 'FlexMatch',
    'softmatch': 'SoftMatch',
    'ds3l':      'DS3L',
}

R_OOD_VALS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2  (2-panel, full-column width ≈ 7.16 inches for IEEE double-col)
# ─────────────────────────────────────────────────────────────────────────────

def fig2_masking_and_ood(fixmatch_svhn: pd.DataFrame,
                          fixmatch_ood:  pd.DataFrame) -> None:
    """
    Panel (a): Accuracy masking — FixMatch × SVHN, dual-axis Acc vs CCI.
    Panel (b): OOD source effect — FixMatch, accuracy drop at r=0.5.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))

    # ── Panel (a): Accuracy masking ──────────────────────────────────────────
    ax_a  = axes[0]
    ax_a2 = ax_a.twinx()

    r = fixmatch_svhn['r_ood'].values
    acc_m  = fixmatch_svhn['best_acc_mean'].values
    acc_s  = fixmatch_svhn['best_acc_std'].values
    cci_m  = fixmatch_svhn['cci_mean'].values
    cci_s  = fixmatch_svhn['cci_std'].values

    l1, = ax_a.plot(r, acc_m, 'o-', color=C_ACC, lw=1.5, ms=4, label='Accuracy (%)')
    ax_a.fill_between(r, acc_m - acc_s, acc_m + acc_s, color=C_ACC, alpha=0.12)

    l2, = ax_a2.plot(r, cci_m, 's--', color=C_CCI, lw=1.5, ms=4, label='CCI')
    ax_a2.fill_between(r, cci_m - cci_s, cci_m + cci_s, color=C_CCI, alpha=0.12)

    # shade "masking regime" r ∈ [0.1, 0.4]
    ax_a.axvspan(0.05, 0.45, alpha=0.07, color='gray')
    ax_a.text(0.25, ax_a.get_ylim()[0] if ax_a.get_ylim()[0] > 0 else 58,
              'masking\nregime', ha='center', va='bottom',
              fontsize=5.5, color='gray', style='italic')

    ax_a.set_xlabel('Contamination ratio $r$')
    ax_a.set_ylabel('Best accuracy (%)', color=C_ACC)
    ax_a2.set_ylabel('CCI', color=C_CCI)
    ax_a.tick_params(axis='y', labelcolor=C_ACC)
    ax_a2.tick_params(axis='y', labelcolor=C_CCI)
    ax_a.set_xticks(R_OOD_VALS)

    legend_handles = [l1, l2,
                      mpatches.Patch(color='gray', alpha=0.25, label='Masking regime')]
    ax_a.legend(handles=legend_handles, loc='lower left',
                framealpha=0.9, borderpad=0.4, labelspacing=0.3)
    ax_a.set_title('(a) Accuracy Masking (FixMatch, SVHN OOD)')

    # ── Panel (b): OOD source effect ─────────────────────────────────────────
    ax_b = axes[1]

    # Get baseline (r=0.0) from the fixmatch_svhn table (only one row at r=0)
    baseline_row = fixmatch_svhn[fixmatch_svhn['r_ood'] == 0.0]
    if baseline_row.empty:
        # Try from ood_type table with r=0.0 rows
        baseline_row = fixmatch_ood[fixmatch_ood['r_ood'] == 0.0]
    baseline = float(baseline_row['best_acc_mean'].iloc[0]) if not baseline_row.empty else None

    # OOD taxonomy
    near_ood = {'cifar100', 'stl10'}
    ood_name_map = {
        'cifar100': 'CIFAR-100',
        'stl10':    'STL-10',
        'svhn':     'SVHN',
        'mnist':    'MNIST',
        'gaussian_noise': 'Gaussian',
        'textures': 'Textures',
    }

    # Extract best_acc at r=0.5 for each OOD type
    r05 = fixmatch_ood[fixmatch_ood['r_ood'] == 0.5].copy()
    r05 = r05[r05['ood_dataset'] != 'none']

    if baseline is None and not r05.empty:
        baseline = float(fixmatch_ood[fixmatch_ood['r_ood'] == 0.0]['best_acc_mean'].mean())

    # Sort by drop (ascending = worst first)
    r05 = r05.copy()
    r05['drop'] = r05['best_acc_mean'] - (baseline or 0.0)
    r05['label'] = r05['ood_dataset'].map(ood_name_map).fillna(r05['ood_dataset'])
    r05['color'] = r05['ood_dataset'].apply(lambda x: C_NEAR if x in near_ood else C_FAR)
    r05 = r05.sort_values('drop')

    x_pos = np.arange(len(r05))
    bars = ax_b.bar(x_pos, r05['drop'].values,
                    color=r05['color'].values,
                    width=0.6, edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars, r05['drop'].values):
        va     = 'bottom' if val >= 0 else 'top'
        offset = 0.3 if val >= 0 else -0.3
        ax_b.text(bar.get_x() + bar.get_width() / 2,
                  val + offset, f'{val:+.1f}%',
                  ha='center', va=va, fontsize=6.5, fontweight='bold')

    ax_b.axhline(0, color='black', lw=0.7)
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(r05['label'].values, rotation=30, ha='right')
    ax_b.set_ylabel('Accuracy drop at $r=0.5$ (%)')
    ax_b.set_title('(b) OOD Source Effect (FixMatch, $r=0.5$)')

    near_patch = mpatches.Patch(color=C_NEAR, label='Near-OOD')
    far_patch  = mpatches.Patch(color=C_FAR,  label='Far-OOD')
    ax_b.legend(handles=[near_patch, far_patch],
                loc='lower right', framealpha=0.9,
                borderpad=0.4, labelspacing=0.3)

    # ── Save ─────────────────────────────────────────────────────────────────
    plt.tight_layout(pad=0.8, w_pad=1.5)
    for ext in ('pdf', 'png'):
        path = OUT_DIR / f'fig2_masking_ood_effect.{ext}'
        plt.savefig(path, dpi=300, bbox_inches='tight', format=ext)
        print(f'Saved {path}')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3  (optional, single-column: algorithm comparison)
# ─────────────────────────────────────────────────────────────────────────────

def fig3_alg_comparison(alg_df: pd.DataFrame) -> None:
    """
    Two sub-panels: (a) Accuracy vs r_ood per algorithm, (b) CCI vs r_ood.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))

    for alg in ('fixmatch', 'flexmatch', 'softmatch', 'ds3l'):
        sub = alg_df[alg_df['algorithm'] == alg].sort_values('r_ood')
        r   = sub['r_ood'].values
        col = ALG_COLORS[alg]
        lbl = ALG_LABELS[alg]

        acc_m = sub['best_acc_mean'].values
        acc_s = sub['best_acc_std'].values
        cci_m = sub['cci_mean'].values
        cci_s = sub['cci_std'].values

        axes[0].plot(r, acc_m, 'o-', color=col, lw=1.4, ms=3.5, label=lbl)
        axes[0].fill_between(r, acc_m - acc_s, acc_m + acc_s, color=col, alpha=0.10)

        axes[1].plot(r, cci_m, 's--', color=col, lw=1.4, ms=3.5, label=lbl)
        axes[1].fill_between(r, cci_m - cci_s, cci_m + cci_s, color=col, alpha=0.10)

    for ax, ylabel, title in zip(
            axes,
            ['Best accuracy (%)', 'CCI'],
            ['(a) Accuracy vs. Contamination Ratio',
             '(b) CCI vs. Contamination Ratio']):
        ax.set_xlabel('Contamination ratio $r$')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(R_OOD_VALS)
        ax.legend(loc='best', framealpha=0.9, borderpad=0.4)

    plt.tight_layout(pad=0.8, w_pad=1.5)
    for ext in ('pdf', 'png'):
        path = OUT_DIR / f'fig3_alg_comparison.{ext}'
        plt.savefig(path, dpi=300, bbox_inches='tight', format=ext)
        print(f'Saved {path}')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not DATA_DIR.exists():
        print(f'[ERROR] paper_data not found: {DATA_DIR}')
        print('  Please run:  python tools/aggregate_results.py')
        return

    # Load aggregated tables
    agg = pd.read_csv(DATA_DIR / 'aggregated.csv')

    # FixMatch × SVHN × CIFAR-10
    fixmatch_svhn = agg[
        (agg['algorithm'] == 'fixmatch') &
        (agg['ood_dataset'] == 'svhn') &
        (agg['dataset'] == 'cifar10')
    ].sort_values('r_ood').reset_index(drop=True)

    # FixMatch × all OOD × CIFAR-10 (for OOD-type comparison)
    fixmatch_ood = agg[
        (agg['algorithm'] == 'fixmatch') &
        (agg['dataset'] == 'cifar10')
    ].sort_values(['ood_dataset', 'r_ood']).reset_index(drop=True)

    # All algorithms × CIFAR-10 (aggregated over OOD types)
    alg_df = pd.read_csv(DATA_DIR / 'alg_comparison.csv')
    alg_df_c10 = alg_df[  # keep only CIFAR-10 rows if column exists
        alg_df.get('dataset', 'cifar10').apply(lambda x: x == 'cifar10')
        if 'dataset' in alg_df.columns else [True] * len(alg_df)
    ].sort_values(['algorithm', 'r_ood']).reset_index(drop=True)

    # ── Print summary tables ─────────────────────────────────────────────────
    print('\n=== FixMatch × SVHN (for Table I in paper) ===')
    cols_show = ['r_ood',
                 'best_acc_mean', 'best_acc_std',
                 'cci_mean', 'cci_std',
                 'ple_mean', 'ple_std',
                 'sem_drift_mean', 'sem_drift_std',
                 'ood_ff_mean', 'ood_ff_std',
                 'grad_align_mean', 'grad_align_std']
    print(fixmatch_svhn[[c for c in cols_show if c in fixmatch_svhn.columns]
                        ].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    print('\n=== FixMatch OOD type at r=0.5 (for Table / Figure) ===')
    r05 = fixmatch_ood[fixmatch_ood['r_ood'] == 0.5].copy()
    baseline_acc = float(fixmatch_ood[fixmatch_ood['r_ood'] == 0.0]['best_acc_mean'].mean())
    r05['drop'] = r05['best_acc_mean'] - baseline_acc
    print(r05[['ood_dataset', 'best_acc_mean', 'best_acc_std', 'drop',
               'cci_mean', 'cci_std']].to_string(index=False,
               float_format=lambda x: f'{x:.4f}'))

    print('\n=== Algorithm comparison × r_ood (for Table II) ===')
    print(alg_df_c10[['algorithm', 'r_ood',
                       'best_acc_mean', 'best_acc_std',
                       'cci_mean', 'cci_std']
                     ].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    # ── CIFAR-100 ──
    if (DATA_DIR / 'cifar100_svhn.csv').exists():
        c100 = pd.read_csv(DATA_DIR / 'cifar100_svhn.csv')
        print('\n=== CIFAR-100 × SVHN × FixMatch ===')
        print(c100[['r_ood', 'best_acc_mean', 'best_acc_std',
                    'cci_mean', 'cci_std', 'sem_drift_mean',
                    'ood_ff_mean']].to_string(index=False,
                    float_format=lambda x: f'{x:.4f}'))

    # ── Generate figures ─────────────────────────────────────────────────────
    print('\nGenerating Figure 2 ...')
    fig2_masking_and_ood(fixmatch_svhn, fixmatch_ood)

    print('Generating Figure 3 ...')
    fig3_alg_comparison(alg_df_c10)

    print('\nDone.')


if __name__ == '__main__':
    main()
