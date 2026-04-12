"""
alg_comparison_bar.py
=====================
Two-panel grouped bar chart comparing FixMatch / FlexMatch / SoftMatch / DS3L
on CIFAR-10 + SVHN contamination.

  Panel (a): Best Accuracy vs. contamination ratio r
  Panel (b): Final-stage CCI    vs. contamination ratio r

Usage
-----
  python tools/alg_comparison_bar.py \
      --base_dir <repo-root> \
      --out_dir  tools/plots/alg_comparison \
      --dpi 300
"""

import argparse
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Configuration ─────────────────────────────────────────────────────────────
RATIOS  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SEEDS   = [0, 10, 20, 30, 40]
LAST_N  = 20          # epochs used to compute final-stage CCI mean

# Algorithm display config  (Paul Tol bright palette – colourblind-safe)
ALG_CFG = {
    'FixMatch':  {'color': '#4477AA', 'edgecolor': '#1a2e47'},
    'FlexMatch': {'color': '#EE6677', 'edgecolor': '#7a1a22'},
    'SoftMatch': {'color': '#228833', 'edgecolor': '#0d3a16'},
    'DS3L':      {'color': '#CCBB44', 'edgecolor': '#6b6010'},
}
ALGS = list(ALG_CFG.keys())


def find_log(base_dir: str, alg: str, ratio: float, seed: int) -> str | None:
    """Return path to log.txt for the given (alg, ratio, seed), or None."""
    r_str = f'{ratio:.1f}'
    s_str = str(seed)

    if alg == 'FixMatch':
        root = os.path.join(base_dir, 'exp_results_ood_type_sweep')
        pattern = os.path.join(root, f'ood_svhn_r{r_str}_s{s_str}_*', 'log.txt')
    else:
        alg_lower = alg.lower()
        root = os.path.join(base_dir, f'exp_results_alg_{alg_lower}_sweep')
        if ratio == 0.0:
            pattern = os.path.join(root, f'alg_{alg_lower}_none_r{r_str}_s{s_str}_*', 'log.txt')
        else:
            pattern = os.path.join(root, f'alg_{alg_lower}_svhn_r{r_str}_s{s_str}_*', 'log.txt')

    matches = glob.glob(pattern)
    if not matches:
        return None
    return sorted(matches)[-1]   # take the latest if duplicates


def parse_log(log_path: str, last_n: int = LAST_N) -> dict:
    """
    Parse a log file and return:
      best_acc  - float from the 'Best accuracy' summary line
      cci_mean  - mean CCI over the last `last_n` Epoch lines
      cci_std   - std  CCI over the last `last_n` Epoch lines
    Returns None if file is not found or parsing fails.
    """
    if log_path is None or not os.path.isfile(log_path):
        return None

    epoch_re = re.compile(
        r'Epoch\s+\[\d+/\d+\].*?Acc:\s*([\d.]+)%.*?CCI:\s*([\d.]+)'
    )
    best_re  = re.compile(r'Best accuracy:\s*([\d.]+)%')

    epoch_rows = []   # list of (acc, cci)
    best_acc   = None

    with open(log_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            m = epoch_re.search(line)
            if m:
                epoch_rows.append((float(m.group(1)), float(m.group(2))))
            m2 = best_re.search(line)
            if m2:
                best_acc = float(m2.group(1))

    if not epoch_rows:
        return None

    # Fall back to max epoch-level acc if summary line is missing
    if best_acc is None:
        best_acc = max(r[0] for r in epoch_rows)

    tail_cci = [r[1] for r in epoch_rows[-last_n:]]
    return {
        'best_acc': best_acc,
        'cci_mean': float(np.mean(tail_cci)),
        'cci_std':  float(np.std(tail_cci)),
    }


def collect_results(base_dir: str) -> dict:
    """
    Returns nested dict:
      results[alg][ratio] = {'acc': [..], 'cci': [..]}
    where each list has one entry per seed.
    """
    results = {alg: {r: {'acc': [], 'cci': []} for r in RATIOS} for alg in ALGS}

    for alg in ALGS:
        for ratio in RATIOS:
            for seed in SEEDS:
                log = find_log(base_dir, alg, ratio, seed)
                parsed = parse_log(log)
                if parsed is None:
                    print(f'  [WARN] missing: {alg} r={ratio} s={seed}')
                    continue
                results[alg][ratio]['acc'].append(parsed['best_acc'])
                results[alg][ratio]['cci'].append(parsed['cci_mean'])

    return results


def compute_stats(results: dict) -> dict:
    """
    Returns nested dict:
      stats[alg][ratio] = {
          'acc_mean', 'acc_std',
          'cci_mean', 'cci_std',
      }
    """
    stats = {}
    for alg in ALGS:
        stats[alg] = {}
        for ratio in RATIOS:
            acc_vals = results[alg][ratio]['acc']
            cci_vals = results[alg][ratio]['cci']
            stats[alg][ratio] = {
                'acc_mean': np.mean(acc_vals) if acc_vals else np.nan,
                'acc_std':  np.std(acc_vals)  if acc_vals else np.nan,
                'cci_mean': np.mean(cci_vals) if cci_vals else np.nan,
                'cci_std':  np.std(cci_vals)  if cci_vals else np.nan,
            }
    return stats


def _draw_panel(ax, stats, m_mean, m_std, title, ylabel, x, bar_w, n_bars,
                show_legend: bool):
    for i, alg in enumerate(ALGS):
        cfg    = ALG_CFG[alg]
        means  = np.array([stats[alg][r][m_mean] for r in RATIOS])
        stds   = np.array([stats[alg][r][m_std]  for r in RATIOS])
        offset = (i - (n_bars - 1) / 2) * bar_w

        ax.bar(
            x + offset, means,
            width=bar_w * 0.88,
            color=cfg['color'],
            edgecolor=cfg['edgecolor'],
            linewidth=0.7,
            label=alg,
            zorder=3,
        )
        ax.errorbar(
            x + offset, means, yerr=stds,
            fmt='none',
            ecolor='#333333',
            elinewidth=0.9,
            capsize=2.2,
            capthick=0.9,
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f'r={r:.1f}' for r in RATIOS], fontsize=15)
    ax.set_ylabel(ylabel, fontsize=16, labelpad=4)
    ax.set_title(title, fontsize=17, fontweight='normal', pad=8)

    ax.yaxis.grid(True, linestyle=':', linewidth=0.6, color='#bbbbbb', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='both', which='both', length=3, width=0.8, labelsize=15)

    if show_legend:
        handles = [
            mpatches.Patch(
                facecolor=ALG_CFG[a]['color'],
                edgecolor=ALG_CFG[a]['edgecolor'],
                linewidth=0.7,
                label=a,
            )
            for a in ALGS
        ]
        ax.legend(
            handles=handles, fontsize=14,
            framealpha=0.92, edgecolor='#cccccc',
            loc='lower right', ncol=2, handlelength=1.4,
            handletextpad=0.5, columnspacing=1.0,
        )


def plot_comparison(stats: dict, out_dir: str, dpi: int = 300):
    os.makedirs(out_dir, exist_ok=True)

    n_groups = len(RATIOS)
    n_bars   = len(ALGS)
    total_w  = 0.75
    bar_w    = total_w / n_bars
    x        = np.arange(n_groups)

    panel_cfg = [
        ('acc_mean', 'acc_std',
         '(a) Best Accuracy vs. Contamination Ratio',
         'Best Accuracy (%)', True),
        ('cci_mean', 'cci_std',
         '(b) CCI vs. Contamination Ratio',
         'Class Concentration Index (CCI)', False),
    ]

    for fmt in ('png', 'pdf'):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
        fig.subplots_adjust(wspace=0.22)

        for (m_mean, m_std, title, ylabel, show_leg), ax in zip(panel_cfg, axes):
            _draw_panel(ax, stats, m_mean, m_std, title, ylabel,
                        x, bar_w, n_bars, show_legend=show_leg)

        out_path = os.path.join(out_dir, f'alg_comparison_svhn.{fmt}')
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'[saved] {out_path}')


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Algorithm comparison bar chart (SVHN)')
    parser.add_argument('--base_dir', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--out_dir',  default='tools/plots/alg_comparison')
    parser.add_argument('--dpi',      type=int, default=600)
    args = parser.parse_args()

    print('[alg_comparison_bar] Collecting results ...')
    results = collect_results(args.base_dir)

    print('[alg_comparison_bar] Computing statistics ...')
    stats = compute_stats(results)

    # Print summary table
    print(f'\n{"Alg":<12}{"r":>6}  {"Acc(mean+-std)":>18}  {"CCI(mean+-std)":>18}')
    print('-' * 60)
    for alg in ALGS:
        for ratio in RATIOS:
            s = stats[alg][ratio]
            print(f'{alg:<12}{ratio:>6.1f}  '
                  f'{s["acc_mean"]:>7.2f}+-{s["acc_std"]:.2f}      '
                  f'{s["cci_mean"]:>6.4f}+-{s["cci_std"]:.4f}')

    print('\n[alg_comparison_bar] Plotting ...')
    plot_comparison(stats, args.out_dir, dpi=args.dpi)
    print('[alg_comparison_bar] Done.')
