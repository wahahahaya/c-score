#!/usr/bin/env python3
"""
Aggregate C-Score metrics from all experiment log files.
Outputs:
  tools/paper_data/all_runs.csv      -- per-seed run metrics
  tools/paper_data/aggregated.csv    -- mean ± std over seeds
  tools/paper_data/ood_type_fixmatch.csv  -- FixMatch best-acc by OOD type
  tools/paper_data/alg_comparison.csv    -- all algorithms across r_ood
  tools/paper_data/cifar100.csv          -- CIFAR-100 sweep

Usage:
  python tools/aggregate_results.py
"""
import re
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Log parsing ──────────────────────────────────────────────────────────────

EPOCH_RE = re.compile(
    r'Epoch \[(\d+)/\d+\].*?'
    r'Acc:\s*([\d.]+)%.*?'
    r'PLE:\s*([\d.]+)\s+'
    r'CCI:\s*([\d.]+)\s+'
    r'S-Drift:\s*([\d.]+)\s+'
    r'OOD-FF:\s*([\d.]+).*?'
    r'G-Align:\s*(-?[\d.]+)'
)
BEST_RE = re.compile(r'Best accuracy:\s*([\d.]+)%')


def parse_log(log_path: Path, last_n: int = 10) -> dict | None:
    """Return mean of last *last_n* epochs + best_acc from a log file."""
    epochs = []
    best_acc = None

    with open(log_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                epochs.append({
                    'epoch':      int(m.group(1)),
                    'acc':        float(m.group(2)),
                    'ple':        float(m.group(3)),
                    'cci':        float(m.group(4)),
                    'sem_drift':  float(m.group(5)),
                    'ood_ff':     float(m.group(6)),
                    'grad_align': float(m.group(7)),
                })
            bm = BEST_RE.search(line)
            if bm:
                best_acc = float(bm.group(1))

    if not epochs:
        return None

    tail = epochs[-last_n:]
    result = {k: float(np.mean([e[k] for e in tail]))
              for k in ('acc', 'ple', 'cci', 'sem_drift', 'ood_ff', 'grad_align')}
    result['best_acc'] = best_acc
    result['n_epochs'] = len(epochs)
    return result


# ── Directory scanner ────────────────────────────────────────────────────────

def scan_dir(exp_dir: Path) -> pd.DataFrame:
    records = []
    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        log_file = run_dir / 'log.txt'
        cfg_file = run_dir / 'config.json'
        if not log_file.exists() or not cfg_file.exists():
            continue

        with open(cfg_file) as f:
            cfg = json.load(f)

        metrics = parse_log(log_file)
        if metrics is None:
            print(f'  [SKIP] no epoch lines in {run_dir.name}')
            continue

        records.append({
            'algorithm':   cfg.get('algorithm', 'unknown'),
            'dataset':     cfg.get('dataset', 'cifar10'),
            'ood_dataset': cfg.get('ood_dataset', 'none'),
            'r_ood':       float(cfg.get('r_ood', 0.0)),
            'seed':        cfg.get('seed', -1),
            **metrics,
        })

    return pd.DataFrame(records)


# ── Aggregation ───────────────────────────────────────────────────────────────

GROUP_COLS = ['algorithm', 'dataset', 'ood_dataset', 'r_ood']
METRIC_COLS = ['best_acc', 'acc', 'ple', 'cci', 'sem_drift', 'ood_ff', 'grad_align']


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in df.groupby(GROUP_COLS):
        row = dict(zip(GROUP_COLS, keys))
        row['n_seeds'] = len(grp)
        for m in METRIC_COLS:
            row[f'{m}_mean'] = grp[m].mean()
            row[f'{m}_std']  = grp[m].std(ddof=1) if len(grp) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    base = Path(__file__).resolve().parent.parent
    out_dir = base / 'tools' / 'paper_data'
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_dirs = {
        'fixmatch':  base / 'exp_results_alg_fixmatch_sweep',
        'flexmatch': base / 'exp_results_alg_flexmatch_sweep',
        'softmatch': base / 'exp_results_alg_softmatch_sweep',
        'ds3l':      base / 'exp_results_alg_ds3l_sweep',
        'cifar100':  base / 'exp_results_cifar100_sweep',
    }

    all_dfs = []
    for name, d in sweep_dirs.items():
        if not d.exists():
            print(f'[SKIP] {name}: directory not found')
            continue
        print(f'Scanning {name} ({d.name}) ...')
        df = scan_dir(d)
        print(f'  -> {len(df)} runs parsed')
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_agg = aggregate(df_all)

    # ── Save full tables ──
    df_all.to_csv(out_dir / 'all_runs.csv', index=False)
    df_agg.to_csv(out_dir / 'aggregated.csv', index=False)
    print(f'\nSaved all_runs.csv ({len(df_all)} rows) and aggregated.csv ({len(df_agg)} rows)')

    # ── Print: FixMatch × SVHN × CIFAR-10 ──
    print('\n' + '='*70)
    print('Table B/C  FixMatch × SVHN × CIFAR-10 (C-Score per r_ood)')
    print('='*70)
    sel = df_agg[
        (df_agg['algorithm'] == 'fixmatch') &
        (df_agg['ood_dataset'] == 'svhn') &
        (df_agg['dataset'] == 'cifar10')
    ].sort_values('r_ood')
    cols = ['r_ood', 'best_acc_mean', 'best_acc_std',
            'cci_mean', 'cci_std', 'ple_mean', 'ple_std',
            'sem_drift_mean', 'sem_drift_std',
            'ood_ff_mean', 'ood_ff_std',
            'grad_align_mean', 'grad_align_std']
    print(sel[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    sel.to_csv(out_dir / 'fixmatch_svhn_cscore.csv', index=False)

    # ── Print: FixMatch best-acc by OOD type at r=0.5 ──
    print('\n' + '='*70)
    print('Table E  FixMatch × CIFAR-10 — best_acc by OOD type')
    print('='*70)
    sel_e = df_agg[
        (df_agg['algorithm'] == 'fixmatch') &
        (df_agg['dataset'] == 'cifar10')
    ][['ood_dataset', 'r_ood', 'best_acc_mean', 'best_acc_std',
       'cci_mean', 'cci_std']].sort_values(['ood_dataset', 'r_ood'])
    print(sel_e.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    sel_e.to_csv(out_dir / 'fixmatch_ood_type.csv', index=False)

    # ── Print: Algorithm comparison (aggregated over all OOD types) ──
    print('\n' + '='*70)
    print('Table D  Algorithm comparison (CIFAR-10, aggregated over OOD types)')
    print('='*70)
    alg_df = df_all[df_all['dataset'] == 'cifar10'].copy()
    alg_agg = alg_df.groupby(['algorithm', 'r_ood'])[METRIC_COLS].agg(['mean','std']).reset_index()
    alg_agg.columns = ['algorithm', 'r_ood'] + \
                      [f'{m}_{s}' for m in METRIC_COLS for s in ('mean', 'std')]
    alg_agg = alg_agg.sort_values(['algorithm', 'r_ood'])
    print(alg_agg[['algorithm', 'r_ood', 'best_acc_mean', 'best_acc_std',
                   'cci_mean', 'cci_std']].to_string(index=False,
                   float_format=lambda x: f'{x:.4f}'))
    alg_agg.to_csv(out_dir / 'alg_comparison.csv', index=False)

    # ── Print: CIFAR-100 ──
    print('\n' + '='*70)
    print('Table E2  CIFAR-100 × SVHN × FixMatch')
    print('='*70)
    sel_c100 = df_agg[
        (df_agg['dataset'] == 'cifar100')
    ].sort_values('r_ood')
    if not sel_c100.empty:
        print(sel_c100[['r_ood', 'best_acc_mean', 'best_acc_std',
                         'cci_mean', 'cci_std',
                         'sem_drift_mean', 'ood_ff_mean']].to_string(
            index=False, float_format=lambda x: f'{x:.4f}'))
        sel_c100.to_csv(out_dir / 'cifar100_svhn.csv', index=False)

    print(f'\nAll CSVs saved to {out_dir}')


if __name__ == '__main__':
    main()
