#!/usr/bin/env python3
"""
Generate a comprehensive LaTeX longtable with ALL experiment conditions and ALL metrics.

Covers
------
* CIFAR-10 as ID (40 labeled): 4 algorithms × 6 OOD types × r ∈ {0,0.1,...,0.5}
* CIFAR-100 as ID (400 labeled): FixMatch × SVHN × r ∈ {0,0.1,...,0.5}
* Supervised baselines: CIFAR-10 fully-supervised (n=1), 40-label (N/A)

All metrics (mean ± std over seeds)
------------------------------------
Acc (best EMA, %), PLE, CCI, Sem-Drift, Grad-Align, OOD-FF

Output
------
  tools/paper_data/overview_table.tex          -- table snippet (include in paper)
  tools/paper_data/overview_table_standalone.tex  -- standalone compilable file

Usage
-----
  python tools/generate_overview_table.py
"""

import re, json, numpy as np
from pathlib import Path
from collections import defaultdict

# ── Paths ────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / 'tools' / 'paper_data'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Regex patterns ───────────────────────────────────────────────────────────
SSL_RE = re.compile(
    r'Epoch \[\d+/\d+\].*?'
    r'Acc:\s*([\d.]+)%.*?'
    r'PLE:\s*([\d.]+)\s+'
    r'CCI:\s*([\d.]+)\s+'
    r'S-Drift:\s*([\d.]+)\s+'
    r'OOD-FF:\s*([\d.]+).*?'
    r'G-Align:\s*(-?[\d.]+)'
)
SUP_RE  = re.compile(r'Epoch \[\d+/\d+\].*?Acc:\s*([\d.]+)%')
BEST_RE = re.compile(r'Best accuracy:\s*([\d.]+)%')

# ── Display names ─────────────────────────────────────────────────────────────
OOD_DISPLAY = {
    'cifar100':       'CIFAR-100',
    'stl10':          'STL-10',
    'svhn':           'SVHN',
    'mnist':          'MNIST',
    'gaussian_noise': 'Gaussian',
    'textures':       'Textures',
    'none':           '\\,---',
}
ALG_DISPLAY = {
    'fixmatch':  'FixMatch',
    'flexmatch': 'FlexMatch',
    'softmatch': 'SoftMatch',
    'ds3l':      'DS3L',
    'supervised':'Supervised',
}

OOD_ORDER = ['cifar100', 'stl10', 'svhn', 'mnist', 'gaussian_noise', 'textures']
ALG_ORDER = ['fixmatch', 'flexmatch', 'softmatch', 'ds3l']
R_VALS    = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# ── Metric field names ────────────────────────────────────────────────────────
METRIC_KEYS = ['best_acc', 'ple', 'cci', 'sem_drift', 'grad_align', 'ood_ff']


# ── Log parsing ───────────────────────────────────────────────────────────────

def parse_log(path: Path, is_sup: bool = False, last_n: int = 10) -> dict | None:
    epochs, best_acc = [], None
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            bm = BEST_RE.search(line)
            if bm:
                best_acc = float(bm.group(1))
            if is_sup:
                m = SUP_RE.search(line)
                if m:
                    epochs.append({k: 0.0 for k in METRIC_KEYS[1:]})
                    epochs[-1]['acc'] = float(m.group(1))
            else:
                m = SSL_RE.search(line)
                if m:
                    epochs.append({
                        'acc':        float(m.group(1)),
                        'ple':        float(m.group(2)),
                        'cci':        float(m.group(3)),
                        'sem_drift':  float(m.group(4)),
                        'ood_ff':     float(m.group(5)),
                        'grad_align': float(m.group(6)),
                    })
    if not epochs:
        return None
    tail = epochs[-last_n:]
    out = {k: float(np.mean([e[k] for e in tail]))
           for k in ('acc', 'ple', 'cci', 'sem_drift', 'grad_align', 'ood_ff')}
    out['best_acc'] = best_acc
    return out


def scan_dir(exp_dir: Path) -> dict:
    """Return { (alg, dataset, ood_dataset, r_ood) -> list[dict] }."""
    raw = defaultdict(list)
    for run in sorted(exp_dir.iterdir()):
        if not run.is_dir():
            continue
        log = run / 'log.txt'
        cfg = run / 'config.json'
        if not log.exists() or not cfg.exists():
            continue
        with open(cfg) as f:
            c = json.load(f)
        is_sup = c.get('algorithm', '') == 'supervised'
        m = parse_log(log, is_sup=is_sup)
        if m is None:
            continue
        key = (c.get('algorithm', '?'),
               c.get('dataset', 'cifar10'),
               c.get('ood_dataset', 'none'),
               float(c.get('r_ood', 0.0)))
        raw[key].append(m)
    return dict(raw)


def aggregate(runs: list[dict]) -> dict:
    out = {'n': len(runs)}
    for k in METRIC_KEYS:
        vals = [r[k] for r in runs if r.get(k) is not None]
        if not vals:
            out[f'{k}_m'] = None
            out[f'{k}_s'] = None
        else:
            out[f'{k}_m'] = float(np.mean(vals))
            out[f'{k}_s'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


# ── LaTeX cell formatters ─────────────────────────────────────────────────────

def cell(m, s, decimals=2) -> str:
    if m is None:
        return '---'
    fmt = f'.{decimals}f'
    if s is None or s == 0.0:
        return f'{m:{fmt}}'
    return f'{m:{fmt}}\\,{{\\tiny$\\pm${s:{fmt}}}}'


def acc_cell(agg: dict) -> str:
    return cell(agg.get('best_acc_m'), agg.get('best_acc_s'), 2)

def metric_cells(agg: dict, is_sup: bool = False) -> list[str]:
    """Return [Acc, PLE, CCI, Sem-Drift, Grad-Align, OOD-FF]."""
    if is_sup:
        return [acc_cell(agg), '---', '---', '---', '---', '---']
    return [
        acc_cell(agg),
        cell(agg.get('ple_m'),        agg.get('ple_s'),        3),
        cell(agg.get('cci_m'),        agg.get('cci_s'),        3),
        cell(agg.get('sem_drift_m'),  agg.get('sem_drift_s'),  3),
        cell(agg.get('grad_align_m'), agg.get('grad_align_s'), 3),
        cell(agg.get('ood_ff_m'),     agg.get('ood_ff_s'),     3),
    ]


# ── Table body builder ────────────────────────────────────────────────────────

def build_body(data: dict) -> str:
    """Build the longtable body rows (between \endfirsthead and \endlastfoot)."""
    lines = []

    def emit(*cols):
        lines.append(' & '.join(str(c) for c in cols) + r' \\')

    def hline(style='midrule'):
        lines.append(f'\\{style}')

    def section_header(text):
        lines.append(f'\\multicolumn{{9}}{{l}}{{\\textbf{{{text}}}}} \\\\')
        hline()

    # ────────────────────────────────────────────────────────────────────────
    # CIFAR-10 as ID
    # ────────────────────────────────────────────────────────────────────────
    section_header('CIFAR-10 as ID dataset (FixMatch/FlexMatch/SoftMatch/DS3L, 40 labeled samples)')

    for alg_idx, alg in enumerate(ALG_ORDER):
        alg_label = ALG_DISPLAY.get(alg, alg)

        # Baseline key: FixMatch uses svhn_r0.0, others use none_r0.0
        base_key = (alg, 'cifar10', 'svhn' if alg == 'fixmatch' else 'none', 0.0)
        base_agg = aggregate(data[base_key]) if base_key in data else {}

        # ── baseline row ──
        lines.append(f'\\multicolumn{{1}}{{l}}{{\\textbf{{{alg_label}}}}} & '
                     f'--- & 0.0 (clean) & '
                     + ' & '.join(metric_cells(base_agg)) + r' \\')
        lines.append(r'\cmidrule{2-9}')

        # ── per-OOD contaminated rows ──
        for ood_idx, ood in enumerate(OOD_ORDER):
            ood_label = OOD_DISPLAY.get(ood, ood)

            # Count available r_ood rows (r > 0) for this OOD source
            avail_r = [r for r in R_VALS[1:]
                       if (alg, 'cifar10', ood, r) in data]
            if not avail_r:
                continue

            for i, r in enumerate(avail_r):
                key  = (alg, 'cifar10', ood, r)
                agg  = aggregate(data[key])
                mcells = metric_cells(agg)

                ood_col = ood_label if i == 0 else ''
                emit('', ood_col, f'{r:.1f}', *mcells)

            # separator between OOD types
            if ood_idx < len(OOD_ORDER) - 1:
                lines.append(r'\cmidrule{2-9}')

        # separator between algorithms
        if alg_idx < len(ALG_ORDER) - 1:
            hline()
        else:
            hline()

    # ────────────────────────────────────────────────────────────────────────
    # CIFAR-100 as ID
    # ────────────────────────────────────────────────────────────────────────
    section_header('CIFAR-100 as ID dataset (FixMatch, SVHN OOD, 400 labeled samples)')

    c100_keys = [(r, (alg, 'cifar100', 'svhn', r))
                 for alg in ['fixmatch']
                 for r in R_VALS
                 if ('fixmatch', 'cifar100', 'svhn', r) in data]

    if c100_keys:
        for i, (r, key) in enumerate(c100_keys):
            agg = aggregate(data[key])
            alg_col = '\\textbf{FixMatch}' if i == 0 else ''
            ood_col = 'SVHN' if i == 0 else ''
            r_label = f'{r:.1f} (clean)' if r == 0.0 else f'{r:.1f}'
            emit(alg_col, ood_col, r_label, *metric_cells(agg))
    else:
        lines.append(r'\multicolumn{9}{c}{\textit{(CIFAR-100 sweep not found)}} \\')
    hline()

    # ────────────────────────────────────────────────────────────────────────
    # Supervised baselines
    # ────────────────────────────────────────────────────────────────────────
    section_header('Supervised baselines (no unlabeled data used)')

    # 40-label supervised (not available)
    emit(r'\textbf{Supervised}', '---', '40 labels (N/A)',
         '---', '---', '---', '---', '---', '---')

    # Fully supervised
    sup_key = ('supervised', 'cifar10', 'none', 0.0)
    if sup_key in data:
        agg = aggregate(data[sup_key])
        n_note  = f'\\,(n={agg["n"]})' if agg['n'] < 5 else ''
        acc_str = cell(agg.get('best_acc_m'), None if agg['n'] < 2 else agg.get('best_acc_s'), 2)
        emit(r'\textbf{Supervised}', '---', f'50000 labels{n_note}',
             acc_str, '---', '---', '---', '---', '---')
    else:
        emit(r'\textbf{Supervised}', '---', r'50\,000 labels (n=1)',
             '92.57', '---', '---', '---', '---', '---')

    return '\n'.join(lines)


# ── LaTeX wrappers ────────────────────────────────────────────────────────────

TABLE_ENV = r"""
\begin{longtable}{%
    l    % (1) Algorithm
    l    % (2) OOD Source
    c    % (3) r
    r    % (4) Acc (%)
    r    % (5) PLE
    r    % (6) CCI
    r    % (7) Sem-Drift
    r    % (8) Grad-Align
    r    % (9) OOD-FF
}
\caption{%
  Comprehensive experiment results: best-epoch EMA accuracy and C-Score
  diagnostic metrics (mean\,$\pm$\,std over 5 seeds unless otherwise noted).
  \textbf{Acc}: best EMA accuracy (\%);
  \textbf{PLE}: Pseudo-label Entropy;
  \textbf{CCI}: Class Concentration Index;
  \textbf{Sem-Drift}: Semantic Drift (logit space);
  \textbf{Grad-Align}: Gradient Alignment;
  \textbf{OOD-FF}: OOD Filtration Failure (oracle, requires ground-truth labels).
  ``---'' = metric not applicable.
  \textit{Near-OOD}: CIFAR-100, STL-10; \textit{Far-OOD}: SVHN, MNIST, Gaussian, Textures.%
}
\label{tab:full_overview}\\
\toprule
\textbf{Algorithm} & \textbf{OOD Source} & $r$ &
\textbf{Acc\,(\%)} & \textbf{PLE} & \textbf{CCI} &
\textbf{Sem-Drift} & \textbf{Grad-Align} & \textbf{OOD-FF} \\
\midrule
\endfirsthead

\multicolumn{9}{r}{\footnotesize\textit{(continued from previous page)}} \\[2pt]
\toprule
\textbf{Algorithm} & \textbf{OOD Source} & $r$ &
\textbf{Acc\,(\%)} & \textbf{PLE} & \textbf{CCI} &
\textbf{Sem-Drift} & \textbf{Grad-Align} & \textbf{OOD-FF} \\
\midrule
\endhead

\midrule
\multicolumn{9}{r}{\footnotesize\textit{(continued on next page)}} \\
\endfoot

\bottomrule
\endlastfoot

%%BODY%%

\end{longtable}
"""

STANDALONE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=1.2cm,landscape]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{multirow}
\usepackage{array}
\usepackage{xcolor}
\usepackage{caption}
\usepackage{lmodern}
\captionsetup{font=small}
\setlength{\LTcapwidth}{\textwidth}
\begin{document}
%%TABLE%%
\end{document}
"""

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sweep_dirs = [
        BASE / 'exp_results_alg_fixmatch_sweep',
        BASE / 'exp_results_alg_flexmatch_sweep',
        BASE / 'exp_results_alg_softmatch_sweep',
        BASE / 'exp_results_alg_ds3l_sweep',
        BASE / 'exp_results_cifar100_sweep',
        BASE / 'exp_results',
    ]

    raw_all: dict = {}
    for d in sweep_dirs:
        if not d.exists():
            print(f'[skip] {d.name}')
            continue
        print(f'Scanning {d.name} ...')
        chunk = scan_dir(d)
        for k, v in chunk.items():
            raw_all.setdefault(k, []).extend(v)
        print(f'  -> {len(chunk)} conditions in this dir')

    print(f'\nTotal unique conditions: {len(raw_all)}')

    # Quick summary
    from collections import Counter
    ctr = Counter((k[0], k[1]) for k in raw_all)
    for (alg, ds), cnt in sorted(ctr.items()):
        print(f'  {alg:12s} x {ds:10s}: {cnt} (alg,ood,r) conditions')

    # Build body
    body = build_body(raw_all)

    # Build table
    table_tex = TABLE_ENV.replace('%%BODY%%', body)
    standalone_tex = STANDALONE.replace('%%TABLE%%', table_tex)

    # Write outputs
    snip_path = OUT_DIR / 'overview_table.tex'
    sa_path   = OUT_DIR / 'overview_table_standalone.tex'

    snip_path.write_text(table_tex, encoding='utf-8')
    sa_path.write_text(standalone_tex, encoding='utf-8')

    print(f'\nSaved:')
    print(f'  {snip_path}')
    print(f'  {sa_path}')
    print('\nTo compile standalone:')
    print(f'  cd {OUT_DIR} && pdflatex overview_table_standalone.tex')


if __name__ == '__main__':
    main()
