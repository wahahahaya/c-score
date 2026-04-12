"""
plot_stats.py — Statistical Plots
======================
Algorithm comparison, OOD type comparison, C-Score vs. accuracy correlation analysis.

Usage (from repo root):
    python tools/plot_stats.py

Output: tools/plots/stats/
"""

import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from itertools import product

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.analysis import parse_log_for_metrics

OUT_DIR = os.path.join(ROOT, "tools", "plots", "stats")
os.makedirs(OUT_DIR, exist_ok=True)

ALG_DIRS = {
    "FlexMatch": os.path.join(ROOT, "exp_results_alg_flexmatch_sweep"),
    "SoftMatch": os.path.join(ROOT, "exp_results_alg_softmatch_sweep"),
    "DS3L":      os.path.join(ROOT, "exp_results_alg_ds3l_sweep"),
}
OOD_TYPE_DIR = os.path.join(ROOT, "exp_results_ood_type_sweep")

ALG_FOLDER_RE = re.compile(r"alg_\w+_\w+_r([0-9.]+)_s([0-9]+)_")
OOD_FOLDER_RE = re.compile(r"ood_(.+?)_r([0-9.]+)_s([0-9]+)_")
BEST_ACC_RE   = re.compile(r"Best accuracy:\s*([0-9.]+)")

OOD_TYPE_ORDER = ["gaussian_noise", "svhn", "stl10", "cifar100", "textures", "mnist"]
OOD_SHORT = {
    "gaussian_noise": "Gauss.", "svhn": "SVHN", "stl10": "STL-10",
    "cifar100": "CIFAR-100", "textures": "Textures", "mnist": "MNIST",
}
ALG_COLORS = {"FlexMatch": "#2196F3", "SoftMatch": "#4CAF50", "DS3L": "#FF9800"}


# ─── Parsers ─────────────────────────────────────────────────────────────────

def _best_acc_from_log(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()
    m = BEST_ACC_RE.search(text)
    if m:
        return float(m.group(1))
    records = parse_log_for_metrics(log_path)
    if records:
        accs = [r["acc"] for r in records if not np.isnan(r["acc"])]
        return max(accs) if accs else np.nan
    return np.nan


def _final_metrics(log_path):
    records = parse_log_for_metrics(log_path)
    return records[-1] if records else {}


def load_alg_records():
    """Returns {alg_name: {r_ood: {"best_acc": [...], "cci": [...], ...}}}"""
    all_data = {}
    for alg_name, alg_dir in ALG_DIRS.items():
        data = defaultdict(lambda: defaultdict(list))
        if not os.path.isdir(alg_dir):
            continue
        for folder in sorted(os.listdir(alg_dir)):
            m = ALG_FOLDER_RE.match(folder)
            if not m:
                continue
            r_ood = float(m.group(1))
            log_path = os.path.join(alg_dir, folder, "log.txt")
            if not os.path.exists(log_path):
                continue
            best_acc = _best_acc_from_log(log_path)
            final    = _final_metrics(log_path)
            data[r_ood]["best_acc"].append(best_acc)
            for k in ["cci", "s_drift", "ood_ff", "ple", "g_align"]:
                data[r_ood][k].append(final.get(k, np.nan))
        all_data[alg_name] = {r: dict(v) for r, v in data.items()}
    return all_data


def load_ood_type_records():
    """Returns {ood_type: {r_ood: {"best_acc": [...], ...}}}"""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    if not os.path.isdir(OOD_TYPE_DIR):
        return data
    for folder in sorted(os.listdir(OOD_TYPE_DIR)):
        m = OOD_FOLDER_RE.match(folder)
        if not m:
            continue
        ood_type, r_ood = m.group(1), float(m.group(2))
        log_path = os.path.join(OOD_TYPE_DIR, folder, "log.txt")
        if not os.path.exists(log_path):
            continue
        best_acc = _best_acc_from_log(log_path)
        final    = _final_metrics(log_path)
        data[ood_type][r_ood]["best_acc"].append(best_acc)
        for k in ["cci", "s_drift", "ood_ff", "ple"]:
            data[ood_type][r_ood][k].append(final.get(k, np.nan))
    return {k: {r: dict(v) for r, v in rv.items()} for k, rv in data.items()}


# ─── Plot 1: Algorithm comparison bar chart (grouped by r_ood)────────────────────────────────

def plot_alg_comparison_bar():
    """Grouped bar: mean best_acc per algorithm per r_ood."""
    alg_data = load_alg_records()
    alg_names = list(ALG_DIRS.keys())
    r_oods = sorted(set(r for d in alg_data.values() for r in d.keys()))

    x = np.arange(len(r_oods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, alg_name in enumerate(alg_names):
        means = []
        stds  = []
        for r_ood in r_oods:
            vals = [v for v in alg_data[alg_name].get(r_ood, {}).get("best_acc", []) if not np.isnan(v)]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        bars = ax.bar(x + i * width, means, width, yerr=stds, capsize=4,
                      label=alg_name, color=ALG_COLORS[alg_name], alpha=0.85, error_kw=dict(elinewidth=1))
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{mean:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("OOD Contamination Ratio (r_ood)", fontsize=11)
    ax.set_ylabel("Best Accuracy (%) ± std", fontsize=11)
    ax.set_title("Algorithm Comparison: Best Accuracy vs OOD Contamination (CIFAR-100 OOD)", fontsize=11, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"r_ood={r:.1f}" for r in r_oods])
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "01_alg_comparison_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 2: OOD type x r_ood heatmap (best_acc)─────────────────────────────

def plot_ood_type_rood_heatmap():
    """Heatmap: rows=OOD type, cols=r_ood, values=mean best_acc."""
    ood_data = load_ood_type_records()
    ood_types = [t for t in OOD_TYPE_ORDER if t in ood_data]
    r_oods    = sorted(set(r for d in ood_data.values() for r in d.keys() if r > 0))

    grid = np.full((len(ood_types), len(r_oods)), np.nan)
    for i, ood_type in enumerate(ood_types):
        for j, r_ood in enumerate(r_oods):
            vals = [v for v in ood_data[ood_type].get(r_ood, {}).get("best_acc", []) if not np.isnan(v)]
            if vals:
                grid[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=30, vmax=90, aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean Best Accuracy (%)")

    ax.set_xticks(range(len(r_oods)))
    ax.set_xticklabels([f"r={r:.1f}" for r in r_oods])
    ax.set_yticks(range(len(ood_types)))
    ax.set_yticklabels([OOD_SHORT.get(t, t) for t in ood_types])
    ax.set_xlabel("OOD Contamination Ratio (r_ood)")
    ax.set_title("Best Accuracy Heatmap: OOD Type × Contamination Ratio", fontsize=12, fontweight="bold")

    for i in range(len(ood_types)):
        for j in range(len(r_oods)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="black" if 40 < grid[i, j] < 80 else "white")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "02_ood_type_rood_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 3: C-Score vs. accuracy correlation scatter plot ──────────────────────

def plot_cscore_vs_acc_scatter():
    """Scatter plots: C-Score metrics vs best_acc, colored by r_ood."""
    alg_data = load_alg_records()

    # Pool all runs across algorithms and r_ood
    metrics   = ["cci", "s_drift", "ood_ff", "ple"]
    m_labels  = ["CCI", "Sem-Drift", "OOD-FF", "PLE"]
    alg_markers = {"FlexMatch": "o", "SoftMatch": "s", "DS3L": "^"}

    rood_cmap = plt.cm.plasma

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("C-Score Metrics vs Best Accuracy\n(each dot = one run, color = r_ood, shape = algorithm)", fontsize=12, fontweight="bold")

    for ax, metric, m_label in zip(axes.flat, metrics, m_labels):
        scatter_plots = {}
        for alg_name, data in alg_data.items():
            r_oods = sorted(data.keys())
            for r_ood in r_oods:
                xs = [v for v in data[r_ood].get(metric, []) if not np.isnan(v)]
                ys = [v for v in data[r_ood].get("best_acc", []) if not np.isnan(v)]
                n  = min(len(xs), len(ys))
                if n == 0:
                    continue
                color = rood_cmap(r_ood / 0.5)
                sc = ax.scatter(xs[:n], ys[:n], c=[color] * n,
                                marker=alg_markers[alg_name],
                                alpha=0.7, s=40, edgecolors="none")

        # Correlation line (all data pooled)
        all_x, all_y = [], []
        for alg_name, data in alg_data.items():
            for r_ood, vals in data.items():
                xs = [v for v in vals.get(metric, []) if not np.isnan(v)]
                ys = [v for v in vals.get("best_acc", []) if not np.isnan(v)]
                n  = min(len(xs), len(ys))
                all_x.extend(xs[:n])
                all_y.extend(ys[:n])

        if len(all_x) > 2:
            coeffs = np.polyfit(all_x, all_y, 1)
            xline  = np.linspace(min(all_x), max(all_x), 100)
            ax.plot(xline, np.polyval(coeffs, xline), "r--", linewidth=1.5, alpha=0.7, label="trend")
            corr   = np.corrcoef(all_x, all_y)[0, 1]
            ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=9,
                    va="top", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        ax.set_xlabel(m_label, fontsize=10)
        ax.set_ylabel("Best Accuracy (%)")
        ax.set_title(f"{m_label} vs Best Accuracy")
        ax.grid(True, alpha=0.3)

        # Shape legend
        for alg_name, marker in alg_markers.items():
            ax.scatter([], [], marker=marker, color="gray", label=alg_name, s=40)
        ax.legend(fontsize=7, loc="upper right")

    # Colorbar for r_ood
    sm = plt.cm.ScalarMappable(cmap=rood_cmap, norm=plt.Normalize(0.1, 0.5))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("r_ood", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    path = os.path.join(OUT_DIR, "03_cscore_vs_acc_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 4: C-Score metric correlation matrix (heatmap)─────────────────────

def plot_correlation_heatmap():
    """Pearson correlation matrix across all final metrics."""
    alg_data = load_alg_records()

    keys = ["best_acc", "cci", "s_drift", "ood_ff", "ple", "g_align"]
    labels = ["BestAcc", "CCI", "Sem-Drift", "OOD-FF", "PLE", "Grad-Align"]

    # Collect all values
    all_vals = {k: [] for k in keys}
    for alg_name, data in alg_data.items():
        for r_ood, vals in data.items():
            n = len(vals.get("best_acc", []))
            for k in keys:
                col = vals.get(k, [np.nan] * n)
                all_vals[k].extend(col[:n])

    # Build matrix
    mat = np.full((len(keys), len(keys)), np.nan)
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            xi = np.array(all_vals[ki], dtype=float)
            xj = np.array(all_vals[kj], dtype=float)
            mask = ~np.isnan(xi) & ~np.isnan(xj)
            if mask.sum() > 3:
                mat[i, j] = np.corrcoef(xi[mask], xj[mask])[0, 1]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Pearson r")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("C-Score Metric Correlation Matrix\n(pooled across all algorithms & r_ood)", fontsize=12, fontweight="bold")

    for i in range(len(keys)):
        for j in range(len(keys)):
            if not np.isnan(mat[i, j]):
                color = "white" if abs(mat[i, j]) > 0.6 else "black"
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, color=color)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "04_correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 5: OOD type x C-Score metric heatmap ────────────────────────────────

def plot_ood_type_cscore_heatmap():
    """Heatmap: rows=OOD type, cols=C-Score metrics (mean across r_ood)."""
    ood_data  = load_ood_type_records()
    ood_types = [t for t in OOD_TYPE_ORDER if t in ood_data]
    metrics   = ["cci", "s_drift", "ood_ff", "ple"]
    m_labels  = ["CCI", "Sem-Drift", "OOD-FF", "PLE"]

    grid = np.full((len(ood_types), len(metrics)), np.nan)
    for i, ood_type in enumerate(ood_types):
        for j, metric in enumerate(metrics):
            all_vals = []
            for r_ood in ood_data[ood_type]:
                if r_ood == 0.0:
                    continue
                all_vals.extend([v for v in ood_data[ood_type][r_ood].get(metric, []) if not np.isnan(v)])
            if all_vals:
                grid[i, j] = np.mean(all_vals)

    # Normalize per column for visualization
    grid_norm = np.copy(grid)
    for j in range(len(metrics)):
        col = grid_norm[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            col_min, col_max = valid.min(), valid.max()
            if col_max > col_min:
                grid_norm[:, j] = (col - col_min) / (col_max - col_min)

    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("C-Score Metrics by OOD Source Type (mean across r_ood > 0)", fontsize=12, fontweight="bold")

    # Raw values
    im1 = ax_raw.imshow(grid, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im1, ax=ax_raw, label="Raw Mean Value")
    ax_raw.set_xticks(range(len(m_labels)))
    ax_raw.set_xticklabels(m_labels)
    ax_raw.set_yticks(range(len(ood_types)))
    ax_raw.set_yticklabels([OOD_SHORT.get(t, t) for t in ood_types])
    ax_raw.set_title("Raw Values")
    for i in range(len(ood_types)):
        for j in range(len(metrics)):
            if not np.isnan(grid[i, j]):
                ax_raw.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=8)

    # Normalized values
    im2 = ax_norm.imshow(grid_norm, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im2, ax=ax_norm, label="Normalized [0,1]")
    ax_norm.set_xticks(range(len(m_labels)))
    ax_norm.set_xticklabels(m_labels)
    ax_norm.set_yticks(range(len(ood_types)))
    ax_norm.set_yticklabels([OOD_SHORT.get(t, t) for t in ood_types])
    ax_norm.set_title("Column-Normalized Values")
    for i in range(len(ood_types)):
        for j in range(len(metrics)):
            if not np.isnan(grid_norm[i, j]):
                ax_norm.text(j, i, f"{grid_norm[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "05_ood_type_cscore_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 6: Algorithm x C-Score summary bar chart ──────────────────────────────

def plot_alg_cscore_summary_bar():
    """Bar chart: mean C-Score metrics per algorithm (averaged over all r_ood > 0)."""
    alg_data  = load_alg_records()
    alg_names = list(ALG_DIRS.keys())
    metrics   = ["cci", "s_drift", "ood_ff"]
    m_labels  = ["CCI", "Sem-Drift", "OOD-FF"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Mean C-Score Metrics by Algorithm (all r_ood > 0, CIFAR-100 OOD)", fontsize=12, fontweight="bold")

    for ax, metric, m_label in zip(axes, metrics, m_labels):
        means, stds = [], []
        for alg_name in alg_names:
            all_vals = []
            for r_ood, vals in alg_data[alg_name].items():
                if r_ood > 0:
                    all_vals.extend([v for v in vals.get(metric, []) if not np.isnan(v)])
            means.append(np.mean(all_vals) if all_vals else 0)
            stds.append(np.std(all_vals) if all_vals else 0)

        colors = [ALG_COLORS[a] for a in alg_names]
        bars = ax.bar(alg_names, means, yerr=stds, capsize=5,
                      color=colors, alpha=0.85, error_kw=dict(elinewidth=1))
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.05,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_ylabel(m_label)
        ax.set_title(m_label)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "06_alg_cscore_summary_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 7: Summary figure (for paper) ──────────────────────────────────────────

def plot_summary_figure():
    """
    Paper-style summary figure:
    Row 1: acc vs r_ood (3 algorithms) + OOD type heatmap
    Row 2: CCI vs r_ood + Sem-Drift vs r_ood
    """
    alg_data = load_alg_records()
    ood_data = load_ood_type_records()

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("C-Score Summary: Accuracy Masking Under Open-World Unlabeled Contamination", fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    alg_names = list(ALG_DIRS.keys())

    # (A) Best Acc vs r_ood per algorithm
    ax_acc = fig.add_subplot(gs[0, :2])
    for alg_name in alg_names:
        data = alg_data[alg_name]
        r_oods = sorted(data.keys())
        means = [np.nanmean(data[r].get("best_acc", [np.nan])) for r in r_oods]
        stds  = [np.nanstd(data[r].get("best_acc", [np.nan]))  for r in r_oods]
        ax_acc.errorbar(r_oods, means, yerr=stds, marker="o", linewidth=2, capsize=4,
                        label=alg_name, color=ALG_COLORS[alg_name])
    ax_acc.set_xlabel("r_ood")
    ax_acc.set_ylabel("Best Accuracy (%)")
    ax_acc.set_title("(A) Best Accuracy vs Contamination Ratio", fontsize=10)
    ax_acc.set_ylim(40, 100)
    ax_acc.legend(fontsize=9)
    ax_acc.grid(True, alpha=0.3)

    # (B) OOD type heatmap (best_acc)
    ax_hm = fig.add_subplot(gs[0, 2:])
    ood_types = [t for t in OOD_TYPE_ORDER if t in ood_data]
    r_oods_ood = sorted(set(r for d in ood_data.values() for r in d.keys() if r > 0))
    grid = np.full((len(ood_types), len(r_oods_ood)), np.nan)
    for i, ood_type in enumerate(ood_types):
        for j, r_ood in enumerate(r_oods_ood):
            vals = [v for v in ood_data[ood_type].get(r_ood, {}).get("best_acc", []) if not np.isnan(v)]
            if vals:
                grid[i, j] = np.mean(vals)
    im = ax_hm.imshow(grid, cmap="RdYlGn", vmin=30, vmax=90, aspect="auto")
    plt.colorbar(im, ax=ax_hm, label="Acc (%)", shrink=0.9)
    ax_hm.set_xticks(range(len(r_oods_ood)))
    ax_hm.set_xticklabels([f"r={r:.1f}" for r in r_oods_ood], fontsize=8)
    ax_hm.set_yticks(range(len(ood_types)))
    ax_hm.set_yticklabels([OOD_SHORT.get(t, t) for t in ood_types], fontsize=8)
    ax_hm.set_title("(B) Accuracy Heatmap: OOD Type × r_ood", fontsize=10)
    for i in range(len(ood_types)):
        for j in range(len(r_oods_ood)):
            if not np.isnan(grid[i, j]):
                ax_hm.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=7,
                           color="black" if 40 < grid[i, j] < 80 else "white")

    # (C) CCI vs r_ood per algorithm
    ax_cci = fig.add_subplot(gs[1, :2])
    for alg_name in alg_names:
        data = alg_data[alg_name]
        r_oods = sorted(data.keys())
        means = [np.nanmean(data[r].get("cci", [np.nan])) for r in r_oods]
        stds  = [np.nanstd(data[r].get("cci", [np.nan]))  for r in r_oods]
        ax_cci.errorbar(r_oods, means, yerr=stds, marker="s", linewidth=2, capsize=4,
                        label=alg_name, color=ALG_COLORS[alg_name])
    ax_cci.set_xlabel("r_ood")
    ax_cci.set_ylabel("CCI")
    ax_cci.set_title("(C) CCI vs Contamination Ratio\n(internal degradation signal)", fontsize=10)
    ax_cci.legend(fontsize=9)
    ax_cci.grid(True, alpha=0.3)
    ax_cci.text(0.02, 0.98, "Higher CCI → class collapse", transform=ax_cci.transAxes,
                fontsize=8, va="top", color="red",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))

    # (D) Sem-Drift vs r_ood per algorithm
    ax_sd = fig.add_subplot(gs[1, 2:])
    for alg_name in alg_names:
        data = alg_data[alg_name]
        r_oods = sorted(data.keys())
        means = [np.nanmean(data[r].get("s_drift", [np.nan])) for r in r_oods]
        stds  = [np.nanstd(data[r].get("s_drift", [np.nan]))  for r in r_oods]
        ax_sd.errorbar(r_oods, means, yerr=stds, marker="^", linewidth=2, capsize=4,
                       label=alg_name, color=ALG_COLORS[alg_name])
    ax_sd.set_xlabel("r_ood")
    ax_sd.set_ylabel("Sem-Drift")
    ax_sd.set_title("(D) Sem-Drift vs Contamination Ratio\n(feature space divergence)", fontsize=10)
    ax_sd.legend(fontsize=9)
    ax_sd.grid(True, alpha=0.3)
    ax_sd.text(0.02, 0.98, "Higher Sem-Drift → feature corruption", transform=ax_sd.transAxes,
               fontsize=8, va="top", color="red",
               bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))

    path = os.path.join(OUT_DIR, "07_summary_figure.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  plot_stats.py — Statistical Plots")
    print(f"  Output: {OUT_DIR}")
    print("=" * 60)

    print("\n[1/7] Algorithm comparison bar chart ...")
    plot_alg_comparison_bar()

    print("\n[2/7] OOD type × r_ood heatmap ...")
    plot_ood_type_rood_heatmap()

    print("\n[3/7] C-Score vs accuracy scatter ...")
    plot_cscore_vs_acc_scatter()

    print("\n[4/7] Correlation heatmap ...")
    plot_correlation_heatmap()

    print("\n[5/7] OOD type C-Score heatmap ...")
    plot_ood_type_cscore_heatmap()

    print("\n[6/7] Algorithm C-Score summary bar ...")
    plot_alg_cscore_summary_bar()

    print("\n[7/7] Summary figure (paper-style) ...")
    plot_summary_figure()

    print(f"\nDone. All plots saved to: {OUT_DIR}")
