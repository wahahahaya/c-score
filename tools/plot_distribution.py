"""
plot_distribution.py — Distribution Plots
==============================
Statistical distributions of metrics across conditions: box plots, violin plots, histograms.

Usage (from repo root):
    python tools/plot_distribution.py

Output: tools/plots/distribution/
"""

import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.analysis import parse_log_for_metrics

OUT_DIR = os.path.join(ROOT, "tools", "plots", "distribution")
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
OOD_TYPE_LABELS = {
    "gaussian_noise": "Gaussian\nNoise",
    "svhn":           "SVHN",
    "stl10":          "STL-10",
    "cifar100":       "CIFAR-100",
    "textures":       "Textures",
    "mnist":          "MNIST",
}
ALG_COLORS = {"FlexMatch": "#2196F3", "SoftMatch": "#4CAF50", "DS3L": "#FF9800"}
ROOD_COLORS = {0.1: "#64B5F6", 0.2: "#81C784", 0.3: "#FFB74D", 0.4: "#E57373", 0.5: "#CE93D8"}


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


def _final_metrics_from_log(log_path):
    records = parse_log_for_metrics(log_path)
    if not records:
        return {}
    return records[-1]


def load_alg_sweep_summary(alg_dir):
    """Returns {r_ood: {"best_acc": [...], "cci": [...], "s_drift": [...], "ood_ff": [...], "ple": [...]}}"""
    data = {}
    if not os.path.isdir(alg_dir):
        return data
    for folder in sorted(os.listdir(alg_dir)):
        m = ALG_FOLDER_RE.match(folder)
        if not m:
            continue
        r_ood = float(m.group(1))
        log_path = os.path.join(alg_dir, folder, "log.txt")
        if not os.path.exists(log_path):
            continue
        best_acc = _best_acc_from_log(log_path)
        final    = _final_metrics_from_log(log_path)
        if r_ood not in data:
            data[r_ood] = {"best_acc": [], "cci": [], "s_drift": [], "ood_ff": [], "ple": [], "g_align": []}
        data[r_ood]["best_acc"].append(best_acc)
        for k in ["cci", "s_drift", "ood_ff", "ple", "g_align"]:
            val = final.get(k, np.nan)
            data[r_ood][k].append(val)
    return data


def load_ood_type_summary():
    """Returns {ood_type: {r_ood: {"best_acc": [...], "cci": [...], ...}}}"""
    data = {}
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
        final    = _final_metrics_from_log(log_path)
        if ood_type not in data:
            data[ood_type] = {}
        if r_ood not in data[ood_type]:
            data[ood_type][r_ood] = {"best_acc": [], "cci": [], "s_drift": [], "ood_ff": [], "ple": []}
        data[ood_type][r_ood]["best_acc"].append(best_acc)
        for k in ["cci", "s_drift", "ood_ff", "ple"]:
            data[ood_type][r_ood][k].append(final.get(k, np.nan))
    return data


# ─── Plot 1: Algorithm x r_ood box plot (best_acc)────────────────────────────────

def plot_alg_bestac_boxplot():
    """Box plot: best_acc distribution per algorithm × r_ood."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Best Accuracy Distribution: Algorithm × OOD Contamination Ratio", fontsize=13, fontweight="bold")

    for ax, (alg_name, alg_dir) in zip(axes, ALG_DIRS.items()):
        data = load_alg_sweep_summary(alg_dir)
        r_oods = sorted(data.keys())
        values = [data[r]["best_acc"] for r in r_oods]
        labels = [f"r={r}" for r in r_oods]
        bp = ax.boxplot(values, labels=labels, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for patch, r_ood in zip(bp["boxes"], r_oods):
            patch.set_facecolor(ROOD_COLORS.get(r_ood, "#BDBDBD"))
            patch.set_alpha(0.7)
        ax.set_title(alg_name, fontsize=11)
        ax.set_xlabel("r_ood")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("Best Accuracy (%)")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "01_alg_bestacc_boxplot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 2: OOD type box plot (best_acc, all r_ood combined)──────────────────

def plot_ood_type_bestac_boxplot():
    """Box plot: best_acc by OOD type (pooled across r_ood)."""
    type_data = load_ood_type_summary()

    ood_types = [t for t in OOD_TYPE_ORDER if t in type_data]
    values = []
    for ood_type in ood_types:
        all_accs = []
        for r_ood, metrics in type_data[ood_type].items():
            all_accs.extend([v for v in metrics["best_acc"] if not np.isnan(v)])
        values.append(all_accs)

    labels = [OOD_TYPE_LABELS.get(t, t) for t in ood_types]
    colors_palette = plt.cm.Set2(np.linspace(0, 1, len(ood_types)))

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(values, labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors_palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_ylabel("Best Accuracy (%)")
    ax.set_title("Best Accuracy Distribution by OOD Source Type\n(all r_ood levels pooled)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "02_ood_type_bestacc_boxplot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 3: OOD type violin plot (best_acc)─────────────────────────────────

def plot_ood_type_violin():
    """Violin plot: best_acc by OOD type."""
    type_data = load_ood_type_summary()
    ood_types = [t for t in OOD_TYPE_ORDER if t in type_data]
    values = []
    for ood_type in ood_types:
        all_accs = []
        for r_ood, metrics in type_data[ood_type].items():
            all_accs.extend([v for v in metrics["best_acc"] if not np.isnan(v)])
        values.append(np.array(all_accs))

    labels = [OOD_TYPE_LABELS.get(t, t) for t in ood_types]
    colors_palette = plt.cm.Set2(np.linspace(0, 1, len(ood_types)))

    fig, ax = plt.subplots(figsize=(10, 5))
    parts = ax.violinplot(values, positions=range(1, len(values) + 1),
                          showmedians=True, showextrema=True)

    for pc, color in zip(parts["bodies"], colors_palette):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Best Accuracy (%)")
    ax.set_title("Accuracy Distribution by OOD Source Type (Violin Plot)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "03_ood_type_bestacc_violin.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 4: C-Score metric distribution (across OOD types)────────────────────────────

def plot_cscore_dist_by_ood_type(metric="cci", metric_label="CCI"):
    """Box plot: a C-Score metric distribution by OOD type."""
    type_data = load_ood_type_summary()
    ood_types = [t for t in OOD_TYPE_ORDER if t in type_data]
    values = []
    for ood_type in ood_types:
        all_vals = []
        for r_ood, metrics in type_data[ood_type].items():
            all_vals.extend([v for v in metrics.get(metric, []) if not np.isnan(v)])
        values.append(all_vals)

    labels = [OOD_TYPE_LABELS.get(t, t) for t in ood_types]
    colors_palette = plt.cm.Set3(np.linspace(0, 1, len(ood_types)))

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(values, labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors_palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} Distribution by OOD Source Type", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"04_ood_type_{metric}_boxplot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 5: Algorithm x r_ood C-Score violin plot (multi-metric)───────────────────

def plot_alg_cscore_violin():
    """Multi-metric violin plots: one row per algorithm, one col per metric."""
    metrics    = ["cci", "s_drift", "ood_ff"]
    m_labels   = ["CCI", "Sem-Drift", "OOD-FF"]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle("C-Score Metric Distributions: Algorithm × Metric", fontsize=13, fontweight="bold")

    for row, (alg_name, alg_dir) in enumerate(ALG_DIRS.items()):
        data = load_alg_sweep_summary(alg_dir)
        r_oods = sorted(data.keys())

        for col, (metric, m_label) in enumerate(zip(metrics, m_labels)):
            ax = axes[row, col]
            values = [[v for v in data[r][metric] if not np.isnan(v)] for r in r_oods]
            # only violin if enough data
            valid = [(v, r) for v, r in zip(values, r_oods) if len(v) >= 2]
            if valid:
                vals, r_vals = zip(*valid)
                parts = ax.violinplot(vals, positions=range(1, len(vals) + 1),
                                      showmedians=True)
                for pc, r_ood in zip(parts["bodies"], r_vals):
                    pc.set_facecolor(ROOD_COLORS.get(r_ood, "#BDBDBD"))
                    pc.set_alpha(0.7)
                ax.set_xticks(range(1, len(vals) + 1))
                ax.set_xticklabels([f"r={r:.1f}" for r in r_vals], fontsize=8)

            if col == 0:
                ax.set_ylabel(alg_name, fontsize=10, fontweight="bold")
            if row == 0:
                ax.set_title(m_label, fontsize=10)
            ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "05_alg_cscore_violin.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 6: OOD-FF distribution (oracle metric)──────────────────────────────────

def plot_oodff_distribution():
    """OOD-FF distribution across OOD types; reveals filtration failure."""
    type_data = load_ood_type_summary()
    ood_types = [t for t in OOD_TYPE_ORDER if t in type_data]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle("OOD-FF Distribution by Contamination Ratio per OOD Type\n(OOD-FF > 1 = OOD leaks more than ID)", fontsize=12, fontweight="bold")

    for ax, ood_type in zip(axes.flat, ood_types):
        r_oods = sorted(type_data[ood_type].keys())
        for r_ood in r_oods:
            vals = [v for v in type_data[ood_type][r_ood]["ood_ff"] if not np.isnan(v)]
            if vals:
                ax.hist(vals, bins=10, alpha=0.6, label=f"r={r_ood}",
                        color=ROOD_COLORS.get(r_ood, "gray"))
        ax.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="OOD-FF=1")
        ax.set_title(OOD_TYPE_LABELS.get(ood_type, ood_type), fontsize=10)
        ax.set_xlabel("OOD-FF")
        ax.set_ylabel("Count")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "06_ood_ff_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 7: Accuracy degradation distribution (relative to r_ood=0 baseline) ────

def plot_acc_drop_distribution():
    """Histogram of accuracy drop relative to r_ood baseline per OOD type."""
    type_data = load_ood_type_summary()
    ood_types = [t for t in OOD_TYPE_ORDER if t in type_data]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_palette = plt.cm.Set2(np.linspace(0, 1, len(ood_types)))

    for ood_type, color in zip(ood_types, colors_palette):
        # find baseline (r_ood=0.0 or smallest)
        r_oods = sorted(type_data[ood_type].keys())
        if len(r_oods) == 0:
            continue
        baseline_r = r_oods[0]
        baseline_accs = [v for v in type_data[ood_type][baseline_r]["best_acc"] if not np.isnan(v)]
        if not baseline_accs:
            continue
        baseline_mean = np.mean(baseline_accs)

        drops = []
        for r_ood in r_oods[1:]:
            accs = [v for v in type_data[ood_type][r_ood]["best_acc"] if not np.isnan(v)]
            for acc in accs:
                drops.append(baseline_mean - acc)

        if drops:
            ax.hist(drops, bins=15, alpha=0.5, label=OOD_TYPE_LABELS.get(ood_type, ood_type),
                    color=color, edgecolor="white")

    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="No drop")
    ax.set_xlabel("Accuracy Drop vs r_ood=0 Baseline (%)")
    ax.set_ylabel("Count (over seeds × r_ood levels)")
    ax.set_title("Distribution of Accuracy Degradation by OOD Type", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, title="OOD Type")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "07_acc_drop_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  plot_distribution.py — Distribution Plots")
    print(f"  Output: {OUT_DIR}")
    print("=" * 60)

    print("\n[1/7] Algorithm best-accuracy box plots ...")
    plot_alg_bestac_boxplot()

    print("\n[2/7] OOD type best-accuracy box plot ...")
    plot_ood_type_bestac_boxplot()

    print("\n[3/7] OOD type best-accuracy violin plot ...")
    plot_ood_type_violin()

    print("\n[4/7] CCI distribution by OOD type ...")
    plot_cscore_dist_by_ood_type("cci", "CCI")

    print("\n[4b] Sem-Drift distribution by OOD type ...")
    plot_cscore_dist_by_ood_type("s_drift", "Sem-Drift")

    print("\n[5/7] Algorithm C-Score violin plots ...")
    plot_alg_cscore_violin()

    print("\n[6/7] OOD-FF distribution ...")
    plot_oodff_distribution()

    print("\n[7/7] Accuracy drop distribution ...")
    plot_acc_drop_distribution()

    print(f"\nDone. All plots saved to: {OUT_DIR}")
