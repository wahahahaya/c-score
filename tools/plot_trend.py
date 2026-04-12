"""
plot_trend.py — Trend Plots
======================
Training dynamics: Accuracy and C-Score metrics as functions of epoch and r_ood.

Usage (from repo root):
    python tools/plot_trend.py

Output: tools/plots/trend/
"""

import os
import re
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict

# ── Add project root so analysis.py can be imported ──────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.analysis import parse_log_for_metrics

OUT_DIR = os.path.join(ROOT, "tools", "plots", "trend")
os.makedirs(OUT_DIR, exist_ok=True)

ALG_DIRS = {
    "FlexMatch": os.path.join(ROOT, "exp_results_alg_flexmatch_sweep"),
    "SoftMatch": os.path.join(ROOT, "exp_results_alg_softmatch_sweep"),
    "DS3L":      os.path.join(ROOT, "exp_results_alg_ds3l_sweep"),
}
OOD_TYPE_DIR = os.path.join(ROOT, "exp_results_ood_type_sweep")

ALG_FOLDER_RE   = re.compile(r"alg_\w+_\w+_r([0-9.]+)_s([0-9]+)_")
OOD_FOLDER_RE   = re.compile(r"ood_(.+?)_r([0-9.]+)_s([0-9]+)_")
BEST_ACC_RE     = re.compile(r"Best accuracy:\s*([0-9.]+)")

COLORS_ROOD = {0.1: "#2196F3", 0.2: "#4CAF50", 0.3: "#FF9800", 0.4: "#F44336", 0.5: "#9C27B0"}
METRIC_LABELS = {
    "acc":     "Accuracy (%)",
    "ple":     "PLE (Pseudo-Label Entropy)",
    "cci":     "CCI (Class Concentration Index)",
    "s_drift": "Sem-Drift",
    "ood_ff":  "OOD-FF",
    "g_align": "Grad-Align",
}


# ─── Parsers ─────────────────────────────────────────────────────────────────

def load_alg_sweep(alg_dir):
    """Load one algorithm sweep directory → dict keyed by r_ood."""
    data = defaultdict(list)
    if not os.path.isdir(alg_dir):
        return data
    for folder in sorted(os.listdir(alg_dir)):
        m = ALG_FOLDER_RE.match(folder)
        if not m:
            continue
        r_ood, seed = float(m.group(1)), int(m.group(2))
        log_path = os.path.join(alg_dir, folder, "log.txt")
        if not os.path.exists(log_path):
            continue
        records = parse_log_for_metrics(log_path)
        if records:
            data[r_ood].append(records)
    return data


def load_ood_type_sweep(ood_dir):
    """Load OOD type sweep → dict keyed by (ood_type, r_ood)."""
    data = defaultdict(list)
    if not os.path.isdir(ood_dir):
        return data
    for folder in sorted(os.listdir(ood_dir)):
        m = OOD_FOLDER_RE.match(folder)
        if not m:
            continue
        ood_type, r_ood, seed = m.group(1), float(m.group(2)), int(m.group(3))
        log_path = os.path.join(ood_dir, folder, "log.txt")
        if not os.path.exists(log_path):
            continue
        records = parse_log_for_metrics(log_path)
        if records:
            data[(ood_type, r_ood)].append(records)
    return data


def get_mean_std_curve(runs, metric):
    """Given list-of-records lists, return (epochs, mean, std)."""
    min_len = min(len(r) for r in runs)
    arr = np.array([[r[i][metric] for i in range(min_len)] for r in runs], dtype=float)
    epochs = np.array([runs[0][i]["epoch"] for i in range(min_len)])
    return epochs, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


# ─── Plot 1: Per-algorithm training curves (acc) across r_ood ────────────────

def plot_alg_acc_curves_by_rood():
    """One subplot per algorithm; lines = different r_ood values."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Training Accuracy vs Epoch (grouped by r_ood)", fontsize=13, fontweight="bold")

    for ax, (alg_name, alg_dir) in zip(axes, ALG_DIRS.items()):
        data = load_alg_sweep(alg_dir)
        for r_ood in sorted(data.keys()):
            runs = data[r_ood]
            epochs, mean, std = get_mean_std_curve(runs, "acc")
            color = COLORS_ROOD.get(r_ood, "gray")
            ax.plot(epochs, mean, label=f"r_ood={r_ood}", color=color, linewidth=1.8)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_title(alg_name, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    axes[0].set_ylabel("Accuracy (%)")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "01_alg_acc_curves_by_rood.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 2: Four C-Score metrics trend (one algorithm, multiple r_ood)────────────────────────

def plot_cscore_curves_one_alg(alg_name="FlexMatch"):
    """4-panel: PLE, CCI, Sem-Drift, OOD-FF vs epoch for FlexMatch."""
    alg_dir = ALG_DIRS[alg_name]
    data = load_alg_sweep(alg_dir)

    metrics = ["ple", "cci", "s_drift", "ood_ff"]
    titles  = ["PLE (Pseudo-Label Entropy)", "CCI (Class Concentration Index)",
               "Sem-Drift", "OOD-FF"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{alg_name}: C-Score Metrics vs Epoch (CIFAR-100 OOD)", fontsize=13, fontweight="bold")

    for ax, metric, title in zip(axes.flat, metrics, titles):
        for r_ood in sorted(data.keys()):
            runs = data[r_ood]
            epochs, mean, std = get_mean_std_curve(runs, metric)
            color = COLORS_ROOD.get(r_ood, "gray")
            ax.plot(epochs, mean, label=f"r_ood={r_ood}", color=color, linewidth=1.8)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"02_cscore_curves_{alg_name.lower()}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 3: Accuracy + C-Score side-by-side (accuracy masking illustration)─────────────────

def plot_accuracy_masking_demo():
    """
    Side-by-side: left=accuracy (looks stable), right=C-Score metrics (degrade).
    Uses FlexMatch with r_ood=0.1 (clean) vs r_ood=0.4 (heavy contamination).
    """
    alg_dir = ALG_DIRS["FlexMatch"]
    data = load_alg_sweep(alg_dir)

    rood_vals = sorted(data.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Accuracy Masking: Accuracy vs Internal C-Score Metrics (FlexMatch)", fontsize=12, fontweight="bold")

    # Left: accuracy
    ax_acc = axes[0]
    for r_ood in rood_vals:
        runs = data[r_ood]
        epochs, mean, std = get_mean_std_curve(runs, "acc")
        color = COLORS_ROOD.get(r_ood, "gray")
        ax_acc.plot(epochs, mean, label=f"r_ood={r_ood}", color=color, linewidth=2)
        ax_acc.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Clean Test Accuracy\n(appears relatively stable)", fontsize=10)
    ax_acc.set_ylim(0, 100)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(fontsize=8)
    ax_acc.text(0.05, 0.95, "← Small drop\neven at r_ood=0.4",
                transform=ax_acc.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # Right: CCI as a representative C-Score metric
    ax_cscore = axes[1]
    for r_ood in rood_vals:
        runs = data[r_ood]
        epochs, mean, std = get_mean_std_curve(runs, "cci")
        color = COLORS_ROOD.get(r_ood, "gray")
        ax_cscore.plot(epochs, mean, label=f"r_ood={r_ood}", color=color, linewidth=2)
        ax_cscore.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)
    ax_cscore.set_xlabel("Epoch")
    ax_cscore.set_ylabel("CCI")
    ax_cscore.set_title("CCI (Class Concentration Index)\n(degrades significantly)", fontsize=10)
    ax_cscore.grid(True, alpha=0.3)
    ax_cscore.legend(fontsize=8)
    ax_cscore.text(0.05, 0.95, "← Large increase\nindicates class collapse",
                   transform=ax_cscore.transAxes, fontsize=8, va="top",
                   bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "03_accuracy_masking_demo.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 4: OOD type x r_ood trend (Acc vs. contamination rate)─────────────────────

def plot_ood_type_acc_trend():
    """Line plot: best_acc vs r_ood for each OOD type."""
    data = load_ood_type_sweep(OOD_TYPE_DIR)

    # collect (ood_type → {r_ood → [best_acc list]})
    ood_acc = defaultdict(lambda: defaultdict(list))
    for (ood_type, r_ood), runs in data.items():
        for records in runs:
            if records:
                # take max acc across epochs as "best_acc"
                best = max(r["acc"] for r in records if not np.isnan(r["acc"]))
                ood_acc[ood_type][r_ood].append(best)

    ood_types = sorted(ood_acc.keys())
    ood_colors = plt.cm.tab10(np.linspace(0, 1, len(ood_types)))

    fig, ax = plt.subplots(figsize=(9, 5))
    for (ood_type, color) in zip(ood_types, ood_colors):
        r_vals = sorted(ood_acc[ood_type].keys())
        means = [np.mean(ood_acc[ood_type][r]) for r in r_vals]
        stds  = [np.std(ood_acc[ood_type][r])  for r in r_vals]
        ax.errorbar(r_vals, means, yerr=stds, label=ood_type,
                    marker="o", linewidth=2, capsize=4, color=color)

    ax.set_xlabel("OOD Contamination Ratio (r_ood)", fontsize=11)
    ax.set_ylabel("Best Accuracy (%)", fontsize=11)
    ax.set_title("Best Accuracy vs OOD Contamination Ratio\nby OOD Source Type (FlexMatch, CIFAR-10)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(title="OOD Type", fontsize=9, title_fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "04_ood_type_acc_trend.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 5: Sem-Drift trend comparison across three algorithms ──────────────────

def plot_alg_semdrift_comparison():
    """Compare Sem-Drift trends across all 3 algorithms."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Sem-Drift vs Epoch: Algorithm Comparison (CIFAR-100 OOD)", fontsize=13, fontweight="bold")

    for ax, (alg_name, alg_dir) in zip(axes, ALG_DIRS.items()):
        data = load_alg_sweep(alg_dir)
        for r_ood in sorted(data.keys()):
            runs = data[r_ood]
            epochs, mean, std = get_mean_std_curve(runs, "s_drift")
            color = COLORS_ROOD.get(r_ood, "gray")
            ax.plot(epochs, mean, label=f"r_ood={r_ood}", color=color, linewidth=1.8)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_title(alg_name, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Sem-Drift")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "05_alg_semdrift_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 6: Grad-Align vs epoch (labeled/unlabeled gradient compatibility) ───────

def plot_galign_curves():
    """Grad-Align for each algorithm across r_ood."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle("Gradient Alignment vs Epoch (Labeled-Unlabeled Optimization Compatibility)", fontsize=12, fontweight="bold")

    for ax, (alg_name, alg_dir) in zip(axes, ALG_DIRS.items()):
        data = load_alg_sweep(alg_dir)
        for r_ood in sorted(data.keys()):
            runs = data[r_ood]
            epochs, mean, std = get_mean_std_curve(runs, "g_align")
            if np.all(np.isnan(mean)):
                continue
            color = COLORS_ROOD.get(r_ood, "gray")
            ax.plot(epochs, mean, label=f"r_ood={r_ood}", color=color, linewidth=1.8)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(alg_name, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Grad-Align (cosine similarity)")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "06_galign_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 7: Four-panel summary (representative experiment) ──────────────────────

def plot_dashboard_one_condition(alg_name="FlexMatch", r_ood_target=0.3):
    """Full diagnostic dashboard for one (algorithm, r_ood) condition."""
    alg_dir = ALG_DIRS[alg_name]
    data = load_alg_sweep(alg_dir)

    if r_ood_target not in data:
        print(f"  [skip] {alg_name} r_ood={r_ood_target} not found")
        return

    runs = data[r_ood_target]
    epochs, acc_m, acc_s   = get_mean_std_curve(runs, "acc")
    _, cci_m,  cci_s        = get_mean_std_curve(runs, "cci")
    _, sd_m,   sd_s         = get_mean_std_curve(runs, "s_drift")
    _, oodff_m, oodff_s     = get_mean_std_curve(runs, "ood_ff")
    _, ple_m,  ple_s        = get_mean_std_curve(runs, "ple")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"C-Score Diagnostic Dashboard — {alg_name}, r_ood={r_ood_target}, CIFAR-100 OOD", fontsize=12, fontweight="bold")

    def _draw(ax, epochs, mean, std, label, color, ylabel):
        ax.plot(epochs, mean, color=color, linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3)

    _draw(axes[0,0], epochs, acc_m,   acc_s,   "Test Accuracy (%)",             "#2196F3", "Accuracy (%)")
    _draw(axes[0,1], epochs, ple_m,   ple_s,   "PLE (Pseudo-Label Entropy)",    "#9C27B0", "PLE")
    _draw(axes[0,2], epochs, cci_m,   cci_s,   "CCI (Class Concentration Index)","#FF9800", "CCI")
    _draw(axes[1,0], epochs, sd_m,    sd_s,    "Sem-Drift",                     "#F44336", "Sem-Drift")
    _draw(axes[1,1], epochs, oodff_m, oodff_s, "OOD-FF (Oracle)",               "#4CAF50", "OOD-FF")

    # Combine all on one normalized axis
    ax = axes[1,2]
    for (mean, std, label, color) in [
        (acc_m / 100, acc_s / 100, "Acc (÷100)", "#2196F3"),
        (ple_m / max(ple_m.max(), 1e-9), ple_s / max(ple_m.max(), 1e-9), "PLE (norm)", "#9C27B0"),
        (cci_m / max(cci_m.max(), 1e-9), cci_s / max(cci_m.max(), 1e-9), "CCI (norm)", "#FF9800"),
    ]:
        ax.plot(epochs, mean, label=label, color=color, linewidth=1.8)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized value")
    ax.set_title("Normalized Overview", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"07_dashboard_{alg_name.lower()}_rood{r_ood_target}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

# ─── Plot 8: C-Score ──────────────────────────────────────

def smooth_curve(y, window=11):
    if window < 3:
        return y
    window = min(window, len(y))
    if window % 2 == 0:
        window += 1  # keep odd
    pad = window // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(y_pad, kernel, mode="valid")

def plot_cscore():
    """4-panel: PLE, CCI, Sem-Drift, G-Align vs epoch for FixMatch."""
    data = load_ood_type_sweep(OOD_TYPE_DIR)

    metrics = ["ple", "cci", "s_drift", "g_align"]
    titles  = ["PLE (Pseudo-Label Entropy)", "CCI (Class Concentration Index)", "Sem-Drift", "G-Align (Grad-Align)"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)

    handles, labels = [], []

    for ax, metric, title in zip(axes.flat, metrics, titles):
        for r_ood in sorted(data.keys()):
            if r_ood[0] == "svhn":
                runs = data[r_ood]
                epochs, mean, std = get_mean_std_curve(runs, metric)
                mean = smooth_curve(mean, window=5)
                std = smooth_curve(std, window=5)
                color = COLORS_ROOD.get(r_ood[1], "gray")

                line, = ax.plot(
                    epochs, mean,
                    label=f"r={r_ood[1]}",
                    color=color,
                    linewidth=2.3
                )
                ax.fill_between(
                    epochs, mean - std, mean + std,
                    color=color, alpha=0.08
                )

                # Collect legend entries only once to avoid duplicates
                if metric == "ple":
                    handles.append(line)
                    labels.append(f"r={r_ood[1]}")

                ax.set_xticks([0, 50, 100, 150, 200])
                ax.tick_params(axis="both", labelsize=14)

        ax.set_title(title, fontsize=20)
        ax.grid(True, alpha=0.15)

    # Only place legend on the G-Align panel
    axes[1, 1].legend(handles, labels, loc="lower right", fontsize=12, frameon=True)
    # axes[1, 0].set_xlabel("Epoch", fontsize=20)
    # axes[1, 1].set_xlabel("Epoch", fontsize=20)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "08_cscore.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  plot_trend.py — Trend Plots")
    print(f"  Output: {OUT_DIR}")
    print("=" * 60)

    # print("\n[1/8] Algorithm accuracy curves by r_ood ...")
    # plot_alg_acc_curves_by_rood()

    # print("\n[2/8] FlexMatch C-Score metrics by r_ood ...")
    # plot_cscore_curves_one_alg("SoftMatch")

    # print("\n[3/8] Accuracy masking demo ...")
    # plot_accuracy_masking_demo()

    # print("\n[4/8] OOD type accuracy trend ...")
    # plot_ood_type_acc_trend()

    # print("\n[5/8] Sem-Drift algorithm comparison ...")
    # plot_alg_semdrift_comparison()

    # print("\n[6/8] Gradient alignment curves ...")
    # plot_galign_curves()

    # print("\n[7/8] Diagnostic dashboard (FlexMatch, r_ood=0.3) ...")
    # plot_dashboard_one_condition("FlexMatch", 0.3)

    print("\n[8/8] FixMatch C-Score metrics by r_ood (SVHN OOD) ...")
    plot_cscore()

    print(f"\nDone. All plots saved to: {OUT_DIR}")
