import os
import re
import csv
import ast
import json
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# Regex helpers
# =========================================================
EPOCH_RE = re.compile(r"Epoch \[(\d+)/(\d+)\]")
ACC_RE = re.compile(r"Acc:\s*([0-9.]+)")
LOSS_RE = re.compile(r"Loss:\s*([0-9.]+)")
IDM_RE = re.compile(r"ID-M:\s*([0-9.]+)%")
OODM_RE = re.compile(r"OOD-M:\s*([0-9.]+)%")
IDPA_RE = re.compile(r"ID-PA:\s*([0-9.]+)%")
PLE_RE = re.compile(r"PLE:\s*([0-9.]+)")
CCI_RE = re.compile(r"CCI:\s*([0-9.]+)")
SDRIFT_RE = re.compile(r"S-Drift:\s*([0-9.]+)")
OODFF_RE = re.compile(r"OOD-FF:\s*([0-9.]+)")
GXNORM_RE  = re.compile(r"G-x:\s*([0-9.]+)")
GUNORM_RE  = re.compile(r"G-u:\s*([0-9.]+)")
GALIGN_RE  = re.compile(r"G-Align:\s*(-?[0-9.]+)")
BEST_ACC_RE = re.compile(r"Best accuracy:\s*([0-9.]+)")

PER_CLASS_ACC_RE = re.compile(r">>\s*Per-Class Acc:\s*(\[[^\]]+\])")
CLASS_MASK_DIST_RE = re.compile(r">>\s*Class Mask Dist:\s*(\[[^\]]+\])")

FOLDER_NEW_RE = re.compile(r"grid_da([TF])_bn([TF])_([0-9.]+)_([0-9.]+)_s([0-9]+)")
FOLDER_OLD_RE = re.compile(r"grid_([0-9.]+)_([0-9.]+)_s([0-9]+)")


# =========================================================
# Core parsing
# =========================================================
def _safe_float(match_obj, default=np.nan):
    return float(match_obj.group(1)) if match_obj else default


def _safe_list_from_str(list_str):
    try:
        return list(ast.literal_eval(list_str))
    except Exception:
        return None


def parse_log_for_metrics(log_path):
    """
    Parse per-epoch metrics from log.txt

    Returns:
        records: list[dict], one record per epoch
    """
    records = []
    pending_per_class_acc = None
    pending_class_mask_dist = None

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # auxiliary lines
        m_pca = PER_CLASS_ACC_RE.search(line)
        if m_pca:
            pending_per_class_acc = _safe_list_from_str(m_pca.group(1))
            continue

        m_cmd = CLASS_MASK_DIST_RE.search(line)
        if m_cmd:
            pending_class_mask_dist = _safe_list_from_str(m_cmd.group(1))
            continue

        if "Epoch [" not in line or "Acc:" not in line:
            continue

        m_epoch = EPOCH_RE.search(line)
        if not m_epoch:
            continue

        epoch = int(m_epoch.group(1))
        total_epochs = int(m_epoch.group(2))

        record = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "acc": _safe_float(ACC_RE.search(line)),
            "loss": _safe_float(LOSS_RE.search(line)),
            "id_m": _safe_float(IDM_RE.search(line)),
            "ood_m": _safe_float(OODM_RE.search(line)),
            "id_pa": _safe_float(IDPA_RE.search(line)),
            "ple": _safe_float(PLE_RE.search(line)),
            "cci": _safe_float(CCI_RE.search(line)),
            "s_drift": _safe_float(SDRIFT_RE.search(line)),
            "ood_ff": _safe_float(OODFF_RE.search(line)),
            "g_x_norm": _safe_float(GXNORM_RE.search(line)),
            "g_u_norm": _safe_float(GUNORM_RE.search(line)),
            "g_align":  _safe_float(GALIGN_RE.search(line)),
            "per_class_acc": pending_per_class_acc,
            "class_mask_dist": pending_class_mask_dist,
        }

        # consume once
        pending_per_class_acc = None
        pending_class_mask_dist = None

        records.append(record)

    return records


def parse_experiment_folder(exp_dir, folder):
    """
    Parse one experiment folder into:
    - meta info
    - per-epoch records
    - final summary
    """
    match_new = FOLDER_NEW_RE.match(folder)
    match_old = FOLDER_OLD_RE.match(folder)

    if match_new:
        da, bn, r_id, r_ood, seed = match_new.groups()
    elif match_old:
        r_id, r_ood, seed = match_old.groups()
        da, bn = "F", "F"
    else:
        return None

    log_path = os.path.join(exp_dir, folder, "log.txt")
    config_path = os.path.join(exp_dir, folder, "config.json")

    if not os.path.exists(log_path):
        return None

    records = parse_log_for_metrics(log_path)
    if not records:
        return None

    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    best_acc_match = BEST_ACC_RE.search(text)
    best_acc = float(best_acc_match.group(1)) if best_acc_match else np.nan

    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    final_record = records[-1]

    # sink metrics from class_mask_dist
    sink_ratio = np.nan
    sink_class = np.nan
    if final_record["class_mask_dist"] is not None:
        arr = np.array(final_record["class_mask_dist"], dtype=float)
        total = arr.sum()
        if total > 0:
            sink_class = int(np.argmax(arr))
            sink_ratio = float(arr.max() / total)

    return {
        "folder": folder,
        "da": da,
        "bn": bn,
        "r_id": float(r_id),
        "r_ood": float(r_ood),
        "seed": int(seed),
        "config": cfg,
        "records": records,
        "best_acc": best_acc,
        "final": final_record,
        "sink_ratio": sink_ratio,
        "sink_class": sink_class,
    }


def load_all_experiments(exp_dir="exp_results_analysis"):
    experiments = []
    if not os.path.exists(exp_dir):
        print(f"[WARN] Directory not found: {exp_dir}")
        return experiments

    for folder in sorted(os.listdir(exp_dir)):
        full_path = os.path.join(exp_dir, folder)
        if not os.path.isdir(full_path):
            continue
        parsed = parse_experiment_folder(exp_dir, folder)
        if parsed is not None:
            experiments.append(parsed)

    print(f"Parsed {len(experiments)} experiments")
    return experiments


# =========================================================
# Export CSV
# =========================================================
def export_epoch_csv(experiments, save_dir="."):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "epoch_metrics.csv")

    fieldnames = [
        "folder", "da", "bn", "r_id", "r_ood", "seed",
        "epoch", "total_epochs",
        "acc", "loss", "id_m", "ood_m", "id_pa",
        "ple", "cci", "s_drift", "ood_ff", "g_x_norm", "g_u_norm", "g_align",
        "per_class_acc", "class_mask_dist"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for exp in experiments:
            for rec in exp["records"]:
                row = {
                    "folder": exp["folder"],
                    "da": exp["da"],
                    "bn": exp["bn"],
                    "r_id": exp["r_id"],
                    "r_ood": exp["r_ood"],
                    "seed": exp["seed"],
                    "epoch": rec["epoch"],
                    "total_epochs": rec["total_epochs"],
                    "acc": rec["acc"],
                    "loss": rec["loss"],
                    "id_m": rec["id_m"],
                    "ood_m": rec["ood_m"],
                    "id_pa": rec["id_pa"],
                    "ple": rec["ple"],
                    "cci": rec["cci"],
                    "s_drift": rec["s_drift"],
                    "ood_ff": rec["ood_ff"],
                    "g_x_norm": rec["g_x_norm"],
                    "g_u_norm": rec["g_u_norm"],
                    "g_align":  rec["g_align"],
                    "per_class_acc": rec["per_class_acc"],
                    "class_mask_dist": rec["class_mask_dist"],
                }
                writer.writerow(row)

    print(f"Epoch-level CSV saved to: {csv_path}")


def export_summary_csv(experiments, save_dir="."):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "summary_metrics.csv")

    fieldnames = [
        "folder", "da", "bn", "r_id", "r_ood", "seed",
        "best_acc",
        "final_acc", "final_loss",
        "final_id_m", "final_ood_m", "final_id_pa",
        "final_ple", "final_cci", "final_s_drift", "final_ood_ff",
        "final_g_x_norm", "final_g_u_norm", "final_g_align",
        "sink_ratio", "sink_class",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for exp in experiments:
            final = exp["final"]
            row = {
                "folder": exp["folder"],
                "da": exp["da"],
                "bn": exp["bn"],
                "r_id": exp["r_id"],
                "r_ood": exp["r_ood"],
                "seed": exp["seed"],
                "best_acc": exp["best_acc"],
                "final_acc": final["acc"],
                "final_loss": final["loss"],
                "final_id_m": final["id_m"],
                "final_ood_m": final["ood_m"],
                "final_id_pa": final["id_pa"],
                "final_ple": final["ple"],
                "final_cci": final["cci"],
                "final_s_drift": final["s_drift"],
                "final_ood_ff": final["ood_ff"],
                "final_g_x_norm": final["g_x_norm"],
                "final_g_u_norm": final["g_u_norm"],
                "final_g_align":  final["g_align"],
                "sink_ratio": exp["sink_ratio"],
                "sink_class": exp["sink_class"],
            }
            writer.writerow(row)

    print(f"Summary CSV saved to: {csv_path}")


# =========================================================
# Plot helpers
# =========================================================
def _group_by_condition(experiments, da="F", bn="F"):
    grouped = defaultdict(list)
    for exp in experiments:
        if exp["da"] == da and exp["bn"] == bn:
            grouped[(exp["r_id"], exp["r_ood"])].append(exp)
    return grouped


def _metric_from_final(exp, metric):
    final = exp["final"]
    metric_map = {
        "acc": final["acc"],
        "best_acc": exp["best_acc"],
        "ple": final["ple"],
        "cci": final["cci"],
        "s_drift": final["s_drift"],
        "ood_ff": final["ood_ff"],
        "id_m": final["id_m"],
        "ood_m": final["ood_m"],
        "id_pa": final["id_pa"],
        "sink_ratio": exp["sink_ratio"],
        "g_align": final["g_align"],
    }
    return metric_map.get(metric, final["acc"])


def generate_heatmap(exp_dir="exp_results_analysis", save_dir=".", metric="acc", da="F", bn="F"):
    """
    Generate heatmap using final metrics.
    """
    print(f"\nGenerating heatmap | metric={metric} | DA={da} | BN={bn}")
    experiments = load_all_experiments(exp_dir)
    if not experiments:
        return

    grouped = _group_by_condition(experiments, da=da, bn=bn)
    if not grouped:
        print("[WARN] No experiments matching DA/BN conditions")
        return

    xs = sorted(list(set(k[1] for k in grouped.keys())))
    ys = sorted(list(set(k[0] for k in grouped.keys())), reverse=True)

    grid = np.full((len(ys), len(xs)), np.nan)

    for (r_id, r_ood), exps in grouped.items():
        vals = [_metric_from_final(exp, metric) for exp in exps]
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) == 0:
            continue
        yi = ys.index(r_id)
        xi = xs.index(r_ood)
        grid[yi, xi] = np.mean(vals)

    plt.figure(figsize=(8, 6))

    if metric in ["acc", "best_acc", "id_m", "ood_m", "id_pa"]:
        cmap = "Reds"
        fmt = ".2f"
        if metric in ["acc", "best_acc"]:
            vmin, vmax = 0, 100
            cbar_label = metric
        else:
            vmin, vmax = 0, 100
            cbar_label = f"{metric} (%)"
    else:
        cmap = "YlGnBu"
        fmt = ".3f"
        vmin, vmax = None, None
        cbar_label = metric

    sns.heatmap(
        grid,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        linecolor="gray",
        xticklabels=[f"{x:.1f}" for x in xs],
        yticklabels=[f"{y:.1f}" for y in ys],
        cbar_kws={"label": cbar_label},
    )

    plt.xlabel("r_ood")
    plt.ylabel("r_id")
    plt.title(f"Heatmap: {metric} [DA={da}, BN={bn}]")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    heatmap_path = os.path.join(save_dir, f"heatmap_{metric}_da{da}_bn{bn}.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    print(f"Heatmap saved to: {heatmap_path}")


def plot_collapse_curves(exp_dir="exp_results_analysis", save_dir="."):
    """
    Plot per-experiment curves:
    - left axis: acc
    - right axis: selected C-score metrics
    """
    print("\nPlotting single-experiment collapse curves ...")
    experiments = load_all_experiments(exp_dir)
    if not experiments:
        return

    curve_dir = os.path.join(save_dir, "curves")
    os.makedirs(curve_dir, exist_ok=True)

    for exp in experiments:
        records = exp["records"]
        epochs = [r["epoch"] for r in records]
        accs = [r["acc"] for r in records]
        cci = [r["cci"] for r in records]
        s_drift = [r["s_drift"] for r in records]
        ood_ff = [r["ood_ff"] for r in records]
        ood_m = [r["ood_m"] for r in records]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(epochs, accs, label="Acc", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_ylim(0, 100)

        ax2 = ax1.twinx()
        ax2.plot(epochs, cci, "--", label="CCI")
        ax2.plot(epochs, s_drift, "-.", label="S-Drift")
        ax2.plot(epochs, ood_ff, ":", label="OOD-FF")
        if not all(np.isnan(x) for x in ood_m):
            ax2.plot(epochs, ood_m, label="OOD-M (%)", alpha=0.7)

        title = (
            f"{exp['folder']} | r_id={exp['r_id']}, r_ood={exp['r_ood']}, "
            f"DA={exp['da']}, BN={exp['bn']}, seed={exp['seed']}"
        )
        plt.title(title)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        plt.tight_layout()
        save_path = os.path.join(curve_dir, f"{exp['folder']}_collapse.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    print(f"Single-experiment curve plots saved to: {curve_dir}")


def plot_mean_curve_by_rood(exp_dir="exp_results_analysis", save_dir=".", da="F", bn="F"):
    """
    Aggregate mean±std curve over seeds for each (r_id, r_ood) condition.
    Useful for baseline sweeps where multiple seeds share the same setup.
    """
    print(f"\nPlotting mean curves | DA={da} | BN={bn}")
    experiments = load_all_experiments(exp_dir)
    if not experiments:
        return

    filtered = [exp for exp in experiments if exp["da"] == da and exp["bn"] == bn]
    if not filtered:
        print("[WARN] No experiments matching conditions")
        return

    grouped = defaultdict(list)
    for exp in filtered:
        grouped[(exp["r_id"], exp["r_ood"])].append(exp)

    out_dir = os.path.join(save_dir, "mean_curves")
    os.makedirs(out_dir, exist_ok=True)

    for (r_id, r_ood), exps in grouped.items():
        if len(exps) == 0:
            continue

        min_len = min(len(exp["records"]) for exp in exps)
        if min_len == 0:
            continue

        # Use the first experiment's epoch sequence as alignment reference
        epochs = np.array([exps[0]["records"][i]["epoch"] for i in range(min_len)])

        metrics = ["acc", "cci", "s_drift", "ood_ff", "ood_m", "id_m", "id_pa"]
        series = {}

        for m in metrics:
            arr = []
            for exp in exps:
                vals = [exp["records"][i][m] for i in range(min_len)]
                arr.append(vals)
            arr = np.array(arr, dtype=float)

            series[m] = {
                "mean": np.nanmean(arr, axis=0),
                "std": np.nanstd(arr, axis=0),
            }

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Left axis: Accuracy
        ax1.plot(epochs, series["acc"]["mean"], label="Acc", linewidth=2)
        ax1.fill_between(
            epochs,
            series["acc"]["mean"] - series["acc"]["std"],
            series["acc"]["mean"] + series["acc"]["std"],
            alpha=0.2
        )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_ylim(0, 100)

        # Right axis: Collapse metrics
        ax2 = ax1.twinx()
        for m in ["cci", "s_drift", "ood_ff"]:
            ax2.plot(epochs, series[m]["mean"], label=m)
            ax2.fill_between(
                epochs,
                series[m]["mean"] - series[m]["std"],
                series[m]["mean"] + series[m]["std"],
                alpha=0.15
            )

        plt.title(f"Mean Curve | r_id={r_id}, r_ood={r_ood}, DA={da}, BN={bn}")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        plt.tight_layout()
        save_path = os.path.join(
            out_dir,
            f"mean_curve_rid{r_id}_rood{r_ood}_da{da}_bn{bn}.png"
        )
        plt.savefig(save_path, dpi=300)
        plt.close()

    print(f"Mean curve plots saved to: {out_dir}")


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="exp_results_analysis")
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument(
        "--metric",
        type=str,
        default="acc",
        choices=[
            "acc", "best_acc", "ple", "cci", "s_drift", "ood_ff",
            "id_m", "ood_m", "id_pa", "g_align", "sink_ratio"
        ],
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "heatmap", "curve", "mean_curve", "export"],
    )
    parser.add_argument("--da", type=str, default="F", choices=["F", "T"])
    parser.add_argument("--bn", type=str, default="F", choices=["F", "T"])
    args = parser.parse_args()

    experiments = load_all_experiments(args.exp_dir)

    if args.mode in ["all", "export"]:
        export_epoch_csv(experiments, save_dir=args.save_dir)
        export_summary_csv(experiments, save_dir=args.save_dir)

    if args.mode in ["all", "curve"]:
        plot_collapse_curves(exp_dir=args.exp_dir, save_dir=args.save_dir)

    if args.mode in ["all", "mean_curve"]:
        plot_mean_curve_by_rood(
            exp_dir=args.exp_dir,
            save_dir=args.save_dir,
            da=args.da,
            bn=args.bn,
        )

    if args.mode in ["all", "heatmap"]:
        generate_heatmap(
            exp_dir=args.exp_dir,
            save_dir=args.save_dir,
            metric=args.metric,
            da=args.da,
            bn=args.bn,
        )
        if args.mode == "all" and args.metric == "acc":
            for m in ["cci", "s_drift", "ood_ff", "ood_m", "id_m", "id_pa", "sink_ratio"]:
                generate_heatmap(
                    exp_dir=args.exp_dir,
                    save_dir=args.save_dir,
                    metric=m,
                    da=args.da,
                    bn=args.bn,
                )