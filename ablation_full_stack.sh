#!/usr/bin/env bash
# ablation_full_stack.sh
# Comprehensive ablation: measure the contribution of each NVIDIA/PyTorch
# acceleration component individually, starting from a bare CPU baseline.
#
# Components under test:
#   1. cpu_only        — no GPU at all (baseline, slowest)
#   2. gpu_baseline    — GPU enabled, cuDNN benchmark OFF, all acceleration OFF
#   3. plus_benchmark  — add cuDNN benchmark mode
#   4. plus_amp        — add AMP (FP16 Tensor Cores)
#   5. plus_compile    — add torch.compile
#   6. plus_fused      — add PyTorch native fused SGD
#   7. plus_dali       — add DALI async GPU data pipeline
#   8. full_stack      — everything ON (fastest)
#
# Run from cscore project root:
#   cd /home/arlenchen/cscore && bash ablation_full_stack.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=/home/arlenchen/venv/safe_sl/bin/python
ABLATION_DIR="ablation_full_stack_results"
mkdir -p "$ABLATION_DIR"

# ── Shared hyperparameters ────────────────────────────────────────────────────
ALGORITHM="fixmatch"
DATASET="cifar10"
OOD_DATASET="cifar100"
R_OOD="0.1"
SEED="0"
NUM_LABELED="40"
EPOCHS="50"
NUM_IT_PER_EPOCH="256"
BATCH_SIZE="64"
MU="7"

# ── Helper: write a YAML config ───────────────────────────────────────────────
make_config() {
    local setting="$1"
    local force_cpu="$2"
    local enable_benchmark="$3"
    local use_amp="$4"
    local use_compile="$5"
    local use_fused="$6"
    local use_dali="$7"
    local out_cfg="/tmp/ablation_full_${setting}.yaml"

    cat > "$out_cfg" <<EOF
algorithm: "${ALGORITHM}"
dataset: "${DATASET}"
num_classes: 10
data_dir: "./data"

seed: ${SEED}
num_labeled: ${NUM_LABELED}

r_id: 1.0
r_ood: ${R_OOD}
ood_dataset: "${OOD_DATASET}"

use_da: false
p_cutoff: 0.95
lambda_u: 1.0
mu: ${MU}
ema_decay: 0.999

epochs: ${EPOCHS}
num_it_per_epoch: ${NUM_IT_PER_EPOCH}
batch_size: ${BATCH_SIZE}
lr: 0.03
weight_decay: 0.0005

model_name: "wrn"
depth: 28
widen_factor: 2

force_cpu: ${force_cpu}
use_amp: ${use_amp}
use_torch_compile: ${use_compile}
compile_mode: "reduce-overhead"
use_fused_optimizer: ${use_fused}
enable_benchmark: ${enable_benchmark}
use_dali: ${use_dali}

num_workers: 4
pin_memory: true
persistent_workers: true
prefetch_factor: 2
eval_batch_size: 256
EOF
    echo "$out_cfg"
}

# ── Helper: run one setting ───────────────────────────────────────────────────
run_setting() {
    local setting="$1"
    local force_cpu="$2"
    local enable_benchmark="$3"
    local use_amp="$4"
    local use_compile="$5"
    local use_fused="$6"
    local use_dali="$7"

    local out_dir="${ABLATION_DIR}/${setting}"
    mkdir -p "$out_dir"

    echo ""
    echo "========================================================"
    echo "  Setting       : ${setting}"
    echo "  force_cpu     : ${force_cpu}"
    echo "  cuDNN benchmark: ${enable_benchmark}"
    echo "  use_amp       : ${use_amp}"
    echo "  torch.compile : ${use_compile}"
    echo "  fused_sgd     : ${use_fused}"
    echo "  DALI          : ${use_dali}"
    echo "  Output        : ${out_dir}/"
    echo "========================================================"

    local cfg
    cfg=$(make_config "$setting" "$force_cpu" "$enable_benchmark" "$use_amp" "$use_compile" "$use_fused" "$use_dali")

    mkdir -p exp_results
    ls exp_results/ 2>/dev/null | sort > /tmp/ablation_full_before_${setting}.txt || true

    # GPU monitor (skip for CPU-only)
    local DMON_PID=""
    if [ "$force_cpu" = "false" ]; then
        nvidia-smi dmon -s u -d 1 > "${out_dir}/gpu_util.log" 2>&1 &
        DMON_PID=$!
        echo "  nvidia-smi dmon started (PID ${DMON_PID})"
    else
        echo "  CPU-only run — nvidia-smi dmon skipped"
    fi

    echo "  Training started at $(date '+%Y-%m-%d %H:%M:%S')"
    "$PYTHON" train_executor.py --config "$cfg"
    local TRAIN_RC=$?
    echo "  Training finished at $(date '+%Y-%m-%d %H:%M:%S') (exit ${TRAIN_RC})"

    if [ -n "$DMON_PID" ]; then
        kill "$DMON_PID" 2>/dev/null || true
        wait "$DMON_PID" 2>/dev/null || true
        echo "  nvidia-smi dmon stopped"
    fi

    ls exp_results/ 2>/dev/null | sort > /tmp/ablation_full_after_${setting}.txt || true
    local EXP_SUBDIR
    EXP_SUBDIR=$(comm -13 /tmp/ablation_full_before_${setting}.txt \
                           /tmp/ablation_full_after_${setting}.txt | head -1 || true)

    if [ -n "$EXP_SUBDIR" ] && [ -f "exp_results/${EXP_SUBDIR}/log.txt" ]; then
        cp "exp_results/${EXP_SUBDIR}/log.txt" "${out_dir}/log.txt"
        echo "  log.txt copied from exp_results/${EXP_SUBDIR}/"
    else
        echo "  WARNING: log.txt not found for ${setting}" >&2
    fi

    rm -f "$cfg" \
          /tmp/ablation_full_before_${setting}.txt \
          /tmp/ablation_full_after_${setting}.txt
}

# ── Run all settings ─────────────────────────────────────────────────────────
echo "Starting full-stack ablation (8 settings × ${EPOCHS} epochs each)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo ""

# Migrate existing gpu_baseline → plus_benchmark (reuse already-completed run)
if [ -d "${ABLATION_DIR}/gpu_baseline" ] && [ ! -d "${ABLATION_DIR}/plus_benchmark" ]; then
    cp -r "${ABLATION_DIR}/gpu_baseline" "${ABLATION_DIR}/plus_benchmark"
    echo "Migrated existing gpu_baseline → plus_benchmark (cuDNN ON, no other acceleration)"
fi

#              setting            cpu     benchmark  amp     compile  fused   dali
run_setting "cpu_only"        "true"  "false"    "false" "false"  "false" "false"
run_setting "gpu_baseline"    "false" "false"    "false" "false"  "false" "false"
# plus_benchmark uses migrated data — skip re-running if already exists
if [ ! -f "${ABLATION_DIR}/plus_benchmark/log.txt" ]; then
    run_setting "plus_benchmark" "false" "true"    "false" "false"  "false" "false"
else
    echo "plus_benchmark already exists (migrated from old gpu_baseline) — skipping"
fi
run_setting "plus_amp"        "false" "true"     "true"  "false"  "false" "false"
run_setting "plus_compile"    "false" "true"     "true"  "true"   "false" "false"
run_setting "plus_fused"      "false" "true"     "true"  "true"   "true"  "false"
run_setting "plus_dali"       "false" "true"     "true"  "true"   "true"  "true"
run_setting "full_stack"      "false" "true"     "true"  "true"   "true"  "true"

# ── Parse and generate summary ───────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Generating summary..."
echo "========================================================"

"$PYTHON" - <<'PYEOF'
import re, os

ABLATION_DIR   = "ablation_full_stack_results"
SETTINGS       = ["cpu_only", "gpu_baseline", "plus_benchmark", "plus_amp",
                  "plus_compile", "plus_fused", "plus_dali", "full_stack"]
NUM_IT         = 256
SAMPLES_PER_IT = 512   # 64 labeled + 7×64 unlabeled

results = {}

for s in SETTINGS:
    log = os.path.join(ABLATION_DIR, s, "log.txt")
    if not os.path.exists(log):
        print(f"  WARNING: {log} not found")
        results[s] = None
        continue
    times = []
    with open(log) as f:
        for line in f:
            m = re.search(r'Epoch \[(\d+)/\d+\] ([\d.]+)s', line)
            if m and int(m.group(1)) >= 2:
                times.append(float(m.group(2)))
    if not times:
        print(f"  WARNING: no epoch times in {log}")
        results[s] = None
        continue
    avg  = sum(times) / len(times)
    tp   = NUM_IT * SAMPLES_PER_IT / avg
    results[s] = (avg, tp)
    print(f"  {s:<16}: avg={avg:.2f}s  tp={tp:,.0f} samp/s")

# Use gpu_baseline as GPU reference and cpu_only as absolute baseline
gpu_ref_tp  = (results.get("gpu_baseline") or (None, None))[1]
cpu_tp      = (results.get("cpu_only")     or (None, None))[1]
full_tp     = (results.get("full_stack")   or (None, None))[1]

rows = []
rows.append(f"{'Setting':<16}| {'Avg Epoch Time':<15}| {'Throughput':>14} | "
            f"{'vs GPU baseline':>16} | {'vs CPU':>10} | Component added")
rows.append("-" * 95)

COMPONENT = {
    "cpu_only":        "—  (CPU reference)",
    "gpu_baseline":    "GPU only (cuDNN benchmark OFF)",
    "plus_benchmark":  "+ cuDNN benchmark mode",
    "plus_amp":        "+ AMP (FP16 Tensor Cores)",
    "plus_compile":    "+ torch.compile",
    "plus_fused":      "+ PyTorch fused SGD",
    "plus_dali":       "+ DALI async pipeline",
    "full_stack":      "= full stack",
}

for s in SETTINGS:
    comp = COMPONENT.get(s, "")
    if results[s] is None:
        rows.append(f"{s:<16}| {'N/A':<15}| {'N/A':>14} | {'N/A':>16} | {'N/A':>10} | {comp}")
        continue
    avg, tp = results[s]
    vs_gpu = (f"{(tp - gpu_ref_tp) / gpu_ref_tp * 100:+.1f}%"
              if gpu_ref_tp and s != "gpu_baseline" else
              "baseline" if s == "gpu_baseline" else "N/A")
    vs_cpu = (f"{tp / cpu_tp:.2f}×" if cpu_tp else "N/A")
    rows.append(f"{s:<16}| {avg:.2f}s{'':<9}| {tp:>14,.0f} | {vs_gpu:>16} | {vs_cpu:>10} | {comp}")

summary = "\n".join(rows)
print()
print(summary)

out = os.path.join(ABLATION_DIR, "summary.txt")
with open(out, "w") as f:
    f.write(summary + "\n")
print(f"\nSummary written to {out}")
PYEOF

echo ""
echo "Ablation complete. Results in ${ABLATION_DIR}/"
