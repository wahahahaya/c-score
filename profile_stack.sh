#!/usr/bin/env bash
# profile_stack.sh
# Full profiling of the NVIDIA software stack on FixMatch/CIFAR-10:
#   1. nsys profile   → CUDA timeline (.nsys-rep) + kernel ranking CSV
#   2. torch.profiler → Chrome trace (open in chrome://tracing or Perfetto)
#   3. Summary report → profiling_results/report.txt
#
# Usage:
#   cd /home/arlenchen/cscore
#   bash profile_stack.sh [setting]
#   setting: full_stack (default) | no_amp | no_benchmark | no_fused
#
# Output: profiling_results/<setting>/
#   nsys_<setting>.nsys-rep      — open with Nsight Systems GUI
#   nsys_kernels.csv             — top CUDA kernels by total time
#   nsys_nvtx.csv                — NVTX range timing (forward/backward/ema/augment)
#   profiler_trace/              — Chrome traces (open in chrome://tracing)
#   report.txt                   — human-readable summary

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=/home/arlenchen/venv/safe_sl/bin/python
SETTING="${1:-full_stack}"
PROFILE_DIR="profiling_results/${SETTING}"
mkdir -p "$PROFILE_DIR"

# ── Per-setting flags ─────────────────────────────────────────────────────────
case "$SETTING" in
  full_stack)    USE_AMP=true;  BENCHMARK=true;  FUSED=true  ;;
  no_amp)        USE_AMP=false; BENCHMARK=true;  FUSED=true  ;;
  no_benchmark)  USE_AMP=true;  BENCHMARK=false; FUSED=true  ;;
  no_fused)      USE_AMP=true;  BENCHMARK=true;  FUSED=false ;;
  *)
    echo "Unknown setting: $SETTING  (valid: full_stack no_amp no_benchmark no_fused)"
    exit 1
    ;;
esac

# ── Profiling config (5 epochs — enough to get stable signal after warmup) ──
CFG=/tmp/profile_cscore_${SETTING}.yaml
cat > "$CFG" <<EOF
algorithm: "fixmatch"
dataset: "cifar10"
num_classes: 10
data_dir: "./data"

seed: 0
num_labeled: 40
r_id: 1.0
r_ood: 0.1
ood_dataset: "cifar100"

use_da: false
p_cutoff: 0.95
lambda_u: 1.0
mu: 7
ema_decay: 0.999

# Short run: torch.profiler records epochs 2-3, nsys captures everything
epochs: 5
num_it_per_epoch: 256
batch_size: 64
lr: 0.03
weight_decay: 0.0005

model_name: "wrn"
depth: 28
widen_factor: 2

use_amp: ${USE_AMP}
use_torch_compile: false
compile_mode: "reduce-overhead"
use_fused_optimizer: ${FUSED}
enable_benchmark: ${BENCHMARK}

num_workers: 4
pin_memory: true
persistent_workers: true
prefetch_factor: 2
eval_batch_size: 256

# Activate torch.profiler inside UniversalTrainer
use_profiler: true
EOF

echo "========================================================"
echo "  Profiling: ${SETTING}"
echo "  use_amp=${USE_AMP}  benchmark=${BENCHMARK}  fused=${FUSED}"
echo "  Output: ${PROFILE_DIR}/"
echo "========================================================"

# ── 1. nsys profile ──────────────────────────────────────────────────────────
# --trace=cuda,nvtx,cudnn,cublas  : capture GPU kernels + our NVTX labels
# --cpuctxsw=none                  : skip CPU context-switch (needs perf_event_paranoid<=2)
# --gpu-metrics-set=ga100           : remove if GPU arch mismatch; nsys will warn
# --force-overwrite=true            : overwrite previous run
NSYS_OUT="${PROFILE_DIR}/nsys_${SETTING}"

echo ""
echo "[1/3] Running nsys profile..."
nsys profile \
  --trace=cuda,nvtx,cudnn,cublas \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output="${NSYS_OUT}" \
  "$PYTHON" train_executor.py --config "$CFG"

echo "  nsys-rep written: ${NSYS_OUT}.nsys-rep"

# ── 2. Export kernel and NVTX stats from .nsys-rep ───────────────────────────
echo ""
echo "[2/3] Exporting nsys stats..."

# Top CUDA kernels by total execution time
nsys stats \
  --report cuda_kern_exec_trace \
  --format csv \
  --output "${PROFILE_DIR}/nsys_kernels" \
  "${NSYS_OUT}.nsys-rep" 2>/dev/null || \
nsys stats \
  --report kernexectrace \
  --format csv \
  --output "${PROFILE_DIR}/nsys_kernels" \
  "${NSYS_OUT}.nsys-rep" 2>/dev/null || \
  echo "  WARNING: kernel stats export failed (nsys version mismatch — try nsys stats --help)"

# NVTX push/pop ranges (forward / backward / ema_update / gpu_augment)
nsys stats \
  --report nvtx_pushpop_trace \
  --format csv \
  --output "${PROFILE_DIR}/nsys_nvtx" \
  "${NSYS_OUT}.nsys-rep" 2>/dev/null || \
  echo "  WARNING: NVTX stats export failed"

# ── 3. Parse results and write report ────────────────────────────────────────
echo ""
echo "[3/3] Generating report..."

"$PYTHON" - <<PYEOF
import os
import glob
import csv

PROFILE_DIR = "${PROFILE_DIR}"
SETTING     = "${SETTING}"

lines = []
lines.append(f"Profiling Report — {SETTING}")
lines.append("=" * 60)

# ── Parse NVTX ranges (forward / backward / ema / augment) ───────────────
nvtx_csv = os.path.join(PROFILE_DIR, "nsys_nvtx.csv")
if os.path.exists(nvtx_csv):
    ranges = {}
    with open(nvtx_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", row.get("name", ""))
            # Prefer "Total Time (ns)" or "Sum" column
            for col in ("Total Time (ns)", "Sum", "Total Duration (ns)"):
                if col in row and row[col]:
                    try:
                        ranges[name] = ranges.get(name, 0) + int(row[col])
                    except ValueError:
                        pass
                    break

    if ranges:
        lines.append("\nNVTX Range Breakdown (total ns across all profiled iterations):")
        lines.append(f"  {'Range':<20} {'Total (ms)':>12}  {'Share':>8}")
        lines.append("  " + "-" * 44)
        total_ns = sum(ranges.values())
        for name, ns in sorted(ranges.items(), key=lambda x: -x[1]):
            pct = ns / total_ns * 100 if total_ns else 0
            lines.append(f"  {name:<20} {ns/1e6:>12.1f}  {pct:>7.1f}%")
    else:
        lines.append("\nNVTX CSV found but no ranges parsed.")
else:
    lines.append("\nNVTX CSV not found — nsys export may have failed.")

# ── Parse top CUDA kernels ────────────────────────────────────────────────
kernel_csv = os.path.join(PROFILE_DIR, "nsys_kernels.csv")
if os.path.exists(kernel_csv):
    kernels = []
    with open(kernel_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", row.get("name", ""))
            for col in ("Total Time (ns)", "Sum", "Total Duration (ns)"):
                if col in row and row[col]:
                    try:
                        kernels.append((name, int(row[col])))
                    except ValueError:
                        pass
                    break

    if kernels:
        kernels.sort(key=lambda x: -x[1])
        lines.append("\nTop 10 CUDA Kernels by Total Time:")
        lines.append(f"  {'Kernel':<55} {'Total (ms)':>10}")
        lines.append("  " + "-" * 67)
        for name, ns in kernels[:10]:
            short = name[:54]
            lines.append(f"  {short:<55} {ns/1e6:>10.1f}")
    else:
        lines.append("\nKernel CSV found but no data parsed.")
else:
    lines.append("\nKernel CSV not found — nsys export may have failed.")

# ── torch.profiler traces ─────────────────────────────────────────────────
traces = glob.glob(os.path.join(PROFILE_DIR, "profiler_trace", "*.pt.trace.json"))
# The exp_results dir (copied by training) may contain the trace
exp_traces = glob.glob("exp_results/*/profiler_trace/*.pt.trace.json")
all_traces = traces + exp_traces
if all_traces:
    lines.append(f"\ntorch.profiler Chrome traces ({len(all_traces)} file(s)):")
    for t in all_traces:
        size_mb = os.path.getsize(t) / 1e6
        lines.append(f"  {t}  ({size_mb:.1f} MB)")
    lines.append("  → Open in: chrome://tracing  or  https://ui.perfetto.dev")
else:
    lines.append("\ntorch.profiler trace not found (may be in exp_results/<run>/profiler_trace/).")

lines.append("")
report = "\n".join(lines)
print(report)

report_path = os.path.join(PROFILE_DIR, "report.txt")
with open(report_path, "w") as f:
    f.write(report + "\n")
print(f"Report written to {report_path}")
PYEOF

rm -f "$CFG"
echo ""
echo "Done. Files in ${PROFILE_DIR}/"
echo "  ${NSYS_OUT}.nsys-rep     ← open with Nsight Systems GUI"
echo "  ${PROFILE_DIR}/nsys_kernels.csv  ← top kernels"
echo "  ${PROFILE_DIR}/nsys_nvtx.csv     ← forward/backward/ema breakdown"
echo "  ${PROFILE_DIR}/report.txt        ← human-readable summary"
echo ""
echo "To profile all 4 settings:"
echo "  for s in full_stack no_amp no_benchmark no_fused; do bash profile_stack.sh \$s; done"
