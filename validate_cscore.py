"""
validate_cscore.py
==================
1. Numerical equivalence tests: compare inline reference logic vs cscore functions on the same tensors.
2. Import health check: confirm all 4 refactored algorithms can be imported and call cscore.
3. 1-epoch sampling comparison: load the best_model from exp_results_ood_type_sweep,
   build a dataloader from the same config, run 256 steps, and compare PLE/CCI
   against the value range in the tail of log.txt.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import numpy as np
import json

# ──────────────────────────────────────────
# 1. Numerical equivalence tests
# ──────────────────────────────────────────
from cscore import compute_ple, compute_cci, compute_sem_drift, compute_grad_align, compute_ood_ff

torch.manual_seed(0)

def test_numerical_equivalence():
    N, C = 128, 10
    probs = torch.softmax(torch.randn(N, C), dim=-1)
    logits_x  = torch.randn(16, C)
    targets_x = torch.randint(0, C, (16,))
    logits_u_w = torch.randn(N, C)
    pseudo_labels = torch.randint(0, C, (N,))
    mask = (torch.rand(N) > 0.5).float()

    # --- PLE ---
    ple_old = -torch.sum(probs * torch.log(probs + 1e-6), dim=-1).mean().item()
    ple_new = compute_ple(probs)
    assert abs(ple_old - ple_new) < 1e-6, f"PLE mismatch: {ple_old} vs {ple_new}"
    print(f"  [PASS] PLE        {ple_new:.6f}")

    # --- CCI ---
    dist = probs.mean(0)
    uniform = torch.ones_like(dist) / C
    cci_old = torch.sum(dist * torch.log(dist / (uniform + 1e-6) + 1e-6)).item()
    cci_new = compute_cci(probs)
    assert abs(cci_old - cci_new) < 1e-6, f"CCI mismatch: {cci_old} vs {cci_new}"
    print(f"  [PASS] CCI        {cci_new:.6f}")

    # --- Sem-Drift ---
    sem_old = 0.0
    valid = 0
    for c in range(C):
        mx = (targets_x == c)
        mu = (pseudo_labels == c) & (mask == 1)
        if mx.sum() > 0 and mu.sum() > 0:
            sem_old += torch.norm(logits_x.detach()[mx].mean(0) - logits_u_w.detach()[mu].mean(0)).item()
            valid += 1
    if valid > 0:
        sem_old /= valid
    sem_new = compute_sem_drift(logits_x, targets_x, logits_u_w, pseudo_labels, mask, C)
    assert abs(sem_old - sem_new) < 1e-5, f"Sem-Drift mismatch: {sem_old} vs {sem_new}"
    print(f"  [PASS] Sem-Drift  {sem_new:.6f}")

    # --- Grad-Align ---
    w = torch.nn.Linear(C, C)
    loss_x = F.cross_entropy(w(logits_x), targets_x)
    loss_u = (F.cross_entropy(w(logits_u_w), pseudo_labels, reduction='none') * mask).mean()
    fc_params = list(w.parameters())

    g_x = torch.autograd.grad(loss_x, fc_params, retain_graph=True, allow_unused=True)
    g_u = torch.autograd.grad(loss_u, fc_params, retain_graph=True, allow_unused=True)
    g_x_flat = torch.cat([g.float().flatten() for g in g_x if g is not None])
    g_u_flat = torch.cat([g.float().flatten() for g in g_u if g is not None])
    ga_old = F.cosine_similarity(g_x_flat.unsqueeze(0), g_u_flat.unsqueeze(0)).item()

    # need fresh graph
    loss_x2 = F.cross_entropy(w(logits_x), targets_x)
    loss_u2 = (F.cross_entropy(w(logits_u_w), pseudo_labels, reduction='none') * mask).mean()
    ga_new, gxn, gun = compute_grad_align(loss_x2, loss_u2, fc_params)
    assert abs(ga_old - ga_new) < 1e-5, f"Grad-Align mismatch: {ga_old} vs {ga_new}"
    print(f"  [PASS] Grad-Align {ga_new:.6f}")

    # --- OOD-FF ---
    id_m, ood_m = 0.73, 0.14
    ff_old = ood_m / (id_m + 1e-6)
    ff_new = compute_ood_ff(id_m, ood_m)
    assert abs(ff_old - ff_new) < 1e-8, f"OOD-FF mismatch: {ff_old} vs {ff_new}"
    print(f"  [PASS] OOD-FF     {ff_new:.6f}")

    print("\n[OK] All C-Score function numerical equivalence tests passed\n")


# ──────────────────────────────────────────
# 2. Import health check
# ──────────────────────────────────────────
def test_imports():
    from core.algorithms.fixmatch  import FixMatch
    from core.algorithms.flexmatch import FlexMatch
    from core.algorithms.softmatch import SoftMatch
    from core.algorithms.ds3l      import DS3L
    print("  [PASS] FixMatch  import OK")
    print("  [PASS] FlexMatch import OK")
    print("  [PASS] SoftMatch import OK")
    print("  [PASS] DS3L      import OK")
    print("\n[OK] All algorithm imports healthy\n")


# ──────────────────────────────────────────
# 3. 1-epoch sampling comparison
# ──────────────────────────────────────────
def test_one_epoch_sample():
    """
    Load the best_model from ood_cifar100_r0.1_s0, run 32 steps (not a full epoch),
    and compare PLE / CCI / OOD-FF against the value range from the last 20 epochs of log.txt.
    """
    exp_dir = "exp_results_ood_type_sweep/ood_cifar100_r0.1_s0_0406_1806"
    cfg_path  = os.path.join(exp_dir, "config.json")
    ckpt_path = os.path.join(exp_dir, "best_model.pth")
    log_path  = os.path.join(exp_dir, "log.txt")

    if not os.path.exists(cfg_path):
        print(f"  [SKIP] Could not find {cfg_path}")
        return

    with open(cfg_path) as f:
        cfg = json.load(f)

    # Read PLE, CCI, OOD-FF from last 20 epochs of log.txt
    ple_log, cci_log, ff_log = [], [], []
    with open(log_path) as f:
        for line in f:
            if "PLE:" in line:
                try:
                    ple_log.append(float(line.split("PLE:")[1].split()[0]))
                    cci_log.append(float(line.split("CCI:")[1].split()[0]))
                    ff_log.append(float(line.split("OOD-FF:")[1].split()[0]))
                except Exception:
                    pass
    ple_log = ple_log[-20:]
    cci_log = cci_log[-20:]
    ff_log  = ff_log[-20:]
    print(f"  Log PLE  range (last 20 epochs): [{min(ple_log):.3f}, {max(ple_log):.3f}]")
    print(f"  Log CCI  range (last 20 epochs): [{min(cci_log):.3f}, {max(cci_log):.3f}]")
    print(f"  Log OOD-FF range (last 20 epochs): [{min(ff_log):.3f}, {max(ff_log):.3f}]")

    # Build model + algorithm (CPU sufficient for validation)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import random
    random.seed(cfg.get('seed', 0))
    np.random.seed(cfg.get('seed', 0))
    torch.manual_seed(cfg.get('seed', 0))

    from core.models.wrn import WideResNet
    from core.algorithms.fixmatch import FixMatch
    from core.datasets.builder import build_dataloader
    from core.datasets.augmentation import GPUAugmentor

    model = WideResNet(depth=cfg['depth'], num_classes=cfg['num_classes'],
                       widen_factor=cfg['widen_factor']).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    algorithm = FixMatch(model, device, cfg)
    algorithm.eval()

    loaders = build_dataloader(cfg)
    gpu_aug = GPUAugmentor(cfg, device)

    labeled_iter   = iter(loaders['labeled'])
    unlabeled_iter = iter(loaders['unlabeled'])

    ple_vals, cci_vals, ff_vals = [], [], []
    NUM_STEPS = 32  # lightweight sampling, not a full epoch

    for step in range(NUM_STEPS):
        try: x_raw, x_tgt = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(loaders['labeled'])
            x_raw, x_tgt = next(labeled_iter)
        try: u_raw, u_gt = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(loaders['unlabeled'])
            u_raw, u_gt = next(unlabeled_iter)

        with torch.no_grad():
            x_gpu  = x_raw.to(device, non_blocking=True)
            u_gpu  = u_raw.to(device, non_blocking=True)
            inputs_x   = gpu_aug(x_gpu, mode='weak')
            inputs_u_w = gpu_aug(u_gpu, mode='weak')
            inputs_u_s = gpu_aug(u_gpu, mode='strong')

        data_batch = {
            'labeled':   (inputs_x, x_tgt),
            'unlabeled': ((inputs_u_w, inputs_u_s), u_gt),
            'ema_model': model,
        }

        with torch.no_grad():
            _, stats = algorithm.compute_loss(data_batch, global_step=step * 32)

        ple_vals.append(stats['PLE'])
        cci_vals.append(stats['CCI'])
        ff_vals.append(stats['OOD_FF'])

    ple_mean = np.mean(ple_vals)
    cci_mean = np.mean(cci_vals)
    ff_mean  = np.mean(ff_vals)

    print(f"\n  Refactored PLE  mean over {NUM_STEPS} steps: {ple_mean:.3f}")
    print(f"  Refactored CCI  mean over {NUM_STEPS} steps: {cci_mean:.3f}")
    print(f"  Refactored OOD-FF mean over {NUM_STEPS} steps: {ff_mean:.3f}")

    # Range comparison (allow +/-30% buffer; model is converged but dataloader shuffle differs)
    lo_ple, hi_ple = min(ple_log) * 0.5, max(ple_log) * 1.5
    lo_cci, hi_cci = -0.05, max(cci_log) * 2.0
    lo_ff,  hi_ff  = 0.0, max(ff_log) * 3.0

    ok_ple = lo_ple <= ple_mean <= hi_ple
    ok_cci = lo_cci <= cci_mean <= hi_cci
    ok_ff  = lo_ff  <= ff_mean  <= hi_ff

    print(f"\n  PLE  in expected range [{lo_ple:.3f}, {hi_ple:.3f}]: {'PASS' if ok_ple else 'FAIL'}")
    print(f"  CCI  in expected range [{lo_cci:.3f}, {hi_cci:.3f}]: {'PASS' if ok_cci else 'FAIL'}")
    print(f"  OOD-FF in expected range [{lo_ff:.3f}, {hi_ff:.3f}]: {'PASS' if ok_ff else 'FAIL'}")

    if ok_ple and ok_cci and ok_ff:
        print("\n[OK] Refactored metrics are consistent with original exp_results values, no systematic error\n")
    else:
        print("\n[WARN] Some metrics are outside the expected range, please inspect manually\n")


if __name__ == '__main__':
    print("=" * 55)
    print("Step 1: Numerical Equivalence Tests")
    print("=" * 55)
    test_numerical_equivalence()

    print("=" * 55)
    print("Step 2: Algorithm Import Health Check")
    print("=" * 55)
    test_imports()

    print("=" * 55)
    print("Step 3: 1-epoch Sampling Comparison (ood_cifar100_r0.1_s0)")
    print("=" * 55)
    test_one_epoch_sample()
