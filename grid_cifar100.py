"""
grid_cifar100.py
================
SafeSSL: CIFAR-100 as In-Distribution dataset sweep.

Mirrors the primary grid.py experiment but with CIFAR-100 as ID:
  - ID: CIFAR-100, 100 classes, 400 labeled samples (4 per class)
  - OOD: SVHN (far-OOD) — same as primary CIFAR-10 experiments
  - Algorithm: FixMatch
  - r_ood in {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}

This validates that the C-Score collapse phenomenon generalizes beyond CIFAR-10.

Usage:
    # All seeds
    python grid_cifar100.py

    # Specific seeds (e.g. run seed 0 and 10 on this machine)
    python grid_cifar100.py --seeds 0 10

    # Specific r_ood values only
    python grid_cifar100.py --r_ood 0.0 0.3 0.5

Results: exp_results_cifar100_sweep/
"""

import os
import sys
import yaml
import torch
import datetime
import time
import argparse
from engines.logger import TeeStream
from train_executor import (
    set_seed, build_dataloader, get_model,
    FixMatch, get_optimizer, get_cosine_schedule_with_warmup, UniversalTrainer,
)

RESULT_DIR    = "exp_results_cifar100_sweep"
TEMPLATE_PATH = "configs/template_grid_sweep.yaml"

DEFAULT_R_OOD = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_SEEDS  = [0, 10, 20, 30, 40]


def run_experiment(config_updates: dict, global_start_str: str):
    start_time = time.time()

    with open(TEMPLATE_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg.update(config_updates)

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    r_ood     = cfg['r_ood']
    seed      = cfg['seed']

    save_dir = os.path.join(
        RESULT_DIR,
        f"cifar100_r{r_ood}_s{seed}_{timestamp}"
    )
    os.makedirs(save_dir, exist_ok=True)

    _log_file    = open(os.path.join(save_dir, "log.txt"), "a")
    _orig_stdout = sys.stdout
    sys.stdout   = TeeStream(_log_file, _orig_stdout)

    try:
        set_seed(cfg['seed'], enable_benchmark=cfg.get('enable_benchmark', True))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        loaders   = build_dataloader(cfg)
        model     = get_model(cfg, device)
        algorithm = FixMatch(model, device, cfg)

        optimizer = get_optimizer(model, cfg)
        num_steps = cfg['epochs'] * cfg.get('num_it_per_epoch', 256)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_steps, cfg.get('num_warmup_steps', 0)
        )

        trainer = UniversalTrainer(algorithm, loaders, optimizer, scheduler, cfg, device, save_dir)
        trainer.run()

        del model, algorithm, optimizer, scheduler, loaders, trainer
        torch.cuda.empty_cache()
    finally:
        sys.stdout = _orig_stdout
        _log_file.close()

    duration = time.time() - start_time
    h = int(duration // 3600)
    m = int((duration % 3600) // 60)
    s = int(duration % 60)
    time_str = f"{h}h {m}m {s}s"

    log_msg = (
        f"Global Start: {global_start_str}\n"
        f"Experiment: dataset=cifar100, r_ood={r_ood}, seed={seed}"
        f" | Duration: {time_str}"
        f" | Finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    with open(os.path.join(save_dir, "time.txt"), "w") as f:
        f.write(log_msg)
    with open(os.path.join(RESULT_DIR, "total_runtime_log.txt"), "a") as f:
        f.write(log_msg)

    print(f"  Duration: {time_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CIFAR-100 as ID sweep")
    parser.add_argument('--seeds',  type=int,   nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--r_ood',  type=float, nargs='+', default=DEFAULT_R_OOD)
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    global_start_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    r_ood_list = sorted(set(args.r_ood))
    seeds      = args.seeds
    total      = len(r_ood_list) * len(seeds)

    print(f"SafeSSL: CIFAR-100 as ID Sweep")
    print(f"  r_ood: {r_ood_list}")
    print(f"  seeds: {seeds}")
    print(f"  total: {total} experiments")
    print(f"  start: {global_start_str}\n")

    completed = 0
    for r_ood in r_ood_list:
        for seed in seeds:
            print(f"\n[{completed+1}/{total}] r_ood={r_ood}, seed={seed}")
            updates = {
                # Dataset
                'dataset':          'cifar100',
                'num_classes':      100,
                'num_labeled':      400,    # 4 per class x 100 classes

                # OOD
                'ood_dataset':      'svhn',
                'r_id':             1.0,
                'r_ood':            r_ood,

                # Training
                'algorithm':        'fixmatch',
                'seed':             seed,
                'use_da':           False,
                'lr':               0.03,
                'epochs':           500,
                'num_it_per_epoch': 512,
                'use_torch_compile': False,
            }
            try:
                run_experiment(updates, global_start_str)
                completed += 1
            except Exception as e:
                err = (
                    f"Global Start: {global_start_str}\n"
                    f"FAILED: r_ood={r_ood}, seed={seed} | Error: {e}\n"
                )
                print(f"  ERROR: {e}")
                with open(os.path.join(RESULT_DIR, "total_runtime_log.txt"), "a") as f:
                    f.write(err)
                continue

    print(f"\nDone. Completed {completed}/{total} experiments.")
    print(f"Results in: {RESULT_DIR}/")
