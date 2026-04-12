"""
grid_ood_type.py
================
Sweep the effect of different OOD dataset types on FixMatch.

Sweep dimensions:
    ood_dataset x r_ood x seed

Special handling:
    r_ood=0.0 is a pure ID baseline (independent of ood_dataset);
    each seed is only run once rather than repeated for all 6 datasets.

Usage:
    # Run all combinations (default)
    python grid_ood_type.py

    # Run specific seeds only
    python grid_ood_type.py --seeds 0 10

    # Run specific datasets only
    python grid_ood_type.py --ood_datasets cifar100 textures

    # Run specific r_ood values only
    python grid_ood_type.py --r_ood 0.0 0.3

Results: exp_results_ood_type_sweep/
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
    Supervised, FixMatch,
    get_optimizer, get_cosine_schedule_with_warmup,
    UniversalTrainer,
)

RESULT_DIR = "exp_results_ood_type_sweep"
TEMPLATE_PATH = "configs/template_grid_sweep.yaml"

# ──────────────────────────────────────────
# Default sweep ranges
# ──────────────────────────────────────────
DEFAULT_OOD_DATASETS = ["svhn", "cifar100", "mnist", "stl10", "textures", "gaussian_noise"]
DEFAULT_R_OOD        = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_SEEDS        = [0, 10, 20, 30, 40]


def run_experiment(config_updates, global_start_str):
    start_time = time.time()

    with open(TEMPLATE_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg.update(config_updates)

    timestamp  = datetime.datetime.now().strftime("%m%d_%H%M")
    ood_name   = cfg.get('ood_dataset', 'svhn')
    r_ood      = cfg['r_ood']
    seed       = cfg['seed']

    save_dir = os.path.join(
        RESULT_DIR,
        f"ood_{ood_name}_r{r_ood}_s{seed}_{timestamp}"
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
        num_training_steps = cfg['epochs'] * cfg.get('num_it_per_epoch', 1024)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps, cfg.get('num_warmup_steps', 0)
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
        f"Global Start Time: {global_start_str}\n"
        f"Experiment: ood_dataset={ood_name}, r_ood={r_ood}, seed={seed}"
        f" | Duration: {time_str}"
        f" | Finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    with open(os.path.join(save_dir, "time.txt"), "w") as f:
        f.write(log_msg)
    with open(os.path.join(RESULT_DIR, "total_runtime_log.txt"), "a") as f:
        f.write(log_msg)

    print(f"Experiment duration: {time_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OOD Type Sweep Grid")
    parser.add_argument('--seeds',        type=int,   nargs='+', default=DEFAULT_SEEDS,
                        help=f'Seeds to run (default: {DEFAULT_SEEDS})')
    parser.add_argument('--ood_datasets', type=str,   nargs='+', default=DEFAULT_OOD_DATASETS,
                        help=f'OOD datasets to sweep (default: all)')
    parser.add_argument('--r_ood',        type=float, nargs='+', default=DEFAULT_R_OOD,
                        help=f'r_ood values to sweep (default: {DEFAULT_R_OOD})')
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    global_start_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"OOD Type Sweep starting | Global start time: {global_start_str}")

    ood_datasets = args.ood_datasets
    r_ood_list   = sorted(set(args.r_ood))   # deduplicate and sort so 0.0 comes first
    seeds        = args.seeds

    # Compute total experiment count (r_ood=0.0 runs only once, independent of ood_dataset)
    baseline_runs = len(seeds) if 0.0 in r_ood_list else 0
    ood_runs      = len(ood_datasets) * len([r for r in r_ood_list if r > 0.0]) * len(seeds)
    total_runs    = baseline_runs + ood_runs
    print(f"Planned experiments: {total_runs}  "
          f"(baseline x seed={baseline_runs}  +  "
          f"dataset x r_ood x seed={ood_runs})\n")

    completed = 0
    for r_ood in r_ood_list:
        # r_ood=0.0: pure ID baseline, ood_dataset is irrelevant, only run once
        datasets_this_round = [ood_datasets[0]] if r_ood == 0.0 else ood_datasets

        for ood_dataset in datasets_this_round:
            for seed in seeds:
                label = f"ood={ood_dataset}, r_ood={r_ood}, seed={seed}"
                if r_ood == 0.0:
                    label = f"[baseline] r_ood=0.0, seed={seed}"

                print(f"\n[{completed+1}/{total_runs}] {label}")

                updates = {
                    'algorithm':        'fixmatch',
                    'ood_dataset':      ood_dataset,
                    'r_id':             1.0,
                    'r_ood':            r_ood,
                    'seed':             seed,
                    'use_da':           False,
                    'lr':               0.03,
                    'epochs':           200,
                    'num_it_per_epoch': 256,
                    'use_torch_compile': False,
                }

                try:
                    run_experiment(updates, global_start_str)
                    completed += 1
                except Exception as e:
                    err_msg = (
                        f"Global Start Time: {global_start_str}\n"
                        f"Experiment crashed: {label} | Error: {e}\n"
                    )
                    print(err_msg)
                    with open(os.path.join(RESULT_DIR, "total_runtime_log.txt"), "a") as f:
                        f.write(err_msg)
                    continue

    print(f"\nOOD Type Sweep complete. Ran {completed}/{total_runs} experiments.")
