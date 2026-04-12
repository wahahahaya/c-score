"""
grid_alg.py
===========
Cross-algorithm x OOD dataset type comparison experiments.

Sweep dimensions:
    algorithm x ood_dataset x r_ood x seed

Design notes:
    - r_ood=0.0 is each algorithm's own baseline (different algorithms behave
      differently even without OOD); only run once per algorithm x seed,
      not repeated for each dataset.
    - Supports --resume: automatically skips experiment directories that already
      contain a log.txt, allowing interrupted runs to be continued.

Results: exp_results_alg_sweep/
    Directory format: alg_{algorithm}_{ood_dataset}_r{r_ood}_s{seed}_{timestamp}
    Baseline:         alg_{algorithm}_none_r0.0_s{seed}_{timestamp}

Usage:
    # Run all combinations (default: r_ood={0.0, 0.3}, 5 seeds)
    python grid_alg.py

    # Compare specific algorithms
    python grid_alg.py --algorithms fixmatch flexmatch

    # Specific OOD datasets
    python grid_alg.py --ood_datasets cifar100 mnist

    # Custom r_ood values
    python grid_alg.py --r_ood 0.0 0.2 0.4

    # Resume after interruption (skip already-completed experiments)
    python grid_alg.py --resume

Estimated experiment count (default):
    baseline : 4 alg x 5 seeds           =  20 runs
    OOD      : 4 alg x 5 ds x 1 r x 5s  = 100 runs
    Total    : 120 runs x ~43 min        ~  86 hours (3.6 days)
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
    get_optimizer, get_cosine_schedule_with_warmup,
    UniversalTrainer,
)
from core.algorithms.fixmatch  import FixMatch
from core.algorithms.flexmatch import FlexMatch
from core.algorithms.softmatch import SoftMatch
from core.algorithms.ds3l      import DS3L

RESULT_DIR    = "exp_results_alg_sweep"
TEMPLATE_PATH = "configs/template_grid_sweep.yaml"

_ALGORITHM_MAP = {
    'fixmatch':  FixMatch,
    'flexmatch': FlexMatch,
    'softmatch': SoftMatch,
    'ds3l':      DS3L,
}

# ──────────────────────────────────────────
# Default sweep ranges
# ──────────────────────────────────────────
DEFAULT_ALGORITHMS  = ['flexmatch', 'softmatch', 'ds3l']
DEFAULT_OOD_DATASETS = ['cifar100', 'mnist', 'stl10', 'textures', 'gaussian_noise']
DEFAULT_R_OOD        = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_SEEDS        = [0, 10, 20, 30, 40]


def _dir_exists_and_done(save_dir: str) -> bool:
    """Check whether an experiment directory is complete (log.txt exists and contains best accuracy)."""
    log_path = os.path.join(save_dir, "log.txt")
    if not os.path.exists(log_path):
        return False
    with open(log_path) as f:
        return "Best accuracy" in f.read()


def _find_existing_dir(result_dir: str, algorithm: str, ood_name: str,
                        r_ood: float, seed: int) -> str | None:
    """Find an existing experiment directory matching the given conditions (ignoring timestamp)."""
    prefix = f"alg_{algorithm}_{ood_name}_r{r_ood}_s{seed}_"
    for d in os.listdir(result_dir):
        if d.startswith(prefix) and os.path.isdir(os.path.join(result_dir, d)):
            return os.path.join(result_dir, d)
    return None


def run_experiment(config_updates: dict, global_start_str: str,
                   resume: bool = False) -> bool:
    """
    Run a single experiment.
    Returns True if ran, False if skipped (resume mode + already done).
    """
    with open(TEMPLATE_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg.update(config_updates)

    algorithm_name = cfg['algorithm']
    ood_name       = cfg.get('ood_dataset', 'none')
    r_ood          = cfg['r_ood']
    seed           = cfg['seed']
    timestamp      = datetime.datetime.now().strftime("%m%d_%H%M")

    save_dir = os.path.join(
        RESULT_DIR,
        f"alg_{algorithm_name}_{ood_name}_r{r_ood}_s{seed}_{timestamp}"
    )

    # --resume: check whether a completed directory already exists
    if resume:
        existing = _find_existing_dir(RESULT_DIR, algorithm_name, ood_name, r_ood, seed)
        if existing and _dir_exists_and_done(existing):
            print(f"  Already complete, skipping: {os.path.basename(existing)}")
            return False

    os.makedirs(save_dir, exist_ok=True)

    _log_file    = open(os.path.join(save_dir, "log.txt"), "a")
    _orig_stdout = sys.stdout
    sys.stdout   = TeeStream(_log_file, _orig_stdout)

    start_time = time.time()
    try:
        set_seed(cfg['seed'], enable_benchmark=cfg.get('enable_benchmark', True))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        loaders   = build_dataloader(cfg)
        model     = get_model(cfg, device)

        AlgClass  = _ALGORITHM_MAP[algorithm_name]
        algorithm = AlgClass(model, device, cfg)

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
        f"Experiment: algorithm={algorithm_name}, ood_dataset={ood_name},"
        f" r_ood={r_ood}, seed={seed}"
        f" | Duration: {time_str}"
        f" | Finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    with open(os.path.join(save_dir, "time.txt"), "w") as f:
        f.write(log_msg)
    with open(os.path.join(RESULT_DIR, "total_runtime_log.txt"), "a") as f:
        f.write(log_msg)

    print(f"  Duration: {time_str}")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Algorithm x OOD Type Sweep")
    parser.add_argument('--algorithms',   type=str,   nargs='+', default=DEFAULT_ALGORITHMS,
                        choices=list(_ALGORITHM_MAP.keys()),
                        help=f'Algorithm list (default: {DEFAULT_ALGORITHMS})')
    parser.add_argument('--ood_datasets', type=str,   nargs='+', default=DEFAULT_OOD_DATASETS,
                        help=f'OOD dataset list (default: all)')
    parser.add_argument('--r_ood',        type=float, nargs='+', default=DEFAULT_R_OOD,
                        help=f'r_ood values (default: {DEFAULT_R_OOD})')
    parser.add_argument('--seeds',        type=int,   nargs='+', default=DEFAULT_SEEDS,
                        help=f'Seeds (default: {DEFAULT_SEEDS})')
    parser.add_argument('--resume',       action='store_true',
                        help='Skip already-completed experiments (based on presence of log.txt)')
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    global_start_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    algorithms   = args.algorithms
    ood_datasets = args.ood_datasets
    r_ood_list   = sorted(set(args.r_ood))
    seeds        = args.seeds
    resume       = args.resume

    # Compute total experiment count
    has_baseline = 0.0 in r_ood_list
    non_zero_r   = [r for r in r_ood_list if r > 0.0]
    baseline_cnt = len(algorithms) * len(seeds) if has_baseline else 0
    ood_cnt      = len(algorithms) * len(ood_datasets) * len(non_zero_r) * len(seeds)
    total_runs   = baseline_cnt + ood_cnt

    print(f"Algorithm Sweep starting | Global start time: {global_start_str}")
    print(f"   Algorithms:   {algorithms}")
    print(f"   OOD datasets: {ood_datasets}")
    print(f"   r_ood:        {r_ood_list}")
    print(f"   Seeds:        {seeds}")
    print(f"   Resume:       {resume}")
    print(f"Planned experiments: {total_runs}  "
          f"(baseline={baseline_cnt} + OOD={ood_cnt})")
    print(f"   Estimated time: {total_runs * 43 / 60:.1f} hours "
          f"({total_runs * 43 / 60 / 24:.1f} days)\n")

    completed = skipped = failed = 0

    for algorithm in algorithms:
        print(f"\n{'='*60}")
        print(f"  Algorithm: {algorithm.upper()}")
        print(f"{'='*60}")

        for r_ood in r_ood_list:

            # r_ood=0.0: per-algorithm baseline, no dataset split
            if r_ood == 0.0:
                for seed in seeds:
                    label = f"[baseline] alg={algorithm}, r_ood=0.0, seed={seed}"
                    print(f"\n[{completed+skipped+failed+1}/{total_runs}] {label}")
                    updates = {
                        'algorithm':         algorithm,
                        'ood_dataset':       'none',   # no OOD, dataset is irrelevant
                        'r_id':              1.0,
                        'r_ood':             0.0,
                        'seed':              seed,
                        'use_da':            False,
                        'lr':                0.03,
                        'epochs':            200,
                        'num_it_per_epoch':  256,
                        'use_torch_compile': False,
                    }
                    try:
                        ran = run_experiment(updates, global_start_str, resume)
                        if ran: completed += 1
                        else:   skipped   += 1
                    except Exception as e:
                        err_msg = (
                            f"Global Start Time: {global_start_str}\n"
                            f"Crashed: {label} | Error: {e}\n"
                        )
                        print(f"  ERROR: {e}")
                        with open(os.path.join(RESULT_DIR, "total_runtime_log.txt"), "a") as f:
                            f.write(err_msg)
                        failed += 1

            # r_ood > 0.0: full dataset x seed sweep
            else:
                for ood_dataset in ood_datasets:
                    for seed in seeds:
                        label = (f"alg={algorithm}, ood={ood_dataset},"
                                 f" r_ood={r_ood}, seed={seed}")
                        print(f"\n[{completed+skipped+failed+1}/{total_runs}] {label}")
                        updates = {
                            'algorithm':         algorithm,
                            'ood_dataset':       ood_dataset,
                            'r_id':              1.0,
                            'r_ood':             r_ood,
                            'seed':              seed,
                            'use_da':            False,
                            'lr':                0.03,
                            'epochs':            200,
                            'num_it_per_epoch':  256,
                            'use_torch_compile': False,
                        }
                        try:
                            ran = run_experiment(updates, global_start_str, resume)
                            if ran: completed += 1
                            else:   skipped   += 1
                        except Exception as e:
                            err_msg = (
                                f"Global Start Time: {global_start_str}\n"
                                f"Crashed: {label} | Error: {e}\n"
                            )
                            print(f"  ERROR: {e}")
                            with open(os.path.join(RESULT_DIR, "total_runtime_log.txt"), "a") as f:
                                f.write(err_msg)
                            failed += 1

    print(f"\nAlgorithm Sweep complete!")
    print(f"   Completed: {completed}  Skipped: {skipped}  Failed: {failed}  Total: {total_runs}")
