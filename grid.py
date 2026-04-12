import os
import yaml
import torch
import copy
import datetime
import time
from train_executor import set_seed, build_dataloader, get_model, Supervised, FixMatch, get_optimizer, UniversalTrainer, get_cosine_schedule_with_warmup
from torch import optim

def run_experiment(config_updates, global_start_str):
    # Start timing
    start_time = time.time()

    # Load base config
    template_path = "configs/template_grid_sweep.yaml"
    with open(template_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Apply sweep parameter overrides
    cfg.update(config_updates)

    # Build experiment save directory
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    da_str = "T" if cfg.get('use_da', False) else "F"

    save_dir = os.path.join(
        "exp_results_ood_sweep",
        f"grid_da{da_str}_{cfg['r_id']}_{cfg['r_ood']}_s{cfg['seed']}_{timestamp}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # 1. Set random seed
    set_seed(cfg['seed'], enable_benchmark=cfg.get('enable_benchmark', True))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Build DataLoaders
    loaders = build_dataloader(cfg)

    # 3. Get model
    model = get_model(cfg, device)

    # 4. Select algorithm
    if cfg['algorithm'] == 'supervised':
        algorithm = Supervised(model, device, cfg)
    elif cfg['algorithm'] == 'fixmatch':
        algorithm = FixMatch(model, device, cfg)

    # 5. Optimizer and scheduler (USB standard: per-iteration cosine with optional warmup)
    optimizer = get_optimizer(model, cfg)
    num_training_steps = cfg['epochs'] * cfg.get('num_it_per_epoch', 1024)
    num_warmup_steps = cfg.get('num_warmup_steps', 0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps, num_warmup_steps)

    # 6. Launch training
    trainer = UniversalTrainer(algorithm, loaders, optimizer, scheduler, cfg, device, save_dir)
    trainer.run()

    # Compute elapsed time
    end_time = time.time()
    duration = end_time - start_time

    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    time_str = f"{hours}h {minutes}m {seconds}s"

    # Log timing (experiment directory and global log)
    log_message = f"Global Start Time: {global_start_str}\n"
    log_message += f"Experiment: da={cfg.get('use_da', False)}, r_id={cfg['r_id']}, r_ood={cfg['r_ood']}, seed={cfg['seed']} | Duration: {time_str} | Finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    # 1. Write to individual experiment directory
    with open(os.path.join(save_dir, "time.txt"), "w") as f:
        f.write(log_message)

    # 2. Append to global runtime log
    with open(os.path.join("exp_results_ood_sweep", "total_runtime_log.txt"), "a") as f:
        f.write(log_message)

    print(f"Experiment duration: {time_str}")

    # Clean up VRAM to avoid OOM in subsequent experiments
    del model, algorithm, optimizer, scheduler, loaders, trainer
    torch.cuda.empty_cache()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='Specify a single seed; runs all seeds if not specified')
    args = parser.parse_args()

    r_id_list = [1.0]
    r_ood_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # r = OOD / (ID + OOD)
    seeds = [args.seed] if args.seed is not None else [30, 40]

    use_da_list = [False]

    # Ensure output directory exists
    os.makedirs("exp_results_ood_sweep", exist_ok=True)

    global_start_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Starting validation sweep... | Global start time: {global_start_str}")

    for use_da in use_da_list:
        for seed in seeds:
                for rid in r_id_list:
                    for rood in r_ood_list:
                        print(f"\nRunning experiment: DA={use_da}, r_id={rid}, r_ood={rood}, seed={seed}")

                        # Dynamically determine algorithm and config
                        is_supervised = (rid == 0.0 and rood == 0.0)
                        current_algorithm = 'supervised' if is_supervised else 'fixmatch'

                        updates = {
                            'r_id': rid,
                            'r_ood': rood,
                            'seed': seed,
                            'algorithm': current_algorithm,

                                    'use_da': False,

                            # Use best learning rate per algorithm (USB standard: 0.03)
                            'lr': 0.1 if is_supervised else 0.03,

                            'epochs': 200,
                            'num_it_per_epoch': 256,
                            'use_torch_compile': False    # disabled for faster startup
                        }

                        try:
                            run_experiment(updates, global_start_str)
                        except Exception as e:
                            error_msg = f"Global Start Time: {global_start_str}\nExperiment crashed: DA={use_da}, rid={rid}, rood={rood}, error: {str(e)}\n"
                            print(error_msg)
                            with open(os.path.join("exp_results_ood_sweep", "total_runtime_log.txt"), "a") as f:
                                f.write(error_msg)
                            continue

    print("All validation sweeps complete.")
