import os
import sys
import math
import yaml
import argparse
import datetime
import random
import platform
import subprocess
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.optim.lr_scheduler import LambdaLR

from core.models.wrn import WideResNet
from core.datasets.builder import build_dataloader
from core.algorithms.supervised import Supervised
from core.algorithms.fixmatch import FixMatch
from core.algorithms.flexmatch import FlexMatch
from core.algorithms.softmatch import SoftMatch
from core.algorithms.ds3l import DS3L
from engines.trainer import UniversalTrainer
from engines.logger import TeeStream

def print_system_info(device):
    """Print CPU, GPU, kernel, and RAM info at experiment start."""
    print(f"Kernel : {platform.uname().release}")
    print(f"OS     : {platform.system()} {platform.version()[:60]}")

    # CPU model and topology via lscpu
    try:
        lscpu = subprocess.check_output(['lscpu'], text=True, stderr=subprocess.DEVNULL)
        keys = ('Model name', 'CPU(s)', 'Thread(s) per core', 'Core(s) per socket',
                'Socket(s)', 'CPU max MHz', 'CPU min MHz')
        for line in lscpu.splitlines():
            if any(line.startswith(k) for k in keys):
                print(f"  {line.strip()}")
    except Exception:
        print(f"CPU    : {platform.processor() or 'unknown'}")

    # RAM
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    kb = int(line.split()[1])
                    print(f"RAM    : {kb / 1e6:.1f} GB")
                    break
    except Exception:
        pass

    # GPU (if CUDA)
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {props.name}")
        print(f"VRAM   : {props.total_memory / 1e9:.2f} GB")
        print(f"SM     : {props.multi_processor_count}  |  "
              f"Compute capability: {props.major}.{props.minor}")
        print(f"CUDA   : {torch.version.cuda}  |  cuDNN: {torch.backends.cudnn.version()}")
    else:
        print("GPU    : none (CPU-only run)")


def set_seed(seed, enable_benchmark=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if enable_benchmark:
        cudnn.benchmark = True
        cudnn.deterministic = False
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print(f"cuDNN benchmark enabled [{gpu_name}]")
    else:
        cudnn.deterministic = True
        cudnn.benchmark = False
        print("Deterministic mode enabled - reproducible results")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

def get_model(cfg, device):
    name = cfg.get('model_name', 'wrn')
    num_classes = cfg['num_classes']

    if name == 'wrn':
        model = WideResNet(
            depth=cfg['depth'],
            num_classes=num_classes,
            widen_factor=cfg['widen_factor'],
        )
    elif name == 'vit':
        model = models.vit_b_16(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")

    model = model.to(device)

    if hasattr(torch, 'compile') and cfg.get('use_torch_compile', False):
        print("Compiling model with torch.compile()...")
        model = torch.compile(
            model,
            mode=cfg.get('compile_mode', 'reduce-overhead'),
            fullgraph=False  # DualBN uses Python control flow; fullgraph=True would fail
        )
        print("Model compilation complete")

    return model

def get_cosine_schedule_with_warmup(optimizer, num_training_steps, num_warmup_steps=0):
    """USB FixMatch standard scheduler: cosine decay with optional warmup, step unit = iteration.
    Terminal LR = cos(pi * 7/16) ~= 0.195 * lr_max (does not decay to 0, consistent with USB).
    """
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, math.cos(math.pi * 7.0 / 16.0 * progress))
    return LambdaLR(optimizer, _lr_lambda)

def get_optimizer(model, cfg):
    no_decay = ["bias", "bn"]
    parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg['weight_decay']},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    use_fused_optimizer = cfg.get('use_fused_optimizer', False)

    if use_fused_optimizer:
        # Priority: PyTorch native fused SGD first (better AMP GradScaler integration
        # in PyTorch >= 2.0), then Apex FusedSGD as fallback, then standard SGD.
        try:
            optimizer = optim.SGD(parameters, lr=cfg['lr'], momentum=0.9, nesterov=True, fused=True)
            print("Using PyTorch native fused SGD")
        except (TypeError, RuntimeError) as e_pt:
            print(f"WARNING: PyTorch native fused SGD unavailable ({e_pt}), trying Apex FusedSGD")
            try:
                from apex.optimizers import FusedSGD
                optimizer = FusedSGD(parameters, lr=cfg['lr'], momentum=0.9, nesterov=True)
                print("Using NVIDIA Apex FusedSGD")
            except (ImportError, RuntimeError) as e_apex:
                print(f"WARNING: Apex FusedSGD unavailable ({e_apex}), falling back to standard SGD")
                optimizer = optim.SGD(parameters, lr=cfg['lr'], momentum=0.9, nesterov=True)
    else:
        optimizer = optim.SGD(parameters, lr=cfg['lr'], momentum=0.9, nesterov=True)

    return optimizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--fast', action='store_true', help='Enable cuDNN benchmark and performance optimizations')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    enable_benchmark = args.fast or cfg.get('enable_benchmark', False)

    # Create save_dir and attach TeeStream so all subsequent prints go to the log
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    save_dir = os.path.join("exp_results", f"{cfg['algorithm']}_{cfg['dataset']}_seed{cfg.get('seed', 42)}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    _log_file    = open(os.path.join(save_dir, "log.txt"), "a")
    _orig_stdout = sys.stdout
    sys.stdout   = TeeStream(_log_file, _orig_stdout)

    try:
        set_seed(cfg.get('seed', 42), enable_benchmark=enable_benchmark)

        # force_cpu: true in config → run entirely on CPU regardless of GPU availability
        force_cpu = cfg.get('force_cpu', False)
        if force_cpu:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print_system_info(device)

        loaders = build_dataloader(cfg)
        model = get_model(cfg, device)

        if cfg['algorithm'] == 'supervised':
            algorithm = Supervised(model, device, cfg)
        elif cfg['algorithm'] == 'fixmatch':
            algorithm = FixMatch(model, device, cfg)

        optimizer = get_optimizer(model, cfg)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

        trainer = UniversalTrainer(algorithm, loaders, optimizer, scheduler, cfg, device, save_dir)
        trainer.run()

    finally:
        sys.stdout = _orig_stdout
        _log_file.close()

if __name__ == '__main__':
    main()
