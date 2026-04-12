#!/usr/bin/env python3
"""
tools/feat_visual.py
====================
Feature space visualization for SafeSSL experiments.

Loads a trained WideResNet from experiment directory and visualizes
the pre-FC embedding space for labeled ID / unlabeled ID / OOD samples.

Usage:
    # Single experiment
    cd <repo-root>
    python tools/feat_visual.py \
        --exp_dir exp_results_ood_sweep/grid_daF_bnF_1.0_0.0_s30_XXXX

    # Side-by-side comparison of multiple r_ood values
    python tools/feat_visual.py \
        --exp_dirs exp_results_ood_sweep/grid_daF_bnF_1.0_0.0_s30_XXXX \
                   exp_results_ood_sweep/grid_daF_bnF_1.0_0.3_s30_XXXX \
                   exp_results_ood_sweep/grid_daF_bnF_1.0_0.5_s30_XXXX \
        --out_dir tools/plots/feat

    # Use PCA (fastest) for quick sanity check
    python tools/feat_visual.py --exp_dir ... --method pca

    # Use UMAP (faster than t-SNE, often better quality)
    python tools/feat_visual.py --exp_dir ... --method umap
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image as PILImage
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.wrn import WideResNet
from core.datasets.cifar10 import CIFAR10Dataset
from core.datasets.OOD import OODDataset

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# 10 distinct colors for ID classes, grey for OOD
_TAB10 = plt.cm.tab10(np.linspace(0, 1, 10))
CLASS_COLORS = [_TAB10[i] for i in range(10)]
OOD_COLOR = (0.55, 0.55, 0.55, 1.0)

CIFAR10_NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2471, 0.2435, 0.2616))
])


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────
def load_model(exp_dir: str, device: torch.device):
    """Load WideResNet + config from an experiment directory."""
    config_path = os.path.join(exp_dir, 'config.json')
    model_path  = os.path.join(exp_dir, 'best_model.pth')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {exp_dir}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"best_model.pth not found in {exp_dir}")

    with open(config_path) as f:
        cfg = json.load(f)

    model = WideResNet(
        depth=cfg.get('depth', 28),
        num_classes=cfg.get('num_classes', 10),
        widen_factor=cfg.get('widen_factor', 2),
        use_dual_bn=False   # EMA model uses single BN for eval
    ).to(device)

    state = torch.load(model_path, map_location=device, weights_only=True)
    # Strip _orig_mod prefix if model was torch.compiled
    if any(k.startswith('_orig_mod.') for k in state.keys()):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────
def extract_features(model: WideResNet,
                     images: torch.Tensor,
                     device: torch.device,
                     batch_size: int = 512) -> np.ndarray:
    """
    Extract pre-FC embeddings via a forward hook on model.fc.
    images: (N, 3, 32, 32) float32 tensor, already normalized.
    Returns: (N, embed_dim) float32 numpy array.
    """
    captured = []

    def _hook(module, inp, out):
        captured.append(inp[0].detach().cpu().float())

    handle = model.fc.register_forward_hook(_hook)
    loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    with torch.no_grad():
        for (batch,) in loader:
            model(batch.to(device))

    handle.remove()
    return torch.cat(captured, dim=0).numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_id_images(data_dir: str, n: int = 2000):
    """CIFAR-10 test set → (images_tensor, labels_array)."""
    dset = CIFAR10Dataset(data_dir, train=False)
    n = min(n, len(dset))
    imgs   = torch.stack([CIFAR10_NORM(PILImage.fromarray(dset.data[i])) for i in range(n)])
    labels = np.array(dset.targets[:n], dtype=np.int64)
    return imgs, labels


def load_ood_images(data_dir: str, ood_name: str, n: int = 1000):
    """OOD pool → images_tensor (labels are implicitly -1)."""
    pool = OODDataset(data_dir, ood_name)
    n    = min(n, len(pool))
    imgs = torch.stack([CIFAR10_NORM(PILImage.fromarray(pool.data[i])) for i in range(n)])
    return imgs


# ──────────────────────────────────────────────────────────────────────────────
# Dimensionality reduction
# ──────────────────────────────────────────────────────────────────────────────
def _reduce(feats: np.ndarray, method: str) -> np.ndarray:
    n = feats.shape[0]
    print(f"  [{method.upper()}] {n} samples × {feats.shape[1]} dims ...", flush=True)

    if method == 'tsne':
        from sklearn.manifold import TSNE
        perp = min(40, n // 4)
        return TSNE(n_components=2, perplexity=perp, n_iter=1000,
                    random_state=42, n_jobs=-1).fit_transform(feats)

    elif method == 'umap':
        import umap as umap_lib
        k = min(30, n // 4)
        return umap_lib.UMAP(n_neighbors=k, min_dist=0.1,
                              random_state=42).fit_transform(feats)

    else:  # pca
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=42).fit_transform(feats)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def _scatter(ax, emb: np.ndarray, labels: np.ndarray,
             num_classes: int = 10, s_id: int = 10, s_ood: int = 14):
    for c in range(num_classes):
        mask = labels == c
        if not mask.any():
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   color=CLASS_COLORS[c], s=s_id, alpha=0.65, linewidths=0,
                   label=CIFAR10_CLASSES[c] if c < len(CIFAR10_CLASSES) else str(c))

    ood_mask = labels == -1
    if ood_mask.any():
        ax.scatter(emb[ood_mask, 0], emb[ood_mask, 1],
                   color=OOD_COLOR, s=s_ood, alpha=0.45,
                   marker='x', linewidths=0.8, label='OOD')

    ax.set_xticks([])
    ax.set_yticks([])


def plot_single_experiment(exp_dir: str, out_dir: str,
                            method: str, n_id: int, n_ood: int,
                            device: torch.device):
    print(f"\n[feat_visual] {os.path.basename(exp_dir)}")
    os.makedirs(out_dir, exist_ok=True)

    model, cfg = load_model(exp_dir, device)
    r_ood    = cfg.get('r_ood', 0.0)
    ood_name = cfg.get('ood_dataset', 'svhn')
    data_dir = cfg.get('data_dir', './data')
    num_cls  = cfg.get('num_classes', 10)

    # Collect images
    id_imgs, id_labels = load_id_images(data_dir, n=n_id)
    all_imgs   = [id_imgs]
    all_labels = [id_labels]

    if r_ood > 0.0 and ood_name != 'none':
        try:
            ood_imgs = load_ood_images(data_dir, ood_name, n=n_ood)
            all_imgs.append(ood_imgs)
            all_labels.append(np.full(len(ood_imgs), -1, dtype=np.int64))
        except Exception as e:
            print(f"  [Warning] Cannot load OOD {ood_name}: {e}")

    all_imgs   = torch.cat(all_imgs, dim=0)
    all_labels = np.concatenate(all_labels)

    # Extract + reduce
    feats = extract_features(model, all_imgs, device)
    emb   = _reduce(feats, method)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter(ax, emb, all_labels, num_classes=num_cls)
    ax.set_title(f"Feature Space | r_ood={r_ood:.1f} | OOD={ood_name} | {method.upper()}",
                 fontsize=11)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7,
              markerscale=2, framealpha=0.9)

    plt.tight_layout()
    tag  = os.path.basename(exp_dir)
    path = os.path.join(out_dir, f"{tag}_{method}.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")


def plot_comparison(exp_dirs: list, out_dir: str,
                    method: str, n_id: int, n_ood: int,
                    device: torch.device):
    """
    Side-by-side feature space panels for multiple experiments.
    Expects experiments that differ only in r_ood (same OOD type, same seed).
    """
    print(f"\n[feat_visual] Comparison: {len(exp_dirs)} experiments")
    os.makedirs(out_dir, exist_ok=True)

    n = len(exp_dirs)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.5))
    if n == 1:
        axes = [axes]

    for i, (exp_dir, ax) in enumerate(zip(exp_dirs, axes)):
        print(f"\n  [{i+1}/{n}] {os.path.basename(exp_dir)}")
        model, cfg = load_model(exp_dir, device)
        r_ood    = cfg.get('r_ood', 0.0)
        ood_name = cfg.get('ood_dataset', 'svhn')
        data_dir = cfg.get('data_dir', './data')
        num_cls  = cfg.get('num_classes', 10)

        id_imgs, id_labels = load_id_images(data_dir, n=n_id)
        all_imgs   = [id_imgs]
        all_labels = [id_labels]

        if r_ood > 0.0 and ood_name != 'none':
            try:
                ood_imgs = load_ood_images(data_dir, ood_name, n=n_ood)
                all_imgs.append(ood_imgs)
                all_labels.append(np.full(len(ood_imgs), -1, dtype=np.int64))
            except Exception as e:
                print(f"    [Warning] {e}")

        all_imgs   = torch.cat(all_imgs, dim=0)
        all_labels = np.concatenate(all_labels)

        feats = extract_features(model, all_imgs, device)
        emb   = _reduce(feats, method)
        _scatter(ax, emb, all_labels, num_classes=num_cls)
        ax.set_title(f"r_ood = {r_ood:.1f}", fontsize=12, fontweight='bold')

    # Legend from last experiment's class list
    patches  = [mpatches.Patch(color=CLASS_COLORS[c], label=CIFAR10_CLASSES[c])
                for c in range(min(10, cfg.get('num_classes', 10)))]
    patches += [mpatches.Patch(color=OOD_COLOR, label='OOD')]
    fig.legend(handles=patches, bbox_to_anchor=(1.0, 0.5), loc='center left',
               fontsize=8, framealpha=0.9)

    title_ood = cfg.get('ood_dataset', 'svhn')
    fig.suptitle(f"Feature Space vs. OOD Contamination | OOD={title_ood} | {method.upper()}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(out_dir, f"comparison_{title_ood}_{method}.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Comparison saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SafeSSL: Feature Space Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--exp_dir',  type=str,
                       help='Single experiment directory')
    group.add_argument('--exp_dirs', type=str, nargs='+',
                       help='Multiple experiment directories (comparison mode)')

    parser.add_argument('--out_dir',   type=str, default='tools/plots/feat')
    parser.add_argument('--method',    type=str, default='tsne',
                        choices=['tsne', 'umap', 'pca'])
    parser.add_argument('--n_id',      type=int, default=2000,
                        help='# ID samples to visualize')
    parser.add_argument('--n_ood',     type=int, default=1000,
                        help='# OOD samples to visualize')
    parser.add_argument('--device',    type=str, default='cuda')
    args = parser.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {dev}')

    if args.exp_dir:
        plot_single_experiment(args.exp_dir, args.out_dir,
                               args.method, args.n_id, args.n_ood, dev)
    elif len(args.exp_dirs) == 1:
        plot_single_experiment(args.exp_dirs[0], args.out_dir,
                               args.method, args.n_id, args.n_ood, dev)
    else:
        plot_comparison(args.exp_dirs, args.out_dir,
                        args.method, args.n_id, args.n_ood, dev)

    print('\n[feat_visual] Done.')
