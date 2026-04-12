#!/usr/bin/env python3
"""
tools/data_visual.py
====================
Extracts features using ImageNet pretrained ResNet-18, then projects to 2D with
UMAP to visualize the semantic distance distribution of each OOD dataset relative
to CIFAR-10 (ID).

No model_path / config_path required; runs directly.

Purpose: Explain why near-OOD (CIFAR-100, STL-10) easily infiltrates the ID pool,
         and why far-OOD (MNIST, Gaussian Noise) is naturally filtered by the
         confidence threshold.

Usage:
    cd <repo-root>

    # Default: UMAP, all OOD types
    python tools/data_visual.py

    # Specific OOD types only
    python tools/data_visual.py --ood_datasets cifar100 stl10 mnist gaussian_noise

    # Custom output directory
    python tools/data_visual.py --out_dir tools/plots/data_umap
"""

import os
import sys
import argparse
import numpy as np
import torch
import torchvision.models as tv_models
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from PIL import Image as PILImage
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.datasets.cifar10 import CIFAR10Dataset
from core.datasets.OOD import OODDataset
from core.models.wrn import WideResNet

# ──────────────────────────────────────────────────────────────────────────────
# OOD metadata
# ──────────────────────────────────────────────────────────────────────────────
OOD_META = {
    # near-OOD: warm orange family
    'cifar100':       {'label': 'CIFAR-100',      'color': '#d94801', 'marker': 's',  'proximity': 'near'},
    'stl10':          {'label': 'STL-10',          'color': '#fd8d3c', 'marker': 'D',  'proximity': 'near'},
    # mid-OOD: blue
    'svhn':           {'label': 'SVHN',            'color': '#2171b5', 'marker': 'o',  'proximity': 'far'},
    # far-OOD: purple / green / pink — visually distinct from near & mid
    'mnist':          {'label': 'MNIST',           'color': '#6a51a3', 'marker': 'v',  'proximity': 'far'},
    'textures':       {'label': 'Textures (DTD)',  'color': '#238b45', 'marker': 'P',  'proximity': 'far'},
    'gaussian_noise': {'label': 'Gaussian Noise',  'color': '#d03992', 'marker': 'X',  'proximity': 'far'},
}

# ordering for legend: near → mid → far
_PROX_RANK = {'near': 0, 'mid': 1, 'far': 2}
ALL_OOD_DATASETS = list(OOD_META.keys())

# ── ImageNet ResNet-18 transforms (resize to 224) ────────────────────────────
IMAGENET_TRANSFORM = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ── CIFAR-10 WRN transforms (32×32 input, CIFAR-10 stats) ───────────────────
CIFAR10_NORM = T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2471, 0.2435, 0.2616])

CIFAR10_TRANSFORM = T.Compose([
    T.Resize(32, interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    CIFAR10_NORM,
])


# ──────────────────────────────────────────────────────────────────────────────
# Feature extractor
# ──────────────────────────────────────────────────────────────────────────────
def build_feature_extractor(device: torch.device,
                             model_path: str | None = None,
                             depth: int = 28,
                             widen_factor: int = 2,
                             num_classes: int = 10) -> torch.nn.Module:
    """
    If model_path is given: load WideResNet-{depth}-{widen_factor} supervised
    on CIFAR-10, strip the FC layer → outputs nChannels-dim features.

    Otherwise: fall back to ImageNet-pretrained ResNet-18 (512-dim).
    Callers should use CIFAR10_TRANSFORM / IMAGENET_TRANSFORM accordingly.
    """
    if model_path is not None:
        model = WideResNet(depth=depth, num_classes=num_classes,
                           widen_factor=widen_factor, drop_rate=0.0,
                           use_dual_bn=False)
        state = torch.load(model_path, map_location='cpu')
        # checkpoint may be bare state_dict or wrapped under 'model' key
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        # torch.compile() prefixes all keys with '_orig_mod.' — strip it
        state = {k.replace('_orig_mod.', '', 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        model.fc = torch.nn.Identity()          # strip classifier → nChannels-dim
        model = model.to(device).eval()
        feat_dim = model.nChannels
        print(f'  Feature extractor: WRN-{depth}-{widen_factor} (CIFAR-10 supervised, '
              f'{feat_dim}-dim), loaded from {model_path}')
        return model
    else:
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
        model = tv_models.resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        model = model.to(device).eval()
        print(f'  Feature extractor: ResNet-18 (ImageNet pretrained, 512-dim), device={device}')
        return model


# ──────────────────────────────────────────────────────────────────────────────
# Data loading: convert to tensor and batch-infer
# ──────────────────────────────────────────────────────────────────────────────
def _pil_to_rgb(arr: np.ndarray) -> PILImage.Image:
    img = PILImage.fromarray(arr)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def load_id_tensors(data_dir: str, n: int, use_wrn: bool = False) -> torch.Tensor:
    tfm  = CIFAR10_TRANSFORM if use_wrn else IMAGENET_TRANSFORM
    dset = CIFAR10Dataset(data_dir, train=False)
    n    = min(n, len(dset))
    return torch.stack([tfm(_pil_to_rgb(dset.data[i])) for i in range(n)])


def load_ood_tensors(data_dir: str, ood_name: str, n: int,
                     use_wrn: bool = False) -> torch.Tensor:
    # WRN expects 32×32; CIFAR10_TRANSFORM already does T.Resize(32)
    # so it handles STL-10 (96×96) and all other sizes correctly.
    tfm  = CIFAR10_TRANSFORM if use_wrn else IMAGENET_TRANSFORM
    pool = OODDataset(data_dir, ood_name)
    n    = min(n, len(pool))
    return torch.stack([tfm(_pil_to_rgb(pool.data[i])) for i in range(n)])


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_features(model: torch.nn.Module,
                     images: torch.Tensor,
                     device: torch.device,
                     batch_size: int = 256) -> np.ndarray:
    loader = DataLoader(TensorDataset(images), batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)
    feats = []
    for (batch,) in loader:
        feats.append(model(batch.to(device)).cpu().float())
    return torch.cat(feats, dim=0).numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Joint UMAP dimensionality reduction
# ──────────────────────────────────────────────────────────────────────────────
def reduce_umap(features_dict: dict,
                n_neighbors: int = 30,
                min_dist: float = 0.1) -> dict:
    import umap as umap_lib

    keys      = list(features_dict.keys())
    all_feats = np.concatenate([features_dict[k] for k in keys], axis=0)
    sizes     = [len(features_dict[k]) for k in keys]
    n_total   = all_feats.shape[0]

    k = min(n_neighbors, n_total // 4)
    print(f'  [UMAP] {n_total} samples × {all_feats.shape[1]} dims, '
          f'n_neighbors={k}, min_dist={min_dist} ...', flush=True)

    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=k,
        min_dist=min_dist,
        metric='cosine',        # cosine distance is more appropriate for semantic features
        random_state=42,
        verbose=False,
    )
    emb = reducer.fit_transform(all_feats)

    result, idx = {}, 0
    for key, size in zip(keys, sizes):
        result[key] = emb[idx: idx + size]
        idx += size
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Dashed confidence ellipses (for paper figures)
# ──────────────────────────────────────────────────────────────────────────────
# Ellipse linestyles per dataset — distinguishable in grayscale print
_ELLIPSE_LINESTYLE: dict[str, tuple] = {
    'id':             (0, ()),            # solid
    'cifar100':       (0, (6, 2)),        # dashed
    'stl10':          (0, (2, 2)),        # densely dotted
    'svhn':           (0, (8, 2, 2, 2)), # dash-dot
    'mnist':          (0, (4, 1, 1, 1)), # dash-dot-dot
    'textures':       (0, (1, 1)),        # dotted
    'gaussian_noise': (0, (10, 3)),       # long dash
}


def _confidence_ellipse(ax: plt.Axes, pts: np.ndarray, color: str,
                        n_std: float = 2.3,
                        dataset_key: str = '') -> None:
    """
    Draw a covariance-based confidence ellipse.
    Each dataset gets a distinct linestyle so the figure is readable in grayscale.
    """
    if len(pts) < 5:
        return
    x, y = pts[:, 0], pts[:, 1]
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)) or np.linalg.matrix_rank(cov) < 2:
        return
    eigvals, eigvecs = np.linalg.eigh(cov)
    order   = eigvals.argsort()[::-1]
    eigvals = np.abs(eigvals[order])
    eigvecs = eigvecs[:, order]
    angle   = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width   = 2.0 * n_std * np.sqrt(eigvals[0])
    height  = 2.0 * n_std * np.sqrt(eigvals[1])

    ls     = _ELLIPSE_LINESTYLE.get(dataset_key, (0, (6, 2)))
    face_c = mcolors.to_rgba(color, alpha=0.07)
    edge_c = mcolors.to_rgba(color, alpha=0.85)

    ell = Ellipse(
        xy=(x.mean(), y.mean()),
        width=width, height=height, angle=angle,
        linewidth=1.8, linestyle=ls,
        facecolor=face_c, edgecolor=edge_c,
        zorder=1,
    )
    ax.add_patch(ell)


# ──────────────────────────────────────────────────────────────────────────────
# Main plot: UMAP scatter (paper quality)
# ──────────────────────────────────────────────────────────────────────────────
def plot_scatter(emb_dict: dict, ood_names: list, out_dir: str,
                 dpi: int = 300) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # ── rcParams for paper aesthetics ────────────────────────────────────────
    plt.rcParams.update({
        'font.family':       'DejaVu Sans',
        'font.size':         12,
        'axes.labelsize':    12,
        'legend.fontsize':   10,
        'legend.title_fontsize': 10,
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })

    fig, ax = plt.subplots(figsize=(9, 8))

    # ── ID (CIFAR-10): small dark background points ──────────────────────────
    ID_COLOR = '#333333'
    if 'id' in emb_dict:
        pts = emb_dict['id']
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=ID_COLOR, s=9, alpha=0.18, linewidths=0,
                   label='CIFAR-10 (In-Distribution)', zorder=2,
                   rasterized=True)
        _confidence_ellipse(ax, pts, ID_COLOR, n_std=2.3, dataset_key='id')

    # ── OOD datasets: sorted near → mid → far ───────────────────────────────
    sorted_ood = sorted(
        [n for n in ood_names if n in emb_dict],
        key=lambda k: (_PROX_RANK.get(OOD_META.get(k, {}).get('proximity', 'far'), 2),
                       OOD_META.get(k, {}).get('label', k))
    )

    for ood_name in sorted_ood:
        meta = OOD_META.get(ood_name,
               {'label': ood_name, 'color': '#cc0000', 'marker': 'o', 'proximity': 'unk'})
        prox = meta['proximity']
        pts  = emb_dict[ood_name]

        # Capitalise proximity label for legend readability
        prox_label = f"{prox.capitalize()}-OOD"
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=meta['color'], s=30, alpha=0.78, marker=meta['marker'],
                   label=f"{meta['label']}  ({prox_label})",
                   linewidths=0.5, edgecolors='white', zorder=3,
                   rasterized=True)
        _confidence_ellipse(ax, pts, meta['color'], n_std=2.3, dataset_key=ood_name)

    # ── axes ─────────────────────────────────────────────────────────────────
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # ── legend ───────────────────────────────────────────────────────────────
    ax.legend(
        loc='upper right',
        fontsize=10,
        markerscale=1.8,
        framealpha=0.93,
        edgecolor='#cccccc',
        title='Dataset',
        title_fontsize=10,
        handletextpad=0.5,
        borderpad=0.8,
    )

    plt.tight_layout(pad=1.2)

    # Save both PDF (vector, for paper) and PNG (raster preview)
    for ext, fmt in [('pdf', 'pdf'), ('png', 'png')]:
        path = os.path.join(out_dir, f'data_distribution_umap.{ext}')
        plt.savefig(path, dpi=dpi, bbox_inches='tight', format=fmt)
        print(f'  Saved → {path}')
    plt.close()

    _plot_proximity_bar(emb_dict, ood_names, out_dir, dpi=dpi)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: distance bar chart
# ──────────────────────────────────────────────────────────────────────────────
def _plot_proximity_bar(emb_dict: dict, ood_names: list, out_dir: str,
                        dpi: int = 300) -> None:
    """Centroid-distance bar chart (supplementary, not main paper figure)."""
    if 'id' not in emb_dict:
        return

    id_centroid = emb_dict['id'].mean(axis=0)
    rows = []
    for ood_name in ood_names:
        if ood_name not in emb_dict:
            continue
        dist = float(np.linalg.norm(emb_dict[ood_name].mean(axis=0) - id_centroid))
        meta = OOD_META.get(ood_name,
               {'label': ood_name, 'color': 'gray', 'proximity': 'unk'})
        rows.append({'label': meta['label'], 'dist': dist,
                     'color': meta['color'],  'proximity': meta['proximity']})

    if not rows:
        return

    rows.sort(key=lambda r: r['dist'])
    labels = [r['label']     for r in rows]
    vals   = [r['dist']      for r in rows]
    colors = [r['color']     for r in rows]

    fig, ax = plt.subplots(figsize=(7, 0.65 * len(rows) + 2))
    bars = ax.barh(labels, vals, color=colors, alpha=0.85, edgecolor='white')

    for bar, val, row in zip(bars, vals, rows):
        ax.text(val + max(vals) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}  [{row["proximity"]}-OOD]',
                va='center', fontsize=9)

    ax.set_xlabel('Centroid Distance from CIFAR-10 (UMAP space)', fontsize=10)
    ax.set_title(
        'OOD Semantic Proximity to ID Distribution\n'
        '(smaller = closer to ID = harder to filter)',
        fontsize=11,
    )
    ax.set_xlim(0, max(vals) * 1.40)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, 'ood_proximity_bar_umap.png')
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {path}')


# ──────────────────────────────────────────────────────────────────────────────
# Semantic Embedding Distance (CLIP text encoder)
# ──────────────────────────────────────────────────────────────────────────────

# Class names for each dataset — used to compute CLIP text embeddings
DATASET_CLASS_NAMES: dict[str, list[str]] = {
    # ID
    'id': [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck',
    ],
    # Near-OOD — shares many visual / semantic categories with CIFAR-10
    'cifar100': [
        'apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
        'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
        'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
        'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
        'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
        'kangaroo', 'keyboard', 'lamp', 'lawn mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple tree', 'motorcycle', 'mountain',
        'mouse', 'mushroom', 'oak tree', 'orange', 'orchid', 'otter',
        'palm tree', 'pear', 'pickup truck', 'pine tree', 'plain', 'plate',
        'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray',
        'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew',
        'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
        'streetcar', 'sunflower', 'sweet pepper', 'table', 'tank',
        'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow tree', 'wolf',
        'woman', 'worm',
    ],
    'stl10': [
        'airplane', 'bird', 'car', 'cat', 'deer',
        'dog', 'horse', 'monkey', 'ship', 'truck',
    ],
    # Far-OOD — semantically distant from natural-object domain
    'svhn': [
        'street number zero', 'street number one', 'street number two',
        'street number three', 'street number four', 'street number five',
        'street number six', 'street number seven', 'street number eight',
        'street number nine',
    ],
    'mnist': [
        'handwritten digit zero', 'handwritten digit one',
        'handwritten digit two', 'handwritten digit three',
        'handwritten digit four', 'handwritten digit five',
        'handwritten digit six', 'handwritten digit seven',
        'handwritten digit eight', 'handwritten digit nine',
    ],
    'textures': [
        'banded texture', 'blotchy texture', 'braided texture',
        'bubbly texture', 'bumpy texture', 'chequered texture',
        'cobwebbed texture', 'cracked texture', 'crosshatched texture',
        'crystalline texture', 'dotted texture', 'fibrous texture',
        'flecked texture', 'freckled texture', 'gauzy texture',
        'grid texture', 'grooved texture', 'honeycombed texture',
        'interlaced texture', 'knitted texture', 'lacelike texture',
        'lined texture', 'marbled texture', 'matted texture',
        'meshed texture', 'paisley texture', 'perforated texture',
        'pitted texture', 'pleated texture', 'polka-dotted texture',
        'porous texture', 'potholed texture', 'scaly texture',
        'smeared texture', 'spiralled texture', 'sprinkled texture',
        'stained texture', 'stratified texture', 'striped texture',
        'studded texture', 'swirly texture', 'veined texture',
        'waffled texture', 'woven texture', 'wrinkled texture',
        'zigzagged texture',
    ],
    'gaussian_noise': [
        'gaussian noise', 'random static noise', 'white noise',
        'random pixel noise', 'image sensor noise',
    ],
}

# Prompt template — standard CLIP zero-shot format
_CLIP_PROMPT = 'a photo of {}'


def build_clip_text_embeddings(datasets: list[str],
                                device: torch.device) -> dict[str, np.ndarray]:
    """
    Encode each dataset's class names with the CLIP text encoder.
    Returns L2-normalised 512-dim embeddings (one per class name).
    """
    from transformers import CLIPTokenizer, CLIPTextModelWithProjection
    import torch.nn.functional as F

    print('  Loading CLIP (openai/clip-vit-base-patch32) ...')
    tokenizer  = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    # CLIPTextModelWithProjection outputs .text_embeds — already in shared 512-dim space
    text_model = CLIPTextModelWithProjection.from_pretrained('openai/clip-vit-base-patch32')
    text_model = text_model.to(device).eval()
    print(f'  CLIP text encoder loaded.')

    result = {}
    with torch.no_grad():
        for key in datasets:
            if key not in DATASET_CLASS_NAMES:
                print(f'  [Warning] No class names defined for {key}, skipping.')
                continue
            names   = DATASET_CLASS_NAMES[key]
            prompts = [_CLIP_PROMPT.format(n) for n in names]
            inputs  = tokenizer(prompts, return_tensors='pt', padding=True,
                                truncation=True, max_length=77).to(device)
            outputs = text_model(**inputs)
            feats   = outputs.text_embeds                      # (N, 512) tensor
            feats   = F.normalize(feats, dim=-1).cpu().numpy()
            result[key] = feats
            print(f'    {key}: {len(names)} classes → {feats.shape}')

    return result


def compute_avg_cosine_similarity(id_embs: np.ndarray,
                                   ood_embs: np.ndarray) -> float:
    """Mean cosine similarity between every ID class and every OOD class."""
    # Both are already L2-normalised → dot product = cosine similarity
    sim_matrix = id_embs @ ood_embs.T          # (n_id_classes, n_ood_classes)
    return float(sim_matrix.mean())


def plot_semantic_mds(emb_dict: dict[str, np.ndarray],
                      ood_names: list[str],
                      out_dir: str,
                      dpi: int = 300) -> None:
    """
    MDS 2D projection of CLIP class-name embeddings.
    Each point = one class name; position faithfully reflects semantic distance.
    """
    from sklearn.manifold import MDS

    os.makedirs(out_dir, exist_ok=True)

    all_keys  = ['id'] + [n for n in ood_names if n in emb_dict]
    all_embs  = np.concatenate([emb_dict[k] for k in all_keys], axis=0)
    sizes     = [len(emb_dict[k]) for k in all_keys]

    # Cosine distance matrix (embeddings already L2-normalised)
    sim   = all_embs @ all_embs.T
    dist  = np.clip(1.0 - sim, 0.0, None)

    print(f'  [MDS] {all_embs.shape[0]} class embeddings → 2D ...', flush=True)
    mds    = MDS(n_components=2, metric=True, dissimilarity='precomputed',
                 random_state=42, n_init=8, max_iter=1000, normalized_stress='auto')
    coords = mds.fit_transform(dist)

    # Split back into per-dataset arrays
    coords_dict, idx = {}, 0
    for key, sz in zip(all_keys, sizes):
        coords_dict[key] = coords[idx: idx + sz]
        idx += sz

    # ── paper-style plot ──────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 12,
        'axes.labelsize': 12, 'legend.fontsize': 10,
        'legend.title_fontsize': 10,
    })
    fig, ax = plt.subplots(figsize=(9, 8))

    ID_COLOR = '#333333'
    # ID
    pts = coords_dict['id']
    ax.scatter(pts[:, 0], pts[:, 1], c=ID_COLOR, s=70, alpha=0.85,
               marker='*', label='CIFAR-10 (In-Distribution)', zorder=4,
               linewidths=0.4, edgecolors='white')
    _confidence_ellipse(ax, pts, ID_COLOR, n_std=2.0, dataset_key='id')

    # OOD
    sorted_ood = sorted(
        [n for n in ood_names if n in coords_dict],
        key=lambda k: (_PROX_RANK.get(OOD_META.get(k, {}).get('proximity', 'far'), 2),
                       OOD_META.get(k, {}).get('label', k))
    )
    for ood_name in sorted_ood:
        meta = OOD_META.get(ood_name, {'label': ood_name, 'color': 'red',
                                        'marker': 'o', 'proximity': 'unk'})
        pts  = coords_dict[ood_name]
        prox_label = f"{meta['proximity'].capitalize()}-OOD"
        ax.scatter(pts[:, 0], pts[:, 1], c=meta['color'], s=55,
                   alpha=0.82, marker=meta['marker'],
                   label=f"{meta['label']}  ({prox_label})",
                   linewidths=0.5, edgecolors='white', zorder=3)
        _confidence_ellipse(ax, pts, meta['color'], n_std=2.0, dataset_key=ood_name)

    ax.set_xlabel('MDS Dimension 1', fontsize=12)
    ax.set_ylabel('MDS Dimension 2', fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(loc='upper right', fontsize=10, markerscale=1.6,
              framealpha=0.93, edgecolor='#cccccc',
              title='Dataset', title_fontsize=10,
              handletextpad=0.5, borderpad=0.8)

    plt.tight_layout(pad=1.2)
    for ext in ('pdf', 'png'):
        path = os.path.join(out_dir, f'semantic_mds.{ext}')
        plt.savefig(path, dpi=dpi, bbox_inches='tight', format=ext)
        print(f'  Saved → {path}')
    plt.close()

    # ── similarity bar chart ─────────────────────────────────────────────────
    _plot_semantic_similarity_bar(emb_dict, ood_names, out_dir, dpi)


def _plot_semantic_similarity_bar(emb_dict: dict[str, np.ndarray],
                                   ood_names: list[str],
                                   out_dir: str,
                                   dpi: int = 300) -> None:
    """Bar chart: average cosine similarity of each OOD dataset to CIFAR-10."""
    if 'id' not in emb_dict:
        return
    id_embs = emb_dict['id']
    rows = []
    for ood_name in ood_names:
        if ood_name not in emb_dict:
            continue
        sim  = compute_avg_cosine_similarity(id_embs, emb_dict[ood_name])
        meta = OOD_META.get(ood_name, {'label': ood_name,
                                        'color': 'gray', 'proximity': 'unk'})
        rows.append({'label': meta['label'], 'sim': sim,
                     'color': meta['color'], 'proximity': meta['proximity']})

    if not rows:
        return

    rows.sort(key=lambda r: r['sim'], reverse=True)   # high sim = near-OOD first
    labels = [r['label']     for r in rows]
    vals   = [r['sim']       for r in rows]
    colors = [r['color']     for r in rows]

    fig, ax = plt.subplots(figsize=(7, 0.65 * len(rows) + 2))
    bars = ax.barh(labels, vals, color=colors, alpha=0.85, edgecolor='white')

    x_max = max(vals)
    for bar, val, row in zip(bars, vals, rows):
        ax.text(val + x_max * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}  [{row["proximity"]}-OOD]',
                va='center', fontsize=9)

    ax.set_xlabel('Avg. Cosine Similarity to CIFAR-10 Classes (CLIP)', fontsize=10)
    ax.set_xlim(0, x_max * 1.35)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, 'semantic_similarity_bar.png')
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {path}')


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SafeSSL: OOD Semantic Proximity Visualization (UMAP)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--mode',         type=str, default='text',
                        choices=['text', 'image'],
                        help='text: CLIP semantic MDS (default); image: image feature UMAP')
    parser.add_argument('--data_dir',     type=str, default='./data',
                        help='Data root directory (default: ./data)')
    parser.add_argument('--out_dir',      type=str, default='tools/plots/data',
                        help='Output directory (default: tools/plots/data)')
    parser.add_argument('--ood_datasets', type=str, nargs='+',
                        default=ALL_OOD_DATASETS,
                        help=f'OOD datasets to visualize (default: all): {ALL_OOD_DATASETS}')
    parser.add_argument('--n_id',         type=int, default=3000,
                        help='Number of ID samples (default: 3000)')
    parser.add_argument('--n_ood',        type=int, default=600,
                        help='Samples per OOD dataset (default: 600)')
    parser.add_argument('--n_neighbors',  type=int, default=30,
                        help='UMAP n_neighbors (default: 30)')
    parser.add_argument('--min_dist',     type=float, default=0.1,
                        help='UMAP min_dist (default: 0.1)')
    parser.add_argument('--model_path',   type=str, default=None,
                        help='WRN-28-2 supervised checkpoint (.pth)。'
                             'If provided, switches to WRN feature extractor (CIFAR-10 stats);'
                             'If omitted, uses ImageNet ResNet-18.')
    parser.add_argument('--depth',        type=int, default=28,
                        help='WideResNet depth (default: 28)')
    parser.add_argument('--widen_factor', type=int, default=2,
                        help='WideResNet widen factor (default: 2)')
    parser.add_argument('--dpi',          type=int, default=300,
                        help='Output DPI (default: 300; recommended 300-600 for publication)')
    parser.add_argument('--device',       type=str, default='cuda')
    args = parser.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'\n[data_visual] Mode: {args.mode} | Device: {dev}')

    # ── Text mode: CLIP semantic MDS ─────────────────────────────────────────
    if args.mode == 'text':
        all_keys    = ['id'] + args.ood_datasets
        print(f'\n[data_visual] Encoding class names with CLIP text encoder ...')
        emb_dict    = build_clip_text_embeddings(all_keys, dev)
        available   = [n for n in args.ood_datasets if n in emb_dict]
        print(f'\n[data_visual] Running MDS + plotting ...')
        plot_semantic_mds(emb_dict, available, args.out_dir, dpi=args.dpi)
        print(f'\n[data_visual] Done. Plots saved to: {args.out_dir}')

    # ── Image mode: image feature UMAP (original) ────────────────────────────
    else:
        use_wrn = args.model_path is not None
        if use_wrn:
            print(f'[data_visual] Backbone: WRN-{args.depth}-{args.widen_factor} '
                  f'from {args.model_path}')
        else:
            print('[data_visual] Backbone: ResNet-18 (ImageNet pretrained)')

        extractor = build_feature_extractor(
            dev, model_path=args.model_path,
            depth=args.depth, widen_factor=args.widen_factor,
        )
        features_dict = {}

        print(f'\n  Loading ID (CIFAR-10, n={args.n_id}) ...')
        id_imgs = load_id_tensors(args.data_dir, args.n_id, use_wrn=use_wrn)
        features_dict['id'] = extract_features(extractor, id_imgs, dev)
        print(f'  ID features: {features_dict["id"].shape}')

        for ood_name in args.ood_datasets:
            print(f'\n  Loading OOD: {ood_name} (n={args.n_ood}) ...')
            try:
                ood_imgs = load_ood_tensors(args.data_dir, ood_name, args.n_ood,
                                            use_wrn=use_wrn)
                features_dict[ood_name] = extract_features(extractor, ood_imgs, dev)
                print(f'  {ood_name} features: {features_dict[ood_name].shape}')
            except Exception as e:
                print(f'  [Warning] Skipping {ood_name}: {e}')

        print(f'\n[data_visual] Running UMAP (cosine metric) ...')
        emb_dict = reduce_umap(features_dict,
                               n_neighbors=args.n_neighbors,
                               min_dist=args.min_dist)
        available = [n for n in args.ood_datasets if n in emb_dict]
        print(f'\n[data_visual] Plotting ...')
        plot_scatter(emb_dict, available, args.out_dir, dpi=args.dpi)
        print(f'\n[data_visual] Done. Plots saved to: {args.out_dir}')
