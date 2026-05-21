import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from .cifar10 import CIFAR10Dataset, split_labeled_unlabeled, ClassBalancedSampler
from .cifar100 import CIFAR100Dataset, split_labeled_unlabeled_c100
from .wrapper import UnifiedDatasetWrapper
from .augmentation import get_augmentation
from .OOD import OODDataset


def _try_build_dali(labeled_subset, mixed_unlb_raw, labeled_batch_size, total_steps, mu):
    """
    Build DALI loaders for labeled + unlabeled data.
    Returns (labeled_loader, unlabeled_loader) or raises ImportError if DALI unavailable.
    """
    from .dali_loader import build_dali_loader

    # Extract numpy arrays from labeled Subset
    labeled_data   = labeled_subset.dataset.data[labeled_subset.indices]    # (N_l, H, W, C) uint8
    labeled_labels = labeled_subset.dataset.targets[labeled_subset.indices]  # (N_l,)

    # Unlabeled: RawMixedDataset already has .data and .targets as numpy arrays
    unlabeled_data   = mixed_unlb_raw.data      # (N_u, H, W, C) uint8
    unlabeled_labels = mixed_unlb_raw.targets   # (N_u,)

    unlabeled_batch_size  = labeled_batch_size * mu
    unlabeled_total_batches = len(unlabeled_data) // unlabeled_batch_size  # drop_last equivalent

    labeled_loader = build_dali_loader(
        labeled_data, labeled_labels,
        batch_size=labeled_batch_size,
        total_batches=total_steps,
        shuffle=True,
    )
    unlabeled_loader = build_dali_loader(
        unlabeled_data, unlabeled_labels,
        batch_size=unlabeled_batch_size,
        total_batches=unlabeled_total_batches,
        shuffle=True,
    )
    return labeled_loader, unlabeled_loader

class RawMixedDataset(Dataset):
    """Safe SSL: Mix ID and OOD samples with controlled ratios

    r_id : fraction of available ID unlabeled data to keep (0.0~1.0)
    r_ood: contamination ratio = OOD / (ID + OOD), so n_ood = n_id * r / (1 - r)
    """
    def __init__(self, id_data, id_targets, ood_data, ood_targets, r_id, r_ood):
        # Sample ID portion according to r_id
        n_id = int(len(id_data) * r_id)
        if n_id > 0:
            id_idx = np.arange(n_id)
            self.id_data = id_data[id_idx]
            self.id_targets = id_targets[id_idx]
        else:
            self.id_data = np.empty((0, 32, 32, 3), dtype=np.uint8)
            self.id_targets = np.empty((0,), dtype=np.int64)

        # Compute OOD count from contamination ratio: r = OOD / (ID + OOD)
        # => n_ood = n_id * r / (1 - r), capped at available OOD pool size
        n_ood = int(n_id * r_ood / (1.0 - r_ood)) if r_ood > 0.0 else 0
        n_ood = min(n_ood, len(ood_data))
        if n_ood > 0:
            ood_idx = np.arange(n_ood)
            self.ood_data = ood_data[ood_idx]
            self.ood_targets = ood_targets[ood_idx]
        else:
            self.ood_data = np.empty((0, 32, 32, 3), dtype=np.uint8)
            self.ood_targets = np.empty((0,), dtype=np.int64)

        self.data = np.concatenate([self.id_data, self.ood_data], axis=0).astype(np.uint8)
        self.targets = np.concatenate([self.id_targets, self.ood_targets], axis=0).astype(np.int64)

        print(f"[Data Info] Unlabeled pool built | ID: {n_id} | OOD: {n_ood} | Total: {n_id + n_ood}")

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        return img, target

    def __len__(self):
        return len(self.targets)

def build_dataloader(cfg):
    """
    Optimized DataLoader builder:
    - persistent_workers to reduce worker respawn overhead
    - prefetch_factor for better pipeline throughput
    - num_workers based on CPU availability
    """
    alg = cfg['algorithm']
    data_dir = cfg['data_dir']

    train_trans, test_trans = get_augmentation(cfg)

    loaders = {}
    dataset_name = cfg.get('dataset', 'cifar10')

    if dataset_name == 'cifar100':
        raw_train_set = CIFAR100Dataset(data_dir, train=True)
        _split_fn = split_labeled_unlabeled_c100
        test_set  = CIFAR100Dataset(data_dir, train=False)
    else:
        raw_train_set = CIFAR10Dataset(data_dir, train=True)
        _split_fn = split_labeled_unlabeled
        test_set  = CIFAR10Dataset(data_dir, train=False)

    workers = cfg.get('num_workers', 4)
    pin_memory = cfg.get('pin_memory', True)
    persistent_workers = cfg.get('persistent_workers', True) if workers > 0 else False
    prefetch_factor = cfg.get('prefetch_factor', 2) if workers > 0 else None

    if alg == 'supervised':
        # Use split_labeled_unlabeled to restrict to the specified number of labeled samples
        labeled_subset, _, _ = _split_fn(raw_train_set, cfg['num_labeled'])
        wrapped_set = UnifiedDatasetWrapper(labeled_subset, mode='supervised', transform=train_trans)

        total_steps = cfg.get('num_it_per_epoch')

        # If num_it_per_epoch is specified (e.g. from grid.py), use ClassBalancedSampler
        # to ensure sufficient iterations in the low-label regime
        if total_steps is not None:
            sampler = ClassBalancedSampler(
                labeled_subset.dataset.targets[labeled_subset.indices],
                num_samples=total_steps * cfg['batch_size']
            )
            loaders['labeled'] = DataLoader(
                wrapped_set,
                batch_size=cfg['batch_size'],
                sampler=sampler,
                pin_memory=pin_memory,
                num_workers=workers,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor
            )
        else:
            # No step count specified: standard one-pass-per-epoch training
            loaders['labeled'] = DataLoader(
                wrapped_set,
                batch_size=cfg['batch_size'],
                shuffle=True,
                pin_memory=pin_memory,
                num_workers=workers,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor
            )

        cfg['data_stats'] = {
            'labeled_total': len(labeled_subset),
            'unlabeled_id': 0,
            'unlabeled_ood': 0,
            'unlabeled_total': 0
        }

    elif alg in ('fixmatch', 'flexmatch', 'softmatch', 'ds3l'):
        labeled_subset, id_unlb_data, id_unlb_targets = _split_fn(raw_train_set, cfg['num_labeled'])

        labeled_batch_size = cfg.get('batch_size', 64)
        total_steps = cfg.get('num_it_per_epoch', 1024)
        sampler = ClassBalancedSampler(
            labeled_subset.dataset.targets[labeled_subset.indices],
            num_samples=total_steps * labeled_batch_size
        )

        loaders['labeled'] = DataLoader(
            UnifiedDatasetWrapper(labeled_subset, mode='supervised', transform=train_trans),
            batch_size=labeled_batch_size,
            sampler=sampler,
            pin_memory=pin_memory,
            num_workers=workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )

        ood_name = cfg.get('ood_dataset', 'svhn')
        if ood_name == 'none' or cfg.get('r_ood', 0.0) == 0.0:
            ood_pool = type('_Empty', (), {'data': np.empty((0, 32, 32, 3), dtype=np.uint8),
                                           'targets': np.empty((0,), dtype=np.int64)})()
        else:
            ood_pool = OODDataset(data_dir, ood_name)

        # Pre-compute pool sizes and write to cfg so they appear in config.json
        r_id  = cfg.get('r_id', 1.0)
        r_ood = cfg.get('r_ood', 0.0)
        n_id  = int(len(id_unlb_data) * r_id)
        n_ood = int(n_id * r_ood / (1.0 - r_ood)) if r_ood > 0.0 else 0
        n_ood = min(n_ood, len(ood_pool.data))

        cfg['data_stats'] = {
            'labeled_total': len(labeled_subset),
            'unlabeled_id': n_id,
            'unlabeled_ood': n_ood,
            'unlabeled_total': n_id + n_ood
        }

        mixed_unlb_raw = RawMixedDataset(
            id_unlb_data, id_unlb_targets, ood_pool.data, ood_pool.targets,
            cfg.get('r_id', 1.0), cfg.get('r_ood', 0.0)
        )

        if cfg.get('use_dali', False):
            try:
                dali_labeled, dali_unlabeled = _try_build_dali(
                    labeled_subset, mixed_unlb_raw,
                    labeled_batch_size, total_steps, cfg.get('mu', 7)
                )
                # Override the DataLoader-based labeled loader built above
                loaders['labeled']   = dali_labeled
                loaders['unlabeled'] = dali_unlabeled
                print("[DataLoader] DALI pipeline active (GPU async prefetch) — both labeled & unlabeled")
            except Exception as e:
                print(f"[DataLoader] WARNING: DALI init failed ({e}), falling back to DataLoader")
                cfg['use_dali'] = False

        if not cfg.get('use_dali', False):
            loaders['unlabeled'] = DataLoader(
                UnifiedDatasetWrapper(mixed_unlb_raw, mode='supervised', transform=train_trans),
                batch_size=labeled_batch_size * cfg.get('mu', 7),
                shuffle=True,
                drop_last=True,
                pin_memory=pin_memory,
                num_workers=workers,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor
            )

    loaders['test'] = DataLoader(
        UnifiedDatasetWrapper(test_set, mode='supervised', transform=test_trans),
        batch_size=cfg.get('eval_batch_size', 256),
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )

    return loaders
