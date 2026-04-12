import torch
import numpy as np
from torchvision.transforms import v2

class GPUAugmentor:
    """Performs image augmentation on the GPU, offloading CPU pressure."""
    def __init__(self, cfg, device):
        self.device = device
        img_size = cfg.get('img_size', 32)
        # CIFAR-10 standard mean and std
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]

        # 1. Weak augmentation
        self.weak_aug = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomCrop(img_size, padding=4, padding_mode='reflect'),
            v2.Normalize(mean=mean, std=std)
        ])

        # 2. Strong augmentation (RandAugment + Cutout)
        # USB FixMatch standard:
        #   - n_ops=2, magnitude=10 (matching USB RandAugmentMC(n=2, m=10))
        #   - Cutout applied before Normalize, fill = CIFAR mean (in [0,1] space)
        #     -> after Normalize the patch becomes 0, consistent with standard Cutout semantics
        #   - p=1.0: always applied
        #   - scale=(0.25, 0.25): fixed 16x16 (= 25% of 32x32), aligned with USB CutoutAbs(size=16)
        self.strong_aug = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomCrop(img_size, padding=4, padding_mode='reflect'),
            v2.RandAugment(num_ops=2, magnitude=10),
            v2.RandomErasing(p=1.0, scale=(0.25, 0.25), ratio=(1, 1), value=mean),  # Fixed 16x16 Cutout BEFORE normalize
            v2.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img_tensor, mode='weak'):
        img_tensor = img_tensor.to(self.device, non_blocking=True)
        if mode == 'weak':
            return self.weak_aug(img_tensor)
        elif mode == 'strong':
            return self.strong_aug(img_tensor)
        return img_tensor

def get_augmentation(cfg):
    """
    Minimal CPU-side transform: convert to tensor only. All augmentation is performed on GPU.
    """
    cpu_train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    # Test set: normalize on CPU (no augmentation needed)
    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    return cpu_train_transform, test_transform
