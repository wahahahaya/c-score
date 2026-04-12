import numpy as np
from torchvision import datasets
from PIL import Image

# Pillow >= 9.1 moved resample filters to Image.Resampling; older versions keep them on Image
_BILINEAR = getattr(getattr(Image, 'Resampling', None), 'BILINEAR', Image.BILINEAR)


def _to_32x32_rgb(pil_img):
    """Convert any PIL Image to a 32x32 RGB uint8 numpy array."""
    return np.array(pil_img.convert('RGB').resize((32, 32), _BILINEAR), dtype=np.uint8)


class OODDataset:
    """
    OOD data pool. All datasets output (N, 32, 32, 3) uint8 numpy arrays with
    targets set to -1 (OOD marker).

    Controlled via the config's ood_dataset field:
        'svhn'           - Street View House Numbers (default)
        'cifar100'       - CIFAR-100 (same domain, near-OOD)
        'mnist'          - MNIST (grayscale -> RGB, resized to 32x32)
        'stl10'          - STL-10 unlabeled split (96x96 -> 32x32)
        'textures'       - DTD Describable Textures (all splits, resized to 32x32)
        'gaussian_noise' - Gaussian random noise (N=50000, mu=128, sigma=64)
    """

    def __init__(self, root, dataset_name='svhn'):
        self.dataset_name = dataset_name

        if dataset_name == 'svhn':
            dset = datasets.SVHN(root, split='train', download=True)
            self.data = np.transpose(dset.data, (0, 2, 3, 1)).astype(np.uint8)
            self.targets = np.full(len(dset.labels), -1, dtype=np.int64)

        elif dataset_name == 'cifar100':
            dset = datasets.CIFAR100(root, train=True, download=True)
            self.data = np.array(dset.data, dtype=np.uint8)
            self.targets = np.full(len(dset.targets), -1, dtype=np.int64)

        elif dataset_name == 'mnist':
            dset = datasets.MNIST(root, train=True, download=True)
            raw = dset.data.numpy()  # (N, 28, 28) uint8
            print(f"[OOD] MNIST: converting {len(raw)} images to 32x32 RGB ...")
            self.data = np.stack([
                np.array(Image.fromarray(img).resize((32, 32), _BILINEAR).convert('RGB'), dtype=np.uint8)
                for img in raw
            ])
            self.targets = np.full(len(raw), -1, dtype=np.int64)

        elif dataset_name == 'stl10':
            dset = datasets.STL10(root, split='unlabeled', download=True)
            raw = np.transpose(dset.data, (0, 2, 3, 1))  # (N, 96, 96, 3) uint8
            print(f"[OOD] STL-10: resizing {len(raw)} images to 32x32 ...")
            self.data = np.stack([
                np.array(Image.fromarray(img).resize((32, 32), _BILINEAR), dtype=np.uint8)
                for img in raw
            ])
            self.targets = np.full(len(raw), -1, dtype=np.int64)

        elif dataset_name == 'textures':
            print("[OOD] Textures (DTD): loading train / val / test splits ...")
            images = []
            for split in ('train', 'val', 'test'):
                dset = datasets.DTD(root, split=split, download=True)
                for i in range(len(dset)):
                    img, _ = dset[i]
                    images.append(_to_32x32_rgb(img))
            self.data = np.stack(images)
            self.targets = np.full(len(self.data), -1, dtype=np.int64)
            print(f"[OOD] Textures: {len(self.data)} samples loaded.")

        elif dataset_name == 'gaussian_noise':
            n = 50000
            rng = np.random.default_rng(seed=42)
            noise = rng.standard_normal((n, 32, 32, 3)) * 64 + 128
            self.data = np.clip(noise, 0, 255).astype(np.uint8)
            self.targets = np.full(n, -1, dtype=np.int64)
            print(f"[OOD] Gaussian noise: generated {n} samples (mu=128, sigma=64).")

        else:
            raise ValueError(
                f"Unsupported OOD dataset: '{dataset_name}'. "
                f"Valid options: svhn, cifar100, mnist, stl10, textures, gaussian_noise"
            )

    def __len__(self):
        return len(self.targets)
