import torch
from torch.utils.data import Dataset

class UnifiedDatasetWrapper(Dataset):
    """
    Wraps a base Dataset and dispatches to the correct return format depending
    on the algorithm mode. This lets a single pipeline serve different algorithms
    that require different data formats.
    """
    def __init__(self, base_dataset, mode='supervised', transform=None, secondary_transform=None):
        self.base_dataset = base_dataset
        self.mode = mode
        self.transform = transform
        self.secondary_transform = secondary_transform

    def __getitem__(self, index):
        # 1. Get the raw image and label from the base dataset
        # base_dataset is responsible only for basic Image.open / array-to-PIL conversion
        img, label = self.base_dataset[index]

        # 2. Dispatch by algorithm mode
        if self.mode == 'supervised':
            # Supervised: one image + one label
            return self.transform(img), label

        elif self.mode == 'fixmatch':
            # Semi-supervised: one image produces weak and strong views
            img_weak = self.transform(img)
            img_strong = self.secondary_transform(img)
            return (img_weak, img_strong), label

        elif self.mode == 'contrastive':
            # Self-supervised (e.g. SimCLR): two independent strong-augmented views (query/key)
            img_q = self.transform(img)
            img_k = self.secondary_transform(img)
            return (img_q, img_k), label

        else:
            # Default: return with base transform
            return self.transform(img), label

    def __len__(self):
        return len(self.base_dataset)
