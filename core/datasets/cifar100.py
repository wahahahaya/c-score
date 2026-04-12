import numpy as np
from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset, Subset

from .cifar10 import ClassBalancedSampler  # reuse existing sampler


class CIFAR100Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.dataset   = datasets.CIFAR100(root=root, train=train, download=True)
        self.transform = transform
        self.data      = self.dataset.data                          # (N,32,32,3) uint8
        self.targets   = np.array(self.dataset.targets, dtype=np.int64)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


def split_labeled_unlabeled_c100(dataset: CIFAR100Dataset, num_labeled: int):
    """
    Stratified split: num_labeled // num_classes samples per class become labeled.
    Returns (labeled_subset, unlabeled_data_array, unlabeled_targets_array).
    """
    indices = np.arange(len(dataset))
    labels  = dataset.targets
    classes = np.unique(labels)
    num_per_class = num_labeled // len(classes)

    labeled_idx   = []
    unlabeled_idx = []

    for c in classes:
        idx = indices[labels == c]
        np.random.shuffle(idx)
        labeled_idx.extend(idx[:num_per_class])
        unlabeled_idx.extend(idx)          # ALL samples go to unlabeled pool too

    return (
        Subset(dataset, labeled_idx),
        dataset.data[unlabeled_idx],
        dataset.targets[unlabeled_idx],
    )
