import numpy as np
import torch
from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset, Subset, Sampler

class CIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform
        self.data = self.dataset.data
        self.targets = np.array(self.dataset.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

class ClassBalancedSampler(Sampler):
    """Ensures each batch includes all classes where possible (designed for very few labels)."""
    def __init__(self, targets, num_samples):
        self.targets = np.array(targets)
        self.classes = np.unique(self.targets)
        self.class_indices = [np.where(self.targets == c)[0] for c in self.classes]
        self.num_samples = num_samples

    def __iter__(self):
        indices = []
        for _ in range(self.num_samples // len(self.classes) + 1):
            for c_idx in self.class_indices:
                indices.append(np.random.choice(c_idx))
        return iter(indices[:self.num_samples])

    def __len__(self):
        return self.num_samples

def split_labeled_unlabeled(dataset, num_labeled):
    indices = np.arange(len(dataset))
    labels = dataset.targets
    classes = np.unique(labels)
    num_per_class = num_labeled // len(classes)

    labeled_idx = []
    unlabeled_idx = []

    for c in classes:
        idx = indices[labels == c]
        np.random.shuffle(idx)
        labeled_idx.extend(idx[:num_per_class])
        unlabeled_idx.extend(idx)

    return Subset(dataset, labeled_idx), dataset.data[unlabeled_idx], dataset.targets[unlabeled_idx]
