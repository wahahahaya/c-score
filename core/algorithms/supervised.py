import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import AlgorithmBase

class Supervised(AlgorithmBase):
    def __init__(self, model, device, cfg):
        super().__init__(model, device, cfg)
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, data_batch, global_step):
        # Extract labeled data from batch dict
        inputs, targets = data_batch['labeled']
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        # Compute batch statistics
        _, predicted = outputs.max(1)
        acc = 100. * predicted.eq(targets).sum().item() / targets.size(0)

        stats = {
            "loss": loss.item(),
            "acc": acc
        }

        return loss, stats
