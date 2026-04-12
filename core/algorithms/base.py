import torch
import torch.nn as nn
import numpy as np

class AlgorithmBase(nn.Module):
    def __init__(self, model, device, cfg):
        super().__init__()
        self.model = model
        self.device = device
        self.cfg = cfg
        self.num_classes = cfg.get('num_classes', 10)

    def compute_loss(self, data_batch, global_step):
        raise NotImplementedError

    def evaluate(self, loader):
        return self.evaluate_with_model(self.model, loader)

    def evaluate_with_model(self, eval_model, loader):
        eval_model.eval()
        actual_model = getattr(eval_model, '_orig_mod', eval_model)
        if hasattr(actual_model, 'set_bn_routing'):
            actual_model.set_bn_routing(mode='labeled')
        correct, total = 0, 0
        per_class_correct = np.zeros(self.num_classes)
        per_class_total = np.zeros(self.num_classes)
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = eval_model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                for c in range(self.num_classes):
                    mask = (targets == c)
                    per_class_total[c] += mask.sum().item()
                    per_class_correct[c] += (predicted[mask] == c).sum().item()
        if hasattr(actual_model, 'set_bn_routing'):
            actual_model.set_bn_routing(mode='mixed')
        per_class_acc = per_class_correct / (per_class_total + 1e-6)
        return {"acc": 100. * correct / total, "per_class_acc": per_class_acc}

    def save_model(self, path):
        actual_model = getattr(self.model, '_orig_mod', self.model)
        torch.save(actual_model.state_dict(), path)

    def _analyze_infiltration(self, mask, pseudo_labels, targets_gt):
        mask_np = mask.cpu().numpy()
        pseudo_labels_np = pseudo_labels.cpu().numpy()
        targets_gt_np = targets_gt.numpy()
        id_idx = np.where(targets_gt_np != -1)[0]
        ood_idx = np.where(targets_gt_np == -1)[0]
        metrics = {"id_mask_ratio": 0.0, "ood_mask_ratio": 0.0, "id_pseudo_acc": 0.0}
        if len(id_idx) > 0:
            metrics["id_mask_ratio"] = float(mask_np[id_idx].mean())
            passed_id = (mask_np[id_idx] > 0)
            if np.sum(passed_id) > 0:
                correct = (pseudo_labels_np[id_idx][passed_id] == targets_gt_np[id_idx][passed_id])
                metrics["id_pseudo_acc"] = float(correct.mean())
        if len(ood_idx) > 0:
            metrics["ood_mask_ratio"] = float(mask_np[ood_idx].mean())
        return metrics
