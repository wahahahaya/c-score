import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import AlgorithmBase
from cscore import compute_ple, compute_cci, compute_sem_drift, compute_grad_align, compute_ood_ff


class FlexMatch(AlgorithmBase):
    """
    FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling.
    Zhang et al., NeurIPS 2021. https://arxiv.org/abs/2110.08263

    Core change: replaces FixMatch's global fixed threshold tau with a per-class
    adaptive threshold. For each class c:
        threshold_c = tau * beta_c / (2 - beta_c)
    where beta_c = class_count_c / max_k(class_count_k) in [0, 1], and
    class_count_c is the EMA-tracked count of samples for class c that exceed
    the initial global threshold tau.

    C-Score metrics: all metrics are fully compatible with FixMatch; additionally
    outputs tau_mean / tau_std to track the per-class threshold distribution.
    """

    def __init__(self, model, device, cfg):
        super().__init__(model, device, cfg)
        self.p_cutoff = cfg.get('p_cutoff', 0.95)
        self.lambda_u = cfg.get('lambda_u', 1.0)
        self.criterion = nn.CrossEntropyLoss()
        # EMA per-class count of samples that pass the global threshold
        self.register_buffer('class_counts', torch.zeros(self.num_classes, device=device))

    def compute_loss(self, data_batch, global_step):
        inputs_x, targets_x = data_batch['labeled']
        (inputs_u_w, inputs_u_s), targets_u_gt = data_batch['unlabeled']

        inputs_x     = inputs_x.to(self.device)
        targets_x    = targets_x.to(self.device)
        inputs_u_w   = inputs_u_w.to(self.device)
        inputs_u_s   = inputs_u_s.to(self.device)

        num_labeled = inputs_x.shape[0]
        num_u       = inputs_u_w.shape[0]

        inputs_combined = torch.cat([inputs_x, inputs_u_w, inputs_u_s], dim=0)
        logits_combined = self.model(inputs_combined)
        logits_x   = logits_combined[:num_labeled]
        logits_u_w = logits_combined[num_labeled:num_labeled + num_u]
        logits_u_s = logits_combined[num_labeled + num_u:]

        loss_x = self.criterion(logits_x, targets_x)

        with torch.no_grad():
            probs = torch.softmax(logits_u_w.detach(), dim=-1)
            max_probs, pseudo_labels = torch.max(probs, dim=-1)

            # CPL: update per-class counts (EMA)
            above_global = (max_probs >= self.p_cutoff)
            for c in range(self.num_classes):
                count_c = (above_global & (pseudo_labels == c)).float().sum()
                self.class_counts[c] = self.class_counts[c] * 0.9 + count_c * 0.1

            # Compute per-class adaptive thresholds
            max_count = self.class_counts.max().clamp(min=1.0)
            beta = self.class_counts / max_count                        # [0, 1]
            class_thresholds = self.p_cutoff * beta / (2.0 - beta + 1e-6)
            class_thresholds = class_thresholds.clamp(min=self.p_cutoff * 0.1)

            # Per-sample mask using class-specific threshold
            sample_thresholds = class_thresholds[pseudo_labels]
            mask = max_probs.ge(sample_thresholds).float()

            do_expensive = (global_step % 32 == 0)

            # C-Score metrics
            ple = compute_ple(probs)
            cci = compute_cci(probs)

            sem_drift = 0.0
            if do_expensive:
                sem_drift = compute_sem_drift(
                    logits_x, targets_x, logits_u_w, pseudo_labels, mask, self.num_classes
                )

        loss_u = (F.cross_entropy(logits_u_s, pseudo_labels, reduction='none') * mask).mean()
        total_loss = loss_x + self.lambda_u * loss_u

        class_mask_counts = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            class_mask_counts[c] = ((pseudo_labels == c) & (mask == 1)).sum().item()

        if do_expensive:
            stats = self._analyze_infiltration(mask, pseudo_labels, targets_u_gt)
        else:
            stats = {"id_mask_ratio": 0.0, "ood_mask_ratio": 0.0, "id_pseudo_acc": 0.0}

        ood_ff = compute_ood_ff(stats['id_mask_ratio'], stats['ood_mask_ratio'])

        # C-Score: Gradient Alignment
        grad_align = 0.0
        grad_x_norm = 0.0
        grad_u_norm = 0.0
        if do_expensive:
            _actual = getattr(self.model, '_orig_mod', self.model)
            fc_params = [p for p in _actual.fc.parameters() if p.requires_grad]
            if fc_params and mask.sum() > 0:
                grad_align, grad_x_norm, grad_u_norm = compute_grad_align(loss_x, loss_u, fc_params)

        stats.update({
            "loss":              total_loss.item(),
            "loss_x":            loss_x.item(),
            "loss_u":            loss_u.item(),
            "mask_ratio":        mask.mean().item(),
            "class_mask_counts": class_mask_counts,
            "PLE":               ple,
            "CCI":               cci,
            "Sem_Drift":         sem_drift,
            "OOD_FF":            ood_ff,
            "Grad_Align":        grad_align,
            "Grad_X_Norm":       grad_x_norm,
            "Grad_U_Norm":       grad_u_norm,
            "tau_mean":          class_thresholds.mean().item(),
            "tau_std":           class_thresholds.std().item(),
        })
        return total_loss, stats
