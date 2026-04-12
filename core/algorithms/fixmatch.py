import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import AlgorithmBase
from cscore import compute_ple, compute_cci, compute_sem_drift, compute_grad_align, compute_ood_ff


class FixMatch(AlgorithmBase):
    def __init__(self, model, device, cfg):
        super().__init__(model, device, cfg)
        self.p_cutoff = cfg.get('p_cutoff', 0.95)
        self.lambda_u = cfg.get('lambda_u', 1.0)
        self.use_da = cfg.get('use_da', False)
        self.criterion = nn.CrossEntropyLoss()

        # Register DA state as a buffer so it is saved/loaded correctly with model state_dict
        self.register_buffer('p_model', torch.ones(self.num_classes, device=self.device) / self.num_classes)

    def compute_loss(self, data_batch, global_step):
        inputs_x, targets_x = data_batch['labeled']
        (inputs_u_w, inputs_u_s), targets_u_gt = data_batch['unlabeled']

        inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)
        inputs_u_w, inputs_u_s = inputs_u_w.to(self.device), inputs_u_s.to(self.device)

        num_labeled = inputs_x.shape[0]

        # Safely unwrap torch.compile wrapper before calling set_bn_routing
        actual_model = getattr(self.model, '_orig_mod', self.model)
        if hasattr(actual_model, 'set_bn_routing'):
            actual_model.set_bn_routing(mode='mixed', split_idx=num_labeled)

        # Standard FixMatch: single combined forward pass through training model
        num_u = inputs_u_w.shape[0]
        inputs_combined = torch.cat([inputs_x, inputs_u_w, inputs_u_s], dim=0)
        logits_combined = self.model(inputs_combined)
        logits_x = logits_combined[:num_labeled]
        logits_u_w = logits_combined[num_labeled:num_labeled + num_u]
        logits_u_s = logits_combined[num_labeled + num_u:]

        loss_x = self.criterion(logits_x, targets_x)

        with torch.no_grad():
            probs = torch.softmax(logits_u_w.detach(), dim=-1)

            # Distribution Alignment (DA) ablation
            if self.use_da:
                self.p_model = self.p_model * 0.999 + probs.mean(dim=0) * 0.001
                probs = probs * (1.0 / self.num_classes) / (self.p_model + 1e-6)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            max_probs, pseudo_labels = torch.max(probs, dim=-1)
            mask = max_probs.ge(self.p_cutoff).float()

            # Throttle expensive metrics: compute every 32 steps, fill 0 otherwise
            do_expensive = (global_step % 32 == 0)

            # C-Score: 1. Pseudo-label Entropy (PLE)
            ple = compute_ple(probs)

            # C-Score: 2. Class Concentration Index (CCI)
            cci = compute_cci(probs)

            # C-Score: Semantic Drift (per-class centroid computation; throttled to save GPU-CPU roundtrips)
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

        # C-Score: Gradient Alignment — cosine similarity between grad(L_x) and grad(L_u) on FC layer
        grad_align = 0.0
        grad_x_norm = 0.0
        grad_u_norm = 0.0
        if do_expensive:
            _actual = getattr(self.model, '_orig_mod', self.model)
            fc_params = [p for p in _actual.fc.parameters() if p.requires_grad]
            if fc_params and mask.sum() > 0:
                grad_align, grad_x_norm, grad_u_norm = compute_grad_align(loss_x, loss_u, fc_params)

        # _analyze_infiltration has CPU numpy roundtrip; throttled
        if do_expensive:
            stats = self._analyze_infiltration(mask, pseudo_labels, targets_u_gt)
        else:
            stats = {"id_mask_ratio": 0.0, "ood_mask_ratio": 0.0, "id_pseudo_acc": 0.0}

        # C-Score: OOD Filtration Failure
        ood_ff = compute_ood_ff(stats['id_mask_ratio'], stats['ood_mask_ratio'])

        stats.update({
            "loss": total_loss.item(),
            "loss_x": loss_x.item(),
            "loss_u": loss_u.item(),
            "mask_ratio": mask.mean().item(),
            "class_mask_counts": class_mask_counts,
            "PLE": ple,
            "CCI": cci,
            "Sem_Drift": sem_drift,
            "OOD_FF": ood_ff,
            "Grad_Align": grad_align,
            "Grad_X_Norm": grad_x_norm,
            "Grad_U_Norm": grad_u_norm,
        })

        return total_loss, stats
