import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .base import AlgorithmBase
from cscore import compute_ple, compute_cci, compute_sem_drift, compute_grad_align, compute_ood_ff


class SoftMatch(AlgorithmBase):
    """
    SoftMatch: Addressing the Quantity-Quality Tradeoff in Semi-supervised Learning.
    Chen et al., ICLR 2023. https://arxiv.org/abs/2301.10921

    Core change: replaces FixMatch's hard binary mask with a Gaussian soft weight.
    For each class c, maintains an EMA mean mu_c and variance sigma_c^2 of the
    confidence scores. Sample i's soft weight is:
        N(score_i; mu_hat_c, sigma_hat_c) / max_density_hat_c  in [0, 1]

    Also uses Uniform Alignment (similar to FixMatch DA) to prevent degenerate collapse.

    C-Score metric adaptations:
    - ID-M / OOD-M: uses effective mask (soft_weight > 0.1) as binary proxy
    - Added SoftW_ID / SoftW_OOD: mean soft weight for ID vs OOD samples.
      Ideally SoftW_OOD << SoftW_ID, indicating OOD samples are naturally downweighted.
    """

    def __init__(self, model, device, cfg):
        super().__init__(model, device, cfg)
        self.p_cutoff   = cfg.get('p_cutoff', 0.95)   # reference baseline only; no hard gate
        self.lambda_u   = cfg.get('lambda_u', 1.0)
        self.ema_m      = 0.999
        self.criterion  = nn.CrossEntropyLoss()

        self.register_buffer('class_score_mean',
                             torch.full((self.num_classes,), 1.0 / self.num_classes, device=device))
        self.register_buffer('class_score_var',
                             torch.ones(self.num_classes, device=device) * 0.01)
        self.register_buffer('p_model',
                             torch.ones(self.num_classes, device=device) / self.num_classes)

    def compute_loss(self, data_batch, global_step):
        inputs_x, targets_x = data_batch['labeled']
        (inputs_u_w, inputs_u_s), targets_u_gt = data_batch['unlabeled']

        inputs_x   = inputs_x.to(self.device)
        targets_x  = targets_x.to(self.device)
        inputs_u_w = inputs_u_w.to(self.device)
        inputs_u_s = inputs_u_s.to(self.device)

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

            # Uniform Alignment: align predicted distribution to uniform to prevent degenerate collapse
            self.p_model = self.p_model * self.ema_m + probs.mean(0) * (1 - self.ema_m)
            probs_aligned = probs * (1.0 / self.num_classes) / (self.p_model + 1e-6)
            probs_aligned = probs_aligned / probs_aligned.sum(-1, keepdim=True)

            max_probs, pseudo_labels = torch.max(probs_aligned, dim=-1)

            # Update per-class score distribution (EMA)
            for c in range(self.num_classes):
                idx_c = (pseudo_labels == c)
                if idx_c.sum() > 1:
                    sc = max_probs[idx_c]
                    self.class_score_mean[c] = (self.class_score_mean[c] * self.ema_m
                                                + sc.mean() * (1 - self.ema_m))
                    self.class_score_var[c]  = (self.class_score_var[c] * self.ema_m
                                                + sc.var().clamp(min=1e-4) * (1 - self.ema_m))

            # Compute soft weights via Gaussian density
            mu     = self.class_score_mean[pseudo_labels]           # (N_u,)
            sigma  = self.class_score_var[pseudo_labels].sqrt().clamp(min=1e-3)
            gauss  = torch.exp(-0.5 * ((max_probs - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
            max_density = 1.0 / (sigma * math.sqrt(2 * math.pi))
            soft_weights = (gauss / (max_density + 1e-6)).clamp(0.0, 1.0)

            do_expensive = (global_step % 32 == 0)

            # C-Score metrics
            ple = compute_ple(probs)
            cci = compute_cci(probs)

            # Effective mask for Sem-Drift / infiltration analysis
            eff_mask = (soft_weights > 0.1).float()

            sem_drift = 0.0
            if do_expensive:
                sem_drift = compute_sem_drift(
                    logits_x, targets_x, logits_u_w, pseudo_labels, eff_mask, self.num_classes
                )

        loss_u = (F.cross_entropy(logits_u_s, pseudo_labels, reduction='none') * soft_weights).mean()
        total_loss = loss_x + self.lambda_u * loss_u

        # C-Score: Gradient Alignment
        grad_align = 0.0
        grad_x_norm = 0.0
        grad_u_norm = 0.0
        if do_expensive:
            _actual = getattr(self.model, '_orig_mod', self.model)
            fc_params = [p for p in _actual.fc.parameters() if p.requires_grad]
            if fc_params and soft_weights.sum() > 0:
                grad_align, grad_x_norm, grad_u_norm = compute_grad_align(loss_x, loss_u, fc_params)

        class_mask_counts = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            class_mask_counts[c] = ((pseudo_labels == c) & (eff_mask == 1)).sum().item()

        if do_expensive:
            stats = self._analyze_infiltration(eff_mask, pseudo_labels, targets_u_gt)
            # SoftMatch-specific: mean soft weight for ID vs OOD
            tgt_np = targets_u_gt.numpy()
            w_np   = soft_weights.cpu().numpy()
            id_idx  = np.where(tgt_np != -1)[0]
            ood_idx = np.where(tgt_np == -1)[0]
            stats['SoftW_ID']  = float(w_np[id_idx].mean())  if len(id_idx)  > 0 else 0.0
            stats['SoftW_OOD'] = float(w_np[ood_idx].mean()) if len(ood_idx) > 0 else 0.0
        else:
            stats = {"id_mask_ratio": 0.0, "ood_mask_ratio": 0.0, "id_pseudo_acc": 0.0,
                     "SoftW_ID": 0.0, "SoftW_OOD": 0.0}

        ood_ff = compute_ood_ff(stats['id_mask_ratio'], stats['ood_mask_ratio'])

        stats.update({
            "loss":              total_loss.item(),
            "loss_x":            loss_x.item(),
            "loss_u":            loss_u.item(),
            "mask_ratio":        eff_mask.mean().item(),
            "soft_weight_mean":  soft_weights.mean().item(),
            "class_mask_counts": class_mask_counts,
            "PLE":               ple,
            "CCI":               cci,
            "Sem_Drift":         sem_drift,
            "OOD_FF":            ood_ff,
            "Grad_Align":        grad_align,
            "Grad_X_Norm":       grad_x_norm,
            "Grad_U_Norm":       grad_u_norm,
        })
        return total_loss, stats
