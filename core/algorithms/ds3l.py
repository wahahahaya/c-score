import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import AlgorithmBase
from cscore import compute_ple, compute_cci, compute_sem_drift, compute_grad_align, compute_ood_ff


class DS3L(AlgorithmBase):
    """
    DS3L: Safe Deep Semi-Supervised Learning for Unseen-Class Unlabeled Data.
    Guo et al., ICML 2020. https://arxiv.org/abs/2004.09515

    The original paper uses bi-level meta-optimization to learn per-sample importance
    weights. This implementation uses a trainable approximation:

        w_i = hard_gate_i x confidence_i x centroid_sim_i

    where:
    - hard_gate_i = 1[max_prob_i >= tau]          (same as FixMatch, initial filtering)
    - confidence_i = max_prob_i                    (higher confidence -> more likely ID)
    - centroid_sim_i = cos(logit_i, centroid_{c})  (cosine similarity to the labeled class
                                                    centroid; OOD sample logits tend to
                                                    deviate from labeled centroids)

    Labeled class centroids are updated each step via EMA.

    C-Score metric adaptations:
    - ID-M / OOD-M: uses effective mask (weight > 0.05)
    - Added DS3L_WID / DS3L_WOOD: mean centroid_sim for ID vs OOD samples,
      directly quantifying whether DS3L successfully suppresses OOD samples
      via geometric distance.
    """

    def __init__(self, model, device, cfg):
        super().__init__(model, device, cfg)
        self.p_cutoff          = cfg.get('p_cutoff', 0.95)
        self.lambda_u          = cfg.get('lambda_u', 1.0)
        self.ema_centroid_m    = 0.99
        self.criterion         = nn.CrossEntropyLoss()

        # EMA labeled class centroids in logit space (num_classes x num_classes)
        self.register_buffer('labeled_centroids',
                             torch.zeros(self.num_classes, self.num_classes, device=device))
        self.register_buffer('centroid_init',
                             torch.zeros(self.num_classes, device=device))

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
            max_probs, pseudo_labels = torch.max(probs, dim=-1)

            # Update labeled centroids (EMA)
            for c in range(self.num_classes):
                idx_c = (targets_x == c)
                if idx_c.sum() > 0:
                    centroid_c = logits_x.detach()[idx_c].mean(0)
                    if self.centroid_init[c] < 0.5:
                        self.labeled_centroids[c] = centroid_c
                        self.centroid_init[c]      = 1.0
                    else:
                        self.labeled_centroids[c] = (self.labeled_centroids[c] * self.ema_centroid_m
                                                     + centroid_c * (1 - self.ema_centroid_m))

            # Compute importance weights (vectorized)
            # Hard gate: completely exclude low-confidence samples
            hard_gate = max_probs.ge(self.p_cutoff).float()

            # Cosine similarity to labeled centroid of the pseudo-labeled class
            logits_u_norm   = F.normalize(logits_u_w.detach(), dim=-1)         # (N_u, C)
            centroids_norm  = F.normalize(self.labeled_centroids, dim=-1)       # (K, C)
            pseudo_centroids = centroids_norm[pseudo_labels]                    # (N_u, C)
            centroid_sim    = (logits_u_norm * pseudo_centroids).sum(-1).clamp(min=0.0)  # (N_u,)

            # Zero out classes not yet initialized
            init_ok = self.centroid_init[pseudo_labels]                         # (N_u,)
            centroid_sim = centroid_sim * (init_ok > 0.5).float()

            # Final weight
            weights = hard_gate * max_probs * centroid_sim

            do_expensive = (global_step % 32 == 0)

            # C-Score metrics
            ple = compute_ple(probs)
            cci = compute_cci(probs)

            eff_mask = (weights > 0.05).float()

            sem_drift = 0.0
            if do_expensive:
                sem_drift = compute_sem_drift(
                    logits_x, targets_x, logits_u_w, pseudo_labels, eff_mask, self.num_classes
                )

        loss_u = (F.cross_entropy(logits_u_s, pseudo_labels, reduction='none') * weights).mean()
        total_loss = loss_x + self.lambda_u * loss_u

        # C-Score: Gradient Alignment
        grad_align = 0.0
        grad_x_norm = 0.0
        grad_u_norm = 0.0
        if do_expensive:
            _actual = getattr(self.model, '_orig_mod', self.model)
            fc_params = [p for p in _actual.fc.parameters() if p.requires_grad]
            if fc_params and weights.sum() > 0:
                grad_align, grad_x_norm, grad_u_norm = compute_grad_align(loss_x, loss_u, fc_params)

        class_mask_counts = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            class_mask_counts[c] = ((pseudo_labels == c) & (eff_mask == 1)).sum().item()

        if do_expensive:
            stats = self._analyze_infiltration(eff_mask, pseudo_labels, targets_u_gt)
            # DS3L-specific: centroid similarity for ID vs OOD
            tgt_np  = targets_u_gt.numpy()
            sim_np  = centroid_sim.cpu().numpy()
            id_idx  = np.where(tgt_np != -1)[0]
            ood_idx = np.where(tgt_np == -1)[0]
            stats['DS3L_WID']  = float(sim_np[id_idx].mean())  if len(id_idx)  > 0 else 0.0
            stats['DS3L_WOOD'] = float(sim_np[ood_idx].mean()) if len(ood_idx) > 0 else 0.0
        else:
            stats = {"id_mask_ratio": 0.0, "ood_mask_ratio": 0.0, "id_pseudo_acc": 0.0,
                     "DS3L_WID": 0.0, "DS3L_WOOD": 0.0}

        ood_ff = compute_ood_ff(stats['id_mask_ratio'], stats['ood_mask_ratio'])

        stats.update({
            "loss":              total_loss.item(),
            "loss_x":            loss_x.item(),
            "loss_u":            loss_u.item(),
            "mask_ratio":        eff_mask.mean().item(),
            "weight_mean":       weights.mean().item(),
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
