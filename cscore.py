"""
cscore.py — C-Score Diagnostic Metrics for Safe Semi-Supervised Learning

Five metrics extracted as reusable functions:
  - compute_ple:        Pseudo-Label Entropy
  - compute_cci:        Class Concentration Index
  - compute_sem_drift:  Semantic Drift
  - compute_grad_align: Gradient Alignment
  - compute_ood_ff:     OOD Filtration Failure

All functions accept raw tensors and return Python floats.
"""

import torch
import torch.nn.functional as F


def compute_ple(probs: torch.Tensor) -> float:
    """
    Pseudo-Label Entropy (PLE)

    Measures the average entropy of the pseudo-label probability distribution
    across unlabeled samples. High PLE → uncertain / spread predictions.
    Low PLE → confident (but possibly collapsed) predictions.

    Args:
        probs: Softmax probabilities for unlabeled samples, shape (N, C).

    Returns:
        Scalar PLE value (>= 0). Upper bound = log(C) when uniform.
    """
    return -torch.sum(probs * torch.log(probs + 1e-6), dim=-1).mean().item()


def compute_cci(probs: torch.Tensor) -> float:
    """
    Class Concentration Index (CCI)

    KL divergence from the average predicted class distribution to a uniform
    distribution. High CCI → predictions are concentrated on few classes
    (class imbalance in pseudo-labels). Zero when perfectly balanced.

    Args:
        probs: Softmax probabilities for unlabeled samples, shape (N, C).

    Returns:
        Scalar CCI value (>= 0).
    """
    num_classes = probs.shape[-1]
    pseudo_class_dist = probs.mean(dim=0)
    uniform = torch.ones_like(pseudo_class_dist) / num_classes
    return torch.sum(pseudo_class_dist * torch.log(pseudo_class_dist / (uniform + 1e-6) + 1e-6)).item()


def compute_sem_drift(
    logits_x: torch.Tensor,
    targets_x: torch.Tensor,
    logits_u_w: torch.Tensor,
    pseudo_labels: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
) -> float:
    """
    Semantic Drift (Sem-Drift)

    Average L2 distance between per-class logit centroids of labeled samples
    and masked pseudo-labeled unlabeled samples. Large drift → the model
    represents the same class very differently for labeled vs unlabeled data.

    Args:
        logits_x:      Logits for labeled samples, shape (B_x, C).
        targets_x:     Ground-truth labels for labeled samples, shape (B_x,).
        logits_u_w:    Logits for unlabeled samples (weak aug), shape (B_u, C).
        pseudo_labels: Hard pseudo-labels for unlabeled samples, shape (B_u,).
        mask:          Binary mask (float) selecting high-confidence samples, shape (B_u,).
        num_classes:   Total number of classes.

    Returns:
        Scalar Sem-Drift value (>= 0). Returns 0.0 when no valid class pair exists.
    """
    sem_drift = 0.0
    valid_classes = 0
    for c in range(num_classes):
        mask_x = (targets_x == c)
        mask_u = (pseudo_labels == c) & (mask == 1)
        if mask_x.sum() > 0 and mask_u.sum() > 0:
            centroid_x = logits_x.detach()[mask_x].mean(dim=0)
            centroid_u = logits_u_w.detach()[mask_u].mean(dim=0)
            sem_drift += torch.norm(centroid_x - centroid_u).item()
            valid_classes += 1
    if valid_classes > 0:
        sem_drift /= valid_classes
    return sem_drift


def compute_grad_align(
    loss_x: torch.Tensor,
    loss_u: torch.Tensor,
    fc_params: list,
) -> tuple[float, float, float]:
    """
    Gradient Alignment (Grad-Align)

    Cosine similarity between the labeled loss gradient and the unlabeled loss
    gradient on the final FC layer parameters.

    Range [-1, +1]:
      +1 = aligned   → unlabeled data reinforces labeled learning
       0 = orthogonal → unlabeled data is neutral
      -1 = conflict  → unlabeled data actively hurts labeled learning

    Args:
        loss_x:    Scalar labeled loss (must still have a live compute graph).
        loss_u:    Scalar unlabeled loss (must still have a live compute graph).
        fc_params: List of FC-layer parameters (from model.fc.parameters()).

    Returns:
        Tuple of (grad_align, grad_x_norm, grad_u_norm).
        All three are 0.0 on failure (e.g. zero-norm gradients).
    """
    grad_align = 0.0
    grad_x_norm = 0.0
    grad_u_norm = 0.0
    try:
        g_x = torch.autograd.grad(loss_x, fc_params, retain_graph=True, allow_unused=True)
        g_u = torch.autograd.grad(loss_u, fc_params, retain_graph=True, allow_unused=True)
        g_x_flat = torch.cat([g.float().flatten() for g in g_x if g is not None])
        g_u_flat = torch.cat([g.float().flatten() for g in g_u if g is not None])
        grad_x_norm = g_x_flat.norm().item()
        grad_u_norm = g_u_flat.norm().item()
        if grad_x_norm > 1e-8 and grad_u_norm > 1e-8:
            grad_align = F.cosine_similarity(
                g_x_flat.unsqueeze(0), g_u_flat.unsqueeze(0)
            ).item()
    except Exception:
        pass
    return grad_align, grad_x_norm, grad_u_norm


def compute_ood_ff(id_mask_ratio: float, ood_mask_ratio: float) -> float:
    """
    OOD Filtration Failure (OOD-FF)

    Ratio of OOD pass-through rate to ID pass-through rate.
    Quantifies how badly the masking mechanism fails to filter OOD samples.

    Interpretation:
      ~0   → OOD samples are almost entirely suppressed (ideal)
      ~1   → OOD leaks at the same rate as ID (filter has collapsed)
      > 1  → OOD leaks more than ID (filter is actively inverting)

    Args:
        id_mask_ratio:  Fraction of ID unlabeled samples that pass the mask.
        ood_mask_ratio: Fraction of OOD unlabeled samples that pass the mask.

    Returns:
        Scalar OOD-FF value (>= 0).
    """
    return ood_mask_ratio / (id_mask_ratio + 1e-6)
