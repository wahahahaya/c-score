# C-Score — Diagnosing Hidden Collapse in Semi-Supervised Learning under OOD Contamination

> **Paper:** *C-Score: Beyond Accuracy for Robustness Assessment in Semi-Supervised Learning under Open-World Unlabeled Contamination*

> **Research question:** When OOD samples contaminate the unlabeled pool of a semi-supervised learner, does test accuracy faithfully reflect what is happening inside the model?

This repository argues the answer is **no** — and provides the **C-Score diagnostic framework** to surface the hidden collapse that accuracy alone cannot see.

---

## Overview

Modern SSL methods (FixMatch, FlexMatch, SoftMatch, DS3L) rely on high-confidence pseudo-labeling over unlabeled data. In open-world settings, the unlabeled pool inevitably contains out-of-distribution (OOD) samples. This work systematically studies how varying OOD contamination ratio (`r`) affects:

1. **Surface-level accuracy** — the metric everyone reports.
2. **Internal representation health** — quantified by the C-Score framework.

The central finding: there exist contamination regimes in which test accuracy remains stable (or even improves) while internal feature alignment, pseudo-label quality, and gradient direction are all degrading — a *robustness illusion*.

---

## C-Score Framework

C-Score consists of five metrics computed per training iteration. All are implemented in [`cscore.py`](cscore.py) as standalone functions and called from each algorithm.

### Prediction Space

#### PLE — Pseudo-Label Entropy

Average Shannon entropy of the unlabeled softmax distribution.

$$\text{PLE} = -\frac{1}{|U|} \sum_{i \in U} \sum_{k=1}^{C} p_k^{(i)} \log(p_k^{(i)} + \varepsilon)$$

| Value | Interpretation |
|---|---|
| Low (≈ 0) | Confident predictions — normal convergence |
| High (≈ log C) | Diffuse predictions — early training or collapse |

#### CCI — Class Concentration Index

KL divergence from the average pseudo-label distribution to a uniform prior.

$$\text{CCI} = \sum_{k=1}^{C} \hat{p}_k \log\frac{\hat{p}_k}{1/C}, \quad \hat{p}_k = \frac{1}{|U|}\sum_{i \in U} p_k^{(i)}$$

| Value | Interpretation |
|---|---|
| ≈ 0 | Balanced pseudo-labels — healthy |
| Rising | Concentration on few classes — feature sink forming |

### Feature & Optimization Space

#### Sem-Drift — Semantic Drift

Average L2 distance between per-class logit centroids of labeled samples and masked pseudo-labeled unlabeled samples.

$$\text{Sem-Drift} = \frac{1}{|C_{\text{valid}}|} \sum_{c \in C_{\text{valid}}} \left\| \bar{z}_x^{(c)} - \bar{z}_u^{(c)} \right\|_2$$

#### Grad-Align — Gradient Alignment

Cosine similarity between the labeled loss gradient and unlabeled loss gradient on the final FC layer.

$$\text{Grad-Align} = \frac{\nabla_{W_{fc}} \mathcal{L}_x \cdot \nabla_{W_{fc}} \mathcal{L}_u}{\|\nabla_{W_{fc}} \mathcal{L}_x\| \cdot \|\nabla_{W_{fc}} \mathcal{L}_u\|}$$

### Oracle Metric

#### OOD-FF — OOD Filtration Failure

$$\text{OOD-FF} = \frac{\text{OOD pass rate}}{\text{ID pass rate} + \varepsilon}$$

Requires ground-truth ID/OOD labels — used for post-hoc analysis only.

### Metric Summary

| Metric | Space | Needs labeled ref | Needs OOD labels |
|---|---|---|---|
| PLE | Prediction | No | No |
| CCI | Prediction | No | No |
| Sem-Drift | Feature | Yes | No |
| Grad-Align | Optimization | Yes | No |
| OOD-FF | Oracle | No | Yes |

---

## Algorithms

All algorithms share the same C-Score infrastructure via `cscore.py`.

| Algorithm | Core Mechanism |
|---|---|
| **FixMatch** | Hard confidence threshold (τ = 0.95) |
| **FlexMatch** | Per-class adaptive threshold (curriculum pseudo-labeling) |
| **SoftMatch** | Soft Gaussian weighting + Uniform Alignment |
| **DS3L** | Centroid cosine similarity × confidence × hard gate |
| **Supervised** | Cross-entropy on labeled only (baseline) |

---

## Key Results

> **Total experiments:** 650 runs — 4 algorithms × 6 OOD types × 6 contamination ratios × 5 seeds on CIFAR-10, plus 30 runs on CIFAR-100.

### Accuracy Masking (FixMatch × SVHN × CIFAR-10)

| r | Best Acc | CCI | OOD-FF |
|:---:|:---:|:---:|:---:|
| 0.0 | 67.67 ± 2.49 | 0.143 ± 0.072 | 0.000 |
| 0.2 | 71.21 ± 4.19 | 0.225 ± 0.037 | 0.037 ± 0.002 |
| 0.3 | 70.74 ± 5.96 | 0.329 ± 0.055 | 0.037 ± 0.005 |
| 0.5 | 68.10 ± 4.81 | 0.552 ± 0.155 | 0.028 ± 0.016 |

At r = 0.2–0.3, accuracy stays within ~1% of clean baseline while **CCI rises +57–130%** — accuracy is masking internal collapse.

### Algorithm Comparison (CIFAR-10, aggregated over 6 OOD types)

| r | FixMatch | FlexMatch | SoftMatch | DS3L |
|:---:|:---:|:---:|:---:|:---:|
| 0.0 | 71.34 | **79.19** | 72.65 | 72.48 |
| 0.5 | 67.77 | **70.09** | 66.90 | 66.51 |
| Δ | −3.57% | −9.10% | −5.75% | −5.97% |

FlexMatch starts highest but degrades most under contamination.

### OOD Source Effect (FixMatch, r = 0.5, CIFAR-10)

| OOD Source | Accuracy Drop | Proximity |
|:---:|:---:|:---:|
| CIFAR-100 | −8.81% | Near-OOD |
| STL-10 | −7.97% | Near-OOD |
| MNIST | −3.50% | Far-OOD |
| SVHN | −0.83% | Far-OOD |
| Gaussian Noise | −2.30% | Far-OOD |
| Textures | +1.98% | Far-OOD |

Near-OOD sources cause the most damage; Textures is effectively filtered by the confidence threshold.

### Generalization: CIFAR-100 as ID (FixMatch × SVHN)

| r | Best Acc | CCI |
|:---:|:---:|:---:|
| 0.0 | 43.21 ± 1.07 | 0.161 ± 0.015 |
| 0.2 | 43.51 ± 1.35 | 0.578 ± 0.072 |
| 0.5 | 39.22 ± 1.26 | 1.430 ± 0.238 |

Accuracy masking replicates on CIFAR-100: CCI rises **+787%** while accuracy drops only ~4%.

---

## Codebase Structure

```
cscore/
├── cscore.py                  # C-Score metric functions (PLE, CCI, Sem-Drift, Grad-Align, OOD-FF)
├── train_executor.py          # Single-experiment entry point
├── grid.py                    # r sweep (CIFAR-10 / SVHN)
├── grid_ood_type.py           # OOD type sweep (6 OOD types × r × seeds)
├── grid_alg.py                # Algorithm sweep (4 algorithms × OOD types × r × seeds)
├── grid_cifar100.py           # CIFAR-100 as ID sweep
├── validate_cscore.py         # Numerical equivalence tests for C-Score functions
│
├── core/
│   ├── algorithms/
│   │   ├── base.py            # AlgorithmBase
│   │   ├── fixmatch.py
│   │   ├── flexmatch.py
│   │   ├── softmatch.py
│   │   ├── ds3l.py
│   │   └── supervised.py
│   ├── datasets/
│   │   ├── cifar10.py
│   │   ├── cifar100.py
│   │   ├── OOD.py             # 6 OOD datasets (SVHN, MNIST, CIFAR-100, STL-10, Textures, Gaussian)
│   │   ├── augmentation.py    # Weak / strong augmentation pipeline
│   │   ├── wrapper.py
│   │   └── builder.py
│   └── models/
│       └── wrn.py             # WideResNet-28-2
│
├── engines/
│   ├── trainer.py             # Training loop, EMA, AMP, logging
│   └── logger.py
│
├── tools/
│   ├── data_visual.py         # CLIP-text MDS / image-feature UMAP visualization
│   ├── feat_visual.py         # Feature space visualization (t-SNE / UMAP / PCA)
│   ├── alg_comparison_bar.py  # Algorithm comparison bar chart
│   ├── analysis.py            # Log parsing → CSV + curves
│   └── aggregate_results.py   # Aggregate mean±std across seeds
│
└── configs/
    ├── template_grid_sweep.yaml
    ├── example_cifar10.yaml
    └── cifar10_sup.yaml
```

---

## Setup

```bash
git clone https://github.com/<your-username>/cscore.git
cd cscore

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires: Python 3.10+, PyTorch 2.0+, CUDA 11.8+.

Download datasets to `data/` (auto-downloaded by torchvision on first run, except DTD Textures).

---

## Usage

### Single experiment

```bash
python train_executor.py --config configs/example_cifar10.yaml
```

### Sweeps

```bash
# OOD type sweep (6 OOD types × 6 ratios × 5 seeds)
python grid_ood_type.py

# Algorithm sweep
python grid_alg.py --resume   # skip already-completed runs

# CIFAR-100 as ID
python grid_cifar100.py
```

### Visualization

```bash
# Semantic MDS via CLIP text encoder (for paper figures)
python tools/data_visual.py --mode text --out_dir tools/plots/semantic_mds --dpi 300

# Image feature UMAP from a trained model
python tools/data_visual.py --mode image \
    --model_path exp_results/.../best_model.pth \
    --out_dir tools/plots/feature_umap --dpi 300

# Algorithm comparison bar chart
python tools/alg_comparison_bar.py --out_dir tools/plots/alg_comparison --dpi 300
```

### Validate C-Score

```bash
python validate_cscore.py
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{chen2026cscore,
  title     = {C-Score: Beyond Accuracy for Robustness Assessment in Semi-Supervised Learning under Open-World Unlabeled Contamination},
  author    = {Chen, Tsao-Lun and Fu, Chi-Cheng and Chou, Han-Yi E. and Su, Shun-Feng},
  year      = {2026}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
