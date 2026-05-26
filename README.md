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

Average Shannon entropy of the unlabeled softmax distribution. $N$ is the number of unlabeled samples in the batch and $K$ is the number of target classes.

$$\text{PLE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{K} p(c|u_i) \log(p(c|u_i) + \varepsilon)$$

| Value | Interpretation |
|---|---|
| Low (≈ 0) | Confident predictions — normal convergence |
| High (≈ log K) | Diffuse predictions — early training or collapse |

#### CCI — Class Concentration Index

KL divergence from the batch-mean soft class distribution $\bar{p}_c$ to the uniform prior.

$$\bar{p}_c = \frac{1}{N} \sum_{i=1}^{N} p(c|u_i)$$

$$\text{CCI} = \sum_{c=1}^{K} \bar{p}_c \log\left(\frac{\bar{p}_c}{1/K}\right)$$

| Value | Interpretation |
|---|---|
| ≈ 0 | Balanced pseudo-labels — healthy |
| Rising | Concentration on few classes — feature sink forming |

### Feature & Optimization Space

#### Sem-Drift — Semantic Drift

Average L2 distance between per-class logit centroids of labeled samples and masked pseudo-labeled unlabeled samples. $z_{x,i}$ and $z_{u,j}$ are the logits of labeled and (weakly augmented) unlabeled samples, respectively; $\hat{y}_j$ is the hard pseudo-label; $m_j \in \{0,1\}$ is the confidence mask.

$$\mu_L^{(c)} = \frac{1}{|\{i: y_i = c\}|} \sum_{i:\, y_i = c} z_{x,i}$$

$$\mu_U^{(c)} = \frac{1}{|\{j: \hat{y}_j = c,\, m_j = 1\}|} \sum_{j:\, \hat{y}_j = c,\, m_j = 1} z_{u,j}$$

$$\text{Sem-Drift} = \frac{1}{|C^{\text{valid}}|} \sum_{c \in C^{\text{valid}}} \left\| \mu_L^{(c)} - \mu_U^{(c)} \right\|_2$$

where $C^{\text{valid}}$ is the set of classes for which both centroids are defined.

#### Grad-Align — Gradient Alignment

Cosine similarity between the labeled and unlabeled loss gradients on the final FC layer $W_{fc}$.

$$g_x = \nabla_{W_{fc}} L_x, \quad g_u = \nabla_{W_{fc}} L_u$$

$$\text{Grad-Align} = \frac{g_x^T g_u}{\|g_x\|_2 \|g_u\|_2 + \varepsilon}$$

### Oracle Metric

#### OOD-FF — OOD Filtration Failure

$r_{\text{ID}}$ and $r_{\text{OOD}}$ are the fractions of ID and OOD unlabeled samples that pass the confidence mask, respectively.

$$\text{OOD-FF} = \frac{r_{\text{OOD}}}{r_{\text{ID}} + \varepsilon}$$

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

## Compute Budget

FixMatch experiments were run on an **NVIDIA RTX PRO 6000 Blackwell Max-Q** Workstation Edition.
FlexMatch, SoftMatch, and DS3L experiments were run on an **NVIDIA L40S** via NVIDIA Brev.

### GPU Hours by Experiment

| Algorithm | Labeled Dataset | Runs | GPU Hours | Throughput |
|---|---|:---:|---:|---:|
| FixMatch | CIFAR-10 | 155 | 116.59 | 9,859 samp/s |
| FlexMatch | CIFAR-10 | 155 | 118.55 | 9,532 samp/s |
| DS3L | CIFAR-10 | 155 | 123.04 | 9,168 samp/s |
| SoftMatch | CIFAR-10 | 155 | 123.53 | 9,130 samp/s |
| **Total (unique)** | | **620** | **481.7** | |

---

## Throughput Ablation: NVIDIA Software Stack

Additive ablation measuring the contribution of each acceleration component.
Hardware: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition.
Configuration: FixMatch, CIFAR-10, seed=0, 50 epochs, epoch 2–50 averaged (epoch 1 excluded as GPU warmup).

| Setting | Avg Epoch Time | Throughput (samp/s) | vs GPU baseline | vs CPU | Component added |
|---|---|---|---|---|---|
| cpu_only | 353.71s | 371 | −94.9% | 1.00× | — (CPU reference) |
| gpu_baseline | 24.96s | 5,252 | baseline | 14.17× | GPU only (cuDNN benchmark OFF) |
| plus_benchmark | 17.90s | 7,324 | +39.4% | 19.76× | + cuDNN benchmark mode |
| plus_amp | 13.20s | 9,933 | +89.1% | 26.80× | + AMP (FP16 Tensor Cores) |
| plus_compile | 10.04s | 13,054 | +148.5% | 35.23× | + torch.compile |
| plus_fused | 9.75s | 13,445 | +156.0% | 36.28× | + PyTorch native fused SGD |
| plus_dali | 9.29s | 14,109 | +168.6% | 38.07× | + NVIDIA DALI async pipeline |
| full_stack | 9.44s | 13,887 | +164.4% | 37.47× | = full stack |

**Key findings:**
- cuDNN benchmark mode alone contributes **+39.4%** by selecting optimal convolution algorithms for WRN-28-2.
- AMP (FP16 Tensor Cores) adds a further **+35.3%** incremental gain.
- torch.compile provides an additional **+31.4%** via kernel fusion and graph optimization.
- NVIDIA DALI contributes **+4.8%** through asynchronous GPU prefetch of the data pipeline.
- The full stack is **37.5× faster than CPU-only**, making the 570-run experimental sweep feasible.

To reproduce:
```bash
bash ablation_full_stack.sh
# results written to ablation_full_stack_results/summary.txt
```

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
