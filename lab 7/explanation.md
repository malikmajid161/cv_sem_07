# Lab 07 — HOG + Genetic Algorithm Feature Selection
### Improving HOG-Based Medical Image Classification Using a Genetic Algorithm

**Student:** Muhammad Majid Ali | FA23-BAI-045  
**Dataset:** BloodMNIST (MedMNIST Benchmark)  
**Course:** Computer Vision

---

## 1. Objective

Investigate whether a **Genetic Algorithm (GA)** can select a smaller, more informative subset of HOG features that maintains competitive classification performance on medical images, compared to using all HOG features.

---

## 2. Dataset — BloodMNIST

| Property | Value |
|---|---|
| Source | MedMNIST v2 Benchmark |
| Task | 8-class blood cell classification |
| Image type | RGB microscopy images |
| Image size | 64 × 64 pixels |
| Train samples | 11,959 |
| Validation samples | 1,715 |
| Test samples | 3,421 |
| Classes | Basophil, Eosinophil, Erythroblast, IG, Lymphocyte, Monocyte, Neutrophil, Platelet |

---

## 3. HOG Feature Extraction Parameters

| Parameter | Value |
|---|---|
| Orientations | 9 |
| Pixels per cell | 8 × 8 |
| Cells per block | 2 × 2 |
| Block normalization | L2-Hys |
| Channel axis | RGB (-1) |
| **Total feature length** | **1,764** |
| Extraction time | 46.2 seconds |

---

## 4. Baseline — HOG + RBF-SVM

All 1,764 HOG features fed directly into an RBF-SVM (C=10, gamma='scale').

### Results on Test Set

| Metric | Score |
|---|---|
| Accuracy | **81.18%** |
| Precision (Macro) | 81.88% |
| Recall (Macro) | 79.80% |
| Macro-F1 | **80.71%** |
| Weighted-F1 | 81.14% |
| Balanced Accuracy | 79.80% |
| Specificity (Macro) | 97.22% |
| Cohen's Kappa | 0.7794 |
| MCC | 0.7797 |
| SVM Train Time | 124.6 s |
| SVM Inference Time | 63.961 s |

---

## 5. Genetic Algorithm Design

### Representation
Each individual (chromosome) is a **binary mask** of length 1,764.  
- `1` = keep this HOG feature  
- `0` = discard this feature

### Fitness Function
```
fitness(mask) = val_Macro-F1 − λ × feature_ratio
```
Where:
- `λ = 0.05` (penalty weight for feature count)
- `feature_ratio = selected_features / 1764`
- Inner classifier: **LinearSVC** (C=1.0, fast proxy for final RBF-SVM)
- Training data: **5% subsample** of training set for fast evaluation

### GA Operators

| Operator | Method | Detail |
|---|---|---|
| Initialisation | Random binary | 50% probability per gene |
| Selection | Tournament (k=3) | Select best of 3 random individuals |
| Crossover | One-point | Single random cut point, prob=0.80 |
| Mutation | Bit-flip | Each bit flips with prob = mut_prob |
| Elitism | Top-2 preserved | Best 2 carried unchanged each generation |

### Two Configurations Tested

| Setting | Config-1 | Config-2 |
|---|---|---|
| Population size | 30 | 50 |
| Generations | 20 | 20 |
| Mutation probability | 0.01 | 0.03 |
| Crossover probability | 0.80 | 0.80 |
| Elitism | 2 | 2 |

---

## 6. GA Results

### Config-1 (pop=30, mut=0.01) — Generation History

| Generation | Best Fitness | Avg Fitness | Features Selected |
|---|---|---|---|
| 1 | 0.4251 | 0.4069 | 888 / 1764 |
| 5 | 0.4392 | 0.4262 | 850 / 1764 |
| 10 | 0.4479 | 0.4379 | 871 / 1764 |
| 15 | 0.4640 | 0.4452 | 876 / 1764 |
| 20 | **0.4702** | 0.4552 | **880 / 1764** |

**GA-1 Search Time: 1713.1 seconds**

### Config-2 (pop=50, mut=0.03) — Generation History

| Generation | Best Fitness | Avg Fitness | Features Selected |
|---|---|---|---|
| 1 | 0.4299 | 0.4096 | 843 / 1764 |
| 5 | 0.4441 | 0.4182 | 913 / 1764 |
| 10 | 0.4485 | 0.4313 | 920 / 1764 |
| 15 | 0.4674 | 0.4365 | 915 / 1764 |
| 20 | **0.4752** | 0.4429 | **915 / 1764** |

**GA-2 Search Time: 3038.4 seconds**

---

## 7. Final Test Set Performance (RBF-SVM on GA Features)

### Table 2 — Main Performance Comparison

| Metric | Baseline (All 1764) | GA-1 (880 features) | Δ GA-1 | GA-2 (915 features) | Δ GA-2 |
|---|---|---|---|---|---|
| Accuracy | 81.18% | 80.42% | −0.76% | 80.30% | −0.88% |
| Precision (Macro) | 81.88% | 81.14% | −0.74% | 80.72% | −1.16% |
| Recall (Macro) | 79.80% | 78.92% | −0.88% | 78.84% | −0.96% |
| Macro-F1 | 80.71% | 79.88% | −0.83% | 79.65% | −1.06% |
| Weighted-F1 | 81.14% | 80.37% | −0.77% | 80.24% | −0.90% |
| Balanced Accuracy | 79.80% | 78.92% | −0.88% | 78.84% | −0.96% |
| Specificity (Macro) | 97.22% | 97.11% | −0.11% | 97.10% | −0.12% |
| Cohen's Kappa | 0.7794 | 0.7704 | −0.0090 | 0.7691 | −0.0103 |
| MCC | 0.7797 | 0.7708 | −0.0089 | 0.7694 | −0.0103 |

### Table 3 — Feature Count and Computational Cost

| Metric | Baseline | GA-1 | GA-2 |
|---|---|---|---|
| Total HOG features | 1,764 | 1,764 | 1,764 |
| Features selected | 1,764 | **880** | **915** |
| Feature reduction | 0.00% | **50.11%** | **48.13%** |
| GA search time | N/A | 1713.1 s | 3038.4 s |
| SVM train time | 124.6 s | **59.2 s** | **51.4 s** |
| SVM inference time | 63.961 s | **31.507 s** | **33.454 s** |

---

## 8. Analysis and Discussion

### Did GA improve classification?
**No — in terms of raw accuracy and F1**, the GA-selected models scored slightly lower than the baseline (by ~0.8–1.0%). However, this minor performance drop comes with significant practical benefits.

### Key Findings

**Feature Reduction:**  
GA-1 reduced features by **50.11%** (from 1,764 → 880) and GA-2 by **48.13%** (→ 915). This means the classifier needs only half the features to achieve nearly the same accuracy.

**Speed Improvement:**  
- SVM training dropped from **124.6s → 59.2s** (GA-1) — a **2.1× speedup**
- Inference dropped from **63.96s → 31.5s** (GA-1) — a **2.03× speedup**

**Tradeoff:**  
The GA search itself is computationally expensive (1713–3038 seconds). This is a one-time cost — once the best feature mask is found, it can be reused for future inference at reduced cost.

### Config Comparison
GA-Config-2 (larger population, higher mutation) converged to a slightly better fitness score (0.4752 vs 0.4702) but selected more features (915 vs 880) and took ~78% longer to run. GA-Config-1 is more efficient overall.

### Strengths of GA Feature Selection
- No domain knowledge needed — purely data-driven search
- Can discover non-obvious feature combinations
- Reduces model complexity and inference time
- Works with any downstream classifier

### Limitations
- GA search is computationally expensive
- With only 5% training subsample for fitness, the proxy may not perfectly rank feature subsets
- Results depend on random seed — different runs may yield slightly different masks
- 20 generations may not be enough for full convergence

---

## 9. Conclusion

GA-based feature selection successfully reduced HOG feature dimensionality by ~50% while sacrificing less than 1% in Macro-F1 and achieving a **2× speedup** in both training and inference. Whether this tradeoff is worthwhile depends on the deployment context — in real-time medical screening systems, inference speed matters critically, making the GA approach practically valuable despite a marginal accuracy drop.

---

## 10. References

- Yang, J., & Honavar, V. (1998). Feature subset selection using a genetic algorithm. *IEEE Intelligent Systems*, 13(2), 44–49.
- Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *CVPR 2005*.
- Yang, J., et al. (2023). MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data*, 10, 41.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
