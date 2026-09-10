# Skin Lesion Classification Using Transfer Learning on HAM10000

## Dataset Overview

The **HAM10000** (Human Against Machine with 10,000 training images) dataset was used for this experiment. It contains **10,015 dermatoscopic images** of skin lesions across **7 classes**:

| Class Code | Disease Name | Type |
|---|---|---|
| akiec | Actinic Keratoses | Pre-cancerous |
| bcc | Basal Cell Carcinoma | Cancerous |
| bkl | Benign Keratosis-like Lesions | Benign |
| df | Dermatofibroma | Benign |
| mel | Melanoma | Cancerous |
| nv | Melanocytic Nevi | Benign |
| vasc | Vascular Lesions | Benign |

### Data Split
| Split | Samples |
|---|---|
| Training | 4,109 |
| Validation | 1,120 |
| Testing | 2,241 |
| **Total** | **7,470** |

---

## Methodology

### Transfer Learning Strategy
All models were pretrained on **ImageNet** (1.2 million images, 1000 classes). A **two-phase training** approach was used:

- **Phase 1 (4 epochs, LR = 1e-3):** Freeze backbone, train only classifier head and last block
- **Phase 2 (6 epochs, LR = 1e-5):** Unfreeze all layers, fine-tune entire network with low learning rate

This prevents destroying pretrained weights while allowing the model to adapt to skin lesion features.

### Training Configuration
| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | CrossEntropy with Label Smoothing (0.1) |
| LR Scheduler | Cosine Annealing |
| Weight Decay | 1e-4 |
| Total Epochs | 10 per model |
| Mixed Precision | Yes (GradScaler) |

### Data Augmentation
To handle class imbalance and improve generalization:
- Random horizontal and vertical flipping
- Random rotation (±30°)
- Color jitter (brightness, contrast, saturation, hue)
- Random crop from 256×256 to 224×224
- Weighted random sampling to balance classes

---

## Table 1 — Transfer Learning Model Comparison

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
|---|---|---|---|---|---|
| AlexNet | 67.54 | 81.51 | 67.54 | 71.90 | 92.38 |
| VGG16 | 68.88 | 81.61 | 68.88 | 72.85 | 92.76 |
| VGG19 | 70.75 | 80.29 | 70.75 | 74.07 | 92.76 |
| ResNet18 | 77.98 | 83.21 | 77.98 | 79.94 | 94.93 |
| ResNet50 | **80.25** | **83.76** | **80.25** | **81.61** | **95.55** |
| ResNet101 | 77.04 | 82.35 | 77.04 | 79.01 | 93.87 |
| DenseNet121 | 76.17 | 82.10 | 76.17 | 78.35 | 94.04 |
| EfficientNet-B0 | 77.91 | 82.66 | 77.91 | 79.64 | 94.05 |

### Analysis

**ResNet50 achieved the best overall accuracy of 80.25%** with an AUC of 95.55%, making it the top performer among all transfer learning models tested.

**Why ResNet50 performed best:**
- Residual connections allow gradients to flow effectively during fine-tuning
- 50-layer depth provides enough capacity to learn complex skin lesion features
- Better balance between model size (23.52M params) and accuracy compared to deeper ResNet101

**AlexNet performed worst (67.54%)** because:
- It is the oldest architecture (2012) with simple sequential convolutions
- No residual connections or batch normalization
- Less effective feature extraction for medical images

**VGG16 and VGG19** performed similarly (68.88% and 70.75%) because they share the same architecture style — just different depths. Their large size (134M and 139M parameters) did not translate to better accuracy, suggesting overfitting on the relatively small HAM10000 dataset.

**EfficientNet-B0 (77.91%)** performed competitively despite having only 4.02M parameters — showing that efficient architecture design matters more than raw size.

**ResNet101 (77.04%)** performed worse than ResNet50 (80.25%) despite being deeper. This is a common phenomenon called **diminishing returns** — more layers do not always help when the dataset is small and training time is limited.

---

## Table 2 — Deep Features + Classical Classifiers

> Features extracted from **ResNet50** (best backbone) penultimate layer (2048-dimensional feature vector)

| Feature Extractor | Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
|---|---|---|---|---|---|---|
| Deep Features | Logistic Regression | 78.54 | 83.99 | 78.54 | 80.38 | 94.26 |
| Deep Features | Decision Tree | 69.08 | 80.46 | 69.08 | 72.54 | 77.24 |
| Deep Features | Random Forest | 78.94 | 86.80 | 78.94 | 81.33 | 95.84 |
| Deep Features | K-Nearest Neighbors | 75.15 | 84.34 | 75.15 | 77.97 | 92.58 |
| Deep Features | Linear SVM | 77.60 | 82.96 | 77.60 | 79.43 | 94.53 |
| Deep Features | RBF-SVM | 78.80 | 84.70 | 78.80 | 80.76 | 94.20 |
| Deep Features | XGBoost | **80.14** | **86.60** | **80.14** | **82.13** | **95.96** |

### Analysis

**XGBoost achieved the highest accuracy (80.14%)** among all classical classifiers, with the best AUC of 95.96%. This is because:
- Gradient boosting builds an ensemble of weak learners sequentially
- Handles high-dimensional features (2048-dim) effectively
- Less prone to overfitting compared to single Decision Trees

**Decision Tree performed worst (69.08%)** because:
- Single trees overfit high-dimensional feature spaces
- No ensemble mechanism to correct errors
- Low AUC (77.24%) means poor class separation

**Random Forest (78.94%)** performed well because it averages many decision trees, reducing variance. Its AUC of 95.84% is the second highest.

**Logistic Regression (78.54%)** surprisingly performed well — showing that ResNet50 features are linearly separable to a high degree, meaning the deep features are already well-organized in feature space.

**Key insight:** Deep features from ResNet50 + XGBoost (80.14%) nearly matched the full end-to-end ResNet50 (80.25%), showing that classical classifiers can be highly effective when given powerful deep features.

---

## Table 3 — Computational Efficiency Comparison

| Model | Parameters (M) | Model Size (MB) | FLOPs (G) | Inference Time (ms) | Accuracy (%) |
|---|---|---|---|---|---|
| AlexNet | 57.03 | 217.56 | 0.71 | 2.12 | 67.54 |
| VGG16 | 134.29 | 512.27 | 15.47 | 12.51 | 68.88 |
| VGG19 | 139.60 | 532.53 | 19.63 | 12.98 | 70.75 |
| ResNet18 | 11.18 | 42.65 | 1.82 | 2.45 | 77.98 |
| ResNet50 | 23.52 | 89.73 | 4.13 | 5.74 | 80.25 |
| ResNet101 | 42.51 | 162.18 | 7.86 | 17.17 | 77.04 |
| DenseNet121 | 6.96 | 26.55 | 2.90 | 21.33 | 76.17 |
| EfficientNet-B0 | 4.02 | 15.32 | 0.38 | 9.02 | 77.91 |

### Analysis

**EfficientNet-B0 is the most efficient model:**
- Smallest model size: **15.32 MB** (35× smaller than VGG16)
- Fewest FLOPs: **0.38G** (41× fewer than VGG16)
- Still achieves **77.91% accuracy** — only 2.34% less than the best model
- Best choice for **deployment on mobile or embedded devices**

**VGG16 and VGG19 are the least efficient:**
- VGG19 uses **532 MB** of storage — largest model
- VGG19 requires **19.63G FLOPs** per inference — most computationally expensive
- Yet only achieves 70.75% accuracy — poor accuracy-to-cost ratio

**ResNet18 offers the best speed:**
- Only **2.45ms inference time** — fastest after AlexNet
- 77.98% accuracy with just 11.18M parameters
- Best choice when **speed is the priority**

**DenseNet121 has a unique property:**
- Smallest parameter count among deep models: **6.96M params**
- But slowest inference among ResNet/Dense family: **21.33ms**
- This is because dense connections require many feature map concatenations

**Accuracy vs Efficiency Trade-off:**

```
High Accuracy  ←————————————————→  High Efficiency
ResNet50       ResNet18    EfficientNet-B0
(80.25%)       (77.98%)    (77.91%, 15MB)
```

---

## Overall Conclusions

### 1. Best Model for Accuracy
**ResNet50** with **80.25% accuracy** and **95.55% AUC** is the top performer. Its residual architecture and balanced depth make it ideal for medical image classification.

### 2. Best Model for Deployment
**EfficientNet-B0** at only **15.32 MB** and **0.38G FLOPs** provides near-best accuracy (77.91%) with minimal computational cost — ideal for real-world clinical tools.

### 3. Best Classical Approach
**ResNet50 features + XGBoost** achieves **80.14% accuracy** — almost matching full end-to-end deep learning. This hybrid approach is useful when compute resources are limited.

### 4. Key Observations
- Deeper is not always better (ResNet101 < ResNet50)
- Large models (VGG) do not guarantee high accuracy on small datasets
- AUC scores (92-95%) are consistently higher than accuracy, confirming good class discrimination despite HAM10000's class imbalance
- Weighted sampling successfully mitigated the imbalance (nv class dominates with 67% of data)

---

## Metrics Explanation

| Metric | Formula | Meaning |
|---|---|---|
| **Accuracy** | Correct / Total | Overall correct predictions |
| **Precision** | TP / (TP + FP) | How many predicted positives are actually positive |
| **Recall** | TP / (TP + FN) | How many actual positives were correctly found |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean of Precision and Recall |
| **AUC** | Area under ROC curve | Ability to distinguish between classes (1.0 = perfect) |

> All metrics computed as **weighted average** across all 7 classes to account for class imbalance.
