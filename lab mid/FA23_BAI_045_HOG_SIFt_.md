# HOG vs SIFT for Blood Cell Classification — Results and Evaluation

| Field | Details |
|---|---|
| Student | Muhammad Majid Ali |
| Registration No. | FA23-BAI-045 |
| Notebook | `FA23_BAI_045_Muhammad_Majid_Ali_LabMid.ipynb` |
| Dataset | BloodMNIST (MedMNIST) |
| Task | 4-class blood cell classification using classical features + Linear SVM |

---

## 1. Objective

This lab compares two classical computer-vision pipelines on four BloodMNIST classes:

1. **HOG + Linear SVM**: Histogram of Oriented Gradients features.
2. **Fixed-grid SIFT + Linear SVM**: SIFT descriptors computed on a fixed 6 × 6 grid of keypoints.

For each pipeline:

- The SVM parameter `C` is selected on the **validation set**.
- The frozen models are then evaluated on **clean**, **rotated (20°)**, and **noisy (σ = 0.05)** test images.

---

## 2. Experimental Setup

### 2.1 Registration-Based Parameters

| Parameter | Value | Rule Applied |
|---|---|---|
| Seed | 45 | From registration number FA23-BAI-0**45** |
| HOG cell size | 16 × 16 pixels | Last digit (5) is odd |
| SIFT grid | 6 × 6 (36 keypoints) | Last digit (5) is in range 5–9 |

### 2.2 Fixed Settings

| Setting | Value |
|---|---|
| Classes (label IDs) | 1 = eosinophil, 5 = monocyte, 6 = neutrophil, 7 = platelet |
| Source image size | 64 × 64 RGB (BloodMNIST `size=64`) |
| Processed image size | 128 × 128 grayscale |
| HOG orientations | 9 |
| HOG block size | 2 × 2 cells |
| HOG block normalization | L2-Hys |
| SIFT keypoint size | 32 |
| SIFT descriptor length | 128 per keypoint |
| Classifier | `LinearSVC` (`dual=True`, `max_iter=20000`, `random_state=45`) |
| Candidate `C` values | 0.1, 1, 10 |
| Rotation angle | 20° about image centre, black (0) border fill |
| Gaussian noise | mean 0, σ = 0.05 on [0, 1] pixel scale, clipped |

---

## 3. Dataset Summary

| Split | Images per Class | Total Images | Image Array Shape | Label Shape |
|---|---|---|---|---|
| Train | 100 | 400 | (400, 128, 128) | (400,) |
| Validation | 30 | 120 | (120, 128, 128) | (120,) |
| Test | 30 | 120 | (120, 128, 128) | (120,) |

All three splits are **class-balanced** (equal images per class), so accuracy is not inflated by a majority class.

---

## 4. Feature Extraction

### 4.1 HOG Features

**Feature length calculation**

| Step | Calculation | Result |
|---|---|---|
| Cells per side | 128 ÷ 16 | 8 |
| Blocks per side (stride = 1 cell) | 8 − 2 + 1 | 7 |
| Total blocks | 7 × 7 | 49 |
| Values per block | 2 × 2 × 9 | 36 |
| **HOG features per image** | 49 × 36 | **1,764** |

Training feature matrix shape: **(400, 1764)**

### 4.2 Detected (Normal) SIFT Keypoints

Standard SIFT detection was run on the 400 training images to check whether its output could feed an SVM directly.

| Statistic | Value |
|---|---|
| Average keypoints per image | 72.67 |
| Minimum keypoints | 8 |
| Maximum keypoints | 229 |
| Images with zero keypoints | 0 / 400 |

**Finding:** The keypoint count varies widely between images (8 to 229). A linear SVM requires one fixed-length vector per sample, so detected SIFT cannot be used directly.

### 4.3 Fixed-Grid SIFT Features

To get a fixed-length vector, one keypoint is placed at the centre of each cell of a 6 × 6 grid. SIFT descriptors are then computed at those fixed positions.

| Step | Calculation | Result |
|---|---|---|
| Grid keypoints | 6 × 6 | 36 |
| Grid spacing | 128 ÷ 6 | ≈ 21.33 pixels |
| Descriptor length per keypoint | — | 128 |
| **SIFT features per image** | 36 × 128 | **4,608** |

Training feature matrix shape: **(400, 4608)**

> Descriptors are divided by 255 for rescaling. This does not guarantee every value lies in [0, 1].

### 4.4 Feature Comparison

| Property | HOG | Fixed-grid SIFT |
|---|---|---|
| Features per image | 1,764 | 4,608 |
| What it describes | Dense gradient-orientation histograms over the whole image | Local gradient descriptors at 36 fixed locations |
| Normalisation | L2-Hys per block | Division by 255 |
| Keypoint orientation | Not applicable | Fixed at 0 (no orientation assignment) |

---

## 5. Model Selection (Validation Set)

Each `C` value was trained on the training set and scored on the validation set. The model with the **highest validation accuracy** was kept. Ties keep the earlier (smaller) `C`.

| `C` | HOG + SVM | SIFT + SVM |
|---|---|---|
| **0.1** | **76.67% (92/120)** | **81.67% (98/120)** |
| 1 | 75.00% (90/120) | 78.33% (94/120) |
| 10 | 75.00% (90/120) | 80.00% (96/120) |
| **Selected `C`** | **0.1** | **0.1** |

**Observation:** Both pipelines selected the strongest regularisation (`C = 0.1`). With 400 training samples and 1,764 or 4,608 features, there are more features than samples. In that setting a smaller `C` (simpler margin) generalising better is the expected result.

> Macro-F1 is printed for each `C`, but selection is made on accuracy only.

---

## 6. Test Results

The models were **frozen** after validation. Rotation and noise were applied only to test images before feature extraction. No retraining was done on transformed data.

| Model | Clean | Rotated 20° | Noisy (σ = 0.05) |
|---|---|---|---|
| HOG + SVM | 72.50% (87/120) | **63.33% (76/120)** | 36.67% (44/120) |
| Grid SIFT + SVM | **78.33% (94/120)** | 49.17% (59/120) | **78.33% (94/120)** |

---

## 7. Evaluation and Analysis

### 7.1 Clean Performance

Grid SIFT outperforms HOG on clean test data (78.33% vs 72.50%). It also led on validation (81.67% vs 76.67%). The ranking is consistent across both splits. The validation-to-test drop is about 3–4 pp for both models, which indicates mild optimism from selecting `C` on validation, not severe overfitting.

### 7.2 Rotation Robustness

HOG is clearly more robust to 20° rotation: it drops by 9.17 pp, while Grid SIFT drops by 29.17 pp. Likely reasons:

- **Grid SIFT lost its rotation invariance.** Standard SIFT is rotation-invariant because each keypoint is assigned a dominant orientation. In this pipeline every grid keypoint has orientation fixed at 0. Rotating the image therefore changes every descriptor, and each grid cell now covers a different part of the cell.
- **Black corners from rotation.** `warpAffine` fills uncovered corners with 0. Corner grid keypoints then describe artificial edges, which affects the fixed-position SIFT layout more directly.
- **HOG pools over large cells.** HOG's 16 × 16 cells and overlapping normalised blocks average gradients over wider regions. This makes a 20° shift partly absorbed rather than fully changing the descriptor.

### 7.3 Noise Robustness

SIFT showed **no accuracy drop** (0.00 pp) under σ = 0.05 Gaussian noise, while HOG dropped by 35.83 pp to 36.67%, close to the 25% chance level. Likely reasons:

- SIFT computes gradients after Gaussian smoothing at the keypoint's scale. With keypoint size 32, smoothing is strong enough to suppress pixel-level noise.
- HOG uses raw pixel gradients with no prior smoothing. Noise adds random gradient directions to every histogram bin, and L2-Hys normalisation then rescales this noise-dominated histogram rather than removing it.

### 7.4 Prediction vs Outcome

| Transformation | Prediction | Outcome | Result |
|---|---|---|---|
| Rotation | SIFT more robust | HOG more robust | Incorrect |
| Noise | HOG weaker | HOG much weaker | Correct |

The prediction assumed SIFT keeps its rotation invariance. The fixed-grid implementation removed it, which explains the incorrect half.

### 7.5 Overall Verdict

| Criterion | Winner |
|---|---|
| Clean accuracy | Grid SIFT |
| Noise robustness | Grid SIFT (strongly) |
| Rotation robustness | HOG |
| Feature compactness | HOG (1,764 vs 4,608) |

Grid SIFT is the better overall pipeline for this setup. It wins two of three test conditions and has the larger margin. Its weakness under rotation is caused by the implementation choice (fixed orientation), not by SIFT itself.

---

## 8. Key Implementation Decisions

| Decision | Reason |
|---|---|
| Balanced sampling from official splits | Prevents class imbalance and keeps test data independent |
| `C` chosen on validation, not test | Using test data for tuning would leak it into model selection and inflate reported performance |
| Fixed-grid SIFT instead of detected SIFT | Detected keypoint counts vary (8–229); the SVM needs a fixed-length vector |
| Same `fit_select_C` function for both pipelines | Ensures a fair, identical selection procedure |
| Models frozen before robustness tests | Measures true robustness, not adaptation to transformed data |
| Seeded generators for sampling and noise | Makes subset selection and noise repeatable |

---

## 9. Limitations

- **Small sample size.** 30 test images per class means one image equals 3.33 pp of per-class accuracy, so small differences are not statistically reliable.
- **Single run.** Results come from one seed. Different subsets may change the margins.
- **Single transformation strength.** Only 20° rotation and σ = 0.05 noise were tested, so these results do not describe robustness curves.
- **Upsampling.** Images are enlarged from 64 × 64 to 128 × 128; no real detail is added.
- **Grayscale conversion.** Colour staining, which differs between blood cell types (e.g. eosinophil granules), is discarded.
- **Selection by accuracy only.** Macro-F1 is reported but not used to choose `C`.
