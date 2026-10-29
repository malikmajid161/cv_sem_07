# CS-482 Computer Vision — Lab Portfolio

**Student:** Muhammad Majid Ali | FA23-BAI-045  
**University:** COMSATS University Islamabad, Wah Campus  
**Course:** CS-482 Computer Vision

---

## Repository Structure

```
cv_sem_07/
├── lab 1/          — Transfer Learning for Skin-Lesion Classification
├── lab 2/          — Effect of Image Filtering on Classification
├── LAB 3/          — Edge Detection Techniques & Classification Impact
├── lab 5/          — Feature Extraction & Dimensionality Reduction
├── lab 6/          — Landmark Recognition Using SIFT
├── lab 7/          — HOG + Genetic Algorithm Feature Selection
└── lab 4/   — Canny Edge Detection on Skin Lesion Images
```

---

## Lab 1 — Transfer Learning for Skin-Lesion Classification

**Notebook:** `lab 1/lab_1.ipynb`  
**Report:** `lab 1/results_explanation.md`

### Overview
Explored 8 pretrained ImageNet models for 7-class skin-lesion classification on the **HAM10000** dataset (10,015 dermoscopic images). Used a two-phase fine-tuning strategy: freeze backbone + train head (Phase 1), then unfreeze all layers and fine-tune end-to-end (Phase 2). Employed weighted random sampling, label smoothing, cosine annealing, and mixed-precision training.

### Key Results

| Model | Accuracy (%) | AUC (%) | Params (M) |
|---|---|---|---|
| ResNet50 | **80.25** | **95.55** | 23.52 |
| EfficientNet-B0 | 77.91 | 94.05 | 4.02 |
| ResNet18 | 77.98 | 94.93 | 11.18 |
| AlexNet | 67.54 | 92.38 | 57.03 |

**Part 2 — Deep Features + Classical Classifiers** (ResNet50 as backbone):

| Classifier | Accuracy (%) | AUC (%) |
|---|---|---|
| XGBoost | **80.14** | **95.96** |
| Random Forest | 78.94 | 95.84 |
| Logistic Regression | 78.54 | 94.26 |
| Decision Tree | 69.08 | 77.24 |

### Key Findings
- **ResNet50** achieved best accuracy (80.25%) — residual connections help gradient flow during fine-tuning
- **EfficientNet-B0** (15 MB, 0.38G FLOPs) achieved 77.91% — best choice for deployment
- **XGBoost on ResNet50 deep features** nearly matched end-to-end deep learning (80.14% vs 80.25%)
- Deeper is not always better: ResNet101 (77.04%) underperformed ResNet50 despite more layers

---

## Lab 2 — Effect of Image Filtering on Classification

**Notebook:** `lab 2/lab 2.ipynb`  
**Report:** `lab 2/explanation.md`

### Overview
Investigated how 5 spatial-domain image filters (Average, Gaussian, Median, Sharpening, Sobel) affect the classification performance of the top 3 models from Lab 1 on the **HAM10000** dataset.

### Key Results

| Model | Best Filter | Best Accuracy (%) | Baseline (%) | Worst Filter |
|---|---|---|---|---|
| ResNet101 + XGBoost | Sharpening | 82.12 | 81.21 | Sobel (73.64%) |
| ResNet101 fine-tuned | Sharpening | 81.72 | 79.29 | Sobel (65.76%) |
| DenseNet121 fine-tuned | No Filter | 81.31 | 81.31 | Sobel (73.84%) |

### Key Findings
- **Sharpening** improved ResNet101 + XGBoost accuracy by +0.91% — enhances structural edges CNNs rely on
- **Sobel** edge filter caused the largest drop across all models (–7.5% to –13.5%) by stripping color/texture
- **DenseNet121** preferred raw images — its dense connections already capture sufficient edge information
- Smoothing filters (Average, Gaussian, Median) uniformly degraded performance by blurring fine lesion textures

---

## LAB 3 — Edge Detection Techniques & Classification Impact

**Notebook:** `LAB 3/lab_3.ipynb`  
**Report:** `LAB 3/Lab_03_Report.md`

### Overview
Systematic study of classical edge detectors (Sobel, Prewitt, Laplacian, LoG, Canny) and their effect on classification when used as the input representation vs. raw images. Tested on SVM, Random Forest, KNN, ResNet101, and DenseNet121.

### Key Results

| Model | Raw Accuracy (%) | Filtered Accuracy (%) | Edge Map Accuracy (%) |
|---|---|---|---|
| DenseNet121 | **81.92** | 80.20 | 46.06 |
| CNN ResNet101 | 79.80 | **80.00** | 35.35 |
| SVM | 77.58 | 77.07 | 67.88 |
| Random Forest | 72.12 | 71.62 | 68.79 |
| KNN | 70.20 | 70.71 | 63.33 |

### Canny Optimal Configuration
- **Best setting:** Low=50, High=150, Kernel=3×3 (Canny-2) — balanced edge quality with minimal noise
- Low thresholds capture too many false edges; high thresholds break true borders into fragments

### Key Findings
- Edge maps caused catastrophic accuracy drops for CNNs (35–46%) since CNNs need color, texture, and gradients
- **Canny** is most robust to noise due to built-in Gaussian smoothing + hysteresis thresholding
- **Laplacian** is most noise-sensitive (second-order derivatives amplify high-frequency noise)
- Pre-smoothing (Gaussian for Gaussian noise, Median for salt-and-pepper) fully restores edge quality
- CNNs automatically learn edge-like filters in early layers — manual edge maps are redundant and lossy

---

## Lab 5 — Feature Extraction & Dimensionality Reduction

**Notebook:** `lab 5/lab05.ipynb`

### Overview
Explored classical and learned feature extraction methods paired with dimensionality reduction techniques for image classification. Covers HOG, LBP, SIFT features combined with PCA and LDA dimensionality reduction, evaluated on downstream classifiers.

---

## Lab 6 — Landmark Recognition Using SIFT

**Notebook:** `lab 6/lab06.ipynb`  
**Summary:** `lab 6/lab06_summary.md`

### Overview
Built an automated landmark recognition and localization system using **Scale-Invariant Feature Transform (SIFT)**. The system matches a query photo against a 5-landmark reference database (Eiffel Tower, Taj Mahal, Great Wall, Statue of Liberty, Colosseum) and localizes the matched region using homography.

### System Pipeline
```
Query Image → SIFT Extraction → BF Matching (KNN + Lowe's Ratio Test)
                                        ↓
              Reference DB → SIFT Features → Ranked Candidates
                                        ↓
                          RANSAC Homography → Localized Region
```

### Parameters

| Parameter | Value | Purpose |
|---|---|---|
| Lowe's ratio | 0.75 | Filter ambiguous matches |
| Min good matches | 10 | Reject unknown landmarks |
| RANSAC reprojection | 5.0 px | Inlier threshold |
| Min RANSAC inliers | 10 | Reject unreliable homography |

### Test Conditions Evaluated
- Original image, Rotated, Resized, Reduced Brightness, Different Viewpoint, Unknown landmark rejection

### Key Findings
- SIFT is invariant to rotation and scale — matched correctly under both transformations
- Brightness reduction decreases keypoint density and may fall below match threshold
- Large viewpoint changes (>45°) produce insufficient RANSAC inliers for reliable homography
- For production scale: FLANN matcher or vocabulary tree indexing needed for speed

---

## Lab 7 — HOG + Genetic Algorithm Feature Selection

**Notebook:** `lab 7/lab07_hog_ga.ipynb`  
**Explanation:** `lab 7/explanation.md`

### Overview
Applied **Histogram of Oriented Gradients (HOG)** feature extraction to BloodMNIST (8-class blood cell classification) and used a **Genetic Algorithm (GA)** to select an optimal subset of HOG features. Goal: maintain classification performance while reducing feature dimensionality by ~50%.

### Dataset — BloodMNIST

| Split | Samples | Classes |
|---|---|---|
| Train | 11,959 | 8 blood cell types |
| Validation | 1,715 | Basophil, Eosinophil, Erythroblast, IG, |
| Test | 3,421 | Lymphocyte, Monocyte, Neutrophil, Platelet |

### HOG Parameters
- Orientations=9, Pixels/cell=8×8, Cells/block=2×2, L2-Hys normalization
- Total feature length: **1,764 features** per image

### Genetic Algorithm Design
- **Chromosome:** Binary mask of length 1,764 (1=keep, 0=discard)
- **Fitness:** `val_Macro-F1 − 0.05 × (selected/1764)`
- **Inner classifier:** LinearSVC (fast proxy for fitness evaluation)
- **Speed tricks:** 5% training subsample + fitness caching
- **Operators:** Tournament selection (k=3), one-point crossover (p=0.80), bit-flip mutation, elitism (top-2)

### Results

| Metric | Baseline (1764 features) | GA-1 (880 features) | GA-2 (915 features) |
|---|---|---|---|
| Accuracy | 81.18% | 80.42% | 80.30% |
| Macro-F1 | 80.71% | 79.88% | 79.65% |
| Feature reduction | — | **50.1%** | **48.1%** |
| SVM train time | 124.6 s | 59.2 s | 51.4 s |
| SVM inference time | 63.96 s | 31.5 s | 33.5 s |

### Key Findings
- GA reduced features by ~50% with only ~0.8% drop in Macro-F1
- Achieved **2× speedup** in both training and inference at minimal accuracy cost
- GA search is a one-time cost (1,713–3,038 s); the saved mask can be reused permanently
- Config-1 (pop=30, mut=0.01) is more efficient than Config-2 despite slightly lower fitness

---

## Lab 4 — Canny Edge Detection on Skin Lesion Images

**Notebook:** `lab 4/canny-edge-det-on-skin-lesion.ipynb`  
**Results:** `lab 4/results.md`

### Overview
Applied preprocessing + edge detection pipelines to 5 dermoscopy skin lesion images. Goal: segment lesion boundaries using Canny and Sobel detectors after various spatial filters, and quantify edge quality and segmentation accuracy.

### Method Comparison

| Method | Noise Handling (σ↓) | Edge F1 (↑) | Boundary IoU (↑) | Overall Rank |
|---|---|---|---|---|
| **Average + Sobel** | **0.267** | **0.246** | 0.509 | **1** |
| **Gaussian + Sobel** | 0.271 | 0.237 | **0.516** | 2 |
| Median + Sobel | 0.325 | 0.235 | 0.506 | 3 |
| Median + Canny | 0.325 | 0.211 | 0.448 | 4 |
| Gaussian + Canny | 0.271 | 0.175 | 0.392 | 5 |
| Original + Canny | 1.007 | 0.163 | 0.429 | 7 |

### Optimal Canny Thresholds per Image

| Image | Best Filter | Edge Method | Area (px) | Perimeter (px) |
|---|---|---|---|---|
| Image 1 | Gaussian 7×7 | Canny 10–30 | 119,405 | 5,175.9 |
| Image 2 | Gaussian 7×7 | Canny 5–15 | 145,493 | 3,168.8 |
| Image 3 | Gaussian 7×7 | Canny 10–30 | 228,267 | 2,727.9 |
| Image 4 | Gaussian 7×7 | Canny 30–70 | 3,079 | 347.4 |
| Image 5 | Gaussian 7×7 | Canny 5–15 | 210,206 | 6,105.0 |

### Key Findings
- **Sobel** consistently outperformed **Canny** in this task — better IoU and edge F1 across all filter pairings
- Gaussian smoothing before edge detection is critical — reduces sensor noise and hair artifacts
- High thresholds (50–100) were too aggressive for already-smoothed images; lower ranges (10–30) worked best
- Challenges: hair artifacts, vignette circles, fuzzy lesion borders, and internal pigmentation variation
- Future improvement: DullRazor hair removal, U-Net segmentation (ISIC-trained), adaptive threshold selection

---

## Dependencies

```bash
pip install torch torchvision medmnist scikit-learn scikit-image matplotlib numpy pandas opencv-contrib-python-headless xgboost
```

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/malikmajid161/cv_sem_07.git
   ```
2. Open any notebook in **Google Colab** or **Jupyter**
3. Run cells sequentially from top to bottom
4. Datasets are downloaded automatically inside each notebook (HAM10000 via kagglehub, BloodMNIST via medmnist)

> **Note:** Lab 7 GA search takes 30–50 minutes on a CPU. Use GPU runtime in Colab for faster execution.

---

## References

- Gonzalez, R. C. & Woods, R. E. (2018). *Digital Image Processing*, 4th ed. Pearson.
- He, K. et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Dalal, N. & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. *CVPR*.
- Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. *IJCV*, 60(2), 91–110.
- Yang, J. et al. (2023). MedMNIST v2. *Scientific Data*, 10, 41.
- Canny, J. (1986). A Computational Approach to Edge Detection. *IEEE TPAMI*.
