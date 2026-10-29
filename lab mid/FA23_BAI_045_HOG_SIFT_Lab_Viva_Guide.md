HOG vs SIFT Blood Cell Classification — Code Explanation for Instructor

Student: Muhammad Majid Ali
Registration: FA23-BAI-045
Notebook: FA23_BAI_045_Muhammad_Majid_Ali_LabMid.ipynb

1. Lab overview

This lab compares two classical computer-vision pipelines for classifying four BloodMNIST blood-cell classes:

HOG feature extraction followed by a linear SVM.

SIFT feature extraction on a fixed grid followed by a linear SVM.

The notebook selects the SVM parameter C using validation data, then evaluates the frozen models on clean, rotated, and noisy test images.

Purpose of the notebook: The notebook implements two classical image-classification pipelines and compares their results. The sections below explain why each code block is present, what the important parameters mean, and how to interpret its output.

2. Libraries and imports

!pip install medmnist opencv-python scikit-image scikit-learn matplotlib numpy

In a notebook, ! runs a shell command. This installs the libraries. medmnist provides the dataset; OpenCV processes images and supplies SIFT; scikit-image supplies HOG; scikit-learn supplies SVM and metrics; Matplotlib draws figures; NumPy handles arrays.

import numpy as np
import cv2
import matplotlib.pyplot as plt
from medmnist import BloodMNIST
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

These imports make the required functions available. Aliases such as np, cv2, and plt are conventional short names.

3. Settings and parameters

REG_NO = "FA23-BAI-045"
SEED = 45
HOG_CELL = (16, 16)
SIFT_GRID = 6

REG_NO identifies the student. SEED=45 follows the registration-number rule and makes random subset selection repeatable. Because the last registration digit is odd, the rule gives a HOG cell of 16 × 16. Since the last digit is in 5–9, the SIFT grid is 6 × 6, which gives 36 keypoints.

CLASSES = {1: "eosinophil", 5: "monocyte", 6: "neutrophil", 7: "platelet"}
N_TRAIN, N_VAL, N_TEST = 100, 30, 30
IMG_SIZE = 128
HOG_ORIENT = 9
HOG_BLOCK = (2, 2)
C_VALUES = [0.1, 1, 10]
ROT_ANGLE = 20
NOISE_SIGMA = 0.05
KP_SIZE = 32

These variables collect the assignment settings in one place. There are four selected label IDs. Each class contributes 100 training, 30 validation, and 30 test images. Images are resized to 128 × 128. HOG uses 9 orientation bins and blocks of 2 × 2 cells. The SVM tries three C values. Robustness tests use 20° rotation and Gaussian noise with sigma 0.05 on normalized pixel values. SIFT grid keypoints have size 32.

np.random.seed(SEED)

Seeds NumPy's global random generator. The data loader below uses its own seeded generator (rng), so that is the one directly controlling its image selection.

4. Loading and preparing BloodMNIST

rng = np.random.RandomState(SEED)

Creates a repeatable random generator. The same seed, dataset, and code produce the same selected indices.

def load_subset(split, n_per_class):
    ds = BloodMNIST(split=split, download=True, size=64)

Defines a reusable loader. split selects train, validation, or test data. download=True downloads missing data. size=64 asks for the dataset's 64 × 64 version; the code later resizes images to 128 × 128.

imgs = ds.imgs
labels = ds.labels.flatten()

ds.imgs accesses the image array, originally shaped (N, 64, 64, 3) for N RGB images. ds.labels contains the associated class IDs. flatten() makes labels one-dimensional, shaped (N,).

X, y = [], []
for lab in CLASSES:
    idx = np.where(labels == lab)[0]
    chosen = rng.choice(idx, n_per_class, replace=False)

X collects images and y collects labels. The loop visits each chosen class. np.where finds indices for that label. rng.choice randomly picks the requested count; replace=False prevents selecting an index twice within that class selection.

for i in chosen:
    gray = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    X.append(gray)
    y.append(lab)

For each selected image, convert RGB to grayscale, resize it to 128 × 128, then append the image and its correct class label. Grayscale follows the assignment and simplifies gradient-based features. Resizing gives every image the same dimensions; it enlarges the 64 × 64 pixels and does not create new biological detail.

return np.array(X, dtype=np.uint8), np.array(y)

Returns NumPy arrays. uint8 stores image intensity as integers from 0 to 255.

X_train, y_train = load_subset("train", N_TRAIN)
X_val, y_val = load_subset("val", N_VAL)
X_test, y_test = load_subset("test", N_TEST)

Builds the three data splits. Four classes × 100 gives 400 training images; four × 30 gives 120 validation and 120 test images.

labs, counts = np.unique(y, return_counts=True)

Finds each unique label in y and counts its occurrences. The notebook prints shapes and per-class counts to verify the data setup.

Expected data shapes

Split

Images

Image-array shape

Label shape

Train

400

(400, 128, 128)

(400,)

Validation

120

(120, 128, 128)

(120,)

Test

120

(120, 128, 128)

(120,)

Each split has equal class counts: 100 train or 30 validation/test images for each of the four classes.

5. Displaying sample images

fig, axes = plt.subplots(len(CLASSES), 4, figsize=(10, 10))

Creates a four-row, four-column figure: one row per class and four example images per row.

The loop finds the first four training images for each label, displays them in grayscale, hides axes, and labels each row. tight_layout() adjusts spacing; show() renders the figure. This visual check helps confirm class selection and preprocessing.

6. HOG feature extraction

HOG means Histogram of Oriented Gradients. It summarizes local edge directions and strengths.

def extract_hog(img):
    return hog(img,
               orientations=HOG_ORIENT,
               pixels_per_cell=HOG_CELL,
               cells_per_block=HOG_BLOCK,
               block_norm="L2-Hys",
               feature_vector=True)

This extracts one feature vector from one image:

orientations=9: uses nine gradient-direction bins.

pixels_per_cell=(16,16): each HOG cell covers 16 × 16 pixels.

cells_per_block=(2,2): normalizes a group of four neighboring cells.

block_norm="L2-Hys": normalizes block values to reduce sensitivity to contrast/brightness changes.

feature_vector=True: returns one flat list of features for the classifier.

def extract_hog_all(X):
    return np.array([extract_hog(im) for im in X])

Applies HOG to every image and forms a feature matrix. Each row corresponds to one image.

F_train_hog = extract_hog_all(X_train)
F_val_hog = extract_hog_all(X_val)

Extracts training and validation features. Training features fit the model; validation features select C.

HOG feature-length calculation

Image = 128 × 128; cell = 16 × 16; block = 2 × 2 cells; orientations = 9.

Cells per side: 128 / 16 = 8.

Blocks per side with one-cell stride: 8 - 2 + 1 = 7.

Total blocks: 7 × 7 = 49.

Values per block: 2 × 2 × 9 = 36.

Total: 49 × 36 = 1,764 HOG features per image.

The HOG visualization calls hog(..., visualize=True). It returns both the descriptor and a visual representation of gradient structure. _, hog_img stores the visualization and ignores the descriptor using _.

7. SVM training and validation selection

def fit_select_C(F_tr, y_tr, F_va, y_va, name):
    best = None

Defines a reusable training routine and starts without a selected best model.

for C in C_VALUES:
    clf = LinearSVC(C=C, dual=True, max_iter=20000, random_state=SEED)
    clf.fit(F_tr, y_tr)
    pred = clf.predict(F_va)

For each C, construct a linear SVM, fit on training features and labels, and predict validation labels. max_iter allows the optimizer enough iterations. C controls the penalty for training errors: small C is more forgiving and regularizes more; large C penalizes errors more strongly. The code tests values instead of assuming one is best.

acc = accuracy_score(y_va, pred)
f1 = f1_score(y_va, pred, average="macro")

Calculates validation accuracy and macro-F1. Macro-F1 computes F1 for each class, then averages the four class scores equally.

if best is None or acc > best[1]:
    best = (C, acc, f1, clf, pred)

Keeps the model with the highest validation accuracy. Because the comparison uses strict >, an earlier smaller C stays selected if accuracy ties. Although macro-F1 is printed, selection in this code is by accuracy.

The remaining code creates and displays the validation confusion matrix, prints it, and returns the best classifier, C, and matrix. confusion_matrix(..., labels=list(CLASSES)) keeps the class order consistent. ConfusionMatrixDisplay draws the matrix with class names.

Why not choose C on test data? The test set is reserved for final evaluation. Tuning on it lets its results influence model selection and makes the reported test performance less independent.

Validation accuracy for each tested C

The notebook's validation outputs for all candidate SVM C values are:

C

HOG + SVM validation accuracy

SIFT + SVM validation accuracy

0.1

76.67% (92/120)

81.67% (98/120)

1

75.00% (90/120)

78.33% (94/120)

10

75.00% (90/120)

80.00% (96/120)

The selection code compares validation accuracy and chooses the highest score. Therefore, both pipelines select C = 0.1 in this run. Macro-F1 is also printed, but it does not determine the selected C.

8. Normal SIFT keypoint inspection

SIFT means Scale-Invariant Feature Transform. It detects distinctive keypoints and computes a 128-value descriptor for each keypoint.

sift = cv2.SIFT_create()

Creates OpenCV's SIFT detector/descriptor.

kps, des = sift.detectAndCompute(im, None)

Detects keypoints and computes descriptors. None means no mask limits the image region. In this keypoint-count loop, the code uses len(kps); descriptors are not used there.

kp_counts.append(len(kps))

Records the number of keypoints found in each image. The notebook then prints average, minimum, maximum, and zero-keypoint image count. In this run: average 72.67, minimum 8, maximum 229, zero images 0/400. These are measured outputs, not fixed settings.

The drawing cell uses cv2.drawKeypoints to overlay detected points on an image. DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS displays keypoint size and orientation as well.

Why is a variable keypoint count a problem? A standard SVM expects every sample to have the same number of features, while normal SIFT can return a different number of 128-value descriptors for each image.

9. Fixed-grid SIFT features

def make_grid_keypoints(n=SIFT_GRID, size=IMG_SIZE, kp_size=KP_SIZE):
    step = size / n
    pts = [(step * (j + 0.5), step * (i + 0.5))
           for i in range(n) for j in range(n)]

Divides the image into a regular n × n grid and places one keypoint at the center of each grid cell. step is the distance between grid positions.

return [cv2.KeyPoint(float(x), float(y), kp_size, 0) for (x, y) in pts]

Creates OpenCV keypoint objects at those coordinates, with size 32 and orientation argument 0.

grid_kps = make_grid_keypoints()

Creates 36 keypoints for the 6 × 6 grid.

def extract_sift_grid(img):
    _, des = sift.compute(img, grid_kps)
    assert des.shape == (SIFT_GRID**2, 128), des.shape
    return (des / 255.0).flatten()

Computes descriptors at the supplied fixed positions. The assertion checks the expected shape: 36 descriptors by 128 values. flatten() joins them into a 4,608-value vector. Division by 255 rescales descriptor values, but should not be described as guaranteeing values between 0 and 1.

def extract_sift_all(X):
    return np.array([extract_sift_grid(im) for im in X])

Extracts a fixed-length SIFT vector for every image and builds the feature matrix.

SIFT feature-length calculation

Grid points: 6 × 6 = 36.

Descriptor length per point: 128.

Total per image: 36 × 128 = 4,608 SIFT features.

The grid illustration draws these fixed points on one image to confirm placement.

10. Train SIFT + SVM

sift_model, sift_C, cm_sift = fit_select_C(
    F_train_sift, y_train, F_val_sift, y_val, "SIFT (grid) + SVM")

Reuses the same SVM selection procedure for SIFT. This makes comparison with HOG more consistent. The chosen value was C=0.1 for both pipelines in this run.

11. Rotation and Gaussian-noise transformations

noise_rng = np.random.RandomState(SEED)

Creates a repeatable random generator for noise.

def rotate_images(X, angle=ROT_ANGLE):
    M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), angle, 1.0)

Creates a rotation matrix around the image center. 1.0 means no resizing during rotation.

return np.array([cv2.warpAffine(im, M, (IMG_SIZE, IMG_SIZE), borderValue=0)
                 for im in X])

Rotates each image into a 128 × 128 output. Newly empty border pixels are filled with 0.

def add_noise(X, sigma=NOISE_SIGMA):
    Xf = X.astype(np.float32) / 255.0
    noisy = Xf + noise_rng.normal(0, sigma, Xf.shape)
    return (np.clip(noisy, 0, 1) * 255).round().astype(np.uint8)

Converts pixels to floating point on the 0–1 range, adds Gaussian random noise with mean 0 and standard deviation 0.05, clips values to the valid range, rescales to 0–255, rounds, and converts back to 8-bit pixels.

The visualization compares one clean, rotated, and noisy image to check that transformations look as expected.

12. Robustness evaluation

def evaluate(model, extractor, X, y):
    F = np.array([extractor(im) for im in X])
    return accuracy_score(y, model.predict(F))

Extracts features from the current images, predicts with an already-trained model, and returns accuracy. The models are frozen; they are not retrained on rotated or noisy test images.

The loop evaluates both pipelines on clean, rotated, and noisy test images. Pandas puts the results into a table. Accuracy drops are calculated as:

drop = clean accuracy - transformed accuracy.

A smaller drop means better robustness to that particular transformation.

Results from this notebook run

Model

Clean test accuracy

Rotated 20° accuracy

Noisy accuracy (sigma 0.05)

HOG + SVM

72.50%

63.33%

36.67%

Grid SIFT + SVM

78.33%

49.17%

78.33%

HOG rotation drop: 0.7250 - 0.6333 = 0.0917, or 9.17 percentage points.
SIFT rotation drop: 0.7833 - 0.4917 ≈ 0.2917, or 29.17 percentage points.
HOG noise drop: 0.7250 - 0.3667 = 0.3583, or 35.83 percentage points.
SIFT noise drop: 0.7833 - 0.7833 = 0, or no drop in this run.

The prediction was half correct: SIFT lost more under rotation, contrary to the prediction; HOG lost more under noise, as predicted. Results describe this dataset and setup, not every possible dataset.

13. Main formulas

Accuracy

Accuracy = correct predictions / total predictions

As a percentage, multiply by 100. HOG clean test accuracy: 87 / 120 × 100 = 72.50%.

Precision and recall for one class

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)

TP means true positives, FP false positives, FN false negatives for the class being considered.

F1 and macro-F1

F1 = 2 × (Precision × Recall) / (Precision + Recall)
Macro-F1 = (F1_class1 + F1_class2 + F1_class3 + F1_class4) / 4

Macro-F1 gives each class equal weight. The notebook prints macro-F1 but does not print each class's individual precision/recall/F1.

Confusion matrix

Rows are actual labels and columns are predicted labels. Diagonal entries are correct classifications; off-diagonal entries are errors. For the HOG validation matrix, diagonal sum is 16+23+23+30=92, so accuracy is 92/120=76.67%.

Accuracy drop

Accuracy drop = clean accuracy - transformed accuracy

This difference is often reported in percentage points when the accuracies are percentages.

14. Values to memorize for the viva

Item

Value

Seed

45

Original BloodMNIST size requested

64 × 64

Processing image size

128 × 128 grayscale

Classes

4: eosinophil, monocyte, neutrophil, platelet

Train / validation / test

400 / 120 / 120 images total

HOG cell

16 × 16 pixels

HOG orientations

9

HOG block

2 × 2 cells

HOG features per image

1,764

Normal SIFT keypoints

avg 72.67; min 8; max 229; zero 0/400

Fixed SIFT grid

6 × 6 = 36 keypoints

SIFT keypoint size

32

SIFT descriptor length

128

SIFT features per image

4,608

Candidate C values

0.1, 1, 10

Selected C

0.1 for both models

Rotation / noise

20° / sigma 0.05

15. Instructor reference: key implementation points

Data split: The code selects balanced subsets from the official train, validation, and test splits. The validation set is used for choosing C; the test set is reserved for final robustness evaluation.

Feature/classifier design: HOG and fixed-grid SIFT each produce one fixed-length vector per image. A linear SVM is trained separately on each feature representation.

Why fixed-grid SIFT: Detected SIFT keypoint counts vary by image (8–229 in this run). A fixed 6 × 6 grid ensures 36 descriptors and a consistent 4,608-value vector.

Parameter selection: The code tests C = 0.1, 1, 10 and selects by validation accuracy. Macro-F1 is reported for comparison but does not drive the current selection.

Robustness protocol: The trained models remain frozen. Rotation and noise are applied to test images before extracting features; no retraining occurs on transformed test data.

Interpretation of results: In this run, SIFT has higher clean and noisy accuracy; HOG has higher rotated accuracy. These results apply to this sample, preprocessing, and fixed-grid implementation.

Reproducibility: A seed controls the random subset selection and the noise generator. Reproducibility also depends on using the same dataset version and software behavior.

16. Outputs and their meaning

Output

What it verifies or measures

X and y shapes

Number and dimensions of images and labels in each split

Per-class counts

Confirms balanced class sampling

HOG feature matrix shape (400, 1764)

One 1,764-value HOG vector for each training image

SIFT feature matrix shape (400, 4608)

One 4,608-value fixed-grid SIFT vector for each training image

Keypoint statistics

Distribution of naturally detected SIFT keypoints before fixed-grid conversion

Validation accuracy and macro-F1

Performance used to compare candidate C values

Validation confusion matrix

Class-specific correct and incorrect predictions during model selection

Test accuracy table

Final clean, rotated, and noisy performance for the frozen models

17. Items to finish/check for full assignment compliance

The supplied notebook reaches Q4, but the exam sheet also requests Q5. Add answers for: why HOG uses block normalization; why C must not be selected using the test set; which classes are most confused and why grayscale may contribute; and the required AI-use log. Base the class-confusion answer on the notebook's actual confusion matrices. The current saved notebook also has an empty final cell, so ensure every required question has a clearly labeled answer before submission.
