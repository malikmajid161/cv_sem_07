# Lab 03: Edge Detection Techniques and Their Impact on Classification Performance

## 1. Introduction
The objective of this laboratory is to explore fundamental concepts of edge detection in image processing and understand its impact on image classification tasks. Edge detection highlights structural information and boundaries of objects while filtering out extraneous details, such as texture and color. This lab covers the implementation of first-order edge detectors (Sobel, Prewitt), second-order detectors (Laplacian, Laplacian of Gaussian - LoG), and multi-stage methods (Canny). The impact of noise on edge detection, the role of pre-smoothing, and how edge maps compare to raw and filtered images for downstream image classification (using SVM, Random Forest, KNN, and CNNs) are systematically evaluated.

## 2. Methodology
The laboratory comprised the following core tasks:
- **Comparative Edge Detection**: Applying Sobel, Prewitt, Laplacian, LoG, and Canny algorithms on representative dataset images to observe their boundary detection characteristics.
- **Noise Analysis**: Introducing Gaussian and Salt-and-Pepper noise to evaluate the noise sensitivity of different edge detectors. Gaussian and Median filters were subsequently applied to evaluate the effectiveness of pre-smoothing.
- **Canny Parameter Analysis**: Investigating the impact of varying Canny low/high hysteresis thresholds and Gaussian kernel sizes on edge quality and edge density.
- **Classification Performance Evaluation**: Training standard classifiers (SVM, RF, KNN) and deep CNNs (ResNet101, DenseNet121) using three distinct input representations: Raw images, Filtered images, and Edge maps, maintaining consistent train/val/test splits.

## 3. Experimental Setup
The experiments were conducted using a designated image classification dataset (from Labs 01 & 02). 
- **Noise Types**: Gaussian noise and Salt-and-Pepper noise.
- **Filters**: Gaussian filter (for Gaussian noise) and Median filter (for Salt-and-Pepper noise).
- **Edge Detectors**: Sobel (x, y, magnitude), Prewitt, Laplacian, LoG, Canny.
- **Models**: Support Vector Machine (SVM), Random Forest (RF), K-Nearest Neighbors (KNN), CNN Model 1 (ResNet101), and CNN Model 2 (DenseNet121).

## 4. Results

### 4.1 Effect of Noise and Preprocessing on Edge Detection
Edge detectors, particularly second-order ones like the Laplacian, were found to be highly sensitive to noise, resulting in severe false positive edges. Gaussian noise heavily deteriorated Sobel and Prewitt outputs. Pre-smoothing the noisy images using a Gaussian filter (for Gaussian noise) or a Median filter (for salt-and-pepper noise) successfully mitigated the false edges, allowing the edge detectors to recover true object boundaries. Canny inherently performed best against noise due to its built-in Gaussian smoothing step.

### 4.2 Canny Parameter Analysis
| Configuration | Low Threshold | High Threshold | Kernel Size | Edge Quality | Observations |
|---------------|---------------|----------------|-------------|--------------|--------------|
| Canny-1       | 30            | 100            | 3x3         | Noisy        | Many weak texture/false edges. High edge density. |
| Canny-2       | 50            | 150            | 3x3         | Balanced     | Continuous borders with few false edges. |
| Canny-3       | 100           | 200            | 3x3         | Sparse       | Broken borders, fine details lost. |
| Canny-4       | 50            | 150            | 5x5         | Balanced     | Smoother than 3x3, but retains main shapes. |
| Canny-5       | 100           | 200            | 5x5         | Sparse       | Severe loss of detail. |

*Observation*: Canny-2 (50, 150, 3x3) yielded the most optimal balance between retaining structural edges and minimizing noise.

### 4.3 Classification Performance Comparison
| Model / Classifier | Accuracy Raw % | Accuracy Filtered % | Accuracy Edge % |
|--------------------|----------------|---------------------|-----------------|
| SVM                | 77.58          | 77.07               | 67.88           |
| Random Forest      | 72.12          | 71.62               | 68.79           |
| KNN                | 70.20          | 70.71               | 63.33           |
| CNN (ResNet101)    | 79.80          | 80.00               | 35.35           |
| CNN (DenseNet121)  | 81.92          | 80.20               | 46.06           |

## 5. Discussion
The results unequivocally demonstrate that while edge maps successfully extract structural shapes, they discard vital visual information (texture, color, and intensity). For machine learning models (SVM, RF, KNN), the accuracy drop is noticeable but somewhat manageable. However, for Convolutional Neural Networks (CNNs), the accuracy drops drastically (from ~80% down to ~35-46%). CNNs rely heavily on learning spatial hierarchies, textures, and color gradients from raw pixels; stripping these out prior to training severely limits the deep networks' feature extraction capabilities. 

## 6. Conclusion
The laboratory confirmed the theoretical behaviors of various edge detectors. First-order methods like Sobel provide thick, reliable edges but are susceptible to noise, whereas second-order methods like Laplacian are highly noise-sensitive unless smoothed (LoG). Canny provides the most robust multi-stage edge detection. Crucially, the classification task revealed that handcrafted feature extraction (such as strict edge maps) is sub-optimal for complex classification tasks, especially when using deep learning models capable of learning their own hierarchical, robust features directly from raw or gently filtered data.

## 7. References
1. R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed. Pearson, 2018.
2. J. Canny, "A Computational Approach to Edge Detection," IEEE Transactions on Pattern Analysis and Machine Intelligence, 1986.

---

# Discussion Questions

**Question 1: Which edge detector was most sensitive to noise? Explain your answer.**
The Laplacian (and other second-order derivative filters) was the most sensitive to noise. Second-order derivatives amplify high-frequency variations in an image more aggressively than first-order derivatives. Because noise consists primarily of high-frequency components, the Laplacian filter amplifies this noise, resulting in an output dominated by false edges.

**Question 2: How did Gaussian and Median filtering affect the quality of detected edges?**
Gaussian filtering effectively blurred the image, smoothing out high-frequency Gaussian noise before edge detection, which significantly reduced false edges but slightly blurred the true edges. Median filtering was highly effective at removing salt-and-pepper noise while preserving the sharpness of actual structural boundaries much better than the Gaussian filter, allowing edge detectors to find continuous, sharp edges without noise artifacts.

**Question 3: How did changing the low and high thresholds affect the number and quality of detected edges?**
Setting both thresholds low (e.g., 30, 100) resulted in high sensitivity, capturing many weak edges, textures, and potential noise (false positives). Setting both thresholds high (e.g., 100, 200) resulted in low sensitivity, causing the true object boundaries to become fragmented and sparse, losing fine details. A balanced configuration (50, 150) connected the strong edges effectively while discarding irrelevant noise gradients.

**Question 4: Did using edge-only images improve or reduce classification accuracy compared with raw images? Explain the possible reasons.**
Using edge-only images significantly *reduced* classification accuracy across all models. The primary reason is that edge detection discards a massive amount of discriminatory information, such as object color, internal texture patterns, and shading. While shape is preserved, shape alone is rarely sufficient to distinguish between complex, real-world object classes that share similar outlines but differ in texture or color.

**Question 5: Edge maps mainly represent object boundaries. What information may be lost when texture, color, and intensity information are removed?**
Information lost includes the material properties of the object (inferred from texture), color profiles (which are critical for distinguishing similar objects like an apple vs. a peach), shading/depth cues (which define the 3D structure of the object), and internal patterns (like text or logos on an object).

**Question 6: What are the advantages of allowing a CNN to learn these features instead of manually providing edge maps?**
When a CNN is given raw images, its early layers naturally learn to act as optimal, oriented edge detectors (similar to Gabor filters) tailored specifically to the dataset. Unlike manual edge maps that forcefully discard color and texture, the CNN retains this information, learning higher-level complex features (textures, shapes, object parts) in deeper layers. Allowing the CNN to learn features ensures no useful discriminatory data is prematurely discarded.

**Question 7: Based on your results, which input representation produced the most useful classification results?**
Raw images (and gently Filtered images) produced the most useful classification results. They retained the highest classification accuracies (around 80-81% for CNNs), demonstrating that preserving the original pixel data—including texture, color, and intensity—provides the richest feature set for machine learning models to differentiate between classes.

---

# Viva Questions & Answers

**1. What is an edge in an image?**
An edge is a local region in an image where there is a sharp change or discontinuity in pixel intensity, often corresponding to object boundaries, shadows, or changes in surface orientation.

**2. What is the difference between first-order and second-order edge detection?**
First-order edge detection (like Sobel, Prewitt) uses the first derivative of the image intensity; edges correspond to local peaks (maxima/minima) in the gradient magnitude. Second-order edge detection (like Laplacian) uses the second derivative; edges correspond to zero-crossings.

**3. What is the difference between Sobel (G_x) and (G_y)?**
Sobel (G_x) computes the gradient in the horizontal direction, highlighting vertical edges. Sobel (G_y) computes the gradient in the vertical direction, highlighting horizontal edges.

**4. Why is the Laplacian more sensitive to noise?**
The Laplacian calculates the second derivative. Taking the derivative amplifies high-frequency signals. Taking it twice amplifies high frequencies (which noise primarily is) even more, overwhelming the true structural edges.

**5. What is the purpose of Gaussian smoothing before edge detection?**
To attenuate high-frequency noise so that the edge detector responds to actual structural changes in the image rather than rapid, spurious intensity jumps caused by noise. 

**6. What is the main advantage of Canny edge detection?**
It is a multi-stage process that provides optimal edge detection by combining noise reduction (Gaussian filter), precise gradient calculation, non-maximum suppression (for thin edges), and hysteresis thresholding (for continuous, connected edges).

**7. What are Canny's low and high thresholds?**
They are used in the hysteresis thresholding step. Any edge gradient above the high threshold is definitively an edge. Any edge gradient below the low threshold is discarded. Edges between the two thresholds are kept only if they are connected to a definite edge pixel.

**8. What is the difference between Gaussian and Salt-and-Pepper noise?**
Gaussian noise adds random variations to pixel values following a normal distribution, making the image look "grainy." Salt-and-Pepper noise introduces stark white and black pixels randomly scattered across the image, caused by sharp data disturbances.

**9. Why is Median filtering useful for Salt-and-Pepper noise?**
Median filtering replaces a pixel's value with the median of its neighbors. It is highly robust to outliers, effectively ignoring the extreme bright/dark spikes of salt-and-pepper noise while preserving the sharpness of valid edges.

**10. Why can edge detection reduce classification performance?**
It acts as a drastic data reduction step. It removes color, shading, and internal textures, leaving only structural outlines, which are often insufficient to distinguish complex classes.

**11. Can a CNN learn edge features automatically?**
Yes. The initial convolutional layers of a CNN automatically learn spatial filters that detect edges, gradients, and lines in various orientations directly from the raw pixel data.

**12. Why might raw images perform better than edge-only images for classification?**
Raw images contain all available visual information (shape, color, depth, texture). Deep learning models have the capacity to determine which features are most important for the specific classification task without forcibly throwing away potentially useful information up front.
