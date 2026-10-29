# Lesion area and perimeter table
| | Image | Best Filter | Edge Method | Area (pixels) | Perimeter (pixels) |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **0** | Image 1 | Gaussian 7x7 | Canny 10–30 | 119405 | 5175.9 |
| **1** | Image 2 | Gaussian 7x7 | Canny 5–15 | 145493 | 3168.8 |
| **2** | Image 3 | Gaussian 7x7 | Canny 10–30 | 228267 | 2727.9 |
| **3** | Image 4 | Gaussian 7x7 | Canny 30–70 | 3079 | 347.4 |
| **4** | Image 5 | Gaussian 7x7 | Canny 5–15 | 210206 | 6105.0 |

# Final comparison
| Method | Noise Handling ($\sigma \downarrow$) | Edge Quality (F1 $\uparrow$) | Boundary Detection (IoU $\uparrow$) | Overall rank | Overall Performance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Average + Sobel** | 0.267 | 0.246 | 0.509 | 1.500 | 1 |
| **Gaussian + Sobel** | 0.271 | 0.237 | 0.516 | 2.167 | 2 |
| **Median + Sobel** | 0.325 | 0.235 | 0.506 | 3.833 | 3 |
| **Median + Canny** | 0.325 | 0.211 | 0.448 | 4.500 | 4 |
| **Gaussian + Canny** | 0.271 | 0.175 | 0.392 | 5.167 | 5 |
| **Average + Canny** | 0.267 | 0.138 | 0.279 | 5.833 | 6 |
| **Original + Canny** | 1.007 | 0.163 | 0.429 | 6.167 | 7 |
| **Original + Sobel** | 1.007 | 0.156 | 0.404 | 6.833 | 8 |

## Questions

**1. Why is Gaussian filtering applied before Canny detection?**
Canny relies on image gradients, which amplify noise. Gaussian smoothing suppresses sensor noise, fine skin texture and thin hairs so that only strong, meaningful intensity transitions survive gradient computation, and it avoids the blocky/shifted edges that average filtering can produce.

**2. How did the three Canny threshold settings affect the result?**
Low thresholds (5–15) keep many weak edges → texture, hair and vignette noise, with a cluttered boundary. High thresholds (30–70) keep only the strongest edges → cleaner map but gaps in low-contrast parts of the border. The middle setting (10–30) balances both. (The originally suggested 50–100 / 100–200 / 150–250 were too high for the smoothed images and returned little or no edge.)

**3. Which threshold produced the best lesion boundary?**
10-30 gave the highest solidity × (1 − noise ratio) and the most continuous closed contour.

**4. Why are edges useful for detecting skin lesions?**
Lesions differ from surrounding skin in pigment/intensity, so their border appears as a strong, closed intensity transition. Edges give a compact representation of that border, from which shape descriptors (area, perimeter, irregularity) used in ABCD-style dermatology analysis can be computed.

**5. Problems observed**
Hair and air bubbles create false edges; low-contrast or fuzzy lesion borders produce gaps; the dark circular vignette in dermoscopy images creates a strong edge; uneven pigmentation inside the lesion creates internal edges; gaps require morphological closing, which can slightly over-/under-estimate the area.

**6. How could the method be improved?**
Hair removal (DullRazor/black-hat + inpainting); illumination correction and vignette cropping; colour-space (e.g., Lab / saturation channel) edges instead of grayscale; automatic/adaptive Canny thresholds (Otsu-based or median-based); active contours / watershed / GrabCut refinement; or a learned segmenter such as U-Net trained on ISIC/HAM10000 masks.
