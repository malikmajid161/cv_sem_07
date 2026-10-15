# Lab 06: Landmark Recognition Using SIFT — Lab Summary

**Course:** CS-482 Computer Vision
**Student:** Muhammad Majid Ali | FA23-BAI-045
**University:** COMSATS University Islamabad, Wah Campus

---

## Objective

Build an automated landmark recognition system using the **Scale-Invariant Feature Transform (SIFT)** algorithm. The system matches a query photograph against a database of reference landmark images, even under changes in scale, rotation, or illumination.

---

## System Overview

```
Query Image → SIFT Extraction → BF Matching (KNN + Lowe's Test)
                                        ↓
              Reference DB → SIFT Features → Ranked Candidates
                                        ↓
                          RANSAC Homography → Localized Region
```

---

## Task Breakdown

### Task 1 — Landmark Image Database
- Reference images organized into per-landmark folders on Google Drive
- Five landmarks: Eiffel Tower, Taj Mahal, Great Wall, Statue of Liberty, Colosseum
- Multiple images per landmark capture different angles and distances

**Drive path:**
```
MyDrive/Landmark_Dataset/landmark_db/
├── eiffel_tower/
├── taj_mahal/
├── great_wall/
├── statue_of_liberty/
└── colosseum/
```

---

### Task 2 — SIFT Feature Extraction
- Each reference image converted to **grayscale**
- `cv2.SIFT_create()` detects keypoints and computes 128-dimensional descriptors
- Descriptors from all images of a landmark are **stacked** into one array
- Keypoints visualized using `DRAW_RICH_KEYPOINTS` (shows scale & orientation)

---

### Task 3 — Query Image Processing
- Query image loaded from Drive, converted to grayscale
- SIFT applied identically to reference images
- Output: keypoint locations + descriptors ready for matching

---

### Task 4 — Feature Matching & Recognition

| Parameter | Value |
|---|---|
| Matcher | Brute-Force (`cv2.BFMatcher`, `NORM_L2`) |
| KNN | k = 2 |
| Lowe's ratio | 0.75 |
| Min matches threshold | 10 |

**Pipeline:**
1. `knnMatch(query_descs, ref_descs, k=2)` for each landmark
2. Keep match `m` only if `m.distance < 0.75 × n.distance`
3. Count good matches per landmark → rank candidates
4. **Recognize** if best count ≥ 10; otherwise **reject** as unknown

---

### Task 5 — Landmark Localization (Homography + RANSAC)

1. Extract corresponding keypoint coordinates from good matches
2. `cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)`
3. Project reference image corners onto query image via homography
4. Draw green polygon around matched landmark region
5. If RANSAC inliers < 10 → display warning, skip polygon

---

### Task 6 — Performance Evaluation

Six test conditions:

| Test Condition | Expected Behavior |
|---|---|
| Original | High match count, easy recognition |
| Rotated | SIFT rotation-invariant → should still recognize |
| Resized | SIFT scale-invariant → should still recognize |
| Reduced Brightness | Fewer keypoints, lower matches |
| Different Viewpoint | Fewer inliers for homography |
| Unknown Landmark | Rejected (below threshold) |

**Metrics recorded:** keypoints detected, good matches, predicted landmark, decision, processing time

---

## Key Parameters

| Parameter | Value | Purpose |
|---|---|---|
| `LOWE_RATIO` | 0.75 | Filter ambiguous matches |
| `MIN_MATCHES` | 10 | Reject unknown landmarks |
| `MIN_RANSAC` | 10 | Reject unreliable homography |
| RANSAC reprojection | 5.0 px | Inlier threshold |

---

## Limitations

- **Brightness reduction** lowers keypoint density → may fall below threshold
- **Large viewpoint changes** (>45°) may produce too few RANSAC inliers
- **Database coverage** directly affects recall — more reference images = better robustness
- **Speed** scales with database size; production systems would use FLANN or a vocabulary tree

---

## Dependencies

```bash
pip install opencv-contrib-python-headless numpy matplotlib
```

> `opencv-contrib-python-headless` is required for SIFT (`cv2.SIFT_create`), which is not available in the standard `opencv-python` package.
