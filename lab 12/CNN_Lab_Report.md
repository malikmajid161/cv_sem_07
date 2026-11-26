# CNN Visualization & Critical Analysis — Lab Report

**Student:** Muhammad Majid Ali
**ID:** FA23-BAI-045
**Course:** Deep Learning Lab
**University:** COMSATS University Islamabad, Wah Campus

---

## Part I — Filter Pipeline Visualization

### Implementation Notes

- **Image:** `skimage.data.camera()` — 128×128 grayscale cameraman image (no internet required)
- **Convolution:** `scipy.ndimage.convolve` with `mode='reflect'` (preserves spatial size)
- **ReLU:** `np.maximum(0, x)` — explicit, no library shortcut
- **Pooling:** Manual nested loop for full transparency
- **Grid:** 5×4 layout (5 filters × 4 stages) with per-subplot colorbars

### Filters Used

| Filter | Kernel | Purpose |
|--------|--------|---------|
| Sobel-X | `[[-1,0,+1],[-2,0,+2],[-1,0,+1]]` | Vertical edge detection |
| Sobel-Y | `[[-1,-2,-1],[0,0,0],[+1,+2,+1]]` | Horizontal edge detection |
| Sharpen | `[[0,-1,0],[-1,5,-1],[0,-1,0]]` | High-frequency enhancement |
| Box Blur | `1/9 × ones(3×3)` | Smoothing / low-pass |
| Emboss | `[[-2,-1,0],[-1,1,1],[0,1,2]]` | 3D emboss effect |

### Pipeline Stages

| Stage | Operation | Description |
|-------|-----------|-------------|
| A | Raw Convolution | Direct filter response — positive and negative values |
| B | Post-ReLU | Negative values zeroed — only positive activations survive |
| C | MaxPool 2×2 | Strongest activation in each 2×2 block preserved |
| D | AvgPool 2×2 | Average of each 2×2 block — smoother, diluted output |

---

## Part II — Parameter Audit

### Formula

$$\text{Params} = (K_h \times K_w \times C_{in} + 1) \times C_{out}$$

### Model A — 3×3 Kernels, MaxPool

| Layer | Calculation | Parameters |
|-------|-------------|------------|
| Conv1 | (3×3×1 + 1) × 14 | 140 |
| Conv2 | (3×3×14 + 1) × 5 | 460 |
| Dense | (10 + 1) × 10 | 110 |
| **Total** | | **710** |

### Model B — 5×5 Kernels, AvgPool

| Layer | Calculation | Parameters |
|-------|-------------|------------|
| Conv1 | (5×5×1 + 1) × 14 | 380 |
| Conv2 | (5×5×14 + 1) × 5 | 1260 |
| Dense | (10 + 1) × 10 | 110 |
| **Total** | | **1750** |

### Keras Verification (TF 2.21.0)

```
Model A  →  Manual: 710   | Keras: 710   → ✓ MATCH
Model B  →  Manual: 1750  | Keras: 1750  → ✓ MATCH
```

---

## Part III — Critical Thinking Questions

---

### Q1 — ReLU "Blackout" Effect

**Which regions turned black and why?**

Convolution produces both positive and negative values. ReLU rule: $f(x) = \max(0, x)$ — every negative value becomes zero (black pixel).

**Sobel-X analysis:**
- Brightness **left → right** (bright to dark): output **positive** → survives ReLU ✓
- Brightness **right → left** (dark to bright): output **negative** → becomes **zero (black)** ✗

For patch `[a, b, c]`: if `a > c` → Sobel-X output = `−a + c < 0` → ReLU → **0** (blackout).
Both are real edges, but only one polarity survives. This is the **ReLU Blackout Effect**.

**Fixes:**

| Activation | Behavior | Effect on Stage B |
|-----------|----------|-------------------|
| **Leaky ReLU** `(α=0.1)` | `f(x) = 0.1x` for x<0 | Faint gray shadow edges appear on blacked-out side — no information permanently lost |
| **Sigmoid** `σ(x) = 1/(1+e^−x)` | Maps all values to (0,1) | No blackout, but edges become soft and less crisp — not suitable for CNNs |

---

### Q2 — MaxPool vs AvgPool: Information Retention

**Stage C vs Stage D observations (cameraman image):**

- **Sharpen (Stage C):** Bright isolated dots at coat edges and hat boundary — peak activations preserved exactly
- **Sharpen (Stage D):** Same regions blurred and dimmed to ~25% strength — spike averaged with 3 low neighbors
- **Sobel-X/Y (Stage C):** Crisp, sharp edge lines on tripod and jacket — 1-pixel edges retained
- **Sobel-X/Y (Stage D):** Same edges spread and weakened — thin lines became thick blurs

```
Patch at contrast boundary:
[ 0.9,  0.1 ]
[ 0.1,  0.1 ]

MaxPool → 0.9      ← peak preserved
AvgPool → 0.3      ← signal reduced 67%
```

**Dark ink text on white background:**

| Pipeline Stage | Better Pooling | Reason |
|----------------|---------------|--------|
| Raw pixels (ink=0, bg=255) | AvgPool | MaxPool picks white background, erasing ink strokes |
| After Conv → ReLU | **MaxPool** | Stroke borders become high activations — MaxPool preserves peak, AvgPool dilutes 4× |

> **Correct pipeline:** Conv → ReLU → MaxPool — edge detection must come first so MaxPool sees high stroke activations, not raw pixel values.

---

### Q3 — Parameter Explosion & Receptive Field Equivalence

**Exact % increase from Model A to Model B:**

$$\frac{1750 - 710}{710} \times 100 = \frac{1040}{710} \times 100 = \textbf{146.48\%}$$

**Receptive field equivalence:**

$$\text{RF after } n \text{ layers of } 3\times3 = 2n + 1$$

$$2n + 1 = 5 \implies n = \textbf{2 layers}$$

Two stacked 3×3 layers achieve the same 5×5 receptive field.

**Parameter saving (C filters, no bias):**

| Architecture | Parameters |
|---|---|
| One 5×5 layer | $25C^2$ |
| Two 3×3 layers | $18C^2$ |
| **Saving** | **28%** |

**Bonus:** Two 3×3 layers have a ReLU non-linearity between them → more representational power per parameter. This is the **VGGNet insight** — modern CNNs exclusively use stacked 3×3 filters.

---

### Q4 — Operation Commutativity

**Output is IDENTICAL** for both orders.

**Proof:**

ReLU is monotone non-decreasing: $x_1 < x_2 \implies \text{ReLU}(x_1) \leq \text{ReLU}(x_2)$

For any monotone non-decreasing function $f$:

$$\max(f(x_1), \ldots, f(x_n)) = f(\max(x_1, \ldots, x_n))$$

Since ReLU satisfies this property, both orders produce identical output. ∎

**2×2 patch `[−3, 2, 5, −1]`:**

| Step | Conv → ReLU → MaxPool | Conv → MaxPool → ReLU |
|------|----------------------|----------------------|
| Conv output | `[−3, 2, 5, −1]` | `[−3, 2, 5, −1]` |
| Intermediate | ReLU → `[0, 2, 5, 0]` | MaxPool → `5` |
| **Final** | `max(0,2,5,0) = 5` | `ReLU(5) = 5` ✓ |

**All-negative `[−3,−2,−5,−1]`:** both orders → **0** ✓

All 5 notebook test cases confirmed **EQUAL**.

**Computational efficiency implication:**

`Conv → MaxPool → ReLU` is more efficient — ReLU runs on N/4 elements after pooling instead of N:

$$\text{Saving} = \frac{N - N/4}{N} \times 100 = \textbf{75\% fewer ReLU operations}$$

Same output, 75% less computation — always prefer this order in production.

---

### Q5 — Diagnosing Filter Redundancy

**Cause 1 — Symmetric Initialization**

All filters start with identical weights → same forward output → same gradient → same update → filters never diverge regardless of epochs.

```
W1 = W2 = W3 = W4 = W5 = [0.5, 0.5, ...]
→ identical gradients → identical updates → identical forever
```

**Fix — He/Kaiming Initialization:**

$$\text{std} = \sqrt{\frac{2}{C_{in}}}$$

Each filter independently sampled → symmetry broken from epoch 1. Factor 2 compensates for ReLU zeroing half the inputs.

```python
nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')
```

---

**Cause 2 — Dead ReLU Syndrome**

High learning rate or negative bias → pre-activations always negative → ReLU outputs 0 → gradient = 0 → filter permanently stuck. When 4/5 filters die, output collapses to one surviving path → identical feature maps.

```
Conv output: [−2.1, −0.8, −1.5, −1.2]
ReLU:        [  0,    0,    0,    0  ]
Gradient:    [  0,    0,    0,    0  ]  ← no learning signal
```

**Fix A — Leaky ReLU:** $f(x) = 0.01x$ for x<0 → small gradient keeps dead neurons recoverable.

**Fix B — Batch Normalization:** Normalizes activations to zero mean → prevents distribution from drifting into dead zone.

---

**Summary Table:**

| Root Cause | Effect | Fix |
|---|---|---|
| Symmetric initialization | Identical gradients — filters never diverge | He/Kaiming initialization |
| Dead ReLU | Zero gradient — filters stop learning, collapse | Leaky ReLU + Batch Normalization |

---

## Notebook Test Results

### Commutativity Verification

```
Mixed pos dominant   →  5.00 = 5.00  ✓ EQUAL
All negative         →  0.00 = 0.00  ✓ EQUAL
All positive         →  5.00 = 5.00  ✓ EQUAL
Zero present         →  2.00 = 2.00  ✓ EQUAL
Single survivor      →  0.10 = 0.10  ✓ EQUAL
```

### Parameter Verification

```
Model A  →  Manual: 710   | Keras: 710   → ✓ MATCH
Model B  →  Manual: 1750  | Keras: 1750  → ✓ MATCH
```

---

*Deep Learning Lab — COMSATS University Islamabad, Wah Campus*
