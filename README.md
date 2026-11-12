# Matrix Compression & Low-Rank SVD

**Algorithms & Data Structures Examination, Group 3**

A step-by-step implementation of low-rank matrix compression, moving from plain dense storage to a truncated, factored SVD data structure. The SVD is computed from scratch using only matrix-vector products and vector norms.

All code, tests and experiments live in a single notebook: [`Problem13_SVD_Compression.ipynb`](Problem13_SVD_Compression.ipynb).

---

## Overview

| Stage | Name | Idea | Storage / cost |
|---|---|---|---|
| **V1** | Dense storage | Keep all entries; naive subsampling as a baseline | `mn` floats, `Ax` in Θ(mn) |
| **V2** | Gram power iteration | Form `G = AᵀA`, power-iterate on `G` | Θ(mn²) to build `G`, `n²` memory |
| **V3** | Matrix-free alternating power iteration | Alternate `Av` and `Aᵀu`, never form `AᵀA` | Θ(mn) per iteration, Θ(m+n) extra space |
| **V4** | Hotelling deflation | `A ← A − σuvᵀ`, repeat `k` times (explicit or implicit) | Θ(k·T·mn) time |
| **V5** | Truncated factored SVD | Store `(U_k, σ_k, V_kᵀ)` instead of `A` | `k(m+n+1)` floats, `Ax` in Θ(k(m+n)) |

**Constraint from the problem sheet:** the V2 to V5 solvers use only matrix-vector products and vector norms. `np.linalg.svd` is used **only as a reference** inside tests and experiments, never inside a solver.

---

## Notebook structure

1. **Part A: hand-worked 3×2 example.** Numerically verifies every number of the handwritten solution (Q1 to Q3) for `A = [[3,0],[4,5],[0,0]]`: Gram matrix, singular values `√45` and `√5`, singular vectors, rank-1 approximation, Hotelling deflation, Frobenius error, and one iteration of the Stage 3 power method.
2. **V1 to V5 implementations.**
   - `DenseMatrix`, `naive_subsample`
   - `power_iteration_sym`, `svd_v2_gram`
   - `alt_power_triplet`, `svd_v3_top1`
   - `svd_v4_deflation` (with `implicit=False` / `True`)
   - `TruncatedSVD` (fit, `matvec`, `rmatvec`, `reconstruct`, `storage`, `fro_error`, `energy`)
3. **Unit tests.** Five tests checking that all versions agree with each other, with `np.linalg.svd`, and with the theory (orthonormality, `Av_i = σ_i u_i`, Eckart–Young error, factored `Ax` equals dense `A_k x`, subsampling is lossy).
4. **Part B: numerical check of Q4.** Energy retained by `A₁`, and storage arithmetic for a 1000×1000 image (`k = 25`, break-even `k = mn/(m+n+1)`).
5. **Experiment 1.** V2 vs V3+V4 on a 2000×400 matrix (top 10): accuracy and wall-clock time. Followed by a scaling benchmark (time and peak memory vs. size) and an ill-conditioning demo showing why forming `AᵀA` loses small singular values.
6. **Experiment 2.** Image compression of a synthetic 256×256 grayscale image for `k ∈ {1, 3, 10, 25, 60}`, with plots of the singular spectrum, the empirical vs. theoretical Eckart–Young error, and the compression ratio.

---

## Key results and takeaways

- **Correctness.** V2, V3+V4 (explicit) and V3+V4 (implicit) all match `np.linalg.svd` on the test matrices (to about 1e-14 in Experiment 1).
- **Eckart–Young holds empirically.** `‖A − A_k‖_F = √(Σ_{i>k} σᵢ²)`, confirmed in tests and in Experiment 2 (b).
- **Compression.** Storage drops from `mn` to `k(m+n+1)` floats. Factoring only pays off while `k < mn/(m+n+1)`; for a 1000×1000 matrix, `k = 499` still saves memory but `k = 500` uses more than the dense matrix.
- **Honest timing result.** On the tall 2000×400 matrix, V2 is faster in wall-clock time, because it pays `mn²` once and then each iteration costs only `n²`. V3/V4 wins on:
  - **Numerical accuracy:** `κ(AᵀA) = κ(A)²`, so V2 loses small singular values (see the ill-conditioning demo).
  - **Memory:** implicit V4 needs only `O(k(m+n))` extra space and works when `A` is available only as an operator.
- **Naive subsampling** gives 4× memory savings but destroys sharp structure, unlike a low-rank factorization.

---

## Requirements

- Python 3.9+
- `numpy`
- `matplotlib`
- Jupyter (`notebook`, `jupyterlab`, or VS Code)

The standard library modules `time` and `tracemalloc` are also used.

```bash
pip install numpy matplotlib jupyter
```

## How to run

```bash
jupyter notebook Problem13_SVD_Compression.ipynb
```

Run all cells from top to bottom (**Kernel → Restart & Run All**). The unit-test cell prints `PASS <test_name>` for each test, and Experiment 1 asserts that every singular value error is below `1e-6`. All random inputs use fixed seeds, so results are reproducible. The benchmark cells (scaling and image experiments) take the longest to run.

---

## Repository contents

```
.
├── Problem13_SVD_Compression.ipynb   # code, tests, experiments
├── README.md                          # this file
└── (handwritten solution is submitted separately)
```

## Authors

Group 3: *add member names here*
