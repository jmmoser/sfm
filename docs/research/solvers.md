# Numerical solvers: measurements (math, bundle adjustment, geometry)

Measurements from 2026-10-16 – 2026-10-21; the code may have changed since — check the source before relying on a number.

Timings are Node/V8 on x86 unless stated; the phone is slower. Accuracy numbers come from the
unit tests in `src/math`, `src/ba` and `src/geometry` (most are asserted there with looser bounds).

## `src/math`

### Dense linear algebra timings (`decomp.ts`)

| Routine | Size | Time |
|---|---|---|
| `nullVector` (QR-of-Aᵀ path) | 8x9 | ≈ 2 µs |
| `svd` | 9x9 | ≈ 20–25 µs |
| `svd` | 100x12 | ≈ 130 µs |
| `svd3` | 3x3 | ≈ 2.5 µs |
| `symEig` | 12x12 | ≈ 35 µs |

Per-call typed-array allocation dominated small SVDs before the module-level scratch buffers.

### Polynomial roots (`poly.ts`)

- Stress test, 40k random polynomials of degree 3–20 with coefficients spanning 1e±4:
  `polyRealRoots` (critical-point isolation + rtsafe) 0 missed / 0 spurious roots; the
  floating-point Sturm variant (`polyRealRootsSturm`) 1 missed + 1 spurious; Cardano/Ferrari
  closed forms failed on ~2 % of badly scaled cubics/quartics. Hence `solveCubic` / `solveQuartic`
  delegate to `polyRealRoots` (≈ 1.4 / 2 µs).
- Extreme scaling verified: cubic with leading coefficient 1e-300, degree 10 with 1e-100,
  degree 20 with 1e-40 (band splitting when deg·log2(bound ratio) > 800 bits).
- Cluster merge at 1.5× the Horner error bound: (x−1)(x−1−5e-7)(x−5) yields 3 roots; no 2–4-fold
  root lost or split in 14k rounded-coefficient trials (factor 8 / 64 merged resolvable roots).
- rtsafe cap 2500: a cap of 300 returned unconverged roots ~1e-90 of the bracket width.

## `src/ba` (bundle adjustment)

### Performance (Node 22, x86, loaded 4-core machine; `solver.perf.test.ts` logs them)

| Problem | Solver | Time per LM iteration |
|---|---|---|
| 100 poses / 20k pts / 98k obs (n = 596) | direct | ≈ 140–210 ms (first call incl. setup + JIT ≈ 600 ms) |
| 500 / 60k / 406k | PCG | ≈ 850 ms (≈ 440 PCG iterations per solve — block-Jacobi is weak on long chains) |
| 500 / 60k / 406k | 'auto' (skyline, exact steps) | ≈ 570 ms |
| local BA: 12 free + 18 fixed poses, 49k obs | auto | ≈ 50 ms after warm-up |
| tracks of ~30 views, 500k obs | auto | ≈ 1.3 s, ~85 MB extra (Schur cost and pair memory grow with Σ m²) |

Hot loops: Schur 6x6 pose blocks unrolled with W in registers; linearisation unrolled for the pose block.

### Memory

No per-observation Jacobians are stored. Per call: W = 18 doubles per (point, distinct camera-side
block) ≈ 144 B × (#obs + #points); Schur pair offsets 4 B × Σ m(m+1)/2 (m = blocks per point); U and
S block-sparse; V / V⁻¹ 12 doubles per point; 2 parameter copies. ≈ 85 MB for 400k observations /
60k points; ≈ 20 MB for 100k / 20k.

### Accuracy (synthetic, `solver.test.ts`, `review.test.ts`)

- Self-calibration from f +10 %, k1 +0.05, perturbed poses/points: f error 1.3e-5; RMS at the
  least-squares noise floor (0.658 vs 0.653 px expected, σ = 0.5 px).
- 10 % gross outliers: Cauchy → f error 3e-5, inlier RMS 0.647 px; trivial loss → f error 12.8 %.
- Depth range 2 m – 2 km in one scene converges; points at 1–1000 km stay finite.
- Dense and PCG solutions agree to 1e-7.
- Robust losses (IRLS) converge linearly (≈ 5–15 iterations); the trivial loss quadratically.
- The LM step equals a brute-force solve of the full damped normal equations (no Schur) to < 1e-7
  relative for every loss, dense and PCG, λ up to 5, with weights, masks, fixed poses/points, 1–3
  cameras with mixed refine flags, centre + relative-rotation priors, plus randomised configurations.

## `src/geometry`

### 5-point essential (`essential.ts`)

- Nistér's degree-10 polynomial (implemented first, rejected): det(B(z)) coefficients cancel
  catastrophically for small baselines — true E lost in 26 % of noise-free samples at
  baseline/depth 1e-3…1e-2.
- Action-matrix solver, noise-free, true E found: 100 % for baseline/depth ≥ 0.01; lost in ~2–10 %
  at 1e-3…1e-2 (up to 30 % for forward motion toward a plane) and 25–80 % at 1e-4…1e-3 (the true
  root's eigenvalue is perturbed into a complex pair by the ill-conditioned elimination template;
  widening the near-real gate does not help). Such parallax is below the noise anyway.
- True solutions polish to rel. residual ≤ 1e-9 (4800 noise-free configurations incl. planar /
  forward motion / baseline/depth 1e-4). Unconverged real roots had σ2/σ1 down to 0.96 (not essential).
- ~45–60 µs per solve warmed, ~4.9 solutions on average.

### Two-view verification (`twoview.ts`)

- Synthetic, RADIAL k1 −0.08, N(0, 0.5–1 px), 30–50 % outliers, 6 seeds each:
  object 0.5 m wide / narrow baseline: rot ≤ 0.03°, t ≤ 0.08° / 0.37°; scene 50 m wide / narrow:
  rot ≤ 0.04°, t ≤ 0.14° / 1.3°; backyard 3–100 m: t ≤ 0.2°; pure rotation R error ≤ 0.01°;
  planar pairs within 0.1° of R or `planarAlternative`.
- Second (Cauchy at σ̂, in-front inliers only) refinement of the E pose, scene 50 m / baseline 1 m:
  t error median 1.9° → 0.4°, max 5.5° → 1.0°.
- Planar branch choice by cheirality support (walking along a facade / sideways over the ground):
  the E-selected branch was wrong (5.7°–19° rotation error) in 3–5 of 6 seeds without it.
- 'panoramic' with a small observed translation (3-D scene 5–15 m, 5 cm step): rotation-only fit
  0.11–0.28° error vs full-model R 0.006–0.04°. Pure rotation puts ≤ 52 % of inliers in front,
  hence the 80 % gate.
- Planar pairs with t ∥ n (walking toward a wall): ≈ 0.2–0.3° rotation / 2–6° translation error
  from two views (ill-conditioned for both H and E).
- 2000 matches / 40 % outliers: ~25–45 ms.
- Hopeless (random) pairs exhaust the budget (~7k 5-point + 10k 7-point samples): ≈ 0.6–1.2 s
  (N = 100–1000).
- `rotationPrior`: identical results with / without prior for general motion.

### PnP (`pnp.ts`)

- Lambda Twist P3P: correct pose among the solutions in 100 % of 5000 random configurations;
  ~10 µs per solve warmed.
