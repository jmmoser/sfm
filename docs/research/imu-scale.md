# IMU metric scale at phone keyframe rates

Measurements from 2026-10 (synthetic unit-test setups and e2e worker runs with CPU features); the code
may have changed since — check the source before relying on a number.

Code: `src/sfm/inertial.ts` (`estimateMetricScale` = band-pass estimator, `estimateMetricScaleFromIntervals`
= keyframe-interval estimator), `src/sfm/world.ts` (which one is used), `src/engine/refine.ts`
(`lowExcitationRetry`).

## 2026-10-21: keyframe-interval preintegration when the band-pass estimate lacks excitation

### Problem

`yard-real --stride 3` (48 keyframes in 40 s, about 1.2 per second: what an iPhone analysing 2–4 frames/s
produces) logged `IMU scale unreliable: insufficient excitation (0.010 m/s² < 0.08)`. The scale then came
from GPS (σ 5.5 %) with an error of −6.4 %, which fails the 5 % check. At stride 1 and 2 the IMU scale is
used (+1.2 % in both clean baselines below).

Cause: the band-pass estimator applies one linear operator to the camera centres and to the
double-integrated IMU acceleration, both sampled at the keyframes: a smoothing-spline second
derivative with cutoff min(3 Hz, 0.4/h) for keyframes h seconds apart, then a 0.3–3 Hz zero-phase
band-pass. At h ≈ 0.83 s only 0.3–0.48 Hz survives. A walk's acceleration is the turns (below 0.3 Hz)
and the sway and bob (1–2 Hz, above the cutoff), so the band holds almost nothing. On the GT poses of
the same 48 keyframes the band-pass estimate is +10.2 % ± 3.2 % (R² 0.97); on the reconstructed poses
it is −4.6 % ± 9.6 % (R² 0.71). The 0.08 m/s² floor was right to reject it. The same collapse is behind
refine's `lowExcitationRetry` (e2e `object`, GT poses: excitation 0.110 / 0.058 / 0.029 m/s² at
200 / 90 / 53 keyframes).

### Estimator

Velocity-free preintegration constraints between keyframes (the linear visual-inertial initialisation
of Mur-Artal & Tardós 2017, "Visual-inertial monocular SLAM with map reuse", §IV-B, and of Qin & Shen's
VINS-Mono, with the velocities eliminated). For keyframes a = k − m, k, b = k + m, the second divided
difference D[f] = (f_b − f_k)/h₂ − (f_k − f_a)/h₁ of any trajectory equals ∫ Λ_k(τ) a(τ) dτ, with Λ_k the
hat function (1 at t_k, 0 outside [t_a, t_b]). So

    D[C] = s₀ D[P] + c_w (h₁ + h₂)/2 + D[∫∫R_wd] c_d

holds exactly up to noise, with P the full-rate (60 Hz) double integral of the world-frame IMU
acceleration (vision-anchored attitude). It goes through the band-pass estimator's regression machinery
(`robustScaleFit`: Huber IRLS, world-frame bias c_w, device-frame bias c_d with its σ = 0.05 m/s² prior,
Newey–West σ). D[P] integrates every IMU sample, so nothing is limited by the keyframe rate. The only
band limit is the hat's low-pass (first null at 1/h). It keeps the slow accelerations that dominate
when keyframes are 0.5–1 s apart.

Choices, with the evidence behind each:

- **Span m.** m = 1 amplifies the vision noise by 1/h². Long spans keep the slow accelerations and
  suppress the IMU noise, but they low-pass the hand motion. The estimator therefore fits spans
  m = c·m₀ (m₀ = the smallest m with a median half-span ≥ 0.25 s), up to a 2 s half-span, and keeps
  the smallest σ (a reliable fit before an unreliable one). The chosen half-span is 0.4–0.8 s for
  orbits (1.3 s at 0.8 Hz) and 1.0–1.6 s for walks. On yard-real stride 3 m = 1 gives +2.0 % ± 3.2 %;
  the chosen m = 2 (1.5 s) gives +0.5 % ± 1.6 %. With 5 mm vision noise (walk, 1.2 Hz) span selection
  lowers the error rms from 1.33 % (m = 1) to 0.89 %.
- **Errors in variables.** White accelerometer noise σ_a puts variance σ_a² Δt (h₁ + h₂)/3 on each
  D[P]. Its share κ of the scale column's partial sum of squares biases the scale by ≈ +κ. On the
  synthetic orbits the mean error tracks κ (κ 0.77 % → mean +0.66 % at 1.2 Hz). κ is computed for an
  assumed σ_a = 0.05 m/s² per sample (the synthetic value; e2e datasets use 0.04; iPhones are probably
  quieter). It is not corrected, because the correction would overshoot on a quieter device. Instead
  κ ≤ 2 % is required and κ is added to σ in quadrature.
- **Device-bias prior.** The ridge prior shrinks ĉ_d. On an orbit the centripetal term is degenerate
  with a device-frame bias along the optical axis, so its spread Σ (A⁻¹)₀ₐ² λ² (s₀σ_b)² is added to
  the variance. A perfectly regular synthetic orbit reports −1.5 % ± 8.3 %, unreliable.
- **σ calibration.** Monte Carlo over the IMU noise of one walk (3 Hz, 1.2 s spans, 40 runs, 3
  trajectories) gave a true σ 1.15–1.25× the Newey–West σ, and a longer HAC lag did not help. The
  factor is 1.2 (the band-pass estimator uses 1.3).
- **Used only as a fallback.** `computeWorldAlignment` runs it only when the band-pass estimate is
  unreliable. Stride-1 captures, where the band-pass estimate is reliable, take exactly the same code
  path as before (the band-pass results are bitwise identical to the old code on 36 synthetic cases).

Rejected alternatives:

- (b) Extra trajectory samples from tracked non-keyframe frames. At stride 3 only 80 frames are
  delivered, 38 of them `tracking_lost` in the run that motivated this change, and those poses belong
  to the live registration code (out of scope for this change).
- (c) Measuring the excitation at the full IMU rate and keeping the R² bound. That changes only the
  gate: the band-pass regression itself sees 0.3–0.48 Hz (−4.6 % ± 9.6 %, R² 0.71 on yard-real
  stride 3).
- Feasible GLS (vision and IMU variance components estimated from the residual moments, banded
  whitening). On yard-real stride 3 the residual lag-1 autocorrelation was −0.51, about 80 % vision
  noise, and GLS cut σ from 2.7 % to 1.8 %. But on the noisy synthetic walks (5 mm) it
  made κ 1–4 % on average (0.3 % without it) and rejected 1–5 of 30 seeds per cell, which the plain fit
  accepts. Span selection gets most of the gain without that fragility.

### Synthetic sweep

Setup: `extras-testutil` hand-held orbit (θ = 0) and 40 m walk (θ = 90). The IMU runs at 60 Hz with
gyro noise 0.005 rad/s, accel noise 0.05 m/s² and the default biases. Keyframes are spaced uniformly in
[0.75, 1.25]/rate; the rotation noise is 0.5 mrad; the vision position noise is 0.5 mm (orbit) / 2 mm
(walk). The time offset, gyro bias and gravity are estimated as in `computeWorldAlignment`. 30 seeds
per cell. Band = `estimateMetricScale`; intervals = `estimateMetricScaleFromIntervals`. "rel" is the
number of reliable seeds; z = error / claimed σ.

| case | band rel | band err rms | band z rms | int rel | int err mean / rms / max | int σ mean | int z rms | κ mean | span |
|---|---|---|---|---|---|---|---|---|---|
| orbit 0.8 Hz | 0 | 3.20 % | 1.74 | 11 | +1.69 / 2.81 / 5.67 % | 3.62 % | 0.83 | 2.30 % | 1.26 s |
| orbit 1.2 Hz | 25 | 1.03 % | 0.88 | 30 | +0.66 / 1.15 / 2.39 % | 1.56 % | 0.68 | 0.77 % | 0.82 s |
| orbit 2 Hz | 30 | 0.70 % | 1.01 | 30 | +0.19 / 0.67 / 1.60 % | 0.87 % | 0.77 | 0.34 % | 0.50 s |
| orbit 3 Hz | 30 | 0.65 % | 1.01 | 30 | −0.01 / 0.56 / 1.12 % | 0.70 % | 0.80 | 0.22 % | 0.37 s |
| orbit 4 Hz | 30 | 0.49 % | 0.82 | 30 | +0.34 / 0.76 / 2.40 % | 0.68 % | 1.07 | 0.27 % | 0.43 s |
| walk 0.8 Hz | 0 | 2.97 % | 1.84 | 30 | +0.03 / 0.72 / 2.20 % | 0.94 % | 0.77 | 0.31 % | 1.25 s |
| walk 1.2 Hz | 0 | 1.52 % | 1.00 | 30 | +0.06 / 0.65 / 1.36 % | 0.82 % | 0.84 | 0.36 % | 1.62 s |
| walk 2 Hz | 30 | 1.53 % | 1.07 | 30 | +0.24 / 0.62 / 1.56 % | 0.70 % | 0.84 | 0.32 % | 1.32 s |
| walk 3 Hz | 30 | 1.04 % | 1.10 | 30 | +0.05 / 0.83 / 1.85 % | 0.63 % | 1.37 | 0.28 % | 1.04 s |
| walk 4 Hz | 30 | 0.73 % | 0.91 | 30 | +0.15 / 0.58 / 1.21 % | 0.62 % | 0.94 | 0.28 % | 0.96 s |

The same sweep with more vision noise (5 mm walk, 1.5 mm orbit):

| case | band rel | band err rms | int rel | int err rms / max | int z rms |
|---|---|---|---|---|---|
| orbit 1.2 Hz | 25 | 1.75 % | 30 | 1.63 / 3.63 % | 0.77 |
| orbit 2 Hz | 30 | 1.70 % | 30 | 1.56 / 5.02 % | 0.97 |
| orbit 4 Hz | 30 | 1.37 % | 30 | 1.35 / 3.60 % | 1.10 |
| walk 0.8 Hz | 0 | 4.72 % | 30 | 1.22 / 3.73 % | 0.80 |
| walk 1.2 Hz | 0 | 3.38 % | 30 | 0.89 / 2.17 % | 0.84 |
| walk 2 Hz | 30 | 3.62 % | 30 | 0.86 / 1.90 % | 0.90 |
| walk 3 Hz | 30 | 2.47 % | 30 | 1.08 / 2.28 % | 1.35 |
| walk 4 Hz | 30 | 1.74 % | 30 | 0.65 / 1.21 % | 0.86 |

Findings:

- At 0.8–1.2 keyframes/s the band-pass estimate is unreliable on every walk. The interval estimate is
  reliable on every walk with rms 0.65–0.72 % and a calibrated σ.
- The interval estimate's z rms is 0.68–1.1 everywhere except the walk at 3 Hz (1.35–1.37). There the
  heavily overlapping 1 s spans leave few effective samples for the HAC σ. The band-pass estimate is
  reliable on those walks, so the fallback does not run.
- Error sources, separated on the walk at 3 Hz with the true time offset: the model is exact (clean
  IMU, no vision noise: rms 0.02 %). With IMU noise only, the minimal span reads −0.2 % low, because
  linear interpolation of 60 Hz point samples attenuates the 2 Hz bob by (π f Δt)²/3 = 0.37 %. The
  band-pass estimator has the same effect. The device-bias prior adds nothing measurable (biases but no
  noise: rms 0.02 %).
- With 5 mm vision noise the interval estimate beats the band-pass one even at 2–4 Hz (walk: 0.65–1.08
  % vs 1.74–3.62 %). That is a candidate for a follow-up (see below).

Unit tests: `src/sfm/inertial.test.ts` "estimateMetricScaleFromIntervals" (walk and orbit at 1.2/s,
calibration over 1–2/s, slow pan, regular orbit, missing accelerations); `src/sfm/world.test.ts`
(the fallback in `computeWorldAlignment`, and no interval estimate when the band-pass one is reliable).

### e2e (sparse, CPU features)

`node e2e/run-e2e.mjs <dataset> --stride N --no-dense --set 'engineConfig={"featureBackend":"cpu"}'`.
"before" is base 18d73c9, run from a clean export; "after" is this change. The IMU-only numbers come
from re-running both estimators offline on the result poses of the same run: truth is the Sim3 scale of
`metrics.json`, and offset, gyro bias and gravity come from `computeWorldAlignment`.

| run | registered | ATE / extent | scale source (method) | claimed σ | scale error | z | checks |
|---|---|---|---|---|---|---|---|
| yard-real s1 before | 82 | 2.78e-4 | IMU+GPS (band-pass) | 1.1 % | +1.17 % | +1.09 | pass |
| yard-real s1 after | 81 | 5.07e-4 | IMU+GPS (band-pass) | 1.2 % | +0.74 % | +0.60 | pass |
| yard-real s2 before | 62 | 3.78e-4 | IMU+GPS (band-pass) | 1.1 % | +1.16 % | +1.06 | pass |
| yard-real s2 after | 62 | 4.53e-4 | IMU+GPS (band-pass) | 1.1 % | +1.52 % | +1.40 | pass |
| yard-real s3 before | 48 | 5.10e-4 | GPS | 5.5 % | −6.38 % | −1.17 | **FAIL** (scale) |
| yard-real s3 after, run 1 | 48 | 5.09e-4 | IMU+GPS (intervals) | 1.5 % | −0.05 % | −0.03 | pass |
| yard-real s3 after, run 2 | 48 | 5.10e-4 | IMU+GPS (intervals) | 1.5 % | −0.04 % | −0.03 | pass |
| object-real s1 before | 89 | 2.67e-4 | IMU (band-pass, low-excitation retry) | 2.0 % | +0.30 % | +0.15 | pass |
| object-real s1 after | 89 | 2.68e-4 | IMU (band-pass, low-excitation retry) | 2.0 % | +0.32 % | +0.16 | pass |
| object-real s2 before | 83 | 2.96e-4 | IMU (band-pass, low-excitation retry) | 2.3 % | −0.52 % | −0.22 | pass |
| object-real s2 after | 83 | 3.03e-4 | IMU (band-pass, low-excitation retry) | 2.3 % | −0.51 % | −0.22 | pass |
| object-real s3 before | 50 | 5.98e-4 | IMU (band-pass, low-excitation retry) | 3.2 % | +1.14 % | +0.36 | pass |
| object-real s3 after | 50 | 6.00e-4 | IMU (band-pass, low-excitation retry) | 3.2 % | +1.14 % | +0.35 | pass |

IMU-only estimates, re-run offline on the result poses ("int" = interval estimate, its chosen span in
brackets):

| run | band-pass | int | int on the GT poses of the same keyframes |
|---|---|---|---|
| yard-real s3 | −4.63 % ± 9.6 %, unreliable (excitation 0.010) | +0.47 % ± 1.6 % (1.5 s; m = 1: +2.0 % ± 3.2 %), κ 0.9 % | +0.93 % ± 1.2 % |
| yard-real s2 | +1.40 % ± 1.2 % | +1.71 % ± 1.2 % (0.67 s) | — |
| yard-real s1 | +1.54 % ± 1.1 % | +0.19 % ± 1.1 % (1.9 s) | −0.13 % ± 0.2 % (band-pass −0.10 %) |
| object-real s1 | +0.49 % (excitation 0.058) | +1.36 % ± 2.4 %, unreliable (κ 2.2 %) | +0.87 % ± 2.4 % |
| object-real s2 | −0.35 % (excitation 0.055) | −0.02 % ± 2.5 %, unreliable (κ 2.3 %) | — |
| object-real s3 | +1.04 % (excitation 0.026) | +0.93 % ± 4.5 %, unreliable (κ 4.2 %) | +1.65 % ± 4.5 % |

- yard-real stride 3 now takes its scale from the IMU, fused with GPS: −0.05 % (z −0.03) in two runs.
  The IMU estimate alone is +0.5 % ± 1.6 % (z 0.3). Both are within the acceptance of |error| ≤ 2.5 %
  and z ≤ 2. On the GT poses it reads +0.9 %, about the assumed κ of 0.9 % (the true κ with the
  dataset's 0.04 m/s² noise is ≈ 0.6 %).
- Stride 1 and 2 take the same code path as before (the band-pass estimate is reliable; the interval
  estimate is not even computed). The yard-real differences (+1.17 → +0.74 %, +1.16 → +1.52 %) are
  run-to-run variation of the live pipeline: ATE/extent changes 2.8 → 5.1e-4 between runs of the same
  code, and an earlier stride-1 run with the same alignment code gave +2.40 %.
- object-real is unchanged at every stride. Its orbit is too gentle for the interval estimate
  (κ 2.2–4.2 % > 2 %), so `lowExcitationRetry` still decides, as before.

### Limits and follow-ups

- The EIV attenuation is bounded and counted in σ, but not corrected. With a measured
  accelerometer noise level (e.g. from the 15–30 Hz band of a recording), κ could be corrected.
- The interval estimate is more precise than the band-pass one in most synthetic cells, not just at
  low rates. Preferring the smaller σ (or combining both, accounting for their correlation) would help
  stride-1 captures too, at the price of changing stride-1 results. Not done here.
- Gentle orbits (object-real: about 0.04 m/s² centripetal) stay unreliable in the interval estimate
  (κ 2.2–4.2 %), and `lowExcitationRetry` keeps deciding object-real.
- A cubic interpolation of the IMU samples in `doubleIntegrateAt` would remove the −0.2 % of 60 Hz
  linear interpolation on walks, in both estimators.
- On yard-real at stride 1 the IMU+GPS scale reads +0.7 %, +1.2 % and +2.4 % in three runs; the
  band-pass IMU estimate on the GT poses of the same keyframes reads −0.1 %. The reconstruction's local (acceleration-level) scale differs
  from its global Sim3 scale by 1–2 %. Not investigated.
