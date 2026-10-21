# Real-capture robustness: gyro gates, forced / best-of-window keyframes, rolling shutter, focus breathing

Measurements from 2026-10 (e2e worker runs, CPU features, and synthetic tests); the code may have changed since — check the source before relying on a number.

Code: `src/sfm/keyframes.ts`, `src/sfm/gyro.ts`, `src/sfm/rolling-shutter.ts`, `src/engine/live.ts`,
`src/engine/refine.ts`, `src/engine/refine-camera.ts`. Per-dataset result tables: e2e/README
"Real-capture robustness" (with its sub-sections "Adversarial review" and "Principal-point prior in the
per-image focal rounds") and "Yard breakdown".

## The problem these address

On e2e `yard-real` / `abl-yard-shake` a fast pan left a 35–45° keyframe gap (blur rejection +
sharpness deferral). The keyframes after it could not register, and the walk was reconnected through
FALSE two-view pairs (repeated fence boards: 60–80 inliers, rotation 100°+ off) that nothing checked
against the gyro: one model made of rigid fragments joined by a wrong rotation (ATE 0.10 of the
extent, rotation 152°). Rolling shutter and focus breathing were absorbed by the principal point
(2.5–8 px ⇒ 0.25–0.7° rotation error) and biased the IMU scale / time offset.

## Gyro gates

- Every pair rejected by the gyro gate on the e2e datasets (12 object, ≈ 105 clean yard, ≈ 130
  yard-real) had a vision rotation ≥ 6° off the ground truth, while the bias-compensated gyro was
  within 0.2–1.6° of the truth. Wrong pairs are 50–140° off; correct ones ≤ 2–3°.
- The tolerance (5° + 0.3°/s·|Δt| + 5 % of the rotation) is NOT tightened when a gyro bias is known:
  the bias estimate comes from rolling-shutter-biased vision rotations (yard-real x component 0.5
  instead of 3.0 mrad/s ⇒ 5° over a 32 s loop closure), and 0.1°/s rejected correct loop-closure pairs.
- Spatial candidates while lost: with the stale keyframe's own view direction the "neighbours" looked
  85–100° away; using the gyro-propagated rotation of the last registered keyframe fixed that.

## Model choice (live vs offline, refine)

Order: fewer gyro breaks (beyond isolated outliers), more registered images, then the share of
verified two-view correspondences whose two features observe the same 3-D point
(`explainedCorrespondenceRatio`) when it differs by > 1.5 points, then the mean reprojection error.

A live model that did not close the 1.03-lap yard loop (start and end tracks stay apart: PnP of the
returning keyframes follows the recent points) had the same reprojection error as the closed
offline one (0.392 vs 0.393 px) but explained fewer correspondences (84.2 vs 87.1 %): ATE 2.5e-3 vs
2.6e-4. Clean captures differ by 0.4–0.8 points (choice unchanged).

## Gyro calibration

- `LiveSession.calibrateGyro` runs `estimateTimeOffset` on the refined live model BEFORE the
  rolling-shutter estimation, which needs ω at the right time: with the offset still 0, yard-real's
  readout came out 19.1 instead of 21.9 ms and the principal point 8.6 instead of 0.5 px off.
- The same calibration every 10 registrations DURING capture (`EngineConfig.liveGyroCalibration`,
  off by default) changes the keyframe selection and measured equal or worse: yard ATE 2.4–2.7e-4
  (3 runs) vs 2.1–2.2e-4, yard-real 7.0e-4 vs 3.3e-4, object 2.19e-4 vs 2.05e-4.
- The offset that the readout passes query ω with comes from the DISTORTED live model and is biased
  by where the features sit in the frame (yard-real −30.5 / abl-yard-rs −27.9 ms, truth −25; the
  world alignment later finds −25.1 on the corrected model). Re-calibrating before the second pass
  gives −25.1 / −24.9 ms but measured neutral on abl-*-rs / object-real and worse on yard-real
  (principal point 0.4 → 1.3 px, rotation 0.10 → 0.13°), so it is not done.
- IMU scale R² is computed with the Huber weights of the fit: one misregistered keyframe had turned
  yard-real's R² into 0.01, rejecting the IMU scale for the far less precise GPS one.

## Rolling shutter

- Rotation-only model on close-range orbits: the sideways camera motion (≈ |ω| × object distance)
  cancels the yaw term at the orbit centre, so the estimate settles on an EFFECTIVE readout
  (abl-object-rs: −16.5 of −22 ms) that under-corrects the roll term. Result: a constant pitch error
  of every camera (−0.20 / −0.26° on abl-object-rs / object-real; rotation p50 0.21 / 0.27°), and the
  object-real dense cloud sat 3.3 mm off the ground truth once aligned on the cameras (the shape
  itself, ICP-aligned, as good as the baseline: 0.35 mm p50). With the translational term: rotation
  p50 0.04 / 0.07°.
- A fixed-points (motion-only) hypothesis test finds axis and direction but only 15–20 % of the
  readout (points triangulated from RS-distorted pans resist), hence the subset BA with free points.
- Search cost: ≈ 12–18 subset BAs, ≈ 3–6 s on SwiftShader (before the refine-performance work, see
  `refine-performance.md`).
- Second pass (narrowed 0.85 / 1 / 1.15 × around the first estimate on the re-adjusted model): the
  short BAs start from the input model, and one built on distorted keypoints pulls the readout towards
  0. Synthetic orbit with 2 % focal error: −18.5 → −21.7 of −22 ms; e2e object-real −21.8 → −21.4,
  abl-object-rs −19.1 → −19.1, yard-real 22.0 → 22.0.
- Dense: the corrected poses are those of the frame stamp while the pixels are still skewed, so
  without warping the views dense got worse although the poses got better (object-real accuracy p50
  1.3 → 2.7 mm, completeness 0.993 → 0.976); hence `rsCorrectPinholeImage` on every undistorted view.
- Exported POINTS2D re-derived after a reload (`modelKeypoints`) are within 1e-3 px of the in-session
  correction on the same model.
- Limit: abl-object-rs still converges to 19.1 of 22 ms (its poses nevertheless reach the clean
  dataset's accuracy, ATE 2.2e-4, rotation 0.04°). The translational term is first order with
  keyframe-rate velocities (hand shake between keyframes is not in it), and the dense images get it
  only at each image's median depth.
- Memory: `rsOriginal` (a copy of every corrected keyframe's keypoints) is ≈ 11 % of the features.

## Per-image focal (focus breathing)

- Acceptance needs the focal lengths to be smooth in time (lag-1 autocorrelation ≥ 0.6): per-image
  focal lengths that absorbed residual rolling shutter correlated −0.84 with |ω| at autocorrelation
  −0.14.
- The per-image focal lengths recover only ≈ 0.6 of the breathing (focal ↔ dolly); a looser 1 %
  prior changes nothing. The shared-camera / per-image alternation converges linearly
  (abl-object-breath, principal-point error after 1 / 2 / 4 / 6 rounds: 4.6 / 3.6 / 2.0 / 1.0 px,
  rotation 0.36 / 0.29 / 0.17 / 0.08°).
- The 6-round cap STOPS the alternation; it has not converged: on object-real / abl-object-breath
  c_y still moves 0.4–0.5 px per round at round 6 (reprojection error flat, 0.509–0.511 px) and walks
  PAST the ground truth. Object-real with 12 rounds stops at round 8 with c_y +0.43 px from the image
  centre against −1.0 px GT: pp error 1.46 px, rotation 0.118° (6 rounds: 0.87 px, 0.077°);
  abl-object-breath extrapolates to ≈ +0.5 px. The fixed point is biased along the orbit's weak
  pitch ↔ c_y mode (the 0.5 % prior keeps part of the breathing in the shared camera), so the cap acts
  as an early-stopping regulariser tuned on these two datasets.
- Weak principal-point prior in the shared re-refinements (`EngineConfig.perImageFocalPPSigmaPx`,
  OFF): does not make the alternation converge. At σ 2.5 px the per-round contraction of the c_y steps
  is unchanged (object-real 0.78 → 0.76, abl-object-breath 0.84 → 0.81); only the extrapolated fixed
  point moves (object-real +1.2 → +0.6 px, GT −1.0). With the 6-round cap it helps object-real (pp
  0.90 → 0.59 px, rotation 0.078 → 0.055°) but hurts abl-object-breath (0.92 → 1.27 px, 0.078 →
  0.105°); anchored at the image centre it is worse (object-real 1.45 px / 0.116°); σ 1 px
  (contraction 0.67) lets the anchor — the breathing-biased input pp — set the fixed point and doubles
  the focal error (0.09 → 0.20 %, pp 0.91 px). So the 6-round cap stays; a joint BA with intrinsics
  shared ACROSS cameras remains the principled fix.
