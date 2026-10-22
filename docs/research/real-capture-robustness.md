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

## Phone-rate capture: keyframes past a fast pan (2026-10-21)

Question: an iPhone analyses only 2–4 frames per second. Does live registration survive the fast
pans then? Replay at a third of the dataset rate: `node e2e/run-e2e.mjs yard-real --stride 3 --no-dense
--set 'engineConfig={"featureBackend":"cpu"}'` (2 analysed frames per second; worker engine; the host
shared, load 5–20 on 4 cores, so refine times are only comparable within a column of similar load).
Before = commit 18d73c9, run back to back with the change.

### Diagnosis (instrumented replays of 18d73c9, debug dump compared with GT)

- Matching is fine: the keyframes past the pan have 200–800 verified inliers with their
  predecessors, and those consecutive pairs are correct (relative rotation 0.02–0.45° off GT; the
  longer-range pairs up to 6°). Frames are reported `lost` because 3 keyframes in a row stay
  unregistered, not because tracking fails. The gyro gates are not the cause either: no
  registration was rejected live (61 verified pairs were).
- Pan 2 (trajectory 15 s) rotates 33.7° between analysed frames 90 and 93 (1.0 rad/s): the hard
  rotation limit (20°) cannot be honoured at this rate. The keyframes after it look at a region only
  the last 2–3 keyframes saw, from 0.4–0.9 m apart (houses 18–47 m away; median triangulation
  angles of those pairs 1.6–2.9°).
- The keyframe at frame 102 sees 36 map points (it has 219 / 171 inliers with the keyframes at
  frames 96 / 93). P3P finds 27 inliers (75 %) and the pose is right (relative rotation 0.69°,
  baseline direction 0.77° off GT), but the 30-inlier floor rejects it. A pending keyframe
  triangulates nothing, so the next ones see 7, 3, 2, 0, 6 map points: 29 of 48 keyframes stay
  pending, 38 of 80 frames report `lost`, and refine's retry of
  the pending keyframes fails too (27 inliers of 49; 27 / 21 / 22 visible points) → offline SfM.
- Lowering the P3P floor is no fix: in that retry the 12–13-inlier P3P poses of the keyframes at
  frames 126 / 132 were 83–87° off.
- Root cause: the map past the pan is a set of TWO-VIEW tracks (features matched in one other
  keyframe only), which the triangulator does not create points from (`ignoreTwoViewTracks`, as
  COLMAP). The batch SfM has every pair from the start; live, the third view arrives later, and
  nothing triangulates those tracks again before the third view registers — on exactly those points.

### Change

1. `LiveMapper.triangulateNeighbours` (src/sfm/live.ts): a keyframe that sees fewer than 30 map
   points, or fails to register, first has up to 6 registered neighbours triangulated again (its
   pairs made their tracks multi-view), then registration is retried once. On the stride-3 replay it
   created +75 / +18 / +114 / +71 / +37 points (visible points 1 → 75, 26 → 37, 29 → 143, 26 → 93,
   3 → 40); without the gyro path the keyframe at frame 102 went 36 → 97 and P3P registered it.
2. Gyro-aided registration (src/geometry/pnp.ts `solvePnPKnownRotationRansac`, src/sfm/mapper.ts
   `registerKnownRotation`, src/engine/live.ts `predictRotation`): when P3P still fails, the rotation
   the gyro propagates from the registered keyframe nearest in time (≤ 3 s; σ = 1° + 0.3°/s·Δt + 3 %
   of the rotation) is kept known: 2-point linear translation LO-MSAC (LO = depth-reweighted least
   squares + 4 robust 6-DOF LM steps), final 2-pass 6-DOF refinement; accepted with ≥ 15 inliers that
   are ≥ 50 % of the correspondences, a refined rotation ≤ 3 σ from the prediction, an inlier spread
   ≥ 5 % of the long image side and the gyro pose check. The keyframe at frame 102: 27 inliers,
   rotation moved 0.54° by the refinement.

Unit evidence: the solver recovers t to 1e-9 from 2 noise-free points; with 12 inliers among 70 %
outliers at 0.5 px it returns the robust optimum (what LM from the true pose reaches) in 200 / 200
seeded trials (rotation exact or 1° off at 2–8 m depth, 1° off at 5–40 m); that optimum is within
1 % of the 1 m baseline at 2–8 m in 200 / 200 (centre error p50 0.35, p95 0.75 cm; at 5–40 m p50 1.1,
p95 2.8 cm: the rotation ↔ translation coupling of 12 points). src/sfm/live.test.ts isolates the
two-view mechanism (a keyframe seeing 10 map points and the walk after it stay pending without
re-triangulation, all register with it); src/sfm/mapper.test.ts the gyro path (26 / 17 good points
register; 14 of 44, a 7 × 9 px cluster, a prediction 20° off or none do not).

### Results (yard-real unless noted; `lost` = frames reported lost; ATE = Sim3 ATE / extent)

| run | live kf reg | lost | offline SfM | model | ATE | rot° p50 | refine s |
|---|---|---|---|---|---|---|---|
| stride 3, before (lead's `s3-base` / `pre-s3`) | 19/48 / 19/48 | 38 / 38 | run (29 not reg.) | offline | 5.10e-4 / 5.41e-4 | 0.111 / 0.115 | 21.4 / 59.3 (load 18) |
| stride 3, after: 9 of 10 runs | 48/48 | 0 | skipped | live | 4.88–5.21e-4 | 0.104–0.113 | 17.0–19.4 |
| stride 3, after: 1 of 10 runs (`fin2`) | 23/48 | 31 | run (25) | offline | 1.22e-3 | 0.376 | 23.4 |
| stride 3, gyro path only (no re-triangulation): 11 of 12 runs | 48/48 | 0 | skipped | live | 5.31–5.62e-4 | 0.089–0.099 | 22–53 |
| stride 3, gyro path only: 1 of 12 runs (`gyro1`, load 20) | 21/48 | 34 | run | offline | 5.16e-4 | 0.124 | 76.3 |
| stride 3, re-triangulation only (`gyroAidedRegistration: false`) | 48/48 | 0 | skipped | live | 5.32e-4 | 0.119 | 21.3 |
| stride 3 from frame 1 (`--start 1`), before → after | 20/49 → 48/48 | 38 → 0 | run → skipped | offline (33/49 reg.) → live | 6.6e-4 → 9.8e-4 | 0.219 → 0.127 | 37.6 → 22.8 |
| stride 3 from frame 2 (`--start 2`), before → after | 20/51 → 34/51 | 39 → 17 | run → run | offline 37/51 both | 1.75e-3 → 1.88e-3 | 0.341 → 0.409 | 35.9 → 39.9 |
| stride 1, before (`pre-s1` / back-to-back `pre2-s1`) | 82/82 / 81/81 | 0 | run / skipped | offline / live | 2.49e-4 / 5.20e-4 | 0.078 / 0.102 | 149 / 51 |
| stride 1, after (`fin-s1` / back-to-back `fin2-s1`) | 81/81 | 0 | skipped | live | 5.46e-4 / 5.51e-4 | 0.107 / 0.103 | 38 / 37 |
| object-real stride 3, before → after | 50/50 | 0 | skipped | live | 5.94e-4 → 6.00e-4 | 0.209 → 0.192 | 49 → 18 |
| object-real stride 1, before → after | 89/89 | 0 | skipped | live | 2.67e-4 → 2.73e-4 | 0.047 → 0.044 | 163 → 51 |

Every run passes the same `thresholds.json` checks before and after except where noted: yard-real
at stride 3 fails `scaleError` before and after (−6.4 %: the IMU scale is rejected for too little
excitation, 0.010 < 0.08 m/s², so GPS sets it); `--start 1` before also failed `registrationRate`
and `gravityErrorDeg`, after only `scaleError`; `--start 2` and `fin2` fail ATE / rotation.
Per run, re-triangulation registered 5 keyframes live at stride 3 and 1 at stride 1, the gyro path
1 at stride 3 (2 and 2 in `fin2`) and none at stride 1 or on object-real.

### Limits

- Live mapping depends on the host's load (time-bounded BAs, the intrinsics the frame tracking
  uses), so replays diverge from frame 57 on and the region past the pan is decided by a few points:
  1 of 10 runs still lost the chain at frame 138. Its live model was poor (0.64 px), and the rolling
  shutter estimated on it biased the keypoints the offline SfM then ran on (0.51 instead of 0.35 px):
  ATE 1.22e-3, worse than the before runs' offline model.
- `--start 2` still loses the loop's corner (frames 179–191, 13–20° per analysed frame for 6 s):
  the keyframes there see 0–11 map points even after re-triangulation (fewer than the gyro path's 15).
- An all-registered live model skips the offline SfM, so `liveModelHealth` alone must catch a live
  loop that did not close: with the gyro path only, `--start 1` registered 49/49 but drifted 0.5 m /
  3° at the loop's ends (ATE 6.9e-3) with just 16 loop-like split correspondences — healthy by its
  rules. With re-triangulation the same replay closed the loop (9.8e-4).
- Not measured on a device; `abl-yard-shake` is not in this checkout (only `object-real` /
  `yard-real` datasets), so it was not replayed.
