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

## Phone-rate capture: keyframes past a fast pan (2026-10-21, review fixes and second review 2026-10-22)

Question: an iPhone analyses only 2–4 frames per second. Does live registration survive the fast
pans then? Replay at a third of the dataset rate: `node e2e/run-e2e.mjs yard-real --stride 3 --no-dense
--set 'engineConfig={"featureBackend":"cpu"}'` (2 analysed frames per second; worker engine; the host
shared, load 3–12 on 4 cores, so refine times are only comparable back to back), also from frames 1
and 2 (`--start 1` / `--start 2`: the other two phases of the same capture).

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

### Diagnosis of the first implementation's failures (2026-10-22, review)

- 1 in 10 stride-3 runs (`fin2`) still lost tracking. Cause: the gyro path ran before
  re-triangulation and registered the keyframe at frame 102 with 26 of 35 inliers, its refinement
  moving the rotation 1.92° — and its baseline 154° off GT: with far points (houses 18–47 m away,
  0.4–0.9 m baselines) rotation and translation couple, and that pose passed every gate (inliers,
  ratio, spread, σ, gyro check). The next keyframes chained off it (baselines 90–162° off) until
  tracking was lost at frame 126. The re-triangulated map gives P3P 84–85 of 99–100 inliers there
  (the correct pose: 0.3° / 3.5° off GT).
- `--start 1` (live 48/48 registered, offline SfM skipped, ATE 9.6e-4–1.0e-3 against 6.6e-4
  before). The before runs' 6.6e-4 is over the 33 of 49 keyframes the offline SfM kept: it dropped
  frames 178–238 — the loop's corner and everything after it, including the loop closure (it
  failed `registrationRate` and `gravityErrorDeg`). On those 32 common frames the new model is
  7.2–7.6e-4 with rotation p50 0.11–0.13° (before 0.22–0.23°). The real hazard was the tail: in 2
  of 13 earlier runs (first implementation and the interrupted fix) refine left the loop open (ATE 1.8e-2, rotation 2.4°, focal
  2.3 % off) and the health check skipped the offline SfM. Why: the live self-calibration of the
  focal wanders on the first ≈ 8 keyframes (weak geometry: far points, small baselines; the
  focal after the global BAs at 3 / 4 / 6 / 8 / 11 / 26 / 41 keyframes, instrumented: 745 / 754 /
  755 / 779 / 774 / 775 / 775 px, truth 753 — the jump between 6 and 8 from the local BAs, which
  refine the focal while they hold every image; from frame 0 it ends at 753, from frame 2 at 742),
  so the live model is mapped with the focal 2.8 % long and its loop's ends are ≈ 11° apart before
  refine (consecutive relative rotations 0.3–1.1° off GT, ≈ 2.8 % of each pan). Refine's global
  refinement corrects the focal (→ 754) and closes that gap when enough loop tracks merge; when
  they do not (split loop pairs 1–42, 1–44, 2–44 at ≈ 210 px), nothing else could: the health
  check's loop-correspondence count (limit 150) does not see 47–69 split correspondences, and the
  offline SfM, run with `refineOfflineSfM: 'always'`, split into 3 models and kept 32 of 48
  keyframes, so the live model would have won on registered images anyway.
- `--start 2` (34/51 live, 37/51 offline both before and after): the loop's corner (frames
  179–191, 13–20° per analysed frame for 6 s, little translation) leaves keyframes that see 0–11
  map points even after re-triangulation — below P3P's 30 and the gyro path's 15; the offline SfM
  (with the gyro path too) also keeps 37. Not a regression: the same keyframes are lost before and
  after. Beyond the corner the only registrable keyframe (frame 212, 118 map points) gets a P3P pose
  152° off the gyro (loop-start points matched to the wrong place) and is rightly rejected. Its
  offline ATE spread (1.70–1.88e-3) is that of the before code too: with both new paths off the
  merged code gave 1.87e-3, the before code 1.70–1.78e-3 (5 runs, also with other offline seeds).

### Change

1. `LiveMapper.triangulateNeighbours` (src/sfm/live.ts): a keyframe that sees fewer than 30 map
   points, or whose P3P fails, has up to 6 registered neighbours triangulated again (its pairs made
   their tracks multi-view), then P3P is tried again. On the stride-3 replay it created +75 / +18 /
   +114 / +71 / +37 points (visible points 1 → 75, 26 → 37, 29 → 143, 26 → 93, 3 → 40); the
   keyframe at frame 102 went 36 → 97 and P3P registered it.
2. Gyro-aided registration (src/geometry/pnp.ts `solvePnPKnownRotationRansac`, src/sfm/mapper.ts
   `registerKnownRotation`, src/engine/live.ts `predictRotation`) only when P3P still fails on the
   re-triangulated map (order: P3P, re-triangulation, P3P again, known rotation): the rotation the
   gyro propagates from the registered keyframe nearest in time (≤ 3 s; σ = 1° + 0.3°/s·Δt + 3 %
   of the rotation) is kept known: 2-point linear translation LO-MSAC (LO = depth-reweighted least
   squares + 4 robust 6-DOF LM steps), final 2-pass 6-DOF refinement; accepted with ≥ 15 inliers that
   are ≥ 50 % of the correspondences, a refined rotation ≤ 3 σ from the prediction, an inlier spread
   ≥ 5 % of the long image side, a baseline within 30° of the direction of the keyframe's best
   verified pair (≥ 1.5° triangulation angle; the `fin2` pose was 154° off it) and the gyro pose
   check. It needs the verified R_cam_device, not the gyro gates (`gyroGate: false` keeps it).
   Live, in refine's retry of the pending keyframes and in refine's offline SfM. (Second review,
   below: without a pair to check the baseline it is now refused, and by default it runs only in
   refine's offline SfM.)
3. `liveModelHealth` (src/engine/refine.ts) no longer skips the offline SfM for a live model with
   a keyframe registered with the gyro-predicted rotation (its rotation agrees with the gyro by
   construction), or with a loop-like split pair (a's points 8–300 px from b's features) between
   keyframes ≥ 1/3 of the capture apart — the ends of an unclosed loop, which a phone-rate capture
   shows on too few correspondences for the count limits. Closed models' loop-like pairs join
   keyframes ≤ 17 % of the capture apart (yard-real: 14 of 81 at stride 1, 7 of 48 at stride 3).
   With the gyro path in the offline SfM (2.) it registers what the live mapper did (`--start 1`:
   1 model of 48 instead of 3 models / 32), so a complete live model with an open loop meets a
   complete offline one and loses on explained correspondences (75.2 vs 78.0 % in run `a1-4`).
   One closed `--start 1` model had a 12 px split across 43 of 48 keyframes: the rule then costs
   the offline SfM's time only. Not used as evidence: the gyro-consistency median and the explained
   share (closed `--start 1` models 0.29–0.31° / 76.5–76.8 %, open ones 0.20–0.22° / 75.0–75.3 %:
   no threshold separates them), and registrations rescued by re-triangulation: P3P poses with
   ≥ 30 inliers that passed the gyro check (6 per stride-3 run: 5 within 0.07–0.30° relative
   rotation of GT, the one at the 5th keyframe 3.6° like the plain P3P one before it, 2.7°, the
   focal still wandering); counting them would run the offline SfM on every phone-rate capture
   (5–8 s of refine's 22) for a model choice that kept the live model in all 4 forced runs.
4. Diagnostics: every registration other than a plain P3P one is logged at debug level (map points
   before / after re-triangulation, P3P and known-rotation inliers, rejections by the gyro); a P3P
   pose the gyro rejected before the known rotation registered the keyframe is reported apart
   (`stats.gyroRecoveredPoses`, mapper `numRotationRejectsRecovered`).

Unit evidence: the solver recovers t to 1e-9 from 2 noise-free points; with 12 inliers among 70 %
outliers at 0.5 px it returns the robust optimum (what LM from the true pose reaches) in 200 / 200
seeded trials (rotation exact or 1° off at 2–8 m depth, 1° off at 5–40 m); that optimum is within
1 % of the 1 m baseline at 2–8 m in 200 / 200 (centre error p50 0.35, p95 0.75 cm; at 5–40 m p50 1.1,
p95 2.8 cm: the rotation ↔ translation coupling of 12 points). src/sfm/live.test.ts: a synthetic
yard walk with a 35° pan between keyframes (K − 2 to K turning to 26°) — keyframe K sees 20 map
points; P3P, the gyro path alone (its inliers lie in a sliver: spread gate) and re-triangulation
alone (27 points < 30) leave K and the rest of the walk pending, both together register all 26 (rotation max
< 0.5°); with the pan 0.5° narrower re-triangulation gives P3P its 30 points and the gyro is not
used; the two-view mechanism in isolation; the order (P3P on the map as it is, re-triangulation,
P3P, only then the gyro). src/sfm/mapper.test.ts the gyro path (26 / 17 good points register; 14 of
44, a 7 × 9 px cluster, a prediction 20° off or none do not; a reversed baseline is refused; a P3P
pose the gyro check rejects falls back to the known rotation). src/engine/gyro-registration.test.ts
`LiveSession.predictRotation` against a synthetic orbit's exact IMU (nearest keyframe, ≤ 3 s, σ,
the verified R_cam_device, independent of the gyro gates, IMU gaps).

### Results (2026-10-22, back to back: before = claude/next-level b2377c7, after = 97d4da1)

Every run (`lost` = frames reported lost; model = live / offline with its registered keyframes;
ATE = Sim3 ATE / scene extent; scale = metric scale error; refine s on the shared host; failed =
`e2e/thresholds.json` checks that fail). The IMU scale estimator of claude/next-level fixed the
stride-3 `scaleError` failure of 2026-10-21 on both sides.

| run | live kf reg | lost | offline SfM | model (reg) | ATE | rot° p50 | scale % | refine s | failed |
|---|---|---|---|---|---|---|---|---|---|
| yard-real s3 from 0, before `b0-1` | 19/48 | 38 | run: 29 of 48 not reg. | offline (48) | 5.10e-04 | 0.111 | -1.4 | 26.0 | – |
| yard-real s3 from 0, before `b0-2` | 19/48 | 38 | run: 29 of 48 not reg. | offline (48) | 5.10e-04 | 0.111 | -1.4 | 25.7 | – |
| yard-real s3 from 0, before `b0-3` | 19/48 | 38 | run: 29 of 48 not reg. | offline (48) | 5.09e-04 | 0.111 | -1.4 | 27.1 | – |
| yard-real s3 from 0, before `b0-4` | 19/48 | 38 | run: 29 of 48 not reg. | offline (48) | 5.10e-04 | 0.111 | -1.4 | 26.5 | – |
| yard-real s3 from 0, before `b0-5` | 19/48 | 38 | run: 29 of 48 not reg. | offline (48) | 5.09e-04 | 0.111 | -1.4 | 27.4 | – |
| yard-real s3 from 0, after `a0-1` | 48/48 | 0 | skipped | live (48) | 5.27e-04 | 0.118 | -0.8 | 22.3 | – |
| yard-real s3 from 0, after `a0-2` | 48/48 | 0 | skipped | live (48) | 5.30e-04 | 0.119 | -0.9 | 21.9 | – |
| yard-real s3 from 0, after `a0-3` | 48/48 | 0 | skipped | live (48) | 5.32e-04 | 0.119 | -0.9 | 22.8 | – |
| yard-real s3 from 0, after `a0-4` | 48/48 | 0 | skipped | live (48) | 5.29e-04 | 0.120 | -0.9 | 22.9 | – |
| yard-real s3 from 0, after `a0-5` | 48/48 | 0 | skipped | live (48) | 5.32e-04 | 0.119 | -0.9 | 23.3 | – |
| yard-real s3 from 0, after `off0` (both new paths off) | 19/48 | 38 | run: 29 of 48 not reg. | offline (48) | 5.09e-04 | 0.112 | -1.4 | 25.4 | – |
| yard-real s3 from 1, before `b1-1` | 20/49 | 38 | run: 29 of 49 not reg. | offline (33) | 6.64e-04 | 0.224 | -2.3 | 31.7 | registrationRate, gravityErrorDeg |
| yard-real s3 from 1, before `b1-2` | 20/49 | 38 | run: 29 of 49 not reg. | offline (33) | 6.55e-04 | 0.229 | -2.4 | 27.4 | registrationRate, gravityErrorDeg |
| yard-real s3 from 1, before `b1-3` | 20/49 | 38 | run: 29 of 49 not reg. | offline (33) | 6.61e-04 | 0.219 | -2.3 | 28.0 | registrationRate, gravityErrorDeg |
| yard-real s3 from 1, before `b1-4` | 20/49 | 38 | run: 29 of 49 not reg. | offline (33) | 6.61e-04 | 0.225 | -2.4 | 31.8 | registrationRate, gravityErrorDeg |
| yard-real s3 from 1, after `a1-1` | 48/48 | 0 | skipped | live (48) | 1.03e-03 | 0.142 | -0.5 | 26.6 | ateRelative |
| yard-real s3 from 1, after `a1-2` | 48/48 | 0 | skipped | live (48) | 9.63e-04 | 0.139 | -0.7 | 22.4 | – |
| yard-real s3 from 1, after `a1-3` | 48/48 | 0 | skipped | live (48) | 1.00e-03 | 0.127 | -0.4 | 27.7 | – |
| yard-real s3 from 1, after `a1-4` | 48/48 | 0 | run: unclosed loop: far loop-like pairs | offline (48) | 9.63e-04 | 0.150 | -0.4 | 28.2 | – |
| yard-real s3 from 2, before `b2-1` | 20/51 | 39 | run: 31 of 51 not reg. | offline (37) | 1.70e-03 | 0.323 | -2.1 | 33.3 | registrationRate, ateRelative, rotErrDegMedian |
| yard-real s3 from 2, before `b2-2` | 20/51 | 39 | run: 31 of 51 not reg. | offline (37) | 1.78e-03 | 0.349 | -1.7 | 33.7 | registrationRate, ateRelative, rotErrDegMedian |
| yard-real s3 from 2, after `a2-1` | 34/51 | 17 | run: 17 of 51 not reg. | offline (37) | 1.85e-03 | 0.405 | -1.4 | 34.5 | registrationRate, ateRelative, rotErrDegMedian |
| yard-real s3 from 2, after `a2-2` | 34/51 | 17 | run: 17 of 51 not reg. | offline (37) | 1.84e-03 | 0.403 | -1.3 | 34.5 | registrationRate, ateRelative, rotErrDegMedian |
| yard-real s3 from 2, before `sd11` (offline seed 11) | 20/51 | 39 | run: 31 of 51 not reg. | offline (37) | 1.78e-03 | 0.349 | -1.7 | 30.3 | registrationRate, ateRelative, rotErrDegMedian |
| yard-real s3 from 2, before `sd12` (offline seed 12) | 20/51 | 39 | run: 31 of 51 not reg. | offline (37) | 1.78e-03 | 0.349 | -1.7 | 31.1 | registrationRate, ateRelative, rotErrDegMedian |
| yard-real s3 from 2, before `sd13` (offline seed 13) | 20/51 | 39 | run: 31 of 51 not reg. | offline (37) | 1.78e-03 | 0.349 | -1.7 | 26.7 | registrationRate, ateRelative, rotErrDegMedian |
| yard-real s3 from 2, after `sd11` (offline seed 11) | 34/51 | 17 | run: 17 of 51 not reg. | offline (37) | 1.84e-03 | 0.403 | -1.3 | 30.0 | registrationRate, ateRelative, rotErrDegMedian |
| yard-real s3 from 2, after `sd12` (offline seed 12) | 34/51 | 17 | run: 17 of 51 not reg. | offline (37) | 1.80e-03 | 0.394 | -1.3 | 32.6 | registrationRate, ateRelative, rotErrDegMedian |
| yard-real s3 from 2, after `sd13` (offline seed 13) | 34/51 | 17 | run: 17 of 51 not reg. | offline (37) | 1.84e-03 | 0.403 | -1.3 | 32.2 | registrationRate, ateRelative, rotErrDegMedian |
| yard-real s3 from 2, before `nrs2` (no rolling shutter) | 20/51 | 39 | run: 31 of 51 not reg. | offline (37) | 1.68e-03 | 1.284 | -10.2 | 27.6 | registrationRate, ateRelative, rotErrDegMedian, scaleError, gravityErrorDeg |
| yard-real s3 from 2, after `ers2` (no rolling shutter) | 34/51 | 17 | run: 17 of 51 not reg. | offline (37) | 1.87e-03 | 1.415 | -10.2 | 29.1 | registrationRate, ateRelative, rotErrDegMedian, scaleError, gravityErrorDeg |
| yard-real s3 from 2, after `off2` (both new paths off) | 20/51 | 39 | run: 31 of 51 not reg. | offline (37) | 1.87e-03 | 0.394 | -2.0 | 29.1 | registrationRate, ateRelative, rotErrDegMedian |
| object-real s3, before `bo3-1` | 50/50 | 0 | skipped | live (50) | 6.00e-04 | 0.198 | +1.1 | 22.9 | – |
| object-real s3, before `bo3-2` | 50/50 | 0 | skipped | live (50) | 5.97e-04 | 0.199 | +1.1 | 22.5 | – |
| object-real s3, after `ao3-1` | 50/50 | 0 | skipped | live (50) | 5.97e-04 | 0.198 | +1.1 | 22.3 | – |
| object-real s3, after `ao3-2` | 50/50 | 0 | skipped | live (50) | 5.96e-04 | 0.201 | +1.1 | 19.3 | – |
| yard-real s1, before `by1-1` | 81/81 | 0 | skipped | live (81) | 4.98e-04 | 0.099 | +0.9 | 37.6 | – |
| yard-real s1, before `by1-2` | 81/81 | 0 | skipped | live (81) | 4.99e-04 | 0.100 | +0.9 | 46.2 | – |
| yard-real s1, before `by1-3` | 81/81 | 0 | skipped | live (81) | 4.98e-04 | 0.101 | +1.0 | 36.5 | – |
| yard-real s1, before `by1-4` | 81/81 | 0 | skipped | live (81) | 4.82e-04 | 0.098 | +0.8 | 61.0 | – |
| yard-real s1, after `ay1-1` | 81/81 | 0 | run: unclosed loop: split counts | offline (81) | 3.63e-04 | 0.091 | +0.1 | 59.0 | – |
| yard-real s1, after `ay1-2` | 81/81 | 0 | skipped | live (81) | 5.72e-04 | 0.100 | +1.2 | 44.6 | – |
| yard-real s1, after `ay1-3` | 81/81 | 0 | skipped | live (81) | 5.67e-04 | 0.095 | +1.0 | 40.7 | – |
| yard-real s1, after `ay1-4` | 81/81 | 0 | skipped | live (81) | 5.61e-04 | 0.107 | +1.2 | 63.0 | – |
| yard-real s1, before `ls1` (live seed 1) | 81/81 | 0 | skipped | live (81) | 5.35e-04 | 0.107 | +1.1 | 50.5 | – |
| yard-real s1, before `ls2` (live seed 2) | 81/81 | 0 | skipped | live (81) | 5.13e-04 | 0.100 | +1.1 | 58.7 | – |
| yard-real s1, before `ls3` (live seed 3) | 81/81 | 0 | skipped | live (81) | 5.29e-04 | 0.102 | +1.2 | 46.2 | – |
| yard-real s1, after `ls1` (live seed 1) | 81/81 | 0 | skipped | live (81) | 5.25e-04 | 0.100 | +0.8 | 60.2 | – |
| yard-real s1, after `ls2` (live seed 2) | 81/81 | 0 | skipped | live (81) | 5.49e-04 | 0.105 | +0.9 | 44.0 | – |
| yard-real s1, after `ls3` (live seed 3) | 81/81 | 0 | skipped | live (81) | 5.28e-04 | 0.098 | +0.9 | 45.1 | – |
| yard-real s1, after `rt0s1` (re-triangulation off) | 81/81 | 0 | skipped | live (81) | 5.25e-04 | 0.101 | +0.9 | 47.5 | – |
| object-real s1, before `bo1-1` | 89/89 | 0 | skipped | live (89) | 2.68e-04 | 0.047 | +0.3 | 57.2 | – |
| object-real s1, before `bo1-2` | 89/89 | 0 | skipped | live (89) | 2.72e-04 | 0.042 | +0.3 | 49.0 | – |
| object-real s1, after `ao1-1` | 89/89 | 0 | skipped | live (89) | 2.69e-04 | 0.045 | +0.4 | 56.5 | – |
| object-real s1, after `ao1-2` | 89/89 | 0 | skipped | live (89) | 2.66e-04 | 0.045 | +0.3 | 56.0 | – |

Summary: yard-real at stride 3 from frame 0 registers 48/48 live in 5 of 5 runs (before 19/48, 38
of 80 frames lost) and skips the offline SfM (refine 22–23 s against 26–27 s); its live model is
5.3e-4 / 0.12° against the before runs' offline model 5.1e-4 / 0.11° (the healthy-live-model
trade-off of refine, as at stride 1; with both new paths off the after code gives the before
result, 5.09e-4). From frame 1: 48/48 in 4 of 4 (before 20/49), all checks
pass but one ATE of 1.03e-3 (limit 1e-3; before: `registrationRate` and `gravityErrorDeg` fail
every run), and in `a1-4` the open loop was caught: offline SfM run, the gyro-aided offline model
(48/48) chosen, 9.6e-4 — the case that ended at 1.8e-2 before this fix. From frame 2: unchanged
failure (37/51 offline in both; ATE 1.80–1.87e-3 after, 1.70–1.78e-3 before, 1.87e-3 for the
after code with both new paths off). object-real at stride 3 and both datasets at stride 1: within
run-to-run noise (see the stride-1 note below); `ay1-1` ran the offline SfM for the older
split-count rule (a live loop that did not close at stride 1) and kept the better offline model.
Correction (second review): this is not "all checks pass but one". The work item's accuracy
criterion — ATE / extent and rotation p50 no worse than before — fails at stride 3: ATE in every
phase (from frame 0 5.27–5.32e-4 vs 5.09–5.10e-4; from frame 1 on the 32 common frames
7.2–7.6e-4 vs 6.6e-4; from frame 2 1.80–1.87e-3 vs 1.70–1.78e-3), rotation p50 from frames 0
(0.118–0.120 vs 0.111°) and 2 (0.39–0.41 vs 0.32–0.35°). From frame 1 the ATE sits at the 1e-3
limit (`a1-1` 1.03e-3, and the reviewer's forced-offline run 1.035e-3). The change buys
registration (no lost frames, 48/48 live), not accuracy.

Stride 1 (yard-real): the default-seed runs gave 5.6–5.7e-4 after against 4.8–5.0e-4 before. The
difference identified there: re-triangulation registers keyframe 64 (frame 192, the loop's corner: 12 map
points → 129, P3P 129/129) at once, the before code three keyframes later in a retry. The live models
before refine are alike (ATE 4.5–4.7e-3 on both sides); the difference arises in refine's loop closure,
which is sensitive to such details: with the live mapper's seed changed (1–3) the before code gives
5.13–5.35e-4 and the after code 5.25–5.49e-4. The after code with re-triangulation off gave 5.25e-4,
not the before's 4.8–5.0e-4 of that session, so this note did not show that re-triangulation was the
only difference (the gyro path registered nothing at stride 1: stats "0 live, 0 in refine" in every
after run, `rt0s1` too). Second review, below: with both live paths off the code is the before code's
(the gyro path is off live by default now), and such runs fall inside the before code's spread, which
is wider across sessions than this note assumed (4.8–5.3e-4). The perturbed runs' spread covers both
sides (means 5.1 vs 5.5e-4 over 7 / 6 live runs — at most a small cost, far inside the 1e-3 limit);
object-real is unchanged (2.66–2.72e-4).

Superseded first implementation (2026-10-21, gyro path before re-triangulation, base 18d73c9):
9 of 10 stride-3 runs 48/48 live (ATE 4.88–5.21e-4), 1 of 10 lost (`fin2`, above); gyro path alone
11 of 12 runs 48/48 (5.31–5.62e-4); re-triangulation alone 48/48 (5.32e-4).

### Second review (2026-10-22): the pair check fails closed, the gyro path offline by default

Findings checked against the code and the runs (all confirmed):

- The pair check failed open. Without a calibrated pair at a median triangulation angle ≥ 1.5°,
  `pairBaselineError` returned NaN, and `NaN > 30` is false, so the pose was accepted with no check.
  That is the typical case past a pan towards far scenery: in the 35° pan test (src/sfm/live.test.ts)
  K's pairs are at 1.26° and 0.57°.
- The live gyro path registered no keyframe in any run of 97d4da1. The stats read "0 live, 0 in
  refine" in every after run above and in the reviewer's 3 runs: re-triangulation gives P3P its
  points. Meanwhile nothing checks the path against translational aliasing. A centre shifted along
  the baseline passes the direction check. In the reviewer's synthetic rig, 20 aliased of 29
  correspondences registered a centre 30–60 % of the baseline off (0.16 / 0.33 m of 0.54 m). This
  change still registers it when a 3° pair qualifies and refuses it with only a 1° pair (Limits).
- The offline SfM tried the known rotation on its top-ranked candidate before it tried the others
  with P3P. Refine's model choice counted the offline model's gyro-aided registrations like any
  other: one such registration joined the 3 models of the `--start 1` offline SfM into 1 of 48.
- `gyroRecoveredPoses` missed one case. When the first call of a LiveMapper attempt had its P3P pose
  rejected and the second call registered the keyframe, the rejection was not counted as
  recovered.
- At stride 1 the gyro path registered nothing either (`rt0s1` too). The stride-1 note above
  therefore did not show that re-triangulation was the only difference. The runs below resolve it.

Change (src/sfm/mapper.ts, src/sfm/live.ts, src/engine/refine.ts, src/engine/live.ts):

1. `registerKnownRotation` refuses an image when no verified pair can check its baseline
   (`RegistrationDiagnostics.knownRotationRefused: 'noPair'`). The refusal comes before the solver
   runs, so it draws nothing from the seeded RNG. Unit evidence (src/sfm/mapper.test.ts) uses a wrong
   pose that passes every other gate: 26 false matches consistent with the centre mirrored through
   the reference image, at a rotation 0.7° from the prediction. It is refused with a 3° pair
   (baseline more than 150° off) and also with only a 1° pair, where no check is possible.
   In the 35° pan test K is now refused. With re-triangulation, K + 1 registers with the gyro
   rotation instead, because its 1.8° pair with K − 2 checks the baseline, and K then registers by
   P3P (all 26 keyframes, ATE / extent 1.0e-4). When the 0.57–1.26° pairs are admitted
   (`knownRotationPairMinTriAngleDeg: 0.5`), K registers with the gyro 0° and 1° off (seeded axes):
   its centre is 1.1 cm off a 0.5 m step right after registration, the rotation change equals the
   injected error, the baseline is 1.6° off the pair's, and the final ATE / extent is 1.0e-4.
2. `EngineConfig.gyroAidedRegistration` takes three values:
   - `'offline'` (default): refine's offline SfM only;
   - `'all'`: also live and in refine's retry of the pending keyframes (the old default);
   - `'off'`.
3. Offline SfM (`IncrementalMapper.registerOneOf`) tries P3P on every candidate first. It tries the
   known rotation only when no candidate registers. When the live model holds gyro-aided keyframes
   (`'all'`), the offline SfM uses P3P only, so it stays their independent check.
   Model choice (`liveModelBetter`): a model's gyro-aided registrations count only when the model
   shows no unclosed loop (`unclosedLoop`, which applies the rules of `liveModelHealth`). A wrong
   join of two fragments leaves the loop of a loop capture open.
4. Diagnostics:
   - LiveMapper reports a P3P pose rejected in an earlier call as recovered when a later call of the
     same attempt registers the keyframe. Rejected minus recovered then counts the rejected poses
     of keyframes that stayed unregistered.
   - `knownRotationRefused` gives the reason for each refusal (e.g. 'noPrediction' when no
     registered keyframe is within 3 s). The debug log prints it.
   - The e2e self-metrics report `gyro.aidedRegistrations` and `retriangulatedRegistrations`.

Results (every run, back to back on the shared host, load 0.8–6.6). Before = claude/next-level
7598a6e (b2377c7 plus a mock-engine fix); after = this change. The gyro-aided column comes from
refine's stats: "off" means the path was not configured there.

| run | live kf reg | lost | offline SfM | gyro-aided | model (reg) | ATE | rot° p50 | scale % | refine s | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| yard-real s3 from 0, before `rb0-1` | 19/48 | 38 | run: 29 of 48 not reg. (1 model) | – | offline (48) | 5.12e-04 | 0.108 | -1.4 | 26.2 | – |
| yard-real s3 from 0, before `rb0-2` | 19/48 | 38 | run: 29 of 48 not reg. (1 model) | – | offline (48) | 5.10e-04 | 0.111 | -1.4 | 22.3 | – |
| yard-real s3 from 0, before `rb0-3` | 19/48 | 38 | run: 29 of 48 not reg. (1 model) | – | offline (48) | 5.10e-04 | 0.111 | -1.4 | 22.0 | – |
| yard-real s3 from 0, after `r2-sanity` | 48/48 | 0 | skipped | off live | live (48) | 5.32e-04 | 0.119 | -0.9 | 21.8 | – |
| yard-real s3 from 0, after `ra0-1` | 48/48 | 0 | skipped | off live | live (48) | 5.30e-04 | 0.119 | -0.8 | 22.4 | – |
| yard-real s3 from 0, after `ra0-2` | 48/48 | 0 | skipped | off live | live (48) | 5.32e-04 | 0.119 | -0.9 | 18.6 | – |
| yard-real s3 from 0, after `ra0-3` | 48/48 | 0 | skipped | off live | live (48) | 5.32e-04 | 0.119 | -0.9 | 18.0 | – |
| yard-real s3 from 0, after, gyroAidedRegistration 'all' `rall0-1` | 48/48 | 0 | skipped | 0 live, 0 in refine | live (48) | 5.30e-04 | 0.118 | -0.9 | 18.0 | – |
| yard-real s3 from 1, before `rb1-1` | 20/49 | 38 | run: 29 of 49 not reg. (3 models) | – | offline (33) | 6.61e-04 | 0.219 | -2.3 | 32.6 | registrationRate, gravityErrorDeg |
| yard-real s3 from 1, before `rb1-2` | 20/49 | 38 | run: 29 of 49 not reg. (3 models) | – | offline (33) | 6.61e-04 | 0.224 | -2.3 | 26.2 | registrationRate, gravityErrorDeg |
| yard-real s3 from 1, before `rb1-3` | 20/49 | 38 | run: 29 of 49 not reg. (3 models) | – | offline (33) | 6.61e-04 | 0.219 | -2.3 | 27.2 | registrationRate, gravityErrorDeg |
| yard-real s3 from 1, before `rb1-4` | 20/49 | 38 | run: 29 of 49 not reg. (3 models) | – | offline (33) | 6.61e-04 | 0.219 | -2.3 | 27.1 | registrationRate, gravityErrorDeg |
| yard-real s3 from 1, after `ra1-1` | 48/48 | 0 | skipped | off live | live (48) | 9.53e-04 | 0.135 | -0.7 | 23.7 | – |
| yard-real s3 from 1, after `ra1-2` | 48/48 | 0 | skipped | off live | live (48) | 9.63e-04 | 0.139 | -0.7 | 22.2 | – |
| yard-real s3 from 1, after `ra1-3` | 48/48 | 0 | skipped | off live | live (48) | 9.63e-04 | 0.139 | -0.7 | 21.1 | – |
| yard-real s3 from 1, after `ra1-4` | 48/48 | 0 | skipped | off live | live (48) | 9.63e-04 | 0.139 | -0.7 | 22.1 | – |
| yard-real s3 from 1, after, offline SfM forced `rx1-1` | 48/48 | 0 | run (1 model) | off live, 1 offline | live (48) | 9.63e-04 | 0.139 | -0.7 | 25.9 | – |
| yard-real s3 from 1, after, offline SfM forced `rx1-2` | 48/48 | 0 | run (1 model) | off live, 1 offline | live (48) | 9.63e-04 | 0.139 | -0.7 | 26.6 | – |
| yard-real s3 from 1, after, offline SfM forced, gyro path off `rxoff1-1` | 48/48 | 0 | run (3 models) | off live, off offline | live (48) | 9.63e-04 | 0.139 | -0.7 | 30.5 | – |
| yard-real s3 from 1, after, gyroAidedRegistration 'all' `rall1-1` | 48/48 | 0 | skipped | 0 live, 0 in refine | live (48) | 9.63e-04 | 0.139 | -0.7 | 21.9 | – |
| yard-real s3 from 2, before `rb2-1` | 20/51 | 39 | run: 31 of 51 not reg. (2 models) | – | offline (37) | 1.78e-03 | 0.349 | -1.7 | 27.6 | registrationRate, ateRelative, rotErrDegMedian |
| yard-real s3 from 2, before `rb2-2` | 20/51 | 39 | run: 31 of 51 not reg. (2 models) | – | offline (37) | 1.78e-03 | 0.349 | -1.7 | 28.9 | registrationRate, ateRelative, rotErrDegMedian |
| yard-real s3 from 2, after `ra2-1` | 34/51 | 17 | run: 17 of 51 not reg. (2 models) | off live, 0 offline | offline (37) | 1.84e-03 | 0.403 | -1.3 | 30.3 | registrationRate, ateRelative, rotErrDegMedian |
| yard-real s3 from 2, after `ra2-2` | 34/51 | 17 | run: 17 of 51 not reg. (2 models) | off live, 0 offline | offline (37) | 1.84e-03 | 0.403 | -1.3 | 30.2 | registrationRate, ateRelative, rotErrDegMedian |
| object-real s3, before `rbo3-1` | 50/50 | 0 | skipped | – | live (50) | 6.00e-04 | 0.192 | +1.1 | 19.2 | – |
| object-real s3, after `rao3-1` | 50/50 | 0 | skipped | off live | live (50) | 6.07e-04 | 0.188 | +1.1 | 19.6 | – |
| yard-real s1, before `rby1-1` | 81/81 | 0 | skipped | – | live (81) | 5.30e-04 | 0.102 | +1.2 | 40.2 | – |
| yard-real s1, before `rby1-2` | 81/81 | 0 | skipped | – | live (81) | 5.32e-04 | 0.102 | +1.1 | 41.3 | – |
| yard-real s1, before `rby1-3` | 81/81 | 0 | skipped | – | live (81) | 5.24e-04 | 0.101 | +1.1 | 39.3 | – |
| yard-real s1, before `rby1-4` | 81/81 | 0 | skipped | – | live (81) | 4.84e-04 | 0.100 | +1.0 | 45.3 | – |
| yard-real s1, after `ray1-1` | 81/81 | 0 | skipped | off live | live (81) | 5.67e-04 | 0.095 | +1.0 | 40.1 | – |
| yard-real s1, after `ray1-2` | 81/81 | 0 | skipped | off live | live (81) | 5.70e-04 | 0.109 | +1.3 | 40.5 | – |
| yard-real s1, after, re-triangulation off (both live paths off) `rrt0s1` | 81/81 | 0 | skipped | off live | live (81) | 5.47e-04 | 0.103 | +1.1 | 37.7 | – |
| yard-real s1, after, re-triangulation off (both live paths off) `rrt0s1-2` | 81/81 | 0 | skipped | off live | live (81) | 5.22e-04 | 0.102 | +1.1 | 42.2 | – |
| yard-real s1, after, re-triangulation off (both live paths off) `rrt0s1-3` | 81/81 | 0 | skipped | off live | live (81) | 5.27e-04 | 0.101 | +1.2 | 40.1 | – |
| object-real s1, before `rbo1-1` | 89/89 | 0 | skipped | – | live (89) | 2.76e-04 | 0.049 | +0.3 | 57.2 | – |
| object-real s1, after `rao1-1` | 89/89 | 0 | skipped | off live | live (89) | 2.72e-04 | 0.047 | +0.3 | 55.1 | – |

Summary:

- **From frame 0:** 48/48 live and the offline SfM skipped, as with 97d4da1: 5.30–5.32e-4 / 0.119°,
  against the before code's offline model at 5.10–5.12e-4 / 0.108–0.111°. `'all'` gives the same
  result with 0 gyro-aided registrations.
- **From frame 1:** the loop closed in all 4 default runs: 48/48 live, 9.53–9.63e-4, offline SfM
  skipped. Before: 6.61e-4 over 33 of 49 keyframes, failing `registrationRate` and `gravityErrorDeg`.
  - On the 32 common frames the after runs give 7.0–7.2e-4 / 0.110–0.113°, against 6.61e-4 /
    0.219–0.224° before.
  - With the offline SfM forced, the fail-closed check and the P3P-first order still join it:
    1 gyro-aided registration and 1 model of 48 (`rx1-*`), against 3 models of which the largest
    has 32 with the path off (`rxoff1-1`). That registration counted, since the model showed no
    unclosed loop.
  - The live model was kept in the forced runs: 48 = 48 registered, explained 76.6 vs 77.9 % (inside
    the 1.5 % gap), then the reprojection error. No open loop occurred in these default runs, so the
    `a1-4` rescue was not exercised. The forced runs show the offline model it needs.
- **From frame 2:** unchanged. 37/51 offline on both sides, 1.84e-3 / 0.403° against 1.78e-3 /
  0.349°, 0 gyro-aided offline.
- **object-real:** within noise (6.07 vs 6.00e-4 at stride 3; 2.72 vs 2.76e-4 at stride 1).
- **Stride 1 (yard-real):** the before code gives 4.84–5.32e-4 here (4 runs; 4.82–4.99e-4 in the
  earlier session). After, the default gives 5.67–5.70e-4. With both live paths off (re-triangulation
  off; the gyro path is off live by default) it gives 5.22–5.47e-4, inside the before spread. Without
  re-triangulation and a live predictor, LiveMapper runs the before code's path: same calls, same
  RNG draws. So the stride-1 shift comes with re-triangulation, i.e. keyframe 64 registering at once.
  Over 2 sessions the after code kept a live model of 5.61–5.72e-4 in 5 of 6 runs (the sixth ran the
  offline SfM: 3.63e-4), against 4.82–5.32e-4 in the 8 before runs. The old `rt0s1` (5.25e-4, gyro path
  configured) falls inside the before code's spread across sessions. Its difference from the 4.8–5.0e-4
  of its own session is unexplained. Candidates are the 15–29-point retries and RNG draws of the
  configured gyro path, both gone by default now.

Acceptance: not met on the e2e accuracy criterion. ATE / extent and rotation p50 are worse than
before at stride 3:

- from frame 0: ATE +4 %, rotation +7–10 %;
- from frame 2: ATE +3 %, rotation +15 %;
- from frame 1, on the comparable 32 frames: ATE +6–8 %, while the rotation p50 halves.

From frame 1 the ATE sits at the 1e-3 limit. Two of the 14 frame-1 runs of 97d4da1 and this change
went over it (`a1-1` 1.03e-3, the reviewer's forced-offline run 1.035e-3: the engineer's 4, the
reviewer's 2, the 8 above); the 8 above give 9.53–9.63e-4. The frame-0 gap is not only the skipped offline SfM: with the offline SfM forced (the
engineer's `x0always`) the choice kept the live model (5.31e-4). What the slice delivers is
registration at a phone's rate: 48/48 live instead of 19–20, no lost frames, refine 18–24 s
instead of 22–33 s. It does not improve accuracy.

### Limits

- Live mapping depends on the host's load (time-bounded BAs, the intrinsics the frame tracking
  uses), so runs diverge after the first pan; the phone-rate live focal is decided by the first
  ≈ 8 keyframes and can be off by ±3 % (`--start 1` / `--start 2`), which only refine's loop closure
  corrects. A loop refine does not close is now caught and handed to the offline SfM; when that
  registers fewer keyframes than the live model (it kept 47 of 48 in one forced stride-3 run), the
  open live model still wins on registered images.
- The gyro path does not detect translational aliasing: false matches on structure that repeats
  along the walking direction can shift the centre along the baseline, and the direction check does
  not see that. It does not detect a wrong offline join in a capture without a loop either (the
  model choice only discounts gyro-aided registrations of a model with an unclosed loop). This is
  one reason the path runs only offline by default. No aliasing capture exists to measure it.
- `--start 2` still loses the loop's corner (frames 179–191): no map points to register on; bridging
  it needs rotation-only tracking (gyro rotation with an unknown, later resolved translation).
- An all-registered healthy live model is less accurate than a from-scratch offline model on some
  inputs (stride 3 from frame 0: 5.3e-4 against 5.1e-4) — the time / accuracy trade-off of
  `liveModelHealth`, unchanged here.
- Not measured on a device; `abl-yard-shake` is not in this checkout (only `object-real` /
  `yard-real` datasets), so it was not replayed.
