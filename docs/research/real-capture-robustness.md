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

## Phone-rate capture: keyframes past a fast pan (2026-10-21, review fixes 2026-10-22)

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
   Live, in refine's retry of the pending keyframes and in refine's offline SfM.
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

Stride 1 (yard-real): the default-seed runs gave 5.6–5.7e-4 after against 4.8–5.0e-4 before. The only
behavioural difference there: re-triangulation registers keyframe 64 (frame 192, the loop's corner: 12 map
points → 129, P3P 129/129) at once, the before code three keyframes later in a retry. The live models
before refine are alike (ATE 4.5–4.7e-3 on both sides); the difference arises in refine's loop closure,
which is sensitive to such details: with the live mapper's seed changed (1–3) the before code gives
5.13–5.35e-4 and the after code 5.25–5.49e-4, and the after code with re-triangulation off 5.25e-4. The
perturbed runs' spread covers both sides (means 5.1 vs 5.5e-4 over 7 / 6 live runs — at most a small
cost, far inside the 1e-3 limit); object-real is unchanged (2.66–2.72e-4).

Superseded first implementation (2026-10-21, gyro path before re-triangulation, base 18d73c9):
9 of 10 stride-3 runs 48/48 live (ATE 4.88–5.21e-4), 1 of 10 lost (`fin2`, above); gyro path alone
11 of 12 runs 48/48 (5.31–5.62e-4); re-triangulation alone 48/48 (5.32e-4).

### Limits

- Live mapping depends on the host's load (time-bounded BAs, the intrinsics the frame tracking
  uses), so runs diverge after the first pan; the phone-rate live focal is decided by the first
  ≈ 8 keyframes and can be off by ±3 % (`--start 1` / `--start 2`), which only refine's loop closure
  corrects. A loop refine does not close is now caught and handed to the offline SfM; when that
  registers fewer keyframes than the live model (it kept 47 of 48 in one forced stride-3 run), the
  open live model still wins on registered images.
- `--start 2` still loses the loop's corner (frames 179–191): no map points to register on; bridging
  it needs rotation-only tracking (gyro rotation with an unknown, later resolved translation).
- An all-registered healthy live model is less accurate than a from-scratch offline model on some
  inputs (stride 3 from frame 0: 5.3e-4 against 5.1e-4) — the time / accuracy trade-off of
  `liveModelHealth`, unchanged here.
- Not measured on a device; `abl-yard-shake` is not in this checkout (only `object-real` /
  `yard-real` datasets), so it was not replayed.
