# Refine performance (Apple M4 / Metal)

Measurements from 2026-10 (e2e worker runs on the real GPU, `E2E_GPU=host`, fast preset, no dense); the code may have changed since — check the source before relying on a number.

Code: `src/engine/refine.ts` (`liveModelHealth`, `splitPairs`, `yieldTask`), `src/engine/cpu-pool.ts`,
`src/workers/cpu.worker.ts`, `src/ba/solver.ts` (`BundleAdjuster`), `src/sfm/mapper.ts`
(`adjustGlobalBundleSliced`), `src/sfm/rolling-shutter.ts` (`rsSearchSteps`,
`searchRollingShutterBatched`). Before / after runs: e2e/README "Refine performance on Metal".

## Where the time went (before)

Host load ≈ 6; `stats.*Ms`:

| | object-real | yard-real |
|---|---|---|
| refine total | 153 s | 97 s |
| matching | 27 s (vocabulary 9: k-means 2 + quantising 90 keyframes 7; CPU verification 11.5; waiting for the GPU matcher 6) | 32 s |
| reconstruction | 125 s (live refinement 1.5, rolling shutter 37 — 28 subset BAs 30 + 2 re-adjustments 7 —, offline SfM 59, principal point 3, per-image focal 24) | 64 s (offline SfM 42) |

All CPU, one thread: BAs everywhere, the offline SfM about half of the reconstruction.

## Offline SfM only for an unhealthy live model

The from-scratch SfM is the independent check that rescued live models made of fragments (gyro
breaks), with misplaced or unregistered keyframes, or with an unclosed loop; a live model without
those symptoms was kept in every run (it ties with the offline one). `splitPairs` evidence:

- closed yard-real live / offline models: 950–1940 split / 60–190 loop-like correspondences of 168 k;
  object-real 0 / 0;
- unclosed live models (live retrieval and spatial pairs off, `livePairs: {retrieval: 0, spatial: 0}`,
  or a loaded host; the loop closes live in some runs, not in others): 2950–5650 split, ≥ 1200
  loop-like when the gap is within 0.4 f (gaps 12–15 px, 50–255 px; rotation 0–8° off the model's; a
  gap beyond it: 98 loop-like, caught by the split total) — the offline model then won (ATE 2.3–2.5e-4
  vs ≈ 3e-4);
- false pairs on repeated structure are split too, but sporadic and mostly hundreds to thousands of
  pixels off.

## Resumable sliced BA

Building the problem, the solver structure and the linearisation once and resuming the LM state
across yields (instead of rebuilding per slice): Node, 90-image orbit, 152 k observations, 3.0–3.9 →
2.3–2.5 s per global BA; results bit-identical to one uninterrupted solve.

## BA kernels

- Schur elimination keeps a point's pose row block in registers for all its column blocks
  (bit-identical).
- `SkylineCholesky.factor` two rows × two columns per step: dense 541² 79 → 38 ms, 1200² 939 → 375 ms
  (Node). The rolling-shutter subset BA (≈ 540 reduced parameters, dense) spent ≈ 60 % of its time in
  the factorisation.

## Rolling-shutter search

±25 ms probes only along a direction clearly best at ±10 ms; a parabola vertex on the best point
confirmed at ±tol/2 instead of bisecting; independent hypotheses scored in parallel on the CPU
workers. Same estimate (Node bench −21.285 ms, curve identical between the sequential and batched
search); e2e 28 → 17–18 subset BAs in ≈ 10 rounds.

## Matching

- The one-deep GPU prefetch never overlapped: the queued match needs the event loop to encode and
  submit, and the synchronous verification ran first. One macrotask (`yieldTask`, a message to
  ourselves — no 4 ms timer clamp) after a match arrives lets the next one reach the GPU: GPU wait
  7.6 → 2.2 s on object-real.
- CPU helper workers (nested in the engine worker, Safari 15.5+; default hardware threads − 2, ≤ 3)
  run pair verification, vocabulary quantisation and rolling-shutter hypotheses.

## Not done, from the measurements

- Pose-guided refine verification ('F' guide from the live poses / seeded LO-MSAC): the verification
  time is in the retrieval candidates (object-real 8.6 of 10.1 s; yard-real 18 of 23 s, 11 s of it in
  pairs that fail), which must stay unguided — on non-overlapping views a guided matcher returns
  band-consistent random matches that then "verify" with the guide's own geometry (false pairs
  agreeing with the model). Sequential + spatial pairs, where guidance would be safe, cost 1.5–2 s in
  total; seeding RANSAC does not lower its iteration count, which is set by the inlier ratio (or the
  cap).
- Per-image focal rounds (cap 6) and the two rolling-shutter passes are kept: both are accuracy-tuned
  (`real-capture-robustness.md`); they only got faster BAs.

## Result

Back-to-back runs at load 8–11: refine object-real 232.8 → 44.8 s, yard-real 101.1 → 29.9 s (whole
e2e 402 → 194 s, 200 → 135 s), accuracy and dense metrics unchanged within the run-to-run noise; a live
model with an open loop still gets the offline SfM.

## Remaining

The per-image focal alternation (13 full BAs, ≈ 16–24 s on object-real) and the rolling-shutter
re-adjustments are sequential global BAs on the engine thread; the next step is a parallel / GPU BA
(point-partitioned linearisation + Schur on the workers or in WGSL).
