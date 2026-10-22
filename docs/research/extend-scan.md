# Extend a scan ("Add more views")

Measurements from 2026-10-22 (Node unit tests and SwiftShader e2e runs, CPU features); the code may
have changed since — check the source before relying on a number. No iPhone data yet.

Code: `src/engine/live.ts` (`reopenCapture`, `segmentTimeShift`, time segments), `src/engine/engine.ts`
(`init` with `extend`), `src/sfm/inertial.ts` (`estimateGravity`), `src/harness/engine/pipeline.ts`
(`extendAt`), `src/app/screens/capture.ts` / `src/app/extend-scan.ts` (UI). Design: docs/ENGINE.md
"Extend".

## The time-base hazard

`performance.now()` restarts with every page load. Fed as is, a second page's IMU samples land
inside the stored range (`ImuLog.append` inserts them at their sorted position) and the gyro
integral spans both captures; stamps that coincide with stored ones are dropped as duplicates.
`src/engine/extend-capture.test.ts` (44-frame 55° synthetic orbit, frames 0–24 in page 1, 27–43 in
page 2 with frame / IMU / GPS times rebased to page 1's clock):

| | session time base (shift) | `extendTimeShift: false` |
|---|---|---|
| IMU strictly increasing, no page-2 sample in the stored range, none dropped | yes (shift 12 450 ms) | no |
| keyframes registered (one model, refine) | 16 / 16 (10 stored + 6 new, all 16 already live) | — |
| gyro breaks (live) / max gravity error / ATE | 0 / 0.19° / 0.49 mm | — |

## Gravity across the segment gap

`estimateGravity` rotates every accelerometer sample into the world with the vision-anchored
attitude: gyro integration from the previous keyframe plus the vision–gyro mismatch of the interval,
spread linearly over it. Across the gap of a continued capture the gyro sees nothing and the whole
unobserved rotation is spread over the interval, so the samples between the segments come out
tilted. Skipping samples inside keyframe intervals with an IMU gap (> 250 ms, except within 100 ms of
the keyframes):

| | gravity error, interpolated across the gap | skipped |
|---|---|---|
| `inertial.test.ts`: orbit, middle 3 s missing, rest 10 s later (IMU 1.5 s before its first keyframe) | 2.64° | 0.34° |
| e2e `extend40` (40-frame synthetic object, 1.6 fps, `--extend-at 0.6 --extend-skip-s 1`) | 5.39° | 0.073° |

## Harness plumbing check (`run-e2e.mjs --extend-at`)

`extend40`, worker engine, CPU features, no dense: reload after 24 frames (first refine 17 / 17
keyframes, 6.7 s), 2 frames skipped, re-init restored 17 keyframes (capture memory 6 of 399 MB, live
model rebuilt in 3.2 s), segment 2 shifted by 34 267 ms; the second segment's 9 keyframes all
registered live and in the final model.

| | single session | extended |
|---|---|---|
| registered / keyframes | 31 / 31 | 26 / 26 |
| ATE / extent | 1.06e-4 | 1.33e-4 (1.25×) |
| gravity / heading error | 0.0076° / −1.6° | 0.073° / −0.2° |

Neither run is metric (the dataset is too short for the IMU scale). The spec's real-dataset check
(`object-real --extend-at 0.6 --extend-skip-s 2` against `e2e/thresholds.json`, ATE within 1.5× of
the single session) and the value check (a higher second pass scored with `nvs.mjs --test-dataset
object-real-up12`) are still to be run.

## Frames during the live-model rebuild (review follow-up, 2026-10-22)

`restore` rebuilds the live model as one mapping job. The mapping back-pressure used to count it: the
second new keyframe of a continued capture waited for the whole rebuild (on a large scan on a phone
plausibly 30–90 s, past EngineClient's 30 s frame timeout). Now the back-pressure counts keyframe jobs
only, at most `maxMappingBacklog` keyframes queue behind the rebuild, and LiveUpdate.modelRebuild
reports it. `extend-capture.test.ts` ("slow rebuild": 10 stored keyframes, 400 ms added per rebuilt
keyframe, background mapping, `maxMappingBacklog` 1, one frame per macrotask like worker messages):

| | old back-pressure | now |
|---|---|---|
| frames of 17 answered while the rebuild (≈ 7–10 s) runs | 5 (one waited for the rest of it) | 9, none waited |
| keyframe decisions dropped behind the rebuild | — | 1 (`timings.rebuildLimited`) |
| largest mapping wait | rest of the rebuild | ≈ 0.5 s (one keyframe job, after the rebuild) |

The rebuild time per keyframe on an iPhone is still unmeasured; the restore log now prints it
("live model rebuilt from N keyframes … (X ms per keyframe)").

## Open

- Pairs across the gap get no gyro check (the IMU does not cover it): a false relocalisation on
  repeated structure is not caught by the gyro gates; one gyro bias is assumed for every segment.
- The world frame is re-estimated by every refine: earlier exports and measurements do not line up
  with the extended model.
- Restore time (rebuilding the live model) on an iPhone is unmeasured (3.2 s for 17 keyframes on
  SwiftShader / CPU features).
