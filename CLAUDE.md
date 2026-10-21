# sfm

On-device photogrammetry in mobile Safari (WebGPU), no server. See README.md for the app and
docs/ARCHITECTURE.md for the conventions and module contracts.

## Commands

* `npm run typecheck`, `npm test` (Node unit tests), `npm run test:gpu` (headless Chromium +
  SwiftShader; keep GPU test images ≤ 640 px), `npm run build`.
* One file: `npx vitest run --project unit <path>` / `--project gpu <path>`.
* End-to-end harnesses: see e2e/README.md.

## Where knowledge lives

* **docs/ARCHITECTURE.md** — goals, coordinate/camera conventions (§2), iOS WebGPU and capture
  rules (§2.1, §2.2), directory layout, module contracts, pipeline overview. Short and stable.
  Update it in the same change when you alter a convention, a public contract or the pipeline.
  Do not append notes, logs or review findings to it.
* **docs/ENGINE.md** — engine threads, worker protocol, UI flow.
* **Source file header comments** — how a module works, why its defaults are what they are,
  platform gotchas and known limits. When you change a file's behaviour, update its header in
  the same change so the comment never describes old code.
* **docs/research/** — dated research and measurement reports (iOS platform facts, tuning
  sweeps, benchmarks). Snapshots: add a new dated report or section rather than rewriting
  history, and cite them from the code they justify.
* **Commit messages** — what changed and the evidence for it.

## Rules

* Follow §2 conventions exactly (row-major matrices, OpenCV camera frame, world-to-camera poses,
  pixel centres at integers, `R_cam_device(θ)`); shared types live in src/core/types.ts.
* Follow the §2.1 iOS WebGPU rules (one heavy command buffer in flight, pipelines created up
  front, no vec3 in buffers, no f32 atomics or subgroups, ~1.2 GB tab budget).
* Seeded `Rng` (src/math/random.ts) for all randomness.
