// Held-out test views of the novel-view evaluation (e2e/nvs.mjs): which dataset frames are scored,
// and their cameras in the RUN's world frame. Pure JS (no Node imports); unit-tested in
// src/harness/nvs/nvs-lib.test.ts.
//
// Held-out frames are the dataset frames that did not become keyframes of the run (every image of
// the result and every live keyframe index is excluded). Each test frame gets:
//  - a world-to-camera pose in the run frame:
//      'relative' (default): nearest registered keyframe (by frame index) k with its ESTIMATED pose,
//        composed with the GT relative motion k → frame, translation divided by the Sim3 scale s
//        (run → GT, Umeyama on the camera centres): R = R_rel·R_k, t = R_rel·t_k + t_rel / s with
//        R_rel = R_f^gt·(R_k^gt)ᵀ, t_rel = t_f^gt − R_rel·t_k^gt. Local SfM errors around k carry over
//        exactly as they do for the keyframe itself;
//      'sim3': the GT pose through the global Sim3 alone: R = R_f^gt·R_s, t = (R_f^gt·t_s + t_f^gt) / s;
//  - distorted RADIAL intrinsics of the run (f, cx, cy, k1, k2) with the focal:
//      'relative' (default): the nearest keyframe's refined per-image focal × f_f^gt / f_k^gt (GT
//        focus breathing of the -real sets; ratio 1 without per-frame GT focal);
//      'keyframe': the nearest keyframe's refined per-image focal;  'shared': the run's camera f;
//  - the rolling-shutter motion of the ground-truth image: camera angular velocity (gyro at the true
//    frame time, camera frame: ω_cam = R_cam_device·ω_dev) and, with a GT focus distance (-real
//    sets), the translational term ν = camera-frame velocity of the centre (central difference of the
//    neighbouring GT frames) / focus distance — the engine corrects its dense views with both.
//
// Test dataset (opts.test, e2e/nvs.mjs --test-dataset): the frames of ANOTHER dataset of the same
// seeded scene (e.g. an off-trajectory walk) are scored instead of the held-out frames. Its GT poses
// share the run dataset's world frame, so 'sim3' maps them directly; 'relative' and the focal use the
// registered keyframe whose GT camera centre is nearest (not the nearest frame index). Every test
// frame is a candidate; ω, ν and the GT image come from the test dataset.
import { mat3Mul, mat3T, mat3Vec, umeyama } from './linalg.mjs';

export function cameraCenter(pose) {
  const c = mat3Vec(mat3T(pose.R), pose.t);
  return [-c[0], -c[1], -c[2]];
}

/**
 * Registered images of a result (`images`, else `sparse.images`), sorted by frame index, with the
 * refined per-image focal `f` of `sparse.images` (the metrics input `images` has none).
 */
export function registeredImages(result) {
  const list = result.images ?? result.sparse?.images ?? [];
  const focal = new Map((result.sparse?.images ?? []).filter((im) => im.f > 0).map((im) => [im.index, im.f]));
  return list.map((im) => (im.f > 0 || !focal.has(im.index) ? im : { ...im, f: focal.get(im.index) })).sort((a, b) => a.index - b.index);
}

/** Sim3 run → GT (X_gt = s·R·X_run + t) from the camera centres; throws with < 3 images. */
export function runToGtSim3(images, framesByIndex) {
  const src = [],
    dst = [];
  for (const im of images) {
    const f = framesByIndex.get(im.index);
    if (!f) continue;
    src.push(cameraCenter(im.pose));
    dst.push(cameraCenter(f.pose));
  }
  const sim = umeyama(src, dst, true);
  if (!sim) throw new Error(`cannot align the run to the ground truth (${src.length} registered images)`);
  return sim;
}

/**
 * Frames that are NOT keyframes of the run. `keepBlurred` = false drops the frames the capture
 * rejected as blurred (`decision` 'reject_blur' in result.frames).
 */
export function heldOutFrames(frames, result, { keepBlurred = false } = {}) {
  const used = new Set(registeredImages(result).map((im) => im.index));
  for (const i of result.live?.keyframeIndices ?? []) used.add(i);
  const blurred = new Set();
  for (const fr of result.frames ?? []) if (fr.decision === 'reject_blur' && Number.isInteger(fr.index)) blurred.add(fr.index);
  return frames.filter((f) => !used.has(f.index) && (keepBlurred || !blurred.has(f.index)));
}

/** n items evenly spread over the list (all when n ≥ length). */
export function selectEvenly(list, n) {
  if (!(n > 0) || n >= list.length) return list.slice();
  const out = [];
  for (let k = 0; k < n; k++) out.push(list[Math.floor(((k + 0.5) * list.length) / n)]);
  return out;
}

/** Registered image nearest in frame index (ties: the earlier one). */
export function nearestImage(sorted, index) {
  let best = null;
  for (const im of sorted) if (!best || Math.abs(im.index - index) < Math.abs(best.index - index)) best = im;
  return best;
}

/**
 * Why a test dataset cannot be scored against a run of `meta`'s dataset: a different scene, seed,
 * texture or camera, or (yard) a different walking loop, which moves the scene's gravel path.
 * Empty when they are the same seeded scene seen by the same camera.
 */
export function sceneMismatch(meta, testMeta) {
  const out = [];
  const same = (what, a, b) => {
    if (JSON.stringify(a ?? null) !== JSON.stringify(b ?? null)) out.push(`${what}: ${JSON.stringify(a ?? null)} vs ${JSON.stringify(b ?? null)}`);
  };
  same('scene', meta.scene, testMeta.scene);
  same('seed', meta.seed, testMeta.seed);
  same('textureScale', meta.config?.textureScale, testMeta.config?.textureScale);
  same('anisotropy', meta.config?.anisotropy, testMeta.config?.anisotropy);
  for (const k of ['width', 'height', 'f', 'cx', 'cy', 'k1', 'k2']) same(`intrinsics.${k}`, meta.intrinsics?.[k], testMeta.intrinsics?.[k]);
  if (meta.scene === 'yard') for (const k of ['center', 'semiAxes']) same(`trajectory.${k}`, meta.config?.trajectory?.[k], testMeta.config?.trajectory?.[k]);
  return out;
}

/** Unit optical axis (camera +z) in the world: third row of R. */
function opticalAxis(pose) {
  return [pose.R[6], pose.R[7], pose.R[8]];
}

/**
 * Registered image whose GT camera centre is nearest to a GT pose, with that distance (GT metres)
 * and the angle between the two optical axes (deg). gtCentres: [{ im, c, axis }].
 */
export function nearestInSpace(gtCentres, pose) {
  const c = cameraCenter(pose),
    a = opticalAxis(pose);
  let best = null;
  for (const k of gtCentres) {
    const d = Math.hypot(k.c[0] - c[0], k.c[1] - c[1], k.c[2] - c[2]);
    if (!best || d < best.distM) best = { im: k.im, distM: d, axis: k.axis };
  }
  if (!best) return null;
  const dot = Math.max(-1, Math.min(1, a[0] * best.axis[0] + a[1] * best.axis[1] + a[2] * best.axis[2]));
  return { im: best.im, distM: best.distM, axisDeg: (Math.acos(dot) * 180) / Math.PI };
}

/** Test pose from a keyframe's estimated pose and the GT relative motion keyframe → frame. */
export function mapPoseRelative(gtFrame, gtKey, runKey, s) {
  const Rrel = mat3Mul(gtFrame.R, mat3T(gtKey.R));
  const Rt = mat3Vec(Rrel, gtKey.t);
  const trel = [gtFrame.t[0] - Rt[0], gtFrame.t[1] - Rt[1], gtFrame.t[2] - Rt[2]];
  const Rk = mat3Vec(Rrel, runKey.t);
  return { R: mat3Mul(Rrel, runKey.R), t: [Rk[0] + trel[0] / s, Rk[1] + trel[1] / s, Rk[2] + trel[2] / s] };
}

/** Test pose from the GT pose and the global Sim3 run → GT. */
export function mapPoseSim3(gtFrame, sim) {
  const Rts = mat3Vec(gtFrame.R, sim.t);
  return {
    R: mat3Mul(gtFrame.R, sim.R),
    t: [(Rts[0] + gtFrame.t[0]) / sim.s, (Rts[1] + gtFrame.t[1]) / sim.s, (Rts[2] + gtFrame.t[2]) / sim.s],
  };
}

/** Linear interpolation of the gyro (rad/s, device frame) at time t (ms, true clock); null outside. */
export function gyroAt(imu, t) {
  if (!imu?.length || t < imu[0].t || t > imu[imu.length - 1].t) return null;
  let lo = 0,
    hi = imu.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (imu[mid].t <= t) lo = mid;
    else hi = mid;
  }
  const a = imu[lo],
    b = imu[hi];
  const w = b.t > a.t ? (t - a.t) / (b.t - a.t) : 0;
  return [0, 1, 2].map((k) => a.gyro[k] + w * (b.gyro[k] - a.gyro[k]));
}

/** Camera-frame angular velocity at a frame (R_cam_device · gyro(tTrue)); null without IMU. */
export function cameraOmega(meta, imu, frame) {
  const g = gyroAt(imu, frame.tTrue ?? frame.t);
  if (!g || !meta.R_cam_device) return null;
  return mat3Vec(meta.R_cam_device, g);
}

/**
 * Translational rolling-shutter term of a frame (`rsCorrectPinholeImage`'s ν, 1/s): camera-frame
 * velocity of the camera centre (central difference of the neighbouring GT frames) divided by the
 * frame's GT focus distance (the AF depth: median z of the central image region). Null without a
 * focus distance or neighbours.
 */
export function cameraNu(framesByIndex, frame) {
  const a = framesByIndex.get(frame.index - 1) ?? frame,
    b = framesByIndex.get(frame.index + 1) ?? frame;
  if (a === b || !(frame.focusDistance > 0)) return null;
  const dt = ((b.tTrue ?? b.t) - (a.tTrue ?? a.t)) / 1000;
  if (!(dt > 0)) return null;
  const ca = cameraCenter(a.pose),
    cb = cameraCenter(b.pose);
  const vc = mat3Vec(frame.pose.R, [(cb[0] - ca[0]) / dt, (cb[1] - ca[1]) / dt, (cb[2] - ca[2]) / dt]);
  return vc.map((v) => v / frame.focusDistance);
}

/**
 * Rolling-shutter model ({ axis: 'rows' | 'cols', readoutMs signed along +v / +u }) of the run
 * (`sparse.stats.rollingShutter`, e.g. "cols -21.5 ms"); null when it estimated none.
 */
export function runRollingShutter(result) {
  const s = result.sparse?.stats?.rollingShutter;
  const m = typeof s === 'string' ? /^(rows|cols)\s+(-?[\d.]+)\s*ms/.exec(s.trim()) : null;
  if (!m) return null;
  const readoutMs = Number(m[2]);
  return Number.isFinite(readoutMs) && readoutMs !== 0 ? { axis: m[1], readoutMs } : null;
}

/** Ground-truth rolling shutter of a -real dataset (meta.captureArtefacts.rollingShutter); null if none. */
export function gtRollingShutter(meta) {
  const rs = meta.captureArtefacts?.rollingShutter;
  if (!rs || !(rs.readoutMs > 0) || (rs.axis !== 'rows' && rs.axis !== 'cols')) return null;
  const sign = /left column|top row/.test(rs.firstLine ?? '') ? 1 : /right column|bottom row/.test(rs.firstLine ?? '') ? -1 : rs.reverse ? -1 : 1;
  return { axis: rs.axis, readoutMs: sign * rs.readoutMs };
}

/**
 * Long side of the run's dense views: the "dense memory plan (V views W×H" log line, else the
 * preset's maxImageSize (src/dense/presets.ts: fast 800, balanced 1280, max 1600), else null.
 */
export function denseImageSize(result) {
  for (const l of result.logs ?? []) {
    const m = /dense memory plan \(\d+ views (\d+)×(\d+)/.exec(l.message ?? '');
    if (m) return Math.max(Number(m[1]), Number(m[2]));
  }
  const q = result.config?.quality;
  const d = result.config?.densify;
  if (d && typeof d === 'object' && d.maxImageSize > 0) return d.maxImageSize;
  return { fast: 800, balanced: 1280, max: 1600 }[q] ?? null;
}

/**
 * Test views of a run. dataset: { meta, frames, imu? } (the run's dataset); result: the run's result.json.
 * opts: { views = 24, pose = 'relative' | 'sim3', focal = 'relative' | 'keyframe' | 'shared',
 *         keepBlurred = false, frames?: number[] (explicit frame indices), rsTranslation = true,
 *         test?: { meta, frames, imu? } (score that dataset's frames instead of the held-out ones) }.
 * Returns { sim3, candidates, views: [{ index, file, tTrue, nearestKeyframe, gap, nearest: { keyframe,
 *   distM, axisDeg }, pose: {R, t}, camera: {f, cx, cy, k1, k2, width, height}, omega, nu,
 *   gt: {f, gain, wb, rsPx, blurPx, omega} }] }. gap: frame-index distance to nearestKeyframe (null
 * for a test dataset); nearest: the registered keyframe nearest in space (GT centres, GT metres).
 */
export function testViews(dataset, result, opts = {}) {
  const { meta, frames } = dataset;
  const test = opts.test ?? null;
  const tMeta = test ? test.meta : meta,
    tFrames = test ? test.frames : frames,
    tImu = test ? test.imu : dataset.imu;
  const byIndex = new Map(frames.map((f) => [f.index, f]));
  const tByIndex = test ? new Map(tFrames.map((f) => [f.index, f])) : byIndex;
  const images = registeredImages(result);
  if (images.length < 3) throw new Error(`the run registered ${images.length} images (need ≥ 3)`);
  const sim3 = runToGtSim3(images, byIndex);
  const intr = result.intrinsics ?? result.sparse?.intrinsics;
  if (!intr) throw new Error('the result has no intrinsics');
  const W = intr.width ?? meta.width,
    H = intr.height ?? meta.height;
  if (W !== meta.width || H !== meta.height) throw new Error(`result intrinsics are ${W}×${H}, the dataset frames ${meta.width}×${meta.height}`);
  if (test && (tMeta.width !== W || tMeta.height !== H)) throw new Error(`test dataset frames are ${tMeta.width}×${tMeta.height}, the run's ${W}×${H}`);
  const gtCentres = images.filter((im) => byIndex.has(im.index)).map((im) => ({ im, c: cameraCenter(byIndex.get(im.index).pose), axis: opticalAxis(byIndex.get(im.index).pose) }));
  // per-image focal of the result (engine frame) relative to its median → the shared f's frame
  const fs = images.map((im) => im.f).filter((f) => f > 0).sort((a, b) => a - b);
  const fMed = fs.length ? fs[fs.length >> 1] : 0;
  const keyFocal = (im) => (im.f > 0 && fMed > 0 ? (intr.f * im.f) / fMed : intr.f);
  const candidates = test ? tFrames.slice() : heldOutFrames(frames, result, opts);
  const chosen = opts.frames?.length ? opts.frames.map((i) => tByIndex.get(i)).filter(Boolean) : selectEvenly(candidates, opts.views ?? 24);
  const poseMode = opts.pose ?? 'relative';
  const focalMode = opts.focal ?? 'relative';
  const views = chosen.map((f) => {
    const near = nearestInSpace(gtCentres, f.pose);
    const k = test ? near.im : nearestImage(images, f.index);
    const gk = byIndex.get(k.index);
    const pose = poseMode === 'sim3' ? mapPoseSim3(f.pose, sim3) : mapPoseRelative(f.pose, gk.pose, k.pose, sim3.s);
    let fx = intr.f;
    if (focalMode === 'keyframe') fx = keyFocal(k);
    else if (focalMode === 'relative') fx = keyFocal(k) * (f.f > 0 && gk.f > 0 ? f.f / gk.f : 1);
    const omega = tImu ? cameraOmega(tMeta, tImu, f) : null;
    const nu = opts.rsTranslation === false ? null : cameraNu(tByIndex, f);
    return {
      index: f.index,
      file: f.file,
      tTrue: f.tTrue,
      nearestKeyframe: k.index,
      gap: test ? null : Math.abs(k.index - f.index),
      nearest: near ? { keyframe: near.im.index, distM: near.distM, axisDeg: near.axisDeg } : null,
      pose,
      camera: { f: fx, cx: intr.cx, cy: intr.cy, k1: intr.k1 ?? 0, k2: intr.k2 ?? 0, width: W, height: H },
      omega,
      nu,
      gt: { f: f.f ?? tMeta.intrinsics?.f ?? null, gain: f.gain ?? null, wb: f.wb ?? null, rsPx: f.rsPx ?? null, blurPx: f.blurPx ?? null, omega: f.omega ?? null },
    };
  });
  return { sim3, candidates: candidates.length, views };
}
