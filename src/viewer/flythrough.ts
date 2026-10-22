/**
 * Fly-through camera path for the video export (pure: no three.js / WebGPU). From the registered
 * keyframe cameras in capture order, it builds N frame poses (ARCHITECTURE §2: world-to-camera,
 * OpenCV camera frame) and the pinhole intrinsics of the video. The path is a stabilised replay of
 * the capture rather than a free orbit. The splats are photoreal only near the photos: on the
 * object-real scene they lose 1.5 dB 5 cm off the trajectory and 8 dB 25 cm off it (e2e/README
 * "Novel views: held-out and off-trajectory"). So the path hugs the trajectory and the view
 * direction follows the photos.
 *
 * Modes (`mode`; 'auto': `convergenceMode` of src/geometry/convergence.ts, the engine's own test,
 * with `depth` as the scene depth):
 *  - object: an orbit about the vertical axis through the optical-axis convergence point. The
 *    capture is cut into passes, runs of keyframes in capture order whose azimuth moves one way: a
 *    turn back by more than 10°, an azimuth gap over 24° or a jump over 6 keyframe steps between
 *    neighbours starts a new one. Keyframes nearer the axis than 0.3 × the median horizontal
 *    radius, or more than 75° above or below the centre (top-down shots, whose azimuth means
 *    nothing), belong to no pass. The pass that covers the most azimuth is replayed in capture
 *    order: its radius, height and azimuth are smoothed over the arc length (Gaussian local-linear,
 *    σ = max(keyframe step, 8° of orbit at the median radius)). So the orbit is circular where the
 *    capture is and follows a rise or a wander, and it never averages two rings or sweeps an
 *    azimuth no photo covers. The orientation is the keyframes' in the frame that turns with the
 *    orbit, smoothed over σ 15° of orbit, so the camera keeps facing the object the way the user
 *    aimed it. A pass under 20° of azimuth: scene mode.
 *  - scene: a centripetal Catmull-Rom spline through the keyframe centres. The centres are first
 *    smoothed (local-linear in arc length, σ = the median keyframe step) and decimated to that
 *    spacing. Orientations are low-pass averaged quaternions (Gaussian over the keyframe sequence,
 *    σ 0.7 keyframes: ROT_SIGMA_KEYFRAMES).
 * The path parameter advances, per keyframe step, by the larger of the step's length and λ × its
 * turning (λ = translation limit / maxTurn: the time the step needs at those limits, in metres), so
 * a pan from one spot advances along the path instead of collapsing into one point. Both modes
 * level the horizon to `up` (the camera's x axis stays horizontal) and keep the smoothed view
 * direction.
 *
 * Timing: each piece of path costs the time it needs at the speed limits, and the clip plays it
 * at that rate. The limits are: translation ≤ 0.2 · depth per s (scene; depth = the typical viewing
 * distance, `viewingDepth`) or ≤ the orbit's speed at the median radius (object), orbit ≤ 40 °/s
 * (object), and view rotation ≤ 30 °/s (object: beyond the orbit's own turning; scene: ≤ 1.8× the
 * path's typical turning per metre at the translation limit, ≥ 10 °/s, so quick pans in the capture
 * do not jerk). The clip is 6–10 s; a shorter path plays slower. In a longer one the pans —
 * stretches where the rotation limit sets the pace — are shortened first: one that would take more
 * than a quarter of the clip is replayed in a quarter (or as fast as 90 °/s at its peak allows)
 * with smoothstep speed ramps, unless the path has almost nothing else to show (< 0.5 s; a pan from
 * one spot). Then it is cropped to the most central window that shows ≥ 90 % of the most
 * translation any window shows (`coverage`). So an out-and-back walk shows walking, not 10 s of
 * turning on the spot. The start and end ease in and out over ≤ 0.6 s (smoothstep speed ramps), so
 * frame steps never exceed the cruise step.
 *
 * Video frame: portrait when the photos' long side is vertical (physical up = `up`), else
 * landscape; 1280 × 720 by default. The focal length is the smallest that keeps the frame inside
 * the photos' median field of view, times `zoom` (1.05: the photo borders are the least
 * constrained parts of the splats).
 */
import type { CameraIntrinsics, Pose, Vec3 } from '../core/types';
import { convergenceMode, opticalAxisConvergence } from '../geometry/convergence';
import { mat3ToQuat, quatToMat3 } from '../math/rotation';
import { median } from '../math/stats';
import type { SplatCamera } from '../splat/types';

const DEG = Math.PI / 180;
/**
 * Scene mode: σ of the orientation low-pass, in keyframes. Keyframes are dense where the view
 * turns fast, so this narrows at quick pans. Wider kernels (or σ in arc length) drift more than 20°
 * from the keyframes there: yard-real swings 34° between keyframes 0.39 m apart.
 */
const ROT_SIGMA_KEYFRAMES = 0.7;
/**
 * Scene mode: the view-rotation limit is lowered to PAN_FACTOR × the path's median turning rate
 * (per metre, at the translation limit), but not below MIN_TURN (rad/s). Quick pans then turn at
 * most ~2× as fast as the rest of the clip: 1.8 leaves room for the eased start and end, which
 * lower the per-frame median. It is computed once over the whole path (not over the shown part,
 * which it selects).
 */
const PAN_FACTOR = 1.8;
const MIN_TURN = 10 * DEG;
/**
 * A path longer than the clip: a pan may take at most PAN_SHARE of the clip, replayed with
 * smoothstep ramps over PAN_RAMP of its time at each end and at most FAST_TURN (rad/s) at its
 * peak; not when the rest of the path takes less than MIN_REST_SEC. Pans closer than
 * PAN_MERGE_SEC are one.
 */
const PAN_SHARE = 0.25;
const PAN_RAMP = 0.3;
const FAST_TURN = 90 * DEG;
const MIN_REST_SEC = 0.5;
const PAN_MERGE_SEC = 0.25;
/**
 * A path longer than the clip: the clip is the most central window that shows at least this share
 * of the most translation a window can show (so it is not centred on a pan).
 */
const WINDOW_SHARE = 0.9;
/** Object mode passes and smoothing (see the file header). */
const ORBIT_MIN_RADIUS = 0.3;
const ORBIT_MAX_ELEVATION = 75 * DEG;
const ORBIT_GAP = 24 * DEG;
const ORBIT_MAX_JUMP_STEPS = 6;
const ORBIT_REVERSAL = 10 * DEG;
/** The azimuth must move this far from a pass's start before its direction counts. */
const ORBIT_DIRECTION = 5 * DEG;
const ORBIT_SIGMA_POS = 8 * DEG;
const ORBIT_SIGMA_ROT = 15 * DEG;
const ORBIT_MIN_SWEEP = 20 * DEG;
/** Samples of the dense path (cost table). */
const SAMPLES = 2000;

export type FlythroughMode = 'object' | 'scene';

/** A registered keyframe camera (capture order). */
export interface FlythroughCamera {
  pose: Pose;
  /** The photo's intrinsics (width, height, f). */
  intrinsics: Pick<CameraIntrinsics, 'width' | 'height' | 'f'>;
}

export interface FlythroughOptions {
  /** 'auto' (default): `convergenceMode` of the cameras (with `depth` as the scene depth). */
  mode?: FlythroughMode | 'auto';
  /** Video size: long and short side (default 1280 × 720). */
  longSide?: number;
  shortSide?: number;
  /** 'auto' (default): portrait when the photos' long side is vertical. */
  orientation?: 'auto' | 'portrait' | 'landscape';
  /** Frames per second (default 30). */
  fps?: number;
  /** Clip length bounds (default 6–10 s). */
  minSeconds?: number;
  maxSeconds?: number;
  /** World up (default +Z, the gravity-aligned world): the horizon is levelled to it. */
  up?: ArrayLike<number>;
  /** Typical viewing distance (world units): speed limit and near plane. Default: from the cameras. */
  depth?: number;
  /** Scene mode: fastest translation per second as a fraction of `depth` (default 0.2). */
  maxSpeedPerDepth?: number;
  /** Fastest view rotation (default 30 °/s; long pans in a cropped clip: see the file header). */
  maxTurnDegPerSec?: number;
  /** Object mode: fastest orbit (default 40 °/s). */
  maxOrbitDegPerSec?: number;
  /** Focal length factor over the widest frame inside the photos' field of view (default 1.05). */
  zoom?: number;
}

export interface FlythroughPath {
  mode: FlythroughMode;
  width: number;
  height: number;
  fps: number;
  /** Pinhole intrinsics of the frames (pixel centres at integers). */
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  /** Near plane (world units, camera z). */
  near: number;
  /** World-to-camera pose of every frame. */
  poses: Pose[];
  /** Object mode: the orbit centre (optical-axis convergence point); null in scene mode. */
  centre: Vec3 | null;
  /** Part of the capture the clip shows: its first and last frame's place in the keyframes (capture order, 0–1). */
  coverage: [number, number];
  /** Typical viewing distance used (world units). */
  depth: number;
}

/** The SplatCamera (world frame, OpenCV) of frame `i`. */
export function flythroughCamera(p: FlythroughPath, i: number): SplatCamera {
  const pose = p.poses[i];
  return { width: p.width, height: p.height, fx: p.fx, fy: p.fy, cx: p.cx, cy: p.cy, R: pose.R, t: pose.t, near: p.near };
}

/**
 * Typical viewing distance of the cameras (the `depth` option): the median, over up to 40 evenly
 * spread cameras, of the median depth of the points (flat xyz, up to `maxPoints` evenly spread)
 * inside their photo (principal point at the image centre). Null with fewer than 20 such points.
 */
export function viewingDepth(cams: readonly FlythroughCamera[], positions: ArrayLike<number>, maxPoints = 4000): number | null {
  const np = Math.floor(positions.length / 3);
  const stride = Math.max(1, Math.floor(np / maxPoints));
  const camStride = Math.max(1, Math.floor(cams.length / 40));
  const perCam: number[] = [];
  for (let c = 0; c < cams.length; c += camStride) {
    const { R, t } = cams[c].pose;
    const { width: W, height: H, f } = cams[c].intrinsics;
    const zs: number[] = [];
    for (let i = 0; i < np; i += stride) {
      const X = positions[3 * i];
      const Y = positions[3 * i + 1];
      const Z = positions[3 * i + 2];
      const z = R[6] * X + R[7] * Y + R[8] * Z + t[2];
      if (!(z > 0)) continue;
      const u = (f * (R[0] * X + R[1] * Y + R[2] * Z + t[0])) / z;
      const v = (f * (R[3] * X + R[4] * Y + R[5] * Z + t[1])) / z;
      if (Math.abs(u) <= W / 2 && Math.abs(v) <= H / 2) zs.push(z);
    }
    if (zs.length >= 20) perCam.push(median(zs));
  }
  return perCam.length ? median(perCam) : null;
}

/** Presentation timestamp of frame `i` (µs), as VideoFrame / EncodedVideoChunk timestamps. */
export function flythroughTimestampUs(p: Pick<FlythroughPath, 'fps'>, i: number): number {
  return Math.round((i * 1e6) / p.fps);
}

// ------------------------------------------------------------------------------- helpers

type V3 = [number, number, number];

const sub = (a: ArrayLike<number>, b: ArrayLike<number>): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: ArrayLike<number>, b: ArrayLike<number>): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: ArrayLike<number>, b: ArrayLike<number>): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: ArrayLike<number>): number => Math.hypot(a[0], a[1], a[2]);
const unit = (a: ArrayLike<number>): V3 => {
  const l = norm(a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};
/** Angle wrapped to (−π, π]. */
const wrapPi = (a: number): number => a - 2 * Math.PI * Math.ceil((a - Math.PI) / (2 * Math.PI));

function centreOf(p: Pose): V3 {
  const { R, t } = p;
  return [-(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]), -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]), -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2])];
}

/** Rotation about unit axis `u` by `a` (row-major). */
function rotAbout(u: ArrayLike<number>, a: number): Float64Array {
  const c = Math.cos(a);
  const s = Math.sin(a);
  const C = 1 - c;
  const [x, y, z] = [u[0], u[1], u[2]];
  return Float64Array.of(c + x * x * C, x * y * C - z * s, x * z * C + y * s, y * x * C + z * s, c + y * y * C, y * z * C - x * s, z * x * C - y * s, z * y * C + x * s, c + z * z * C);
}

/** a · b (row-major 3×3). */
function mul3(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const o = new Float64Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[3 * r + c] = a[3 * r] * b[c] + a[3 * r + 1] * b[3 + c] + a[3 * r + 2] * b[6 + c];
  return o;
}

/** a · bᵀ (row-major 3×3). */
function mulT3(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const o = new Float64Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[3 * r + c] = a[3 * r] * b[3 * c] + a[3 * r + 1] * b[3 * c + 1] + a[3 * r + 2] * b[3 * c + 2];
  return o;
}

/** Angle between two rotations (rad). */
export function rotationDistance(A: ArrayLike<number>, B: ArrayLike<number>): number {
  let tr = 0;
  for (let i = 0; i < 9; i++) tr += A[i] * B[i];
  return Math.acos(Math.min(1, Math.max(-1, (tr - 1) / 2)));
}

/**
 * Gaussian-weighted local-linear fit of `d`-vectors `ys` (flat) over the sample positions `xs`,
 * evaluated at `x0` (σ `sigma`). Unbiased at the ends and on linear trends, unlike a moving
 * average. A small ridge on the slope keeps it continuous where the weighted samples barely
 * spread (beside a gap); the nearest sample only where no sample has weight.
 */
function localLinear(xs: ArrayLike<number>, ys: ArrayLike<number>, d: number, x0: number, sigma: number, out: Float64Array): Float64Array {
  let s0 = 0;
  let s1 = 0;
  let s2 = 0;
  const t0 = new Float64Array(d);
  const t1 = new Float64Array(d);
  let nearest = 0;
  for (let k = 0; k < xs.length; k++) {
    const dx = xs[k] - x0;
    if (Math.abs(dx) < Math.abs(xs[nearest] - x0)) nearest = k;
    const w = Math.exp(-0.5 * (dx / sigma) ** 2);
    if (w < 1e-12) continue;
    s0 += w;
    s1 += w * dx;
    s2 += w * dx * dx;
    for (let j = 0; j < d; j++) {
      t0[j] += w * ys[d * k + j];
      t1[j] += w * dx * ys[d * k + j];
    }
  }
  const s2r = s2 + 1e-3 * s0 * sigma * sigma;
  const det = s0 * s2r - s1 * s1;
  for (let j = 0; j < d; j++) out[j] = s0 < 1e-9 ? ys[d * nearest + j] : (s2r * t0[j] - s1 * t1[j]) / det;
  return out;
}

/** Gaussian-weighted mean of sign-aligned quaternions `qs` (flat [w, x, y, z]) at `x0`, normalised. */
function quatAverage(xs: ArrayLike<number>, qs: ArrayLike<number>, x0: number, sigma: number): Float64Array {
  const out = new Float64Array(4);
  let nearest = 0;
  let wsum = 0;
  for (let k = 0; k < xs.length; k++) {
    const dx = xs[k] - x0;
    if (Math.abs(dx) < Math.abs(xs[nearest] - x0)) nearest = k;
    const w = Math.exp(-0.5 * (dx / sigma) ** 2);
    if (w < 1e-12) continue;
    wsum += w;
    for (let j = 0; j < 4; j++) out[j] += w * qs[4 * k + j];
  }
  if (wsum < 1e-9) for (let j = 0; j < 4; j++) out[j] = qs[4 * nearest + j];
  const l = Math.hypot(out[0], out[1], out[2], out[3]) || 1;
  for (let j = 0; j < 4; j++) out[j] /= l;
  return out;
}

/** Quaternions of rotations, each sign-aligned with the previous one (a smooth sequence). */
function alignedQuats(Rs: ArrayLike<number>[]): Float64Array {
  const qs = new Float64Array(4 * Rs.length);
  const q = new Float64Array(4);
  for (let k = 0; k < Rs.length; k++) {
    mat3ToQuat(Rs[k], q);
    if (k > 0 && q[0] * qs[4 * k - 4] + q[1] * qs[4 * k - 3] + q[2] * qs[4 * k - 2] + q[3] * qs[4 * k - 1] < 0) for (let j = 0; j < 4; j++) q[j] = -q[j];
    qs.set(q, 4 * k);
  }
  return qs;
}

/** R with its x axis made horizontal (perpendicular to `up`), keeping the view direction. */
function level(R: ArrayLike<number>, up: V3): Float64Array {
  const d = unit([R[6], R[7], R[8]]);
  let x = cross(d, up);
  if (norm(x) < Math.sin(5 * DEG)) {
    // looking (nearly) along up: keep the camera's own x axis, made perpendicular to d
    const x0: V3 = [R[0], R[1], R[2]];
    x = sub(x0, d.map((v) => v * dot(x0, d)) as V3);
  }
  x = unit(x);
  const y = cross(d, x);
  return Float64Array.of(x[0], x[1], x[2], y[0], y[1], y[2], d[0], d[1], d[2]);
}

/** Smoothstep speed ramps: progress (0–1) at normalised time `tau` with ramps of length `r`. */
export function easedProgress(tau: number, r: number): number {
  if (r <= 0) return tau;
  const head = (t: number): number => {
    const x = t / r;
    return r * (x * x * x - 0.5 * x * x * x * x);
  };
  let p: number;
  if (tau <= r) p = head(tau);
  else if (tau >= 1 - r) p = 1 - r - head(1 - tau);
  else p = r / 2 + (tau - r);
  return Math.min(1, Math.max(0, p / (1 - r)));
}

let panEase: { tau: Float64Array; p: Float64Array } | null = null;

/** Inverse of `easedProgress(·, PAN_RAMP)`: the normalised time at which a pan has turned `p`. */
function panTime(p: number): number {
  if (!panEase) {
    const tau = Float64Array.from({ length: 513 }, (_, i) => i / 512);
    panEase = { tau, p: Float64Array.from(tau, (t) => easedProgress(t, PAN_RAMP)) };
  }
  return interp(panEase.p, panEase.tau, p);
}

/** Centripetal Catmull-Rom point between P1 and P2 (knots from |ΔP|^½). */
function catmullRom(P0: V3, P1: V3, P2: V3, P3: V3, s: number): V3 {
  const kn = (a: V3, b: V3): number => Math.max(1e-9, Math.sqrt(norm(sub(b, a))));
  const t0 = 0;
  const t1 = t0 + kn(P0, P1);
  const t2 = t1 + kn(P1, P2);
  const t3 = t2 + kn(P2, P3);
  const t = t1 + (t2 - t1) * s;
  const lerp = (a: V3, b: V3, ta: number, tb: number): V3 => {
    const w = (t - ta) / (tb - ta);
    return [a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w, a[2] + (b[2] - a[2]) * w];
  };
  const A1 = lerp(P0, P1, t0, t1);
  const A2 = lerp(P1, P2, t1, t2);
  const A3 = lerp(P2, P3, t2, t3);
  const B1 = lerp(A1, A2, t0, t2);
  const B2 = lerp(A2, A3, t1, t3);
  return lerp(B1, B2, t1, t2);
}

/** Linear interpolation in a monotone table (xs ascending) at x. */
function interp(xs: ArrayLike<number>, ys: ArrayLike<number>, x: number): number {
  const n = xs.length;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (xs[m] <= x) lo = m;
    else hi = m;
  }
  const w = xs[hi] > xs[lo] ? (x - xs[lo]) / (xs[hi] - xs[lo]) : 0;
  return ys[lo] + (ys[hi] - ys[lo]) * w;
}

/** First index with xs[i] ≥ x (xs ascending). */
function upperIndex(xs: Float64Array, x: number): number {
  let lo = 0;
  let hi = xs.length - 1;
  if (x <= xs[0]) return 0;
  if (x > xs[hi]) return hi;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (xs[m] < x) lo = m;
    else hi = m;
  }
  return hi;
}

/**
 * A dense path: SAMPLES + 1 parameter values and what each step (sample i − 1 → i; index 0 unused)
 * needs: its translation, its rotation against the turn limit and any further time.
 */
interface DensePath {
  param: Float64Array;
  dl: Float64Array;
  dr: Float64Array;
  /** Time a step needs besides translation and rotation (s; object: the orbit limit). */
  fixed: Float64Array;
  /** Translation limit (world units / s; 0: rotation only). */
  vmax: number;
  /** Place in the capture (keyframe index / (n − 1)) at a parameter value. */
  progress(x: number): number;
  /** Position at a parameter value. */
  position(x: number): V3;
  /** Smoothed, levelled world-to-camera rotation at a parameter value. */
  rotation(x: number): Float64Array;
}

interface DensifyOptions {
  vmax: number;
  progress(x: number): number;
  /** The rotation that counts against the turn limit (default: the view rotation). */
  turning?: (x: number, R: Float64Array) => Float64Array;
  /** Further time a step needs (s). */
  extra?: (x0: number, x1: number) => number;
}

/** Samples a parameterised path between `x0` and `x1` (see DensePath). */
function densify(x0: number, x1: number, position: (x: number) => V3, rotation: (x: number) => Float64Array, o: DensifyOptions): DensePath {
  const S = SAMPLES;
  const param = new Float64Array(S + 1);
  const dl = new Float64Array(S + 1);
  const dr = new Float64Array(S + 1);
  const fixed = new Float64Array(S + 1);
  const turning = o.turning ?? ((_x: number, R: Float64Array): Float64Array => R);
  let prevP = position(x0);
  let prevR = turning(x0, rotation(x0));
  param[0] = x0;
  for (let i = 1; i <= S; i++) {
    const x = x0 + ((x1 - x0) * i) / S;
    param[i] = x;
    const P = position(x);
    const R = turning(x, rotation(x));
    dl[i] = norm(sub(P, prevP));
    dr[i] = rotationDistance(R, prevR);
    fixed[i] = o.extra ? o.extra(param[i - 1], x) : 0;
    prevP = P;
    prevR = R;
  }
  return { param, dl, dr, fixed, vmax: o.vmax, progress: o.progress, position, rotation };
}

/**
 * Scene mode: the view-rotation limit, PAN_FACTOR × the median turning rate at the translation
 * limit over the whole path's arc length, within [MIN_TURN, turn] (`turn` with no translation).
 */
function typicalTurnLimit(d: DensePath, turn: number): number {
  if (!(d.vmax > 0)) return turn;
  const idx: number[] = [];
  let total = 0;
  for (let i = 1; i < d.dl.length; i++)
    if (d.dl[i] > 0) {
      idx.push(i);
      total += d.dl[i];
    }
  if (!idx.length) return turn;
  const rate = (i: number): number => (d.dr[i] * d.vmax) / d.dl[i];
  idx.sort((a, b) => rate(a) - rate(b));
  let acc = 0;
  let med = rate(idx[idx.length - 1]);
  for (const i of idx) {
    acc += d.dl[i];
    if (acc >= total / 2) {
      med = rate(i);
      break;
    }
  }
  return Math.min(turn, Math.max(MIN_TURN, PAN_FACTOR * med));
}

/**
 * Cumulative time of the samples at the limits (rotation limit `w`), with the pans of a path longer
 * than `budget` shortened, and the shown window [a, b] of it (see the file header).
 */
function timeline(d: DensePath, w: number, budget: number): { cost: Float64Array; a: number; b: number } {
  const S = d.dl.length - 1;
  const lin = (i: number): number => (d.vmax > 0 ? d.dl[i] / d.vmax : 0);
  const turning = (i: number): boolean => d.dr[i] / w > Math.max(lin(i), d.fixed[i]);
  const c = new Float64Array(S + 1);
  let total = 0;
  for (let i = 1; i <= S; i++) {
    c[i] = Math.max(lin(i), d.fixed[i], d.dr[i] / w, 1e-12);
    total += c[i];
  }
  if (total > budget) {
    // pans: runs of steps whose rotation sets the pace, joined across short gaps
    const runs: [number, number][] = [];
    for (let i = 1; i <= S; i++) {
      if (!turning(i)) continue;
      let j = i;
      while (j < S && turning(j + 1)) j++;
      const prev = runs[runs.length - 1];
      let gap = Infinity;
      if (prev) {
        gap = 0;
        for (let k = prev[1] + 1; k < i; k++) gap += c[k];
      }
      if (gap < PAN_MERGE_SEC) prev[1] = j;
      else runs.push([i, j]);
      i = j;
    }
    const sum = (v: Float64Array, [i0, i1]: [number, number]): number => {
      let acc = 0;
      for (let k = i0; k <= i1; k++) acc += v[k];
      return acc;
    };
    const cap = PAN_SHARE * budget;
    const longest = runs.map((run) => ({ run, cost: sum(c, run) })).sort((x, y) => y.cost - x.cost || x.run[0] - y.run[0]);
    for (const { run, cost: was } of longest) {
      if (total <= budget || was <= cap || total - was < MIN_REST_SEC) continue;
      const angle = sum(d.dr, run);
      const L = Math.max(cap, angle / (FAST_TURN * (1 - PAN_RAMP)));
      if (!(L < was)) continue;
      // replayed in L: the turned fraction follows the eased progress of the pan's time
      let acc = 0;
      let tPrev = 0;
      for (let k = run[0]; k <= run[1]; k++) {
        acc += d.dr[k];
        const t = L * panTime(Math.min(1, acc / angle));
        c[k] = Math.max(lin(k), d.fixed[k], t - tPrev, 1e-12);
        tPrev = t;
      }
      total += sum(c, run) - was;
    }
  }
  const cost = new Float64Array(S + 1);
  for (let i = 1; i <= S; i++) cost[i] = cost[i - 1] + c[i];
  total = cost[S];
  if (total <= budget) return { cost, a: 0, b: total };
  // the most central window with at least WINDOW_SHARE of the most translation any window shows
  const moved = new Float64Array(S + 1);
  for (let i = 1; i <= S; i++) moved[i] = moved[i - 1] + d.dl[i];
  const span = total - budget;
  const G = 400;
  const shown = Float64Array.from({ length: G + 1 }, (_, g) => interp(cost, moved, (span * g) / G + budget) - interp(cost, moved, (span * g) / G));
  let most = 0;
  for (const v of shown) most = Math.max(most, v);
  let bestG = -1;
  for (let g = 0; g <= G; g++) if (shown[g] >= WINDOW_SHARE * most && (bestG < 0 || Math.abs(g - G / 2) < Math.abs(bestG - G / 2))) bestG = g;
  const a = (span * bestG) / G;
  return { cost, a, b: a + budget };
}

// ------------------------------------------------------------------------------- builder

/** Fly-through path of the registered cameras (capture order); null with fewer than 2. */
export function buildFlythrough(cams: readonly FlythroughCamera[], o: FlythroughOptions = {}): FlythroughPath | null {
  const n = cams.length;
  if (n < 2) return null;
  const up = unit(o.up ?? [0, 0, 1]);
  const fps = o.fps ?? 30;
  const minS = o.minSeconds ?? 6;
  const maxS = Math.max(minS, o.maxSeconds ?? 10);
  const turn = (o.maxTurnDegPerSec ?? 30) * DEG;
  const orbitRate = (o.maxOrbitDegPerSec ?? 40) * DEG;
  const poses = cams.map((c) => c.pose);
  const C = poses.map(centreOf);
  const steps: number[] = [];
  for (let k = 1; k < n; k++) steps.push(norm(sub(C[k], C[k - 1])));
  const conv = opticalAxisConvergence(poses);
  let mode: FlythroughMode = o.mode === 'object' || o.mode === 'scene' ? o.mode : convergenceMode(conv, o.depth ?? 0);
  if (mode === 'object' && !conv.point) mode = 'scene';
  let ext = 0;
  for (const c of C) ext = Math.max(ext, norm(sub(c, C[0])));
  const depth = o.depth && o.depth > 0 ? o.depth : mode === 'object' ? conv.distMedian : Math.max(5 * (median(steps) || 0), 0.1 * ext, 1e-3);

  let path: DensePath | null = mode === 'object' ? objectPath(poses, C, steps, conv.point!, up, orbitRate, turn) : null;
  if (!path) {
    mode = 'scene';
    path = scenePath(poses, C, steps, up, (o.maxSpeedPerDepth ?? 0.2) * depth, turn);
  }

  // timing: T − 0.6 s plays the path at the limits (eased start and end); shorter paths play
  // slower over minS, longer ones are cropped
  const rampSec = 0.6;
  const budget = maxS * (1 - Math.min(0.15, rampSec / maxS));
  const w = mode === 'scene' ? typicalTurnLimit(path, turn) : turn;
  const { cost, a, b } = timeline(path, w, budget);
  const T = Math.min(maxS, Math.max(minS, b - a + rampSec));
  const r = Math.min(0.15, rampSec / T);
  const N = Math.max(2, Math.round(T * fps));
  const out: Pose[] = [];
  for (let i = 0; i < N; i++) {
    const m = a + (b - a) * easedProgress(i / (N - 1), r);
    const x = interp(cost, path.param, m);
    const R = path.rotation(x);
    const P = path.position(x);
    out.push({ R, t: Float64Array.of(-(R[0] * P[0] + R[1] * P[1] + R[2] * P[2]), -(R[3] * P[0] + R[4] * P[1] + R[5] * P[2]), -(R[6] * P[0] + R[7] * P[1] + R[8] * P[2])) });
  }

  // video frame: orientation and focal length from the photos' field of view about `up`
  const halfV: number[] = [];
  const halfH: number[] = [];
  let portraitVotes = 0;
  for (const c of cams) {
    const R = c.pose.R;
    // physical up in the camera frame: is the image's y axis (rows) or x axis vertical?
    const wy = Math.abs(R[3] * up[0] + R[4] * up[1] + R[5] * up[2]);
    const wx = Math.abs(R[0] * up[0] + R[1] * up[1] + R[2] * up[2]);
    const { width: W, height: H, f } = c.intrinsics;
    const vert = wy >= wx ? H : W;
    const horz = wy >= wx ? W : H;
    halfV.push(Math.atan(vert / 2 / f));
    halfH.push(Math.atan(horz / 2 / f));
    if (vert >= horz) portraitVotes++;
  }
  const portrait = o.orientation === 'portrait' || (o.orientation !== 'landscape' && portraitVotes * 2 >= n);
  const long = o.longSide ?? 1280;
  const short = o.shortSide ?? 720;
  const width = portrait ? short : long;
  const height = portrait ? long : short;
  const f = (o.zoom ?? 1.05) * Math.max(height / 2 / Math.tan(median(halfV)), width / 2 / Math.tan(median(halfH)));
  return {
    mode,
    width,
    height,
    fps,
    fx: f,
    fy: f,
    cx: (width - 1) / 2,
    cy: (height - 1) / 2,
    near: 0.02 * depth,
    poses: out,
    centre: mode === 'object' ? Float64Array.from(conv.point!) : null,
    coverage: [path.progress(interp(cost, path.param, a)), path.progress(interp(cost, path.param, b))],
    depth,
  };
}

/** Scene mode: smoothed, decimated centres → centripetal Catmull-Rom (path parameter: see the file header). */
function scenePath(poses: Pose[], C: V3[], steps: number[], up: V3, vmax: number, turn: number): DensePath {
  const n = C.length;
  const s = new Float64Array(n);
  for (let k = 1; k < n; k++) s[k] = s[k - 1] + steps[k - 1];
  const len = s[n - 1];
  const qs = alignedQuats(poses.map((p) => p.R));
  const keyIdx = Float64Array.from({ length: n }, (_, k) => k);
  if (len < 1e-9) {
    // no translation (a pan from one spot): the orientations alone, by keyframe index
    const fixed: V3 = C[0];
    return densify(0, n - 1, () => fixed, (x) => level(quatToMat3(quatAverage(keyIdx, qs, x, ROT_SIGMA_KEYFRAMES)), up), { vmax: 0, progress: (x) => x / (n - 1) });
  }
  const step = median(steps.filter((v) => v > 0)) || len / Math.max(1, n - 1);
  const sigma = step;
  const flatC = new Float64Array(3 * n);
  C.forEach((c, k) => flatC.set(c, 3 * k));
  // smoothed centres, decimated to ≥ one step apart (first and last kept)
  const ctrl: V3[] = [];
  const ctrlS: number[] = [];
  const tmp = new Float64Array(3);
  for (let k = 0; k < n; k++) {
    if (k > 0 && k < n - 1 && s[k] - ctrlS[ctrlS.length - 1] < step) continue;
    if (k === n - 1 && ctrlS.length > 1 && s[k] - ctrlS[ctrlS.length - 1] < 0.5 * step) {
      ctrl.pop();
      ctrlS.pop();
    }
    localLinear(s, flatC, 3, s[k], sigma, tmp);
    ctrl.push([tmp[0], tmp[1], tmp[2]]);
    ctrlS.push(s[k]);
  }
  // dense spline samples with arc length u; keyframe arc length s → spline arc length
  const m = ctrl.length;
  const SUB = 48;
  const pts: V3[] = [];
  const knotU: number[] = [0];
  const u: number[] = [];
  for (let j = 0; j < m - 1; j++) {
    const P0 = j > 0 ? ctrl[j - 1] : (sub(ctrl[0], sub(ctrl[1], ctrl[0])) as V3);
    const P3 = j + 2 < m ? ctrl[j + 2] : (sub(ctrl[m - 1], sub(ctrl[m - 2], ctrl[m - 1])) as V3);
    for (let i = j === 0 ? 0 : 1; i <= SUB; i++) {
      const p = catmullRom(P0, ctrl[j], ctrl[j + 1], P3, i / SUB);
      u.push(pts.length ? u[u.length - 1] + norm(sub(p, pts[pts.length - 1])) : 0);
      pts.push(p);
    }
    knotU.push(u[u.length - 1]);
  }
  const uOfS = (x: number): number => interp(ctrlS, knotU, x);
  const uArr = Float64Array.from(u);
  const atU = (x: number): V3 => {
    const k = Math.min(pts.length - 1, Math.max(0, upperIndex(uArr, x)));
    const k0 = Math.max(0, k - 1);
    const w = uArr[k] > uArr[k0] ? (x - uArr[k0]) / (uArr[k] - uArr[k0]) : 0;
    const p0 = pts[k0];
    const p1 = pts[k];
    return [p0[0] + (p1[0] - p0[0]) * w, p0[1] + (p1[1] - p0[1]) * w, p0[2] + (p1[2] - p0[2]) * w];
  };
  // path parameter per keyframe: max(arc length, λ · turning), so a pan from one spot still advances
  const lam = vmax / turn;
  const keyP = new Float64Array(n);
  for (let k = 1; k < n; k++) keyP[k] = keyP[k - 1] + Math.max(steps[k - 1], lam * rotationDistance(poses[k].R, poses[k - 1].R));
  const position = (x: number): V3 => atU(uOfS(interp(keyP, s, x)));
  // orientations are smoothed over the keyframe sequence (fractional keyframe index along the path)
  const rotation = (x: number): Float64Array => level(quatToMat3(quatAverage(keyIdx, qs, interp(keyP, keyIdx, x), ROT_SIGMA_KEYFRAMES)), up);
  return densify(0, keyP[n - 1], position, rotation, { vmax, progress: (x) => interp(keyP, keyIdx, x) / (n - 1) });
}

/**
 * Object mode: the pass covering the most azimuth about the vertical through `centre`, replayed in
 * cylindrical coordinates (see the file header). Null when no pass covers 20° of azimuth.
 */
function objectPath(poses: Pose[], C: V3[], steps: number[], centre: ArrayLike<number>, up: V3, orbitRate: number, turn: number): DensePath | null {
  const n = C.length;
  // horizontal basis: e1 towards the first camera off the axis
  let e1: V3 | null = null;
  for (const c of C) {
    const v = sub(c, centre);
    const hv = sub(v, up.map((x) => x * dot(v, up)) as V3);
    if (norm(hv) > 1e-9) {
      e1 = unit(hv);
      break;
    }
  }
  e1 ??= unit(Math.abs(up[0]) < 0.9 ? cross(up, [1, 0, 0]) : cross(up, [0, 1, 0]));
  const e2 = cross(up, e1);
  const az = new Float64Array(n);
  const rad = new Float64Array(n);
  const hgt = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const v = sub(C[k], centre);
    const x = dot(v, e1);
    const y = dot(v, e2);
    hgt[k] = dot(v, up);
    rad[k] = Math.hypot(x, y);
    az[k] = Math.atan2(y, x);
  }
  const rAll = median(Array.from(rad));
  const valid = (k: number): boolean => rad[k] >= ORBIT_MIN_RADIUS * rAll && Math.atan2(Math.abs(hgt[k]), rad[k]) <= ORBIT_MAX_ELEVATION;
  const maxJump = ORBIT_MAX_JUMP_STEPS * (median(steps.filter((v) => v > 0)) || Infinity);
  // passes: keyframe runs (capture order) whose unwrapped azimuth moves one way
  const passes: { ks: number[]; phi: number[] }[] = [];
  let ks: number[] = [];
  let phi: number[] = [];
  /** Direction of the current pass (0: not yet known) and the index of its farthest azimuth. */
  let dir = 0;
  let far = 0;
  for (let k = 0; k <= n; k++) {
    const last = ks.length - 1;
    const ok = k < n && valid(k);
    const d = ok && last >= 0 ? wrapPi(az[k] - az[ks[last]]) : 0;
    if (!ok || (last >= 0 && (Math.abs(d) > ORBIT_GAP || norm(sub(C[k], C[ks[last]])) > maxJump))) {
      if (ks.length >= 2) passes.push({ ks, phi });
      ks = ok ? [k] : [];
      phi = ok ? [az[k]] : [];
      dir = 0;
      far = 0;
      continue;
    }
    const p = last >= 0 ? phi[last] + d : az[k];
    if (dir !== 0 && dir * (phi[far] - p) > ORBIT_REVERSAL) {
      // turned back: this pass ends at its farthest keyframe, the next one starts there
      passes.push({ ks: ks.slice(0, far + 1), phi: phi.slice(0, far + 1) });
      ks = ks.slice(far);
      phi = phi.slice(far);
      dir = 0;
      far = 0;
    }
    ks.push(k);
    phi.push(p);
    if (dir === 0 && Math.abs(p - phi[0]) > ORBIT_DIRECTION) {
      dir = Math.sign(p - phi[0]);
      for (let j = 1; j < phi.length; j++) if (dir * (phi[j] - phi[far]) > 0) far = j;
    } else if (dir !== 0 && dir * (p - phi[far]) > 0) far = phi.length - 1;
  }
  let best: { ks: number[]; phi: number[] } | null = null;
  let bestSpan = 0;
  for (const ps of passes) {
    const span = Math.max(...ps.phi) - Math.min(...ps.phi);
    if (span > bestSpan + DEG || (best && Math.abs(span - bestSpan) <= DEG && ps.ks.length > best.ks.length)) {
      best = ps;
      bestSpan = span;
    }
  }
  if (!best || bestSpan < ORBIT_MIN_SWEEP) return null;
  const pass = best;
  const K = pass.ks;
  const m = K.length;
  const sK = new Float64Array(m);
  for (let j = 1; j < m; j++) sK[j] = sK[j - 1] + norm(sub(C[K[j]], C[K[j - 1]]));
  const rMed = median(K.map((k) => rad[k]));
  // the orientation in the frame turning with the orbit: R0 = R · Rot(φ)
  const R0s = K.map((k, j) => mul3(poses[k].R, rotAbout(up, pass.phi[j])));
  const qs = alignedQuats(R0s);
  const vmax = orbitRate * rMed;
  const lam = vmax / turn;
  const keyP = new Float64Array(m);
  for (let j = 1; j < m; j++) keyP[j] = keyP[j - 1] + Math.max(sK[j] - sK[j - 1], lam * rotationDistance(R0s[j], R0s[j - 1]));
  const passStep = median(Array.from(sK.subarray(1), (v, j) => v - sK[j]).filter((v) => v > 0)) || sK[m - 1] / Math.max(1, m - 1);
  const sigP = Math.max(passStep, rMed * ORBIT_SIGMA_POS);
  const sigQ = Math.max(passStep, rMed * ORBIT_SIGMA_ROT);
  const cyl = new Float64Array(3 * m);
  K.forEach((k, j) => cyl.set([rad[k], hgt[k], pass.phi[j]], 3 * j));
  const tmp = new Float64Array(3);
  /** Smoothed [radius, height, azimuth] at a parameter value (a shared buffer). */
  const at = (x: number): Float64Array => localLinear(sK, cyl, 3, interp(keyP, sK, x), sigP, tmp);
  const position = (x: number): V3 => {
    const [r, h, f] = at(x);
    const c = Math.cos(f);
    const s = Math.sin(f);
    return [centre[0] + r * (c * e1[0] + s * e2[0]) + h * up[0], centre[1] + r * (c * e1[1] + s * e2[1]) + h * up[1], centre[2] + r * (c * e1[2] + s * e2[2]) + h * up[2]];
  };
  const orbit = (x: number): Float64Array => rotAbout(up, at(x)[2]);
  const rotation = (x: number): Float64Array => level(mulT3(quatToMat3(quatAverage(keyP, qs, x, sigQ)), orbit(x)), up);
  const keyK = Float64Array.from(K);
  return densify(0, keyP[m - 1], position, rotation, {
    vmax,
    progress: (x) => interp(keyP, keyK, x) / (n - 1),
    // the orbit itself, or turning beyond it (the rotation in the orbit's turning frame)
    turning: (x, R) => mul3(R, orbit(x)),
    extra: (x0, x1) => {
      const f1 = at(x1)[2];
      return Math.abs(f1 - at(x0)[2]) / orbitRate;
    },
  });
}
