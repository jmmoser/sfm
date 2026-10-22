/**
 * Fly-through camera path for the video export (pure: no three.js / WebGPU). From the registered
 * keyframe cameras in capture order, it builds N frame poses (ARCHITECTURE §2: world-to-camera,
 * OpenCV camera frame) and the pinhole intrinsics of the video. The path is a stabilised replay of
 * the capture rather than a free orbit. The splats are photoreal only near the photos: on the
 * synthetic object scene they lose ~1 dB at +5 cm off the trajectory and ~9 dB at +25 cm (e2e/README
 * "Novel views"). So the path hugs the trajectory and the view direction follows the photos.
 *
 * Modes ('auto': object when the optical axes converge, src/geometry/convergence.ts):
 *  - object: an orbit about the vertical axis through the optical-axis convergence point, swept
 *    at a constant rate over the azimuth range the keyframes cover (all of it, and several turns
 *    for a multi-loop capture). The radius and height follow the keyframes, smoothed over azimuth
 *    (Gaussian local-linear fit, σ 12°), so a steady orbit is an arc at the median radius and
 *    elevation, while a capture that rises or wanders keeps hugging it. The orientation is the
 *    keyframes' smoothed over azimuth (σ 15°) in the frame that turns with the orbit, so the
 *    camera keeps facing the object the way the user aimed it.
 *  - scene: a centripetal Catmull-Rom spline through the keyframe centres. The centres are first
 *    smoothed (local-linear in arc length, σ = the median keyframe step) and decimated to that
 *    spacing. The spline is arc-length parameterised; orientations are low-pass averaged
 *    quaternions (Gaussian over the keyframe sequence, σ 0.7 keyframes: ROT_SIGMA_KEYFRAMES).
 * Both modes level the horizon to `up` (the camera's x axis stays horizontal) and keep the
 * smoothed view direction.
 *
 * Timing: each piece of path costs the time it needs at the speed limits, and the clip plays it
 * at that rate. The limits are: translation ≤ 0.2 · depth per s (scene; depth = the typical viewing
 * distance), view rotation ≤ 30 °/s, and orbit ≤ 40 °/s (object; the view rotation then counts only
 * beyond the orbit's own turning). The duration is clamped to 6–10 s. A
 * longer capture is cropped to its middle part (`coverage`); a shorter one plays slower. The start
 * and end ease in and out over ≤ 0.6 s (smoothstep speed ramps), so frame steps never exceed the
 * cruise step.
 *
 * Video frame: portrait when the photos' long side is vertical (physical up = `up`), else
 * landscape; 1280 × 720 by default. The focal length is the smallest that keeps the frame inside
 * the photos' median field of view, times `zoom` (1.05: the photo borders are the least
 * constrained parts of the splats).
 */
import type { CameraIntrinsics, Pose, Vec3 } from '../core/types';
import { opticalAxisConvergence } from '../geometry/convergence';
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

export type FlythroughMode = 'object' | 'scene';

/** A registered keyframe camera (capture order). */
export interface FlythroughCamera {
  pose: Pose;
  /** The photo's intrinsics (width, height, f). */
  intrinsics: Pick<CameraIntrinsics, 'width' | 'height' | 'f'>;
}

export interface FlythroughOptions {
  /** 'auto' (default): 'object' when the optical axes converge, else 'scene'. */
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
  /** Fastest view rotation (default 30 °/s). */
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
  /** Part of the capture the clip shows (fraction of the path, 0–1). */
  coverage: [number, number];
  /** Typical viewing distance used (world units). */
  depth: number;
}

/** The SplatCamera (world frame, OpenCV) of frame `i`. */
export function flythroughCamera(p: FlythroughPath, i: number): SplatCamera {
  const pose = p.poses[i];
  return { width: p.width, height: p.height, fx: p.fx, fy: p.fy, cx: p.cx, cy: p.cy, R: pose.R, t: pose.t, near: p.near };
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
 * average.
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
  const det = s0 * s2 - s1 * s1;
  for (let j = 0; j < d; j++) {
    if (s0 < 1e-9) out[j] = ys[d * nearest + j];
    else if (det > 1e-9 * s0 * s0 * sigma * sigma) out[j] = (s2 * t0[j] - s1 * t1[j]) / det;
    else out[j] = t0[j] / s0;
  }
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

/** A dense path: `samples` parameter values with their positions and the cost (s) of each step. */
interface DensePath {
  /** Cumulative cost (s at the speed limits) per sample. */
  cost: Float64Array;
  /** Path parameter per sample (scene: arc length; object: unwrapped azimuth). */
  param: Float64Array;
  /** Position at a parameter value. */
  position(x: number): V3;
  /** Smoothed, levelled world-to-camera rotation at a parameter value. */
  rotation(x: number): Float64Array;
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
  const objectOk = !!conv.point && conv.converging;
  let mode: FlythroughMode = o.mode === 'object' || o.mode === 'scene' ? o.mode : objectOk ? 'object' : 'scene';
  if (mode === 'object' && !conv.point) mode = 'scene';
  let ext = 0;
  for (const c of C) ext = Math.max(ext, norm(sub(c, C[0])));
  const depth = o.depth && o.depth > 0 ? o.depth : objectOk ? conv.distMedian : Math.max(5 * (median(steps) || 0), 0.1 * ext, 1e-3);

  let path: DensePath | null = mode === 'object' ? objectPath(poses, C, conv.point!, up, orbitRate, turn) : null;
  if (!path) {
    mode = 'scene';
    path = scenePath(poses, C, steps, up, (o.maxSpeedPerDepth ?? 0.2) * depth, turn);
  }

  // timing: the middle of the path when it needs more than maxS, eased start and end
  // (T − 0.6 s plays the path at the limits: shorter paths play slower over minS, longer ones are cropped)
  const total = path.cost[path.cost.length - 1];
  const rampSec = 0.6;
  const T = Math.min(maxS, Math.max(minS, total + rampSec));
  const r = Math.min(0.15, rampSec / T);
  const window = Math.min(total, T * (1 - r));
  const a = 0.5 * (total - window);
  const b = a + window;
  const N = Math.max(2, Math.round(T * fps));
  const out: Pose[] = [];
  for (let i = 0; i < N; i++) {
    const m = a + (b - a) * easedProgress(i / (N - 1), r);
    const x = interp(path.cost, path.param, m);
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
  const L = path.param[path.param.length - 1] - path.param[0];
  const toFrac = (x: number): number => (L > 0 ? (x - path.param[0]) / L : 0);
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
    coverage: [toFrac(interp(path.cost, path.param, a)), toFrac(interp(path.cost, path.param, b))],
    depth,
  };
}

/** Scene mode: smoothed, decimated centres → centripetal Catmull-Rom; arc-length parameter. */
function scenePath(poses: Pose[], C: V3[], steps: number[], up: V3, vmax: number, turn: number): DensePath {
  const n = C.length;
  const s = new Float64Array(n);
  for (let k = 1; k < n; k++) s[k] = s[k - 1] + steps[k - 1];
  const len = s[n - 1];
  const qs = alignedQuats(poses.map((p) => p.R));
  if (len < 1e-9) {
    // no translation (a pan from one spot): the orientations alone, by keyframe index
    const idx = Float64Array.from({ length: n }, (_, k) => k);
    const fixed: V3 = C[0];
    return densify(idx, () => fixed, (x) => level(quatToMat3(quatAverage(idx, qs, x, ROT_SIGMA_KEYFRAMES)), up), 0, turn);
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
  // orientations are smoothed over the keyframe sequence (fractional keyframe index along the path)
  const keyU = Float64Array.from(s, uOfS);
  const keyIdx = Float64Array.from({ length: n }, (_, k) => k);
  const uArr = Float64Array.from(u);
  const position = (x: number): V3 => {
    const k = Math.min(pts.length - 1, Math.max(0, upperIndex(uArr, x)));
    const k0 = Math.max(0, k - 1);
    const w = uArr[k] > uArr[k0] ? (x - uArr[k0]) / (uArr[k] - uArr[k0]) : 0;
    const p0 = pts[k0];
    const p1 = pts[k];
    return [p0[0] + (p1[0] - p0[0]) * w, p0[1] + (p1[1] - p0[1]) * w, p0[2] + (p1[2] - p0[2]) * w];
  };
  const rotation = (x: number): Float64Array => level(quatToMat3(quatAverage(keyIdx, qs, interp(keyU, keyIdx, x), ROT_SIGMA_KEYFRAMES)), up);
  return densify(uArr, position, rotation, vmax, turn);
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
 * Cost table of a parameterised path: ~2000 samples between the parameter ends; each step costs
 * max(translation / vmax, rotation / turn) seconds (vmax 0: rotation only).
 */
function densify(param: ArrayLike<number>, position: (x: number) => V3, rotation: (x: number) => Float64Array, vmax: number, turn: number): DensePath {
  const x0 = param[0];
  const x1 = param[param.length - 1];
  const S = 2000;
  const xs = new Float64Array(S + 1);
  const cost = new Float64Array(S + 1);
  let prevP = position(x0);
  let prevR = rotation(x0);
  for (let i = 1; i <= S; i++) {
    const x = x0 + ((x1 - x0) * i) / S;
    xs[i] = x;
    const P = position(x);
    const R = rotation(x);
    const dl = norm(sub(P, prevP));
    const dr = rotationDistance(R, prevR);
    cost[i] = cost[i - 1] + Math.max(vmax > 0 ? dl / vmax : 0, dr / turn, 1e-12);
    prevP = P;
    prevR = R;
  }
  xs[0] = x0;
  return { cost, param: xs, position, rotation };
}

/**
 * Object mode: orbit about the vertical through the convergence point over the azimuth range
 * covered. Null when the keyframes cover less than 20° of azimuth (no orbit to show).
 */
function objectPath(poses: Pose[], C: V3[], centre: ArrayLike<number>, up: V3, orbitRate: number, turn: number): DensePath | null {
  const n = C.length;
  // horizontal basis: e1 towards the first camera
  const v0 = sub(C[0], centre);
  let e1 = sub(v0, up.map((x) => x * dot(v0, up)) as V3);
  if (norm(e1) < 1e-9) e1 = Math.abs(up[0]) < 0.9 ? cross(up, [1, 0, 0]) : cross(up, [0, 1, 0]);
  e1 = unit(e1);
  const e2 = cross(up, e1);
  const phi = new Float64Array(n);
  const rh = new Float64Array(2 * n);
  for (let k = 0; k < n; k++) {
    const v = sub(C[k], centre);
    const h = dot(v, up);
    const a = Math.atan2(dot(v, e2), dot(v, e1));
    if (k === 0) phi[k] = a;
    else {
      let d = a - (phi[k - 1] % (2 * Math.PI));
      d = ((((d + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;
      phi[k] = phi[k - 1] + d;
    }
    rh[2 * k] = Math.hypot(dot(v, e1), dot(v, e2));
    rh[2 * k + 1] = h;
  }
  let lo = 0;
  let hi = 0;
  for (let k = 1; k < n; k++) {
    if (phi[k] < phi[lo]) lo = k;
    if (phi[k] > phi[hi]) hi = k;
  }
  if (phi[hi] - phi[lo] < 20 * DEG) return null;
  // the orientation in the frame turning with the orbit: R0 = R · Rot(φ)
  const R0s = poses.map((p, k) => mul3(p.R, rotAbout(up, phi[k])));
  // smoothing is by azimuth: sort the keyframes by φ (a back-and-forth capture interleaves)
  const order = Array.from({ length: n }, (_, k) => k).sort((x, y) => phi[x] - phi[y]);
  const ph = Float64Array.from(order, (k) => phi[k]);
  const rhs = new Float64Array(2 * n);
  order.forEach((k, i) => rhs.set([rh[2 * k], rh[2 * k + 1]], 2 * i));
  const qs = alignedQuats(order.map((k) => R0s[k]));
  const sP = 12 * DEG;
  const sQ = 15 * DEG;
  const tmp = new Float64Array(2);
  const position = (x: number): V3 => {
    localLinear(ph, rhs, 2, x, sP, tmp);
    const c = Math.cos(x);
    const s = Math.sin(x);
    return [centre[0] + tmp[0] * (c * e1[0] + s * e2[0]) + tmp[1] * up[0], centre[1] + tmp[0] * (c * e1[1] + s * e2[1]) + tmp[1] * up[1], centre[2] + tmp[0] * (c * e1[2] + s * e2[2]) + tmp[1] * up[2]];
  };
  const rotation = (x: number): Float64Array => level(mulT3(quatToMat3(quatAverage(ph, qs, x, sQ)), rotAbout(up, x)), up);
  // sweep in the capture's direction: from the end it started nearer to
  const forward = Math.abs(phi[0] - phi[lo]) <= Math.abs(phi[0] - phi[hi]);
  const a = forward ? phi[lo] : phi[hi];
  const b = forward ? phi[hi] : phi[lo];
  // cost: the orbit itself, or turning beyond it (the rotation in the orbit's turning frame)
  const S = 2000;
  const cost = new Float64Array(S + 1);
  let prevR0 = mul3(rotation(a), rotAbout(up, a));
  for (let i = 1; i <= S; i++) {
    const x = a + ((b - a) * i) / S;
    const R0 = mul3(rotation(x), rotAbout(up, x));
    cost[i] = cost[i - 1] + Math.max(Math.abs(b - a) / S / orbitRate, rotationDistance(R0, prevR0) / turn);
    prevR0 = R0;
  }
  // `interp` needs an ascending parameter: map the cost to the sweep fraction, then to φ
  const frac = Float64Array.from({ length: S + 1 }, (_, i) => i / S);
  return {
    cost,
    param: frac,
    position: (f) => position(a + (b - a) * f),
    rotation: (f) => rotation(a + (b - a) * f),
  };
}
