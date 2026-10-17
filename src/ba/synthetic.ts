/**
 * Synthetic bundle-adjustment problems for tests and benchmarks (not part of the
 * runtime pipeline; deterministic through the seeded Rng).
 *
 * Layouts:
 *  - 'orbit':  cameras on a circle (radius `radius`, gentle height variation) looking
 *              at the origin — close-range object capture.
 *  - 'strafe': cameras along +X looking at +Y (small attitude jitter), points 5–100 m
 *              away by default — large-scene / backyard capture.
 * Each point is generated in the frustum of an anchor camera and observed by the
 * cameras within ±`trackWindow` of it (when it projects inside the image in front
 * of the camera).
 */
import type { CameraIntrinsics, Pose } from '../core/types';
import { mat3Mul } from '../math/mat3';
import { Rng } from '../math/random';
import { rotationAngleBetween, rotvecToMat3 } from '../math/rotation';
import { radialResidual } from './camera-model';
import type { BAProblem } from './problem';

export interface SyntheticBAConfig {
  numPoses: number;
  numPoints: number;
  layout?: 'orbit' | 'strafe';
  seed?: number;
  width?: number;
  height?: number;
  /** Focal length (px). Default: 70° HFOV. */
  f?: number;
  k1?: number;
  k2?: number;
  /** Number of distinct cameras (poses assigned round-robin). Default 1. */
  numCameras?: number;
  /** Observe each point in anchor ± trackWindow (default 3). */
  trackWindow?: number;
  /** Orbit radius (default 4) or strafe camera spacing (default 1). */
  radius?: number;
  /** Fraction of the circle covered by the orbit (default 1). */
  orbitArc?: number;
  /** Depth range of points in the anchor camera (default orbit: [r−1.5, r+1.5], strafe: [5, 100]). */
  depthRange?: [number, number];
  /** Gaussian pixel noise σ (default 0.5). */
  noisePx?: number;
  /** Fraction of observations replaced by uniform random pixels (default 0). */
  outlierFraction?: number;
  /** Attitude jitter (rad, default 0.02). */
  attitudeJitter?: number;
}

export interface SyntheticBA {
  problem: BAProblem;
  gtPoses: Pose[];
  gtPoints: Float64Array;
  gtCameras: CameraIntrinsics[];
  /** 1 for observations replaced by gross outliers. */
  outlier: Uint8Array;
}

function lookAt(C: ArrayLike<number>, target: ArrayLike<number>, up: ArrayLike<number>): Float64Array {
  let fx = target[0] - C[0],
    fy = target[1] - C[1],
    fz = target[2] - C[2];
  let n = Math.hypot(fx, fy, fz);
  fx /= n;
  fy /= n;
  fz /= n;
  // right = forward × up
  let rx = fy * up[2] - fz * up[1],
    ry = fz * up[0] - fx * up[2],
    rz = fx * up[1] - fy * up[0];
  n = Math.hypot(rx, ry, rz);
  rx /= n;
  ry /= n;
  rz /= n;
  // down = forward × right
  const dx = fy * rz - fz * ry,
    dy = fz * rx - fx * rz,
    dz = fx * ry - fy * rx;
  return Float64Array.of(rx, ry, rz, dx, dy, dz, fx, fy, fz);
}

function poseFromRC(R: Float64Array, C: ArrayLike<number>): Pose {
  const t = new Float64Array(3);
  for (let r = 0; r < 3; r++) t[r] = -(R[3 * r] * C[0] + R[3 * r + 1] * C[1] + R[3 * r + 2] * C[2]);
  return { R, t };
}

/** Deep copy of a BA problem (all arrays and objects duplicated). */
export function cloneBAProblem(p: BAProblem): BAProblem {
  return {
    cameras: p.cameras.map((c) => ({ ...c })),
    cameraRefine: p.cameraRefine.map((c) => ({ ...c })),
    poses: p.poses.map(clonePose),
    imageCamera: p.imageCamera.slice(),
    poseFixed: p.poseFixed.slice(),
    translationFixedMask: p.translationFixedMask?.slice(),
    points: p.points.slice(),
    pointFixed: p.pointFixed?.slice(),
    obsPose: p.obsPose.slice(),
    obsPoint: p.obsPoint.slice(),
    obsUV: p.obsUV.slice(),
    obsWeight: p.obsWeight?.slice(),
    centerPriors: p.centerPriors?.map((c) => ({ pose: c.pose, center: Float64Array.from(c.center), sigma: c.sigma })),
    relRotationPriors: p.relRotationPriors?.map((r) => ({ i: r.i, j: r.j, R_ji: Float64Array.from(r.R_ji), sigma: r.sigma })),
    focalPriors: p.focalPriors?.map((f) => ({ ...f })),
    principalPointPriors: p.principalPointPriors?.map((c) => ({ ...c })),
  };
}

/** Camera centre C = −Rᵀ t. */
export function poseCentre(p: Pose, out: Float64Array = new Float64Array(3)): Float64Array {
  for (let c = 0; c < 3; c++) out[c] = -(p.R[c] * p.t[0] + p.R[3 + c] * p.t[1] + p.R[6 + c] * p.t[2]);
  return out;
}

function clonePose(p: Pose): Pose {
  return { R: Float64Array.from(p.R), t: Float64Array.from(p.t) };
}

/** Generate a synthetic problem (problem parameters = ground truth; perturb with `perturbBA`). */
export function makeSyntheticBA(cfg: SyntheticBAConfig): SyntheticBA {
  const rng = new Rng(cfg.seed ?? 1);
  const layout = cfg.layout ?? 'orbit';
  const W = cfg.width ?? 1920,
    H = cfg.height ?? 1440;
  const f = cfg.f ?? W / 2 / Math.tan((70 * Math.PI) / 360);
  const nCam = cfg.numCameras ?? 1;
  const cams: CameraIntrinsics[] = [];
  for (let c = 0; c < nCam; c++)
    cams.push({ width: W, height: H, f: f * (1 + 0.05 * c), cx: (W - 1) / 2 + 3 * c, cy: (H - 1) / 2 - 2 * c, k1: cfg.k1 ?? 0, k2: cfg.k2 ?? 0 });
  const nP = cfg.numPoses;
  const radius = cfg.radius ?? (layout === 'orbit' ? 4 : 1);
  const jit = cfg.attitudeJitter ?? 0.02;
  const poses: Pose[] = [];
  const up = Float64Array.of(0, 0, 1);
  const w = new Float64Array(3);
  const Rj = new Float64Array(9);
  for (let i = 0; i < nP; i++) {
    let C: Float64Array, R: Float64Array;
    if (layout === 'orbit') {
      const arc = cfg.orbitArc ?? 1;
      const th = (2 * Math.PI * arc * i) / (arc >= 1 ? nP : Math.max(1, nP - 1));
      C = Float64Array.of(radius * Math.cos(th), radius * Math.sin(th), 0.25 * radius * Math.sin(3 * th) + 0.5);
      R = lookAt(C, [0, 0, 0], up);
    } else {
      C = Float64Array.of(i * radius, 0.3 * Math.sin(i * 0.37), 1.5 + 0.2 * Math.cos(i * 0.23));
      R = lookAt(C, [C[0], C[1] + 10, C[2]], up);
    }
    w[0] = jit * rng.gaussian();
    w[1] = jit * rng.gaussian();
    w[2] = jit * rng.gaussian();
    rotvecToMat3(w, Rj);
    R = mat3Mul(Rj, R);
    poses.push(poseFromRC(R, C));
  }
  const depth = cfg.depthRange ?? (layout === 'orbit' ? [radius - 1.5, radius + 1.5] : [5, 100]);
  const win = cfg.trackWindow ?? 3;
  const noise = cfg.noisePx ?? 0.5;
  const outFrac = cfg.outlierFraction ?? 0;
  const imageCamera = new Uint32Array(nP);
  for (let i = 0; i < nP; i++) imageCamera[i] = i % nCam;

  const K = new Float64Array(5 * nCam);
  for (let c = 0; c < nCam; c++) K.set([cams[c].f, cams[c].cx, cams[c].cy, cams[c].k1, cams[c].k2], 5 * c);
  const Rf = new Float64Array(9 * nP),
    Tf = new Float64Array(3 * nP);
  for (let i = 0; i < nP; i++) {
    Rf.set(poses[i].R, 9 * i);
    Tf.set(poses[i].t, 3 * i);
  }

  const pts = new Float64Array(3 * cfg.numPoints);
  const obsPose: number[] = [];
  const obsPoint: number[] = [];
  const obsUV: number[] = [];
  const res = new Float64Array(2);
  const X = new Float64Array(3);
  const cand: number[] = [];
  const candUV: number[] = [];
  for (let j = 0; j < cfg.numPoints; j++) {
    for (let attempt = 0; ; attempt++) {
      const a = rng.int(nP);
      const pa = poses[a];
      const cam = cams[imageCamera[a]];
      // random normalized direction inside ~90 % of the image
      const u = rng.uniform(0.05 * W, 0.95 * W),
        v = rng.uniform(0.05 * H, 0.95 * H);
      const xn = (u - cam.cx) / cam.f,
        yn = (v - cam.cy) / cam.f;
      // inverse-depth-uniform sampling covers near and far evenly
      const d = 1 / rng.uniform(1 / depth[1], 1 / depth[0]);
      const xc = xn * d,
        yc = yn * d,
        zc = d;
      for (let c = 0; c < 3; c++)
        X[c] = pa.R[c] * (xc - pa.t[0]) + pa.R[3 + c] * (yc - pa.t[1]) + pa.R[6 + c] * (zc - pa.t[2]);
      cand.length = 0;
      candUV.length = 0;
      for (let di = -win; di <= win; di++) {
        let i = a + di;
        if (layout === 'orbit' && (cfg.orbitArc ?? 1) >= 1) i = ((i % nP) + nP) % nP;
        if (i < 0 || i >= nP) continue;
        if (di !== 0 && i === a) continue;
        const c = imageCamera[i];
        // in front?
        const zc2 = Rf[9 * i + 6] * X[0] + Rf[9 * i + 7] * X[1] + Rf[9 * i + 8] * X[2] + Tf[3 * i + 2];
        if (!(zc2 > 1e-3)) continue;
        radialResidual(Rf, 9 * i, Tf, 3 * i, X, 0, K, 5 * c, 0, 0, res);
        if (res[0] < 0 || res[0] > W - 1 || res[1] < 0 || res[1] > H - 1) continue;
        if (cand.includes(i)) continue;
        cand.push(i);
        candUV.push(res[0], res[1]);
      }
      if (cand.length >= 2 || attempt > 50) break;
    }
    pts.set(X, 3 * j);
    for (let k = 0; k < cand.length; k++) {
      obsPose.push(cand[k]);
      obsPoint.push(j);
      obsUV.push(candUV[2 * k], candUV[2 * k + 1]);
    }
  }
  const nO = obsPose.length;
  const uv = Float64Array.from(obsUV);
  const outlier = new Uint8Array(nO);
  for (let o = 0; o < nO; o++) {
    if (outFrac > 0 && rng.next() < outFrac) {
      outlier[o] = 1;
      uv[2 * o] = rng.uniform(0, W - 1);
      uv[2 * o + 1] = rng.uniform(0, H - 1);
    } else {
      uv[2 * o] += noise * rng.gaussian();
      uv[2 * o + 1] += noise * rng.gaussian();
    }
  }
  const problem: BAProblem = {
    cameras: cams.map((c) => ({ ...c })),
    cameraRefine: cams.map(() => ({ focal: false, principalPoint: false, distortion: false })),
    poses: poses.map(clonePose),
    imageCamera,
    poseFixed: new Uint8Array(nP),
    points: Float64Array.from(pts),
    obsPose: Uint32Array.from(obsPose),
    obsPoint: Uint32Array.from(obsPoint),
    obsUV: uv,
  };
  return { problem, gtPoses: poses.map(clonePose), gtPoints: pts, gtCameras: cams.map((c) => ({ ...c })), outlier };
}

export interface PerturbOptions {
  seed?: number;
  /** σ of the rotation perturbation (rad, per axis). */
  rotSigma?: number;
  /** σ of the camera-centre perturbation (world units, per axis). */
  centreSigma?: number;
  /** σ of the point perturbation (world units, per axis) or, if `pointRelative`, relative to depth from origin. */
  pointSigma?: number;
  /** Relative focal perturbation (e.g. 0.1 = +10 %) applied to cameras with focal refinement. */
  focalScale?: number;
  /** Added to k1 of cameras with distortion refinement. */
  k1Delta?: number;
}

/**
 * Perturb the FREE parameters of `s.problem` in place (fixed poses / points and
 * pinned translation components are left untouched).
 */
export function perturbBA(s: SyntheticBA, o: PerturbOptions): void {
  const rng = new Rng(o.seed ?? 7);
  const p = s.problem;
  const w = new Float64Array(3);
  const Rd = new Float64Array(9);
  const C = new Float64Array(3);
  for (let i = 0; i < p.poses.length; i++) {
    if (p.poseFixed[i]) continue;
    const ps = p.poses[i];
    poseCentre(ps, C);
    const rs = o.rotSigma ?? 0,
      cs = o.centreSigma ?? 0;
    w[0] = rs * rng.gaussian();
    w[1] = rs * rng.gaussian();
    w[2] = rs * rng.gaussian();
    rotvecToMat3(w, Rd);
    const R = mat3Mul(Rd, ps.R);
    for (let k = 0; k < 3; k++) C[k] += cs * rng.gaussian();
    const tOld = Float64Array.from(ps.t);
    const np = poseFromRC(R, C);
    ps.R.set(R);
    ps.t.set(np.t);
    const m = p.translationFixedMask ? p.translationFixedMask[i] : 0;
    for (let k = 0; k < 3; k++) if (m & (1 << k)) ps.t[k] = tOld[k];
  }
  const ps = o.pointSigma ?? 0;
  for (let j = 0; j < p.points.length / 3; j++) {
    if (p.pointFixed && p.pointFixed[j]) continue;
    for (let k = 0; k < 3; k++) p.points[3 * j + k] += ps * rng.gaussian();
  }
  for (let c = 0; c < p.cameras.length; c++) {
    if (p.cameraRefine[c].focal && o.focalScale) p.cameras[c].f *= 1 + o.focalScale;
    if (p.cameraRefine[c].distortion && o.k1Delta) p.cameras[c].k1 += o.k1Delta;
  }
}

/** Errors of the problem's current poses / points / focal vs ground truth (no alignment). */
export function syntheticErrors(s: SyntheticBA): {
  maxRotErr: number;
  maxCentreErr: number;
  rmsPointErr: number;
  medianPointErr: number;
  maxFocalRelErr: number;
} {
  const p = s.problem;
  let maxRot = 0,
    maxC = 0;
  const c1 = new Float64Array(3),
    c2 = new Float64Array(3);
  for (let i = 0; i < p.poses.length; i++) {
    maxRot = Math.max(maxRot, rotationAngleBetween(p.poses[i].R, s.gtPoses[i].R));
    poseCentre(p.poses[i], c1);
    poseCentre(s.gtPoses[i], c2);
    maxC = Math.max(maxC, Math.hypot(c1[0] - c2[0], c1[1] - c2[1], c1[2] - c2[2]));
  }
  const nX = p.points.length / 3;
  let se = 0;
  const errs = new Float64Array(nX);
  for (let j = 0; j < nX; j++) {
    const e = Math.hypot(
      p.points[3 * j] - s.gtPoints[3 * j],
      p.points[3 * j + 1] - s.gtPoints[3 * j + 1],
      p.points[3 * j + 2] - s.gtPoints[3 * j + 2],
    );
    errs[j] = e;
    se += e * e;
  }
  errs.sort();
  let mf = 0;
  for (let c = 0; c < p.cameras.length; c++) mf = Math.max(mf, Math.abs(p.cameras[c].f / s.gtCameras[c].f - 1));
  return {
    maxRotErr: maxRot,
    maxCentreErr: maxC,
    rmsPointErr: Math.sqrt(se / Math.max(1, nX)),
    medianPointErr: nX ? errs[nX >> 1] : 0,
    maxFocalRelErr: mf,
  };
}

/**
 * Fix the gauge like an incremental mapper: pose `a` fixed, and the translation
 * component of pose `b` with the largest sensitivity to scale pinned.
 */
export function fixGaugeTwoPoses(p: BAProblem, a = 0, b = 1): void {
  p.poseFixed[a] = 1;
  const mask = p.translationFixedMask ?? new Uint8Array(p.poses.length);
  const Ca = poseCentre(p.poses[a]),
    Cb = poseCentre(p.poses[b]);
  const d = [Cb[0] - Ca[0], Cb[1] - Ca[1], Cb[2] - Ca[2]];
  const R = p.poses[b].R;
  let best = 0,
    bestAbs = -1;
  for (let k = 0; k < 3; k++) {
    const v = Math.abs(R[3 * k] * d[0] + R[3 * k + 1] * d[1] + R[3 * k + 2] * d[2]);
    if (v > bestAbs) {
      bestAbs = v;
      best = k;
    }
  }
  mask[b] = 1 << best;
  p.translationFixedMask = mask;
}
