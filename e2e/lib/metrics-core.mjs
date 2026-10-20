// Evaluation of a reconstruction result against a synthetic dataset (pure; no fs).
//
// Result format: { intrinsics: {f,cx,cy,k1,k2}, images: [{index, pose: {R: number[9], t: number[3]}}],
//   metric: {metric, gravityAligned, northAligned}, pointsFile?, numInputImages? }
// Poses are world-to-camera (x_cam = R X + t), row-major, OpenCV camera frame.

import {
  applySim3,
  chordalMean,
  mat3Mul,
  mat3T,
  mean,
  percentile,
  rms,
  rotationAngleBetween,
  umeyama,
} from './linalg.mjs';
import { HashGrid } from './grid.mjs';
import { KdTree } from './kdtree.mjs';

const DEG = 180 / Math.PI;

function center(pose) {
  const R = pose.R,
    t = pose.t;
  return [-(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]), -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]), -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2])];
}

function angleDeg(a, b) {
  const d = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (Math.hypot(...a) * Math.hypot(...b));
  return Math.acos(Math.min(1, Math.max(-1, d))) * DEG;
}

/** Signed heading error (deg, + = result north rotated clockwise / towards east). */
function headingErrorDeg(Rmap) {
  const v = [Rmap[1], Rmap[4], Rmap[7]]; // result +Y expressed in the GT frame
  return Math.atan2(v[0], v[1]) * DEG;
}

function inBox(box, x, y, z) {
  return x >= box.min[0] && x <= box.max[0] && y >= box.min[1] && y <= box.max[1] && z >= box.min[2] && z <= box.max[2];
}

function distortionDisplacementPx(cam, k1, k2, r) {
  const r2 = r * r;
  return cam.f * r * (k1 * r2 + k2 * r2 * r2);
}

/** Undistorted normalized radius r with r·(1 + k1 r² + k2 r⁴) = rd (Newton). */
function undistortRadius(k1, k2, rd) {
  let r = rd;
  for (let i = 0; i < 30; i++) {
    const r2 = r * r;
    const step = (r * (1 + k1 * r2 + k2 * r2 * r2) - rd) / (1 + 3 * k1 * r2 + 5 * k2 * r2 * r2);
    r -= step;
    if (Math.abs(step) < 1e-15) break;
  }
  return r;
}

/**
 * @param {object} dataset  { meta, frames, gt?: {points, normals?, visibility?}, gtFocus?: {...} }
 * @param {object} result   parsed result JSON
 * @param {object} [opts]   { points?: Float32Array, tau?, roi?: boolean, minVisibility?, denseAlign?: 'sim3'|'sim3+icp'|'se3'|'none',
 *                            maxPoints?, maxDistFactor? }
 */
export function computeMetrics(dataset, result, opts = {}) {
  const { meta, frames } = dataset;
  const byIndex = new Map(frames.map((f) => [f.index, f]));
  const out = { dataset: meta.name, scene: meta.scene, numFrames: frames.length };

  // ---------------- registration ----------------
  const pairs = [];
  const unknown = [];
  for (const im of result.images ?? []) {
    const f = byIndex.get(im.index);
    if (!f) {
      unknown.push(im.index);
      continue;
    }
    pairs.push({ index: im.index, gt: f.pose, res: { R: im.pose.R, t: im.pose.t } });
  }
  pairs.sort((a, b) => a.index - b.index);
  const numInput = result.numInputImages ?? null;
  out.registration = {
    numFrames: frames.length,
    numRegistered: pairs.length,
    rate: pairs.length / frames.length,
    numInput,
    rateOfInput: numInput ? pairs.length / numInput : null,
    unknownIndices: unknown.length,
  };

  // ---------------- trajectory (Sim3) ----------------
  const Cg = pairs.map((p) => center(p.gt));
  const Cr = pairs.map((p) => center(p.res));
  const sim = umeyama(Cr, Cg, true);
  const extent = meta.sceneExtent;
  out.trajectory = null;
  if (sim) {
    const err = Cr.map((c, i) => {
      const a = applySim3(sim, c);
      return Math.hypot(a[0] - Cg[i][0], a[1] - Cg[i][1], a[2] - Cg[i][2]);
    });
    const RsimT = mat3T(sim.R);
    const rot = pairs.map((p) => rotationAngleBetween(p.gt.R, mat3Mul(p.res.R, RsimT)) * DEG);
    // relative rotation error between consecutive registered images
    const rpe = [];
    for (let i = 1; i < pairs.length; i++) {
      const relG = mat3Mul(pairs[i].gt.R, mat3T(pairs[i - 1].gt.R));
      const relR = mat3Mul(pairs[i].res.R, mat3T(pairs[i - 1].res.R));
      rpe.push(rotationAngleBetween(relG, relR) * DEG);
    }
    out.trajectory = {
      sim3: sim,
      ateRmse: rms(err),
      ateMedian: percentile(err, 50),
      ateMax: Math.max(...err),
      sceneExtent: extent,
      ateRelative: rms(err) / extent,
      rotErrDegMean: mean(rot),
      rotErrDegMedian: percentile(rot, 50),
      rotErrDegMax: Math.max(...rot),
      rpeRotDegMean: rpe.length ? mean(rpe) : null,
      rpeRotDegMax: rpe.length ? Math.max(...rpe) : null,
    };
  }

  // ---------------- metric scale ----------------
  const claims = result.metric ?? { metric: false, gravityAligned: false, northAligned: false };
  out.metric = null;
  // Rigid (no scale) fit of the result centres to GT: used for the metric ATE and for
  // `denseAlign: 'se3'` (dropping the scale of the Sim3 would leave its translation,
  // which was solved for the scaled points, off by (s − 1)·R·mean — metres at ENU offsets).
  const se3 = sim ? umeyama(Cr, Cg, false) : null;
  if (claims.metric && sim) {
    const errSe3 = se3
      ? Cr.map((c, i) => {
          const a = applySim3(se3, c);
          return Math.hypot(a[0] - Cg[i][0], a[1] - Cg[i][1], a[2] - Cg[i][2]);
        })
      : [];
    out.metric = { scale: sim.s, scaleError: 1 / sim.s - 1, ateSe3Rmse: se3 ? rms(errSe3) : null, ateSe3Relative: se3 ? rms(errSe3) / extent : null };
  }

  // ---------------- gravity / north (orientation-only frame alignment) ----------------
  // A_i = R_gtᵀ R_res maps result-frame vectors to the GT frame; A = chordal mean.
  out.frame = null;
  if (pairs.length > 0) {
    const A = chordalMean(pairs.map((p) => mat3Mul(mat3T(p.gt.R), p.res.R)));
    const upA = [A[2], A[5], A[8]];
    out.frame = {
      gravityAligned: !!claims.gravityAligned,
      northAligned: !!claims.northAligned,
      gravityErrorDeg: angleDeg(upA, [0, 0, 1]),
      gravityErrorDegSim3: sim ? angleDeg([sim.R[2], sim.R[5], sim.R[8]], [0, 0, 1]) : null,
      headingErrorDeg: headingErrorDeg(A),
      headingErrorDegSim3: sim ? headingErrorDeg(sim.R) : null,
    };
  }

  // ---------------- intrinsics ----------------
  out.intrinsics = null;
  const ri = result.intrinsics;
  const gi = meta.intrinsics;
  if (ri && typeof ri.f === 'number') {
    let rCorner = 0;
    for (const [u, v] of [
      [-0.5, -0.5],
      [meta.width - 0.5, -0.5],
      [-0.5, meta.height - 0.5],
      [meta.width - 0.5, meta.height - 0.5],
    ])
      rCorner = Math.max(rCorner, Math.hypot((u - gi.cx) / gi.f, (v - gi.cy) / gi.f));
    // The corner pixel is at DISTORTED radius rCorner; compare both models on the GT ray
    // through it (undistorted radius), i.e. the pixel error of the result's distortion there.
    rCorner = undistortRadius(gi.k1, gi.k2, rCorner);
    out.intrinsics = {
      fGt: gi.f,
      fRes: ri.f,
      focalErrorPct: (100 * (ri.f - gi.f)) / gi.f,
      ppErrorPx: typeof ri.cx === 'number' ? Math.hypot(ri.cx - gi.cx, ri.cy - gi.cy) : null,
      k1Error: typeof ri.k1 === 'number' ? ri.k1 - gi.k1 : null,
      k2Error: typeof ri.k2 === 'number' ? ri.k2 - gi.k2 : null,
      cornerDistortionErrorPx:
        typeof ri.k1 === 'number'
          ? Math.abs(distortionDisplacementPx(gi, ri.k1, ri.k2 ?? 0, rCorner) - distortionDisplacementPx(gi, gi.k1, gi.k2, rCorner))
          : null,
      perFrameFocal: null,
    };
    // Focus breathing (informational): datasets with real-capture artefacts carry a GT focal per
    // frame (frames.json f; meta f = f0). Compare the single estimated focal with their mean over
    // the registered frames, and report the spread.
    const fAll = frames.map((f) => f.f).filter((v) => typeof v === 'number');
    if (fAll.length === frames.length && fAll.length > 0) {
      const fReg = pairs.map((p) => byIndex.get(p.index).f);
      const fRegMean = fReg.length ? mean(fReg) : null;
      out.intrinsics.perFrameFocal = {
        min: Math.min(...fAll),
        max: Math.max(...fAll),
        mean: mean(fAll),
        registeredMean: fRegMean,
        focalErrorPctVsRegisteredMean: fRegMean ? (100 * (ri.f - fRegMean)) / fRegMean : null,
      };
    }
  }

  // ---------------- dense ----------------
  out.dense = null;
  if (opts.points && dataset.gt) out.dense = denseMetrics(dataset, opts.points, sim, se3, opts);
  return out;
}

const DENSE_ALIGN_MODES = ['sim3', 'sim3+icp', 'se3', 'none'];

function denseMetrics(dataset, rawPoints, sim, se3, opts) {
  const { meta } = dataset;
  const tau = opts.tau ?? meta.denseTau ?? (meta.scene === 'object' ? 0.01 * meta.sceneExtent : 0.1);
  const useRoi = opts.roi !== false && meta.roi;
  const minVis = opts.minVisibility ?? 1;
  const mode = opts.denseAlign ?? 'sim3';
  if (!DENSE_ALIGN_MODES.includes(mode)) throw new Error(`unknown dense-align mode '${mode}' (expected ${DENSE_ALIGN_MODES.join(' | ')})`);
  if (!(tau > 0)) throw new Error(`dense tau must be > 0 (got ${tau})`);
  const maxDist = (opts.maxDistFactor ?? 10) * tau;
  let T = null;
  if (mode === 'sim3' || mode === 'sim3+icp') T = sim;
  else if (mode === 'se3') T = se3;
  // ---- transform + ROI-filter result points; accuracy / precision use a ≤ maxPoints subsample,
  // completeness every point (a strided subsample of a large cloud thins its coverage: a 3.5 M
  // point yard cloud read every 2nd point scored C 0.434 instead of 0.484) ----
  const nRaw = rawPoints.length / 3;
  const maxPoints = opts.maxPoints ?? 3_000_000;
  const step = Math.max(1, Math.ceil(nRaw / maxPoints));
  const all = new Float64Array(3 * nRaw);
  let nAll = 0;
  const pts = [];
  for (let i = 0; i < nRaw; i++) {
    let p = [rawPoints[3 * i], rawPoints[3 * i + 1], rawPoints[3 * i + 2]];
    if (!Number.isFinite(p[0] + p[1] + p[2])) continue;
    if (T) p = applySim3(T, p);
    if (useRoi && !inBox(meta.roi, p[0], p[1], p[2])) continue;
    all[3 * nAll] = p[0];
    all[3 * nAll + 1] = p[1];
    all[3 * nAll + 2] = p[2];
    nAll++;
    if (i % step === 0) pts.push(p[0], p[1], p[2]);
  }
  const P = Float64Array.from(pts);
  const nP = P.length / 3;
  // ---- GT sample sets: base (+ dense focus set), one k-d tree each ----
  const sets = [];
  const addSet = (set, spacing, maxD) => {
    if (!set) return;
    const n = set.points.length / 3;
    const N = new Float32Array(3 * n);
    if (set.normals) for (let i = 0; i < 3 * n; i++) N[i] = set.normals[i] / 127;
    sets.push({ G: set.points, N: set.normals ? N : null, h: spacing, tree: new KdTree(set.points), maxD });
  };
  const spacingBase = meta.gt?.spacing ?? tau / 2;
  addSet(dataset.gt, spacingBase, maxDist);
  if (dataset.gtFocus) {
    const sf = meta.gt?.focus?.spacing ?? spacingBase;
    addSet(dataset.gtFocus, sf, maxDist);
  }
  // ---- optional ICP refinement (decouples dense quality from camera-pose error) ----
  let icp = null;
  if (mode === 'sim3+icp' && nP >= 10) {
    const stride = Math.max(1, Math.floor(nP / 50000));
    let total = { s: 1, R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0] };
    let it = 0;
    for (; it < 15; it++) {
      const src = [],
        dst = [],
        dist = [];
      for (let i = 0; i < nP; i += stride) {
        const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
        let best = null;
        for (const S of sets) {
          const nn = S.tree.nearest(x, y, z, Math.min(3 * tau, best ? best.d : Infinity));
          if (nn.index >= 0 && (!best || nn.dist < best.d)) best = { d: nn.dist, p: [S.G[3 * nn.index], S.G[3 * nn.index + 1], S.G[3 * nn.index + 2]] };
        }
        if (best) {
          src.push([x, y, z]);
          dst.push(best.p);
          dist.push(best.d);
        }
      }
      if (src.length < 10) break;
      const cut = percentile(dist, 80);
      const s2 = [], d2 = [];
      for (let k = 0; k < src.length; k++) if (dist[k] <= cut) { s2.push(src[k]); d2.push(dst[k]); }
      const step = umeyama(s2, d2, false);
      if (!step) break;
      for (let i = 0; i < nP; i++) {
        const q = applySim3(step, [P[3 * i], P[3 * i + 1], P[3 * i + 2]]);
        P[3 * i] = q[0]; P[3 * i + 1] = q[1]; P[3 * i + 2] = q[2];
      }
      total = { s: 1, R: mat3Mul(step.R, total.R), t: applySim3(step, total.t) };
      const ang = rotationAngleBetween(step.R, [1, 0, 0, 0, 1, 0, 0, 0, 1]);
      if (ang < 1e-6 && Math.hypot(...step.t) < 1e-4 * tau) break;
    }
    icp = { iterations: it, rotationDeg: rotationAngleBetween(total.R, [1, 0, 0, 0, 1, 0, 0, 0, 1]) * DEG, translation: Math.hypot(...total.t) };
  }
  // ---- accuracy: result → GT surface (nearest sample, point-to-plane refined) ----
  const acc = new Float64Array(nP);
  let within = 0;
  for (let i = 0; i < nP; i++) {
    const x = P[3 * i],
      y = P[3 * i + 1],
      z = P[3 * i + 2];
    let dBest = Infinity; // refined distance so far
    // base set first, then the dense focus set. A sample farther than dBest + h (Euclidean)
    // cannot refine below dBest (the lateral slack is at most h), so that bounds the search.
    for (let k = 0; k < sets.length; k++) {
      const S = sets[k];
      const nn = S.tree.nearest(x, y, z, Math.min(S.maxD, dBest + S.h));
      if (nn.index < 0) continue;
      let d = nn.dist;
      const j = nn.index;
      if (S.N) {
        const nx = S.N[3 * j],
          ny = S.N[3 * j + 1],
          nz = S.N[3 * j + 2];
        if (nx || ny || nz) {
          const ex = x - S.G[3 * j],
            ey = y - S.G[3 * j + 1],
            ez = z - S.G[3 * j + 2];
          const dn = Math.abs(ex * nx + ey * ny + ez * nz);
          const dl = Math.sqrt(Math.max(0, d * d - dn * dn));
          const lat = Math.max(0, dl - S.h);
          d = Math.sqrt(dn * dn + lat * lat);
        }
      }
      if (d < dBest) dBest = d;
    }
    acc[i] = dBest;
    if (dBest <= tau) within++;
  }
  const precision = nP ? within / nP : 0;
  // ---- completeness: GT samples (visible, in ROI) with a result point within tau ----
  const compSet = dataset.gtFocus ?? dataset.gt;
  const compRoi = dataset.gtFocus ? false : useRoi;
  const rGrid = nAll > 0 ? new HashGrid(step > 1 ? all.subarray(0, 3 * nAll) : P, tau) : null;
  let nEval = 0,
    covered = 0;
  const nC = compSet.points.length / 3;
  for (let i = 0; i < nC; i++) {
    if (compSet.visibility && compSet.visibility[i] < minVis) continue;
    const x = compSet.points[3 * i],
      y = compSet.points[3 * i + 1],
      z = compSet.points[3 * i + 2];
    if (compRoi && !inBox(meta.roi, x, y, z)) continue;
    nEval++;
    if (rGrid && rGrid.hasWithin(x, y, z, tau)) covered++;
  }
  const completeness = nEval ? covered / nEval : 0;
  const fscore = precision + completeness > 0 ? (2 * precision * completeness) / (precision + completeness) : 0;
  return {
    tau,
    align: T ? mode : 'none',
    icp,
    roi: !!useRoi,
    numPointsRaw: nRaw,
    numPointsEvaluated: nP,
    accuracyMedian: percentile(acc, 50),
    accuracyP90: percentile(acc, 90),
    accuracyMeanCapped: mean(Array.from(acc, (d) => Math.min(d, maxDist))),
    precision,
    completeness,
    completenessSet: dataset.gtFocus ? 'focus' : 'base',
    numGtEvaluated: nEval,
    fscore,
  };
}

const fmt = (v, d = 4) => (v === null || v === undefined ? '—' : typeof v === 'number' ? (Number.isFinite(v) ? (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : v.toFixed(d)) : String(v)) : String(v));

/** Human-readable table of the metrics. */
export function formatTable(m) {
  const rows = [];
  const add = (k, v, unit = '') => rows.push([k, fmt(v) + (unit && v !== null && v !== undefined ? ` ${unit}` : '')]);
  add('dataset', m.dataset);
  add('registered / frames', `${m.registration.numRegistered} / ${m.registration.numFrames}`);
  add('registration rate', m.registration.rate);
  if (m.registration.numInput) add('registration rate (of input)', m.registration.rateOfInput);
  if (m.trajectory) {
    add('ATE RMSE (Sim3)', m.trajectory.ateRmse, 'm');
    add('ATE / scene extent', m.trajectory.ateRelative);
    add('ATE median / max', `${fmt(m.trajectory.ateMedian)} / ${fmt(m.trajectory.ateMax)}`, 'm');
    add('rotation error mean / max', `${fmt(m.trajectory.rotErrDegMean, 3)} / ${fmt(m.trajectory.rotErrDegMax, 3)}`, 'deg');
    add('relative rotation error mean', m.trajectory.rpeRotDegMean, 'deg');
    add('Sim3 scale', m.trajectory.sim3.s);
  } else add('trajectory', 'n/a (< 3 registered or degenerate)');
  if (m.metric) {
    add('scale error (1/s − 1, + = too large)', m.metric.scaleError);
    add('ATE RMSE (SE3)', m.metric.ateSe3Rmse, 'm');
  }
  if (m.frame) {
    add(`gravity error${m.frame.gravityAligned ? '' : ' (not claimed)'}`, m.frame.gravityErrorDeg, 'deg');
    add(`heading error${m.frame.northAligned ? '' : ' (not claimed)'}`, m.frame.headingErrorDeg, 'deg');
  }
  if (m.intrinsics) {
    add('focal error', m.intrinsics.focalErrorPct, '%');
    const pf = m.intrinsics.perFrameFocal;
    if (pf) {
      add('per-frame GT focal min / mean / max', `${fmt(pf.min, 2)} / ${fmt(pf.mean, 2)} / ${fmt(pf.max, 2)}`, 'px');
      add('focal error vs mean GT focal of registered', pf.focalErrorPctVsRegisteredMean, '%');
    }
    add('principal point error', m.intrinsics.ppErrorPx, 'px');
    add('corner distortion error', m.intrinsics.cornerDistortionErrorPx, 'px');
  }
  if (m.dense) {
    add('dense tau', m.dense.tau, 'm');
    add('dense points evaluated', m.dense.numPointsEvaluated);
    add('accuracy median / p90', `${fmt(m.dense.accuracyMedian)} / ${fmt(m.dense.accuracyP90)}`, 'm');
    add('precision (< tau)', m.dense.precision);
    add(`completeness (${m.dense.completenessSet})`, m.dense.completeness);
    add('F-score', m.dense.fscore);
  }
  const w = Math.max(...rows.map((r) => r[0].length));
  return rows.map(([k, v]) => `${k.padEnd(w)}  ${v}`).join('\n');
}
