/**
 * Tests of the evaluation library e2e/lib/*.mjs (plain JS, typed via .d.mts).
 */
import { describe, expect, it } from 'vitest';
import { HashGrid } from '../../../e2e/lib/grid.mjs';
import { axisAngle, mat3Mul, mat3T, rotationAngleBetween, svd3, umeyama, applySim3 } from '../../../e2e/lib/linalg.mjs';
import { computeMetrics, formatTable, type LoadedDataset } from '../../../e2e/lib/metrics-core.mjs';
import { parsePoints } from '../../../e2e/lib/points.mjs';
import { SynthRng } from './rng';

function randRot(rng: SynthRng): number[] {
  const a = [rng.gaussian(), rng.gaussian(), rng.gaussian()];
  const n = Math.hypot(a[0], a[1], a[2]);
  return axisAngle([a[0] / n, a[1] / n, a[2] / n], rng.uniform(-Math.PI, Math.PI));
}

describe('linear algebra', () => {
  it('svd3 reconstructs A (incl. rank-deficient)', () => {
    const rng = new SynthRng(4);
    for (let k = 0; k < 200; k++) {
      const A = Array.from({ length: 9 }, () => rng.gaussian());
      if (k % 5 === 0) for (let c = 0; c < 3; c++) A[6 + c] = 2 * A[c] - A[3 + c]; // rank 2
      if (k % 7 === 0) for (let r = 0; r < 3; r++) A[3 * r + 2] = A[3 * r + 1] = A[3 * r] * 0.5; // rank 1
      const { U, S, V } = svd3(A);
      const B = mat3Mul(mat3Mul(U, [S[0], 0, 0, 0, S[1], 0, 0, 0, S[2]]), mat3T(V));
      for (let i = 0; i < 9; i++) expect(B[i]).toBeCloseTo(A[i], 9);
      expect(S[0]).toBeGreaterThanOrEqual(S[1]);
      expect(S[1]).toBeGreaterThanOrEqual(S[2]);
      const UtU = mat3Mul(mat3T(U), U);
      for (let i = 0; i < 9; i++) expect(UtU[i]).toBeCloseTo(i % 4 === 0 ? 1 : 0, 9);
    }
  });

  it('Umeyama recovers random similarity transforms exactly (and rigid with scale 1)', () => {
    const rng = new SynthRng(9);
    for (let k = 0; k < 50; k++) {
      const R = randRot(rng);
      const s = Math.exp(rng.uniform(-3, 3));
      const t = [rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(-50, 50)];
      const n = 3 + rng.int(40);
      const src = Array.from({ length: n }, () => [rng.gaussian(), rng.gaussian(), rng.gaussian() * (k % 3 === 0 ? 0 : 1)]); // some planar sets
      const dst = src.map((p) => applySim3({ s, R, t }, p));
      const est = umeyama(src, dst, true)!;
      expect(est).not.toBeNull();
      expect(est.s / s).toBeCloseTo(1, 9);
      expect(rotationAngleBetween(est.R, R)).toBeLessThan(1e-9);
      for (let i = 0; i < 3; i++) expect(est.t[i]).toBeCloseTo(t[i], 6);
      const rigid = umeyama(src, src.map((p) => applySim3({ s: 1, R, t }, p)), false)!;
      expect(rigid.s).toBe(1);
      expect(rotationAngleBetween(rigid.R, R)).toBeLessThan(1e-9);
    }
    // degenerate: collinear points → null
    expect(umeyama([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 0, 0], [1, 0, 0], [2, 0, 0]], true)).toBeNull();
  });

  it('Umeyama with noise: least-squares optimum is unbiased', () => {
    const rng = new SynthRng(10);
    const R = randRot(rng),
      s = 2.5,
      t = [1, 2, 3];
    const src = Array.from({ length: 2000 }, () => [rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-1, 1)]);
    const dst = src.map((p) => applySim3({ s, R, t }, p).map((v) => v + 0.01 * rng.gaussian()));
    const est = umeyama(src, dst, true)!;
    expect(Math.abs(est.s - s)).toBeLessThan(1e-3);
    expect(rotationAngleBetween(est.R, R)).toBeLessThan(1e-4);
  });
});

describe('hash grid', () => {
  it('nearest / hasWithin agree with brute force', () => {
    const rng = new SynthRng(11);
    const n = 3000;
    const P = new Float32Array(3 * n);
    for (let i = 0; i < 3 * n; i++) P[i] = rng.uniform(-1, 1);
    const g = new HashGrid(P, 0.07);
    for (let q = 0; q < 300; q++) {
      const x = rng.uniform(-1.2, 1.2),
        y = rng.uniform(-1.2, 1.2),
        z = rng.uniform(-1.2, 1.2);
      let best = Infinity,
        bi = -1;
      for (let i = 0; i < n; i++) {
        const d = Math.hypot(P[3 * i] - x, P[3 * i + 1] - y, P[3 * i + 2] - z);
        if (d < best) {
          best = d;
          bi = i;
        }
      }
      const r = g.nearest(x, y, z, 0.5);
      if (best <= 0.5) {
        expect(r.index).toBe(bi);
        expect(r.dist).toBeCloseTo(best, 6);
      } else expect(r.index).toBe(-1);
      expect(g.hasWithin(x, y, z, 0.1)).toBe(best <= 0.1);
    }
  });
});

describe('point file parsing', () => {
  it('binary little-endian PLY with extra properties, ascii PLY, raw Float32', () => {
    const header = 'ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty double conf\nend_header\n';
    const h = new TextEncoder().encode(header);
    const body = new DataView(new ArrayBuffer(2 * (12 + 3 + 8)));
    const vals = [
      [1, 2, 3],
      [-4, 5.5, 6],
    ];
    vals.forEach((v, i) => {
      const o = i * 23;
      body.setFloat32(o, v[0], true);
      body.setFloat32(o + 4, v[1], true);
      body.setFloat32(o + 8, v[2], true);
      body.setFloat64(o + 15, 0.5, true);
    });
    const buf = new Uint8Array(h.length + body.byteLength);
    buf.set(h);
    buf.set(new Uint8Array(body.buffer), h.length);
    expect(Array.from(parsePoints(buf))).toEqual([1, 2, 3, -4, 5.5, 6]);
    const ascii = new TextEncoder().encode('ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n1 2 3\n4 5 6\n');
    expect(Array.from(parsePoints(ascii))).toEqual([1, 2, 3, 4, 5, 6]);
    const raw = new Uint8Array(new Float32Array([7, 8, 9]).buffer);
    expect(Array.from(parsePoints(raw))).toEqual([7, 8, 9]);
  });
});

/** Small synthetic dataset: a ring of cameras looking at the origin; GT samples on a unit sphere. */
function makeDataset(): LoadedDataset {
  const frames = [];
  for (let i = 0; i < 24; i++) {
    const a = (i / 24) * 2 * Math.PI;
    const C = [3 * Math.cos(a), 3 * Math.sin(a), 0.5 + 0.2 * Math.sin(3 * a)];
    // look at origin: forward f, right r = f × up, down = f × r
    const f = [-C[0], -C[1], -C[2]];
    const nf = Math.hypot(...f);
    for (let k = 0; k < 3; k++) f[k] /= nf;
    let r = [f[1] * 1 - f[2] * 0, f[2] * 0 - f[0] * 1, 0];
    const nr = Math.hypot(...r);
    r = r.map((v) => v / nr);
    const d = [f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0]];
    const R = [...r, ...d, ...f];
    const t = [0, 1, 2].map((k) => -(R[3 * k] * C[0] + R[3 * k + 1] * C[1] + R[3 * k + 2] * C[2]));
    frames.push({ index: i, pose: { R, t } });
  }
  const rng = new SynthRng(12);
  const n = 20000;
  const pts = new Float32Array(3 * n),
    nrm = new Int8Array(3 * n);
  for (let i = 0; i < n; i++) {
    const v = [rng.gaussian(), rng.gaussian(), rng.gaussian()];
    const l = Math.hypot(...v);
    for (let k = 0; k < 3; k++) {
      pts[3 * i + k] = v[k] / l;
      nrm[3 * i + k] = Math.round((127 * v[k]) / l);
    }
  }
  return {
    meta: {
      name: 'test',
      scene: 'object',
      width: 640,
      height: 480,
      intrinsics: { f: 500, cx: 319.5, cy: 239.5, k1: -0.03, k2: 0.005 },
      sceneExtent: 6,
      roi: { min: [-2, -2, -2], max: [2, 2, 2] },
      denseTau: 0.02,
      gt: { spacing: Math.sqrt((4 * Math.PI) / n) },
    },
    frames,
    gt: { points: pts, normals: nrm, visibility: null },
  };
}

describe('computeMetrics', () => {
  const ds = makeDataset();
  // result = GT transformed by an arbitrary similarity X_res = s R X_gt + t
  const S = { s: 0.37, R: axisAngle([0.48, 0.6, 0.64], 1.1), t: [3, -1, 2] };
  const toResult = (pose: { R: number[]; t: number[] }) => {
    // x_cam = R_gt X_gt + t_gt; X_gt = (Rsᵀ (X_res − t))/s ⇒ R_res = R_gt Rsᵀ, t_res = s (t_gt) − R_res t … scaled
    const Rres = mat3Mul(pose.R, mat3T(S.R));
    const Rt = [0, 1, 2].map((k) => Rres[3 * k] * S.t[0] + Rres[3 * k + 1] * S.t[1] + Rres[3 * k + 2] * S.t[2]);
    return { R: Rres, t: [0, 1, 2].map((k) => S.s * pose.t[k] - Rt[k]) };
  };

  it('perfect (similarity-transformed) result: zero errors, full registration', () => {
    const result = {
      intrinsics: { f: 505, cx: 319.5, cy: 239.5, k1: -0.03, k2: 0.005 },
      images: ds.frames.map((f) => ({ index: f.index, pose: toResult(f.pose) })),
      metric: { metric: false, gravityAligned: false, northAligned: false },
    };
    const gt = ds.gt!.points;
    const pts = new Float32Array(gt.length);
    for (let i = 0; i < gt.length / 3; i++) {
      const p = applySim3(S, [gt[3 * i], gt[3 * i + 1], gt[3 * i + 2]]);
      pts.set(p, 3 * i);
    }
    const m = computeMetrics(ds, result, { points: pts });
    expect(m.registration.rate).toBe(1);
    expect(m.trajectory!.ateRmse).toBeLessThan(1e-5);
    expect(m.trajectory!.sim3.s).toBeCloseTo(1 / S.s, 6);
    expect(m.trajectory!.rotErrDegMax).toBeLessThan(1e-4);
    expect(m.intrinsics!.focalErrorPct).toBeCloseTo(1, 9);
    expect(m.intrinsics!.cornerDistortionErrorPx).toBeCloseTo(0, 9);
    expect(m.intrinsics!.perFrameFocal).toBeNull(); // no per-frame focal in a clean dataset
    expect(m.dense!.accuracyMedian).toBeLessThan(1e-5);
    expect(m.dense!.completeness).toBe(1);
    expect(m.dense!.fscore).toBe(1);
    expect(formatTable(m)).toContain('ATE RMSE');
  });

  it('focus breathing datasets: focal error vs the registered frames\' mean GT focal (informational)', () => {
    const frames = ds.frames.map((f) => ({ ...f, f: 500 * (1 + 0.006 * Math.sin(f.index)) }));
    const reg = frames.filter((f) => f.index < 12);
    const regMean = reg.reduce((a, f) => a + f.f, 0) / reg.length;
    const m = computeMetrics(
      { ...ds, frames },
      { intrinsics: { f: regMean * 1.01 }, images: reg.map((f) => ({ index: f.index, pose: toResult(f.pose) })), metric: { metric: false, gravityAligned: false, northAligned: false } },
    );
    const pf = m.intrinsics!.perFrameFocal!;
    expect(pf.registeredMean).toBeCloseTo(regMean, 9);
    expect(pf.focalErrorPctVsRegisteredMean).toBeCloseTo(1, 9);
    expect(pf.max).toBeLessThanOrEqual(503 + 1e-9);
    expect(pf.min).toBeGreaterThanOrEqual(497 - 1e-9);
    expect(m.intrinsics!.focalErrorPct).toBeCloseTo((100 * (regMean * 1.01 - 500)) / 500, 9); // still vs meta f0
    expect(formatTable(m)).toContain('focal error vs mean GT focal of registered');
  });

  it('metric + gravity + heading: detects a scale error, a tilt and a yaw', () => {
    const tilt = axisAngle([1, 0, 0], (2 * Math.PI) / 180);
    const yaw = axisAngle([0, 0, 1], (-10 * Math.PI) / 180); // result north rotated 10° clockwise
    const Rerr = mat3Mul(yaw, tilt);
    // result frame: X_res = 1.05 · Rerrᵀ X_gt (so result +Z = Rerr ẑ in GT coords)
    const s = 1.05;
    const images = ds.frames.map((f) => {
      const R = mat3Mul(f.pose.R, Rerr);
      return { index: f.index, pose: { R, t: f.pose.t.map((v) => v * s) } };
    });
    const m = computeMetrics(ds, { images, metric: { metric: true, gravityAligned: true, northAligned: true } });
    expect(m.metric!.scaleError).toBeCloseTo(s - 1, 9); // result is 5 % too large
    expect(m.trajectory!.ateRmse).toBeLessThan(1e-6);
    expect(m.metric!.ateSe3Rmse!).toBeGreaterThan(0.1);
    expect(m.frame!.gravityErrorDeg).toBeCloseTo(2, 6);
    expect(m.frame!.headingErrorDeg).toBeCloseTo(10, 1);
    expect(m.frame!.gravityErrorDegSim3!).toBeCloseTo(2, 4);
  });

  it('partial registration, trajectory noise, and noisy/incomplete dense points', () => {
    const rng = new SynthRng(13);
    const images = ds.frames
      .filter((f) => f.index % 2 === 0)
      .map((f) => {
        const p = toResult(f.pose);
        return { index: f.index, pose: { R: p.R, t: p.t.map((v) => v + 0.003 * rng.gaussian()) } };
      });
    const gt = ds.gt!.points;
    const keep: number[] = [];
    for (let i = 0; i < gt.length / 3; i++) {
      if (gt[3 * i + 2] < -0.5) continue; // missing bottom cap → incomplete
      const p = [gt[3 * i], gt[3 * i + 1], gt[3 * i + 2]].map((v) => v + 0.004 * rng.gaussian());
      keep.push(...applySim3(S, p));
    }
    const m = computeMetrics(ds, { images, metric: { metric: false, gravityAligned: false, northAligned: false } }, { points: new Float32Array(keep) });
    expect(m.registration.numRegistered).toBe(12);
    expect(m.registration.rate).toBe(0.5);
    expect(m.trajectory!.ateRmse).toBeGreaterThan(0.001);
    expect(m.trajectory!.ateRmse).toBeLessThan(0.03);
    // gaussian noise σ=4 mm per axis (in GT units after alignment) → normal component median ≈ 0.674σ
    expect(m.dense!.accuracyMedian).toBeGreaterThan(0.0015);
    expect(m.dense!.accuracyMedian).toBeLessThan(0.006);
    expect(m.dense!.completeness).toBeGreaterThan(0.7);
    expect(m.dense!.completeness).toBeLessThan(0.8);
    expect(m.dense!.precision).toBeGreaterThan(0.95);
  });
});

describe('k-d tree', () => {
  it('nearest agrees with brute force at any distance', async () => {
    const { KdTree } = await import('../../../e2e/lib/kdtree.mjs');
    const rng = new SynthRng(21);
    const n = 5000;
    const P = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      // clustered on a surface + some outliers
      const a = rng.next() * 6.28,
        b = rng.next() * 3.14;
      const r = i % 10 === 0 ? rng.uniform(0, 3) : 1;
      P[3 * i] = r * Math.sin(b) * Math.cos(a);
      P[3 * i + 1] = r * Math.sin(b) * Math.sin(a);
      P[3 * i + 2] = r * Math.cos(b);
    }
    const t = new KdTree(P);
    for (let q = 0; q < 500; q++) {
      const x = rng.uniform(-4, 4),
        y = rng.uniform(-4, 4),
        z = rng.uniform(-4, 4);
      let best = Infinity,
        bi = -1;
      for (let i = 0; i < n; i++) {
        const d = Math.hypot(P[3 * i] - x, P[3 * i + 1] - y, P[3 * i + 2] - z);
        if (d < best) {
          best = d;
          bi = i;
        }
      }
      const r = t.nearest(x, y, z);
      expect(r.index).toBe(bi);
      expect(r.dist).toBeCloseTo(best, 5);
      const lim = t.nearest(x, y, z, best * 0.99);
      expect(lim.index).toBe(-1);
    }
  });
});
