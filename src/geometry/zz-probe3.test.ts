import { it } from 'vitest';
import { Rng } from '../math/random';
import { createIntrinsics, distortNormalized, undistortNormalized } from './camera';
import { estimateTwoViewGeometry } from './twoview';
import { makeTwoViewScene, toPixels, rotErrDeg, dirErrDeg, randomRotation, randomUnitVector } from './testutil';
import { p3p, epnp } from './pnp';
import type { Pose } from '../core/types';

const CAM = createIntrinsics(1920, 1440, (70 * Math.PI) / 180, -0.08, 0.015);

function problem(rng: Rng, o: Parameters<typeof makeTwoViewScene>[1], sigma: number, outl: number) {
  const sc = makeTwoViewScene(rng, o);
  const n = sc.n;
  const nOut = Math.round((n * outl) / (1 - outl));
  const N = n + nOut;
  const uv1 = toPixels(rng, CAM, sc.x1, n, sigma), uv2 = toPixels(rng, CAM, sc.x2, n, sigma);
  const kp1 = new Float32Array(4 * N), kp2 = new Float32Array(4 * N);
  const m = new Uint32Array(2 * N);
  for (let i = 0; i < N; i++) {
    if (i < n) {
      kp1[4 * i] = uv1[2 * i]; kp1[4 * i + 1] = uv1[2 * i + 1];
      kp2[4 * i] = uv2[2 * i]; kp2[4 * i + 1] = uv2[2 * i + 1];
    } else {
      kp1[4 * i] = rng.uniform(0, 1920); kp1[4 * i + 1] = rng.uniform(0, 1440);
      kp2[4 * i] = rng.uniform(0, 1920); kp2[4 * i + 1] = rng.uniform(0, 1440);
    }
    m[2 * i] = m[2 * i + 1] = i;
  }
  return { sc, kp1, kp2, m };
}

it('probe twoview edge scenes', () => {
  const cases: [string, Parameters<typeof makeTwoViewScene>[1], number, number][] = [
    ['pure rot sigma0', { n: 400, depthMin: 1, depthMax: 50, baseline: 0, maxAngle: 0.4 }, 0, 0.3],
    ['pure rot sigma0.05', { n: 400, depthMin: 1, depthMax: 50, baseline: 0, maxAngle: 0.4 }, 0.05, 0.3],
    ['far 2000 baselines sigma0.02', { n: 400, depthMin: 150, depthMax: 300, baseline: 0.1, maxAngle: 0.3 }, 0.02, 0.3],
    ['far 2000 baselines sigma0.3', { n: 400, depthMin: 150, depthMax: 300, baseline: 0.1, maxAngle: 0.3 }, 0.3, 0.3],
    ['forward motion', { n: 400, depthMin: 2, depthMax: 20, baseline: 1, maxAngle: 0.05, centerDir: [0, 0, 1] }, 0.5, 0.3],
    ['backward motion', { n: 400, depthMin: 2, depthMax: 20, baseline: 1, maxAngle: 0.05, centerDir: [0, 0, -1] }, 0.5, 0.3],
    ['forward small', { n: 400, depthMin: 2, depthMax: 20, baseline: 0.2, maxAngle: 0.05, centerDir: [0, 0, 1] }, 0.5, 0.3],
    ['sideways rot 1.2rad', { n: 300, depthMin: 1, depthMax: 3, baseline: 1, maxAngle: 1.2 }, 0.5, 0.3],
    ['planar close', { n: 400, depthMin: 0.3, depthMax: 0.6, baseline: 0.1, maxAngle: 0.3, planar: true }, 0.5, 0.3],
    ['planar far small baseline', { n: 400, depthMin: 30, depthMax: 60, baseline: 0.5, maxAngle: 0.2, planar: true }, 0.5, 0.3],
  ];
  for (const [name, o, sigma, outl] of cases) {
    const res: string[] = [];
    for (let seed = 0; seed < 6; seed++) {
      const rng = new Rng(77 + seed * 13);
      const p = problem(rng, o, sigma, outl);
      const g = estimateTwoViewGeometry(p.kp1, p.kp2, p.m, CAM, CAM, { rng: new Rng(seed) });
      const tn = Math.hypot(p.sc.t[0], p.sc.t[1], p.sc.t[2]);
      const re = g.R ? rotErrDeg(g.R, p.sc.R).toFixed(3) : '-';
      const te = g.t && tn > 0 && Math.hypot(g.t[0], g.t[1], g.t[2]) > 0 ? dirErrDeg(g.t, p.sc.t).toFixed(2) : '-';
      const alt = g.planarAlternative ? `/alt ${rotErrDeg(g.planarAlternative.R, p.sc.R).toFixed(3)},${dirErrDeg(g.planarAlternative.t, p.sc.t).toFixed(2)}` : '';
      res.push(`${g.config}(${re},${te}${alt}) E${g.numInliersE} F${g.numInliersF} H${g.numInliersH}`);
    }
    console.log(`${name}:\n  ${res.join('\n  ')}`);
  }
});

it('probe p3p stress', () => {
  const rng = new Rng(3);
  const kinds = ['far', 'tiny', 'offset', 'nearcollinear', 'wideangle', 'danger'];
  for (const kind of kinds) {
    let ok = 0, tot = 0, maxE = 0;
    for (let k = 0; k < 3000; k++) {
      const R = randomRotation(rng, Math.PI);
      const t = new Float64Array(3);
      // camera-frame points
      const P: number[][] = [];
      for (let i = 0; i < 3; i++) {
        let p: number[];
        if (kind === 'far') { const z = 500 + 500 * rng.next(); p = [rng.uniform(-0.5, 0.5) * 2, rng.uniform(-0.5, 0.5) * 2, z]; }
        else if (kind === 'tiny') { const z = 1e-3 * (1 + rng.next()); p = [rng.uniform(-0.6, 0.6) * z, rng.uniform(-0.45, 0.45) * z, z]; }
        else if (kind === 'wideangle') { const z = 1 + rng.next(); p = [rng.uniform(-3, 3) * z, rng.uniform(-3, 3) * z, z]; }
        else { const z = 1 + 9 * rng.next(); p = [rng.uniform(-0.6, 0.6) * z, rng.uniform(-0.45, 0.45) * z, z]; }
        P.push(p);
      }
      if (kind === 'nearcollinear') { // third point almost on segment 1-2
        const s = rng.next();
        for (let c = 0; c < 3; c++) P[2][c] = P[0][c] + s * (P[1][c] - P[0][c]) + 1e-3 * rng.gaussian();
      }
      if (kind === 'danger') { // camera nearly on the circumscribed cylinder: put the 3 points on a vertical plane through camera?
        // points on a plane that passes close to the camera centre
        const nrm = randomUnitVector(rng);
        for (let i = 0; i < 3; i++) { const d = P[i][0] * nrm[0] + P[i][1] * nrm[1] + P[i][2] * nrm[2] - 1e-3; for (let c = 0; c < 3; c++) P[i][c] -= d * nrm[c]; if (P[i][2] < 0.1) P[i][2] = 0.1 + rng.next(); }
      }
      const off = kind === 'offset' ? 1e5 : 0;
      // world X = R^T (P - t) + off ; t such that world offsets
      const X = new Float64Array(9), B = new Float64Array(9);
      const Cw = [off, -off, 0.5 * off];
      for (let i = 0; i < 3; i++) {
        for (let c = 0; c < 3; c++) X[3 * i + c] = R[c] * P[i][0] + R[3 + c] * P[i][1] + R[6 + c] * P[i][2] + Cw[c];
        const nn = Math.hypot(P[i][0], P[i][1], P[i][2]);
        for (let c = 0; c < 3; c++) B[3 * i + c] = P[i][c] / nn;
      }
      // t = -R Cw
      for (let r = 0; r < 3; r++) t[r] = -(R[3 * r] * Cw[0] + R[3 * r + 1] * Cw[1] + R[3 * r + 2] * Cw[2]);
      const sols = p3p(X, B);
      tot++;
      let best = Infinity;
      for (const s of sols) best = Math.min(best, rotErrDeg(s.R, R));
      if (best < 1e-3) ok++;
      else maxE = Math.max(maxE, best);
    }
    console.log(`p3p ${kind}: ${ok}/${tot}`);
  }
});

it('probe epnp', () => {
  const rng = new Rng(9);
  for (const planar of [false, true]) for (const n of [4, 5, 6, 8, 20]) for (const off of [0, 1e5]) {
    let ok = 0, nul = 0;
    for (let k = 0; k < 300; k++) {
      const R = randomRotation(rng, Math.PI);
      const C = [off + rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5)];
      const t = Float64Array.of(-(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]), -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]), -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2]));
      const pose: Pose = { R, t };
      const X = new Float64Array(3 * n), xn = new Float64Array(2 * n);
      const tilt = randomRotation(rng, 1.4);
      for (let i = 0; i < n; i++) {
        let Pc: number[];
        if (planar) { const a = rng.uniform(-1, 1), b = rng.uniform(-1, 1); Pc = [tilt[0] * a + tilt[1] * b, tilt[3] * a + tilt[4] * b, 4 + tilt[6] * a + tilt[7] * b]; }
        else { const z = rng.uniform(2, 6); Pc = [rng.uniform(-0.6, 0.6) * z, rng.uniform(-0.45, 0.45) * z, z]; }
        xn[2 * i] = Pc[0] / Pc[2]; xn[2 * i + 1] = Pc[1] / Pc[2];
        const d = [Pc[0] - t[0], Pc[1] - t[1], Pc[2] - t[2]];
        for (let c = 0; c < 3; c++) X[3 * i + c] = R[c] * d[0] + R[3 + c] * d[1] + R[6 + c] * d[2];
      }
      const est = epnp(X, xn, n);
      if (!est) { nul++; continue; }
      if (rotErrDeg(est.R, pose.R) < 1e-4) ok++;
    }
    console.log(`epnp planar=${planar} n=${n} off=${off}: ok ${ok}/300 null ${nul}`);
  }
});

it('probe undistort', () => {
  const ks: [number, number][] = [[-0.45, 0], [-0.3, 0.05], [0.5, 0.5], [0.4, -0.1], [0, -0.3], [-0.49, 0.49], [0.3, -0.49], [-0.2, -0.2]];
  const out = new Float64Array(2), d = new Float64Array(2);
  for (const [k1, k2] of ks) {
    const cam = { ...CAM, k1, k2 };
    let maxErr = 0, maxR = 0;
    // valid radius
    const rng = new Rng(1);
    for (let i = 0; i < 20000; i++) {
      const r = 1.5 * Math.pow(rng.next(), 0.5);
      const a = 2 * Math.PI * rng.next();
      const x = r * Math.cos(a), y = r * Math.sin(a);
      // check monotone branch
      const s = r * r; const gp = 1 + 3 * k1 * s + 5 * k2 * s * s;
      let mono = true; for (let q = 0; q <= 50; q++) { const rr = (r * q) / 50, ss = rr * rr; if (1 + 3 * k1 * ss + 5 * k2 * ss * ss <= 0) mono = false; }
      if (!mono || gp <= 0) continue;
      distortNormalized(cam, x, y, d);
      undistortNormalized(cam, d[0], d[1], out);
      const e = Math.hypot(out[0] - x, out[1] - y) / Math.max(1e-300, r);
      if (e > maxErr) { maxErr = e; maxR = r; }
    }
    console.log(`undistort k1=${k1} k2=${k2}: max rel err ${maxErr.toExponential(2)} at r=${maxR.toFixed(3)}`);
  }
});
