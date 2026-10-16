import { describe, expect, it } from 'vitest';
import { mat3Mul } from '../math/mat3';
import { Rng } from '../math/random';
import { rotvecToMat3 } from '../math/rotation';
import { radialResidual, radialResidualJacobian, RJ_INTR, RJ_POINT, RJ_POSE, RJ_SIZE } from './camera-model';

function randomRotation(rng: Rng, scale = Math.PI): Float64Array {
  return rotvecToMat3([scale * rng.uniform(-1, 1), scale * rng.uniform(-1, 1), scale * rng.uniform(-1, 1)]);
}

/** Independent re-implementation of ARCHITECTURE §2 RADIAL projection. */
function projectRef(R: Float64Array, t: Float64Array, X: ArrayLike<number>, K: ArrayLike<number>): [number, number] {
  const Xc = [0, 1, 2].map((r) => R[3 * r] * X[0] + R[3 * r + 1] * X[1] + R[3 * r + 2] * X[2] + t[r]);
  const x = Xc[0] / Xc[2],
    y = Xc[1] / Xc[2];
  const r2 = x * x + y * y;
  const d = 1 + K[3] * r2 + K[4] * r2 * r2;
  return [K[0] * d * x + K[1], K[0] * d * y + K[2]];
}

interface Case {
  R: Float64Array;
  t: Float64Array;
  X: Float64Array;
  K: Float64Array;
  uv: [number, number];
}

function randomCase(rng: Rng, k1: number, k2: number): Case {
  const R = randomRotation(rng);
  // camera-frame point inside a ~70° FOV, depth 0.5..50, then to world
  const z = 0.5 + 50 * rng.next();
  const Xc = [z * rng.uniform(-0.65, 0.65), z * rng.uniform(-0.5, 0.5), z];
  const t = Float64Array.of(rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-3, 3));
  const X = new Float64Array(3);
  for (let c = 0; c < 3; c++) X[c] = R[c] * (Xc[0] - t[0]) + R[3 + c] * (Xc[1] - t[1]) + R[6 + c] * (Xc[2] - t[2]);
  const K = Float64Array.of(1300 + 300 * rng.next(), 959.5 + 10 * rng.gaussian(), 719.5 + 10 * rng.gaussian(), k1, k2);
  return { R, t, X, K, uv: [rng.uniform(0, 1919), rng.uniform(0, 1439)] };
}

function residualOf(c: Case, out: Float64Array): Float64Array {
  radialResidual(c.R, 0, c.t, 0, c.X, 0, c.K, 0, c.uv[0], c.uv[1], out);
  return out;
}

describe('RADIAL projection', () => {
  it('matches the §2 formula', () => {
    const rng = new Rng(11);
    const out = new Float64Array(2);
    for (let n = 0; n < 200; n++) {
      const c = randomCase(rng, rng.uniform(-0.3, 0.3), rng.uniform(-0.1, 0.1));
      const [u, v] = projectRef(c.R, c.t, c.X, c.K);
      residualOf(c, out);
      expect(out[0]).toBeCloseTo(u - c.uv[0], 8);
      expect(out[1]).toBeCloseTo(v - c.uv[1], 8);
      const J = new Float64Array(RJ_SIZE);
      expect(radialResidualJacobian(c.R, 0, c.t, 0, c.X, 0, c.K, 0, c.uv[0], c.uv[1], J)).toBe(true);
      expect(J[0]).toBe(out[0]);
      expect(J[1]).toBe(out[1]);
    }
  });

  it('principal point is pixel (cx, cy) for the optical axis; offsets are honoured', () => {
    const R = rotvecToMat3([0, 0, 0]);
    const t = Float64Array.of(0, 0, 0);
    const K = Float64Array.of(1000, 959.5, 719.5, 0.1, 0.01);
    const out = new Float64Array(2);
    const big = new Float64Array(10);
    big.set([1, 2, 3], 4);
    const Rb = new Float64Array(20);
    Rb.set(R, 7);
    const tb = new Float64Array(8);
    tb.set(t, 2);
    const Kb = new Float64Array(12);
    Kb.set(K, 5);
    radialResidual(Rb, 7, tb, 2, big, 4, Kb, 5, 0, 0, out);
    const [u, v] = projectRef(R, t, [1, 2, 3], K);
    expect(out[0]).toBeCloseTo(u, 10);
    expect(out[1]).toBeCloseTo(v, 10);
    radialResidual(R, 0, t, 0, [0, 0, 5], 0, K, 0, 0, 0, out);
    expect(out[0]).toBe(959.5);
    expect(out[1]).toBe(719.5);
  });

  it('residual stays finite at the camera centre and reports non-finite input', () => {
    const R = rotvecToMat3([0, 0, 0]);
    const t = Float64Array.of(0, 0, 0);
    const K = Float64Array.of(1000, 0, 0, 0, 0);
    const out = new Float64Array(2);
    expect(radialResidual(R, 0, t, 0, [0, 0, 0], 0, K, 0, 1, 1, out)).toBe(true);
    expect(Number.isFinite(out[0])).toBe(true);
    expect(radialResidual(R, 0, t, 0, [NaN, 0, 1], 0, K, 0, 1, 1, out)).toBe(false);
    const J = new Float64Array(RJ_SIZE);
    expect(radialResidualJacobian(R, 0, t, 0, [Infinity, 0, 1], 0, K, 0, 1, 1, J)).toBe(false);
  });
});

describe('RADIAL analytic Jacobians vs central differences', () => {
  const labels = ['dw0', 'dw1', 'dw2', 'dt0', 'dt1', 'dt2', 'X', 'Y', 'Z', 'f', 'cx', 'cy', 'k1', 'k2'];

  /** Numeric Jacobian (2 x 14) w.r.t. [δω, δt, X, f, cx, cy, k1, k2]. */
  function numericJacobian(c: Case): Float64Array {
    const Jn = new Float64Array(28);
    const rp = new Float64Array(2),
      rm = new Float64Array(2);
    const dR = new Float64Array(9);
    for (let k = 0; k < 14; k++) {
      // step relative to the parameter's natural scale
      const h =
        k < 3 ? 1e-6 : k < 6 ? 1e-6 * (1 + Math.abs(c.t[k - 3])) : k < 9 ? 1e-6 * (1 + Math.abs(c.X[k - 6])) : k === 9 ? 1e-3 : k < 12 ? 1e-3 : 1e-7;
      for (const [sgn, out] of [
        [1, rp],
        [-1, rm],
      ] as const) {
        const cc: Case = { R: c.R.slice(), t: c.t.slice(), X: c.X.slice(), K: c.K.slice(), uv: c.uv };
        if (k < 3) {
          const w = [0, 0, 0];
          w[k] = sgn * h;
          rotvecToMat3(w, dR);
          cc.R = mat3Mul(dR, c.R);
        } else if (k < 6) cc.t[k - 3] += sgn * h;
        else if (k < 9) cc.X[k - 6] += sgn * h;
        else cc.K[k - 9] += sgn * h;
        residualOf(cc, out);
      }
      Jn[k] = (rp[0] - rm[0]) / (2 * h);
      Jn[14 + k] = (rp[1] - rm[1]) / (2 * h);
    }
    return Jn;
  }

  function analytic(c: Case): Float64Array {
    const J = new Float64Array(RJ_SIZE);
    radialResidualJacobian(c.R, 0, c.t, 0, c.X, 0, c.K, 0, c.uv[0], c.uv[1], J);
    const Ja = new Float64Array(28);
    for (let row = 0; row < 2; row++) {
      for (let k = 0; k < 6; k++) Ja[14 * row + k] = J[RJ_POSE + 6 * row + k];
      for (let k = 0; k < 3; k++) Ja[14 * row + 6 + k] = J[RJ_POINT + 3 * row + k];
      for (let k = 0; k < 5; k++) Ja[14 * row + 9 + k] = J[RJ_INTR + 5 * row + k];
    }
    return Ja;
  }

  for (const [name, k1, k2] of [
    ['pinhole', 0, 0],
    ['barrel', -0.12, 0.03],
    ['pincushion', 0.2, -0.05],
  ] as const) {
    it(`all parameter blocks (${name})`, () => {
      const rng = new Rng(name.length * 101);
      let worst = 0;
      for (let n = 0; n < 100; n++) {
        const c = randomCase(rng, k1, k2);
        const Ja = analytic(c);
        const Jn = numericJacobian(c);
        for (let q = 0; q < 28; q++) {
          const scale = Math.max(1, Math.abs(Ja[q]));
          const err = Math.abs(Ja[q] - Jn[q]) / scale;
          if (err > worst) worst = err;
          if (err > 2e-6) throw new Error(`Jacobian ${labels[q % 14]} row ${q < 14 ? 'u' : 'v'}: ${Ja[q]} vs ${Jn[q]}`);
        }
      }
      expect(worst).toBeLessThan(2e-6);
    });
  }

  it('also correct for points behind the camera (mirrored projection)', () => {
    const rng = new Rng(5);
    for (let n = 0; n < 20; n++) {
      const c = randomCase(rng, -0.05, 0.01);
      // move the point behind: X ← C + (C − X)
      const C = [0, 1, 2].map((k) => -(c.R[k] * c.t[0] + c.R[3 + k] * c.t[1] + c.R[6 + k] * c.t[2]));
      for (let k = 0; k < 3; k++) c.X[k] = 2 * C[k] - c.X[k];
      const Ja = analytic(c);
      const Jn = numericJacobian(c);
      for (let q = 0; q < 28; q++) expect(Math.abs(Ja[q] - Jn[q]) / Math.max(1, Math.abs(Ja[q]))).toBeLessThan(2e-6);
    }
  });
});
