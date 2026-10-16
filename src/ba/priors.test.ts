import { describe, expect, it } from 'vitest';
import { mat3Mul } from '../math/mat3';
import { Rng } from '../math/random';
import { mat3ToRotvec, rotvecToMat3 } from '../math/rotation';
import { centerPriorResidualJacobian, relRotationPriorResidualJacobian, so3LeftJacobianInverse } from './priors';

function randVec(rng: Rng, s: number): number[] {
  return [s * rng.uniform(-1, 1), s * rng.uniform(-1, 1), s * rng.uniform(-1, 1)];
}

describe('so3LeftJacobianInverse', () => {
  it('log(exp(δ) exp(φ)) ≈ φ + J_l⁻¹(φ) δ for small δ', () => {
    const rng = new Rng(3);
    for (const mag of [1e-9, 1e-5, 1e-3, 0.3, 1.5, 2.8]) {
      for (let n = 0; n < 20; n++) {
        const dir = randVec(rng, 1);
        const nn = Math.hypot(...dir);
        const phi = dir.map((v) => (v / nn) * mag);
        const Jinv = so3LeftJacobianInverse(phi);
        const h = 1e-6;
        for (let k = 0; k < 3; k++) {
          const dp = [0, 0, 0];
          dp[k] = h;
          const dm = [0, 0, 0];
          dm[k] = -h;
          const lp = mat3ToRotvec(mat3Mul(rotvecToMat3(dp), rotvecToMat3(phi)));
          const lm = mat3ToRotvec(mat3Mul(rotvecToMat3(dm), rotvecToMat3(phi)));
          for (let r = 0; r < 3; r++) {
            const num = (lp[r] - lm[r]) / (2 * h);
            expect(Math.abs(num - Jinv[3 * r + k])).toBeLessThan(2e-7);
          }
        }
      }
    }
  });

  it('is the identity at φ = 0', () => {
    const J = so3LeftJacobianInverse([0, 0, 0]);
    expect(Array.from(J)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });
});

describe('centre prior', () => {
  it('residual is (C − c)/σ with C = −Rᵀt and the Jacobian matches central differences', () => {
    const rng = new Rng(17);
    const r = new Float64Array(3),
      J = new Float64Array(18);
    const rp = new Float64Array(3),
      rm = new Float64Array(3),
      Jd = new Float64Array(18);
    for (let n = 0; n < 50; n++) {
      const R = rotvecToMat3(randVec(rng, 3));
      const t = Float64Array.from(randVec(rng, 20));
      const c = Float64Array.from(randVec(rng, 20));
      const sigma = 0.1 + rng.next();
      centerPriorResidualJacobian(R, 0, t, 0, c, 1 / sigma, r, J);
      for (let k = 0; k < 3; k++) {
        const Ck = -(R[k] * t[0] + R[3 + k] * t[1] + R[6 + k] * t[2]);
        expect(r[k]).toBeCloseTo((Ck - c[k]) / sigma, 10);
      }
      for (let k = 0; k < 6; k++) {
        const h = 1e-6 * (k < 3 ? 1 : 1 + Math.abs(t[k - 3]));
        for (const [sg, out] of [
          [1, rp],
          [-1, rm],
        ] as const) {
          let R2 = R,
            t2 = t;
          if (k < 3) {
            const w = [0, 0, 0];
            w[k] = sg * h;
            R2 = mat3Mul(rotvecToMat3(w), R);
          } else {
            t2 = t.slice();
            t2[k - 3] += sg * h;
          }
          centerPriorResidualJacobian(R2, 0, t2, 0, c, 1 / sigma, out, Jd);
        }
        for (let row = 0; row < 3; row++) {
          const num = (rp[row] - rm[row]) / (2 * h);
          expect(Math.abs(num - J[6 * row + k])).toBeLessThan(1e-6 * Math.max(1, Math.abs(num)));
        }
      }
    }
  });
});

describe('relative-rotation prior', () => {
  it('residual = log(R_j R_iᵀ R_jiᵀ)/σ and exact Jacobians match central differences (incl. large residuals)', () => {
    const rng = new Rng(23);
    const r = new Float64Array(3),
      Ji = new Float64Array(9),
      Jj = new Float64Array(9);
    const rp = new Float64Array(3),
      rm = new Float64Array(3),
      d1 = new Float64Array(9),
      d2 = new Float64Array(9);
    for (const resMag of [0, 1e-4, 0.05, 0.8, 2.5]) {
      for (let n = 0; n < 20; n++) {
        const Ri = rotvecToMat3(randVec(rng, 3));
        const Rji = rotvecToMat3(randVec(rng, 1));
        // R_j = exp(e) R_ji R_i with |e| = resMag
        const e = randVec(rng, 1);
        const en = Math.hypot(...e) || 1;
        const E = rotvecToMat3(e.map((v) => (v / en) * resMag));
        const Rj = mat3Mul(E, mat3Mul(Rji, Ri));
        const sigma = 0.01 + rng.next() * 0.1;
        relRotationPriorResidualJacobian(Ri, 0, Rj, 0, Rji, 1 / sigma, r, Ji, Jj);
        const phi = mat3ToRotvec(E);
        for (let k = 0; k < 3; k++) expect(r[k]).toBeCloseTo(phi[k] / sigma, 7);
        const h = 1e-7;
        for (const which of ['i', 'j'] as const) {
          for (let k = 0; k < 3; k++) {
            for (const [sg, out] of [
              [1, rp],
              [-1, rm],
            ] as const) {
              const w = [0, 0, 0];
              w[k] = sg * h;
              const dR = rotvecToMat3(w);
              const Ri2 = which === 'i' ? mat3Mul(dR, Ri) : Ri;
              const Rj2 = which === 'j' ? mat3Mul(dR, Rj) : Rj;
              relRotationPriorResidualJacobian(Ri2, 0, Rj2, 0, Rji, 1 / sigma, out, d1, d2);
            }
            const Ja = which === 'i' ? Ji : Jj;
            for (let row = 0; row < 3; row++) {
              const num = (rp[row] - rm[row]) / (2 * h);
              expect(Math.abs(num - Ja[3 * row + k])).toBeLessThan(2e-5 * Math.max(1, Math.abs(num)));
            }
          }
        }
      }
    }
  });
});
