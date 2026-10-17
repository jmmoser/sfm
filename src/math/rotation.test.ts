import { describe, expect, it } from 'vitest';
import { mat3Det, mat3Mul, mat3MulTranspose, mat3MulVec, mat3Transpose } from './mat3';
import { Rng } from './random';
import {
  axisAngleToMat3,
  mat3ToQuat,
  mat3ToRotvec,
  orthonormalize,
  quatConj,
  quatMul,
  quatNormalize,
  quatRotateVec,
  quatSlerp,
  quatToMat3,
  rotationAngle,
  rotationAngleBetween,
  rotationBetweenVectors,
  rotvecToMat3,
} from './rotation';

function randUnit(rng: Rng): Float64Array {
  for (;;) {
    const v = new Float64Array([rng.gaussian(), rng.gaussian(), rng.gaussian()]);
    const n = Math.hypot(v[0], v[1], v[2]);
    if (n > 1e-6) return v.map((x) => x / n);
  }
}

function randQuat(rng: Rng): Float64Array {
  const q = new Float64Array([rng.gaussian(), rng.gaussian(), rng.gaussian(), rng.gaussian()]);
  return quatNormalize(q);
}

function maxDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

/** max |RᵀR − I| and |det R − 1| */
function rotationError(R: ArrayLike<number>): number {
  const RtR = mat3Mul(mat3Transpose(R), R);
  let e = Math.abs(mat3Det(R) - 1);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) e = Math.max(e, Math.abs(RtR[3 * i + j] - (i === j ? 1 : 0)));
  return e;
}

/** Reference Rodrigues formula (direct, for moderate angles). */
function rodriguesRef(axis: Float64Array, angle: number): Float64Array {
  const [x, y, z] = axis;
  const c = Math.cos(angle),
    s = Math.sin(angle),
    C = 1 - c;
  return new Float64Array([
    c + x * x * C,
    x * y * C - z * s,
    x * z * C + y * s,
    y * x * C + z * s,
    c + y * y * C,
    y * z * C - x * s,
    z * x * C - y * s,
    z * y * C + x * s,
    c + z * z * C,
  ]);
}

const ANGLES = [
  0, 1e-300, 1e-15, 1e-12, 1e-9, 1e-7, 1e-6, 1e-5, 1e-3, 0.1, 1, 2, 2.0943951, 2.5, 3, Math.PI - 1e-3, Math.PI - 1e-6,
  Math.PI - 1e-9, Math.PI - 1e-12,
];

describe('quaternions', () => {
  it('quatToMat3 gives rotations; mat3ToQuat inverts it with w ≥ 0', () => {
    const rng = new Rng(1);
    for (let i = 0; i < 5000; i++) {
      const q = randQuat(rng);
      const R = quatToMat3(q);
      expect(rotationError(R)).toBeLessThan(1e-14);
      const q2 = mat3ToQuat(R);
      expect(q2[0]).toBeGreaterThanOrEqual(0);
      const sgn = q[0] >= 0 ? 1 : -1;
      // Near w = 0 both signs are valid; compare up to sign.
      const d1 = maxDiff(q2, q.map((v) => v * sgn));
      const d2 = maxDiff(q2, q.map((v) => -v * sgn));
      expect(Math.min(d1, d2)).toBeLessThan(1e-14);
      expect(Math.abs(Math.hypot(...q2) - 1)).toBeLessThan(1e-15);
    }
  });

  it('round trip near 180° (w ≈ 0) and identity', () => {
    const rng = new Rng(2);
    for (const angle of ANGLES) {
      for (let i = 0; i < 50; i++) {
        const axis = randUnit(rng);
        const R = axisAngleToMat3(axis, angle);
        const q = mat3ToQuat(R);
        expect(maxDiff(quatToMat3(q), R)).toBeLessThan(1e-14);
      }
    }
    const I = mat3ToQuat(new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]));
    expect(Array.from(I)).toEqual([1, 0, 0, 0]);
    // Exactly 180° about x: [0, ±1, 0, 0]
    const qx = mat3ToQuat(new Float64Array([1, 0, 0, 0, -1, 0, 0, 0, -1]));
    expect(Math.abs(qx[1])).toBeCloseTo(1, 15);
    expect(qx[0]).toBe(0);
  });

  it('convention: q = [cos(θ/2), sin(θ/2)·n] is the right-handed rotation by θ about n', () => {
    const rng = new Rng(30);
    for (let i = 0; i < 500; i++) {
      const n = randUnit(rng);
      const th = rng.uniform(-Math.PI, Math.PI);
      const q = [Math.cos(th / 2), ...n.map((v) => v * Math.sin(th / 2))];
      expect(maxDiff(quatToMat3(q), axisAngleToMat3(n, th))).toBeLessThan(1e-15 * 4);
      expect(maxDiff(quatToMat3(q), rotvecToMat3(n.map((v) => v * th)))).toBeLessThan(1e-15 * 4);
    }
    // +90° about z maps x to y.
    const qz = [Math.SQRT1_2, 0, 0, Math.SQRT1_2];
    expect(maxDiff(quatRotateVec(qz, [1, 0, 0]), [0, 1, 0])).toBeLessThan(1e-15);
  });

  it('non-unit quaternions are handled as normalised; zero → identity', () => {
    const rng = new Rng(3);
    const q = randQuat(rng);
    const R = quatToMat3(q);
    expect(maxDiff(quatToMat3(q.map((v) => v * 7.5)), R)).toBeLessThan(1e-15);
    expect(Array.from(quatToMat3([0, 0, 0, 0]))).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    expect(Array.from(quatNormalize([0, 0, 0, 0]))).toEqual([1, 0, 0, 0]);
  });

  it('quatMul composes like matrix products; conj is the inverse; rotateVec matches', () => {
    const rng = new Rng(4);
    for (let i = 0; i < 1000; i++) {
      const a = randQuat(rng),
        b = randQuat(rng);
      const ab = quatMul(a, b);
      expect(maxDiff(quatToMat3(ab), mat3Mul(quatToMat3(a), quatToMat3(b)))).toBeLessThan(1e-14);
      const id = quatMul(a, quatConj(a));
      expect(maxDiff(id, [1, 0, 0, 0])).toBeLessThan(1e-15);
      const v = [rng.gaussian(), rng.gaussian(), rng.gaussian()];
      expect(maxDiff(quatRotateVec(a, v), mat3MulVec(quatToMat3(a), v))).toBeLessThan(1e-14);
      // aliasing
      const c = a.slice();
      quatMul(c, b, c);
      expect(maxDiff(c, ab)).toBe(0);
    }
  });

  it('slerp: endpoints, constant angular speed, shortest path', () => {
    const rng = new Rng(5);
    for (let i = 0; i < 500; i++) {
      const a = randQuat(rng);
      const b = randQuat(rng);
      const Ra = quatToMat3(a),
        Rb = quatToMat3(b);
      expect(maxDiff(quatToMat3(quatSlerp(a, b, 0)), Ra)).toBeLessThan(1e-14);
      expect(maxDiff(quatToMat3(quatSlerp(a, b, 1)), Rb)).toBeLessThan(1e-14);
      const total = rotationAngleBetween(Ra, Rb);
      for (const t of [0.25, 0.5, 0.8]) {
        const Rt = quatToMat3(quatSlerp(a, b, t));
        expect(Math.abs(rotationAngleBetween(Ra, Rt) - t * total)).toBeLessThan(1e-12);
        expect(Math.abs(rotationAngleBetween(Rt, Rb) - (1 - t) * total)).toBeLessThan(1e-12);
      }
      // Same rotation with negated b: identical result (shortest path).
      const s1 = quatToMat3(quatSlerp(a, b, 0.3));
      const s2 = quatToMat3(quatSlerp(a, b.map((v) => -v), 0.3));
      expect(maxDiff(s1, s2)).toBeLessThan(1e-14);
    }
    // Nearly identical inputs.
    const a = randQuat(rng);
    const b = quatNormalize(a.map((v, k) => v + (k === 1 ? 1e-12 : 0)));
    const m = quatSlerp(a, b, 0.5);
    expect(Math.abs(Math.hypot(...m) - 1)).toBeLessThan(1e-15);
  });
});

describe('exp / log maps', () => {
  it('rotvecToMat3 matches Rodrigues and is a rotation', () => {
    const rng = new Rng(10);
    for (let i = 0; i < 2000; i++) {
      const axis = randUnit(rng);
      const angle = rng.uniform(0, Math.PI);
      const R = rotvecToMat3(axis.map((v) => v * angle));
      expect(maxDiff(R, rodriguesRef(axis, angle))).toBeLessThan(1e-15 * 4);
      expect(rotationError(R)).toBeLessThan(1e-14);
    }
  });

  it('round trip rotvec → R → rotvec for angles from 0 to π−1e-12', () => {
    const rng = new Rng(11);
    for (const angle of ANGLES) {
      for (let i = 0; i < 200; i++) {
        const axis = randUnit(rng);
        const w = axis.map((v) => v * angle);
        const R = rotvecToMat3(w);
        expect(rotationError(R)).toBeLessThan(1e-14);
        const w2 = mat3ToRotvec(R);
        // Absolute rotvec error. Near π the axis is determined only up to rounding
        // of R (≈1e-16 relative); |Δw| ≈ π·|Δaxis| stays ~1e-15..1e-8 depending on
        // how close to π (the sign of w at exactly π is ambiguous).
        const err = maxDiff(w2, w);
        const tol = angle < 3 ? 2e-15 * Math.max(1, angle) : 1e-15 / Math.max(Math.PI - angle, 1e-8) + 1e-14;
        expect(err, `angle ${angle}`).toBeLessThan(tol);
        expect(Math.abs(Math.hypot(...w2) - angle)).toBeLessThan(1e-14);
        // And the matrices agree tightly regardless of the axis conditioning.
        expect(maxDiff(rotvecToMat3(w2), R)).toBeLessThan(1e-14);
      }
    }
  });

  it('small-angle Taylor branch is accurate (relative)', () => {
    const rng = new Rng(12);
    for (const angle of [1e-300, 1e-200, 1e-20, 1e-12, 1e-8, 9.9e-7, 1.01e-6, 1e-5]) {
      const axis = randUnit(rng);
      const w = axis.map((v) => v * angle);
      const R = rotvecToMat3(w);
      // antisymmetric part = sin(angle)·axis to relative precision
      expect(Math.abs((R[7] - R[5]) / 2 - Math.sin(angle) * axis[0])).toBeLessThanOrEqual(1e-15 * angle + 1e-320);
      const w2 = mat3ToRotvec(R);
      for (let k = 0; k < 3; k++) expect(Math.abs(w2[k] - w[k])).toBeLessThanOrEqual(2e-16 * angle + 1e-320);
    }
    expect(Array.from(mat3ToRotvec(new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1])))).toEqual([0, 0, 0]);
  });

  it('exactly π about coordinate and generic axes', () => {
    const cases: [number[], number[]][] = [
      [[1, 0, 0, 0, -1, 0, 0, 0, -1], [Math.PI, 0, 0]],
      [[-1, 0, 0, 0, 1, 0, 0, 0, -1], [0, Math.PI, 0]],
      [[-1, 0, 0, 0, -1, 0, 0, 0, 1], [0, 0, Math.PI]],
    ];
    for (const [R, w] of cases) {
      const w2 = mat3ToRotvec(R);
      expect(Math.min(maxDiff(w2, w), maxDiff(w2, w.map((v) => -v)))).toBeLessThan(1e-15);
    }
    const rng = new Rng(13);
    for (let i = 0; i < 200; i++) {
      const axis = randUnit(rng);
      const R = axisAngleToMat3(axis, Math.PI);
      const w2 = mat3ToRotvec(R);
      const a = axis.map((v) => v * Math.PI);
      expect(Math.min(maxDiff(w2, a), maxDiff(w2, a.map((v) => -v)))).toBeLessThan(1e-14);
    }
  });

  it('log map tolerates slightly non-orthogonal input', () => {
    const rng = new Rng(14);
    for (let i = 0; i < 200; i++) {
      const w = randUnit(rng).map((v) => v * rng.uniform(0, 3.1));
      const R = rotvecToMat3(w);
      for (let k = 0; k < 9; k++) R[k] += 1e-10 * rng.gaussian();
      const w2 = mat3ToRotvec(R);
      expect(maxDiff(w2, w)).toBeLessThan(1e-8);
    }
  });
});

describe('angles, axis-angle, orthonormalize', () => {
  it('axisAngleToMat3 normalises the axis; zero axis → identity', () => {
    const R = axisAngleToMat3([0, 0, 5], Math.PI / 2);
    expect(maxDiff(R, [0, -1, 0, 1, 0, 0, 0, 0, 1])).toBeLessThan(1e-16);
    expect(Array.from(axisAngleToMat3([0, 0, 0], 1))).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    // right-handed: rotating x by +90° about z gives y
    expect(maxDiff(mat3MulVec(R, [1, 0, 0]), [0, 1, 0])).toBeLessThan(1e-16);
  });

  it('rotationAngle / rotationAngleBetween accurate near 0 and π', () => {
    const rng = new Rng(20);
    for (const angle of ANGLES) {
      const axis = randUnit(rng);
      const R = axisAngleToMat3(axis, angle);
      expect(Math.abs(rotationAngle(R) - angle)).toBeLessThan(angle < 1e-6 ? 1e-15 * angle + 1e-300 : 2e-15);
      const Ra = rotvecToMat3(randUnit(rng).map((v) => v * rng.uniform(0, 3)));
      const Rb = mat3Mul(R, Ra);
      expect(Math.abs(rotationAngleBetween(Rb, Ra) - angle)).toBeLessThan(angle < 1e-6 ? 3e-16 + 1e-15 * angle : 1e-14);
    }
  });

  it('orthonormalize: nearest rotation, det +1 even for reflections', () => {
    const rng = new Rng(21);
    for (let i = 0; i < 1000; i++) {
      const w = randUnit(rng).map((v) => v * rng.uniform(0, 3.1));
      const R = rotvecToMat3(w);
      const noisy = R.map((v) => v + 1e-6 * rng.gaussian());
      const Rn = orthonormalize(noisy);
      expect(rotationError(Rn)).toBeLessThan(1e-14);
      expect(rotationAngleBetween(Rn, R)).toBeLessThan(1e-5);
      // Idempotent on rotations.
      expect(maxDiff(orthonormalize(R), R)).toBeLessThan(1e-15 * 8);
    }
    // Reflection input → proper rotation output.
    const F = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, -1]);
    const Rf = orthonormalize(F);
    expect(rotationError(Rf)).toBeLessThan(1e-15);
    // Rank-deficient input still yields a rotation.
    const Rz = orthonormalize(new Float64Array([1, 0, 0, 0, 0, 0, 0, 0, 0]));
    expect(rotationError(Rz)).toBeLessThan(1e-15);
    // Scaled rotation (polar factor).
    const S = mat3Mul(rotvecToMat3([0.3, -0.2, 0.9]), new Float64Array([2, 0, 0, 0, 3, 0, 0, 0, 0.5]));
    const Rs = orthonormalize(S);
    expect(rotationError(Rs)).toBeLessThan(1e-14);
  });

  it('rotationBetweenVectors: random, parallel, antiparallel, near-antiparallel, non-unit', () => {
    const rng = new Rng(22);
    const check = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number): void => {
      const R = rotationBetweenVectors(a, b);
      expect(rotationError(R)).toBeLessThan(1e-14);
      const na = Math.hypot(a[0], a[1], a[2]),
        nb = Math.hypot(b[0], b[1], b[2]);
      const Ra = mat3MulVec(R, [a[0] / na, a[1] / na, a[2] / na]);
      expect(maxDiff(Ra, [b[0] / nb, b[1] / nb, b[2] / nb])).toBeLessThan(tol);
      // Minimal: the rotation angle equals the angle between a and b.
      const ang = Math.atan2(
        Math.hypot(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]),
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
      );
      expect(Math.abs(rotationAngle(R) - ang)).toBeLessThan(1e-11);
    };
    for (let i = 0; i < 2000; i++) {
      const a = randUnit(rng);
      const b = randUnit(rng);
      check(a, b, 1e-15 * 4);
      check(a.map((v) => v * 3.7), b.map((v) => v * 0.01), 1e-15 * 4);
      check(a, a, 1e-15);
      check(a, a.map((v) => -v), 1e-15 * 4);
      // Near-antiparallel: b = −a + ε⊥
      for (const eps of [1e-6, 1e-10, 1e-13, 1e-16]) {
        const p = randUnit(rng);
        const d = p[0] * a[0] + p[1] * a[1] + p[2] * a[2];
        const perp = p.map((v, k) => v - d * a[k]);
        const bn = a.map((v, k) => -v + eps * perp[k]);
        check(a, bn, 1e-14);
      }
    }
    // Coordinate axes antiparallel.
    check([1, 0, 0], [-1, 0, 0], 1e-15);
    check([0, 0, 1], [0, 0, -1], 1e-15);
    // Zero input → identity.
    expect(Array.from(rotationBetweenVectors([0, 0, 0], [1, 0, 0]))).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });

  it('aliasing: out may equal the input', () => {
    const rng = new Rng(23);
    const R = rotvecToMat3([0.1, 0.2, -0.3]);
    const S = R.map((v) => v + 1e-4 * rng.gaussian());
    const ref = orthonormalize(S);
    orthonormalize(S, S);
    expect(maxDiff(S, ref)).toBe(0);
    const w = new Float64Array([0.3, -1, 2]);
    const Rw = rotvecToMat3(w);
    const v = mat3ToRotvec(Rw);
    expect(maxDiff(v, w)).toBeLessThan(1e-14);
    const RtR = mat3MulTranspose(Rw, Rw);
    expect(maxDiff(RtR, [1, 0, 0, 0, 1, 0, 0, 0, 1])).toBeLessThan(1e-15 * 4);
  });
});
