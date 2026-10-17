import { describe, expect, it } from 'vitest';
import { Rng } from '../math/random';
import {
  poseCenter,
  poseCompose,
  poseCopy,
  poseFromCenter,
  poseFromMat34,
  poseIdentity,
  poseInverse,
  poseRelative,
  poseToMat34,
  poseTransformPoint,
} from './pose';
import { randomPoseLookingAtOrigin } from './testutil';

function expectClose(a: ArrayLike<number>, b: ArrayLike<number>, tol: number): void {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol);
}

describe('pose utilities', () => {
  const rng = new Rng(7);

  it('identity, copy, center and fromCenter', () => {
    const I = poseIdentity();
    expectClose(I.R, [1, 0, 0, 0, 1, 0, 0, 0, 1], 0);
    expectClose(I.t, [0, 0, 0], 0);
    const p = randomPoseLookingAtOrigin(rng, 4);
    const c = poseCopy(p);
    expect(c.R).not.toBe(p.R);
    expectClose(c.R, p.R, 0);
    const C = poseCenter(p);
    expect(Math.hypot(C[0], C[1], C[2])).toBeCloseTo(4, 12);
    // the centre maps to the camera origin
    expectClose(poseTransformPoint(p, C), [0, 0, 0], 1e-12);
    const q = poseFromCenter(p.R, C);
    expectClose(q.t, p.t, 1e-12);
  });

  it('compose / inverse / relative', () => {
    for (let k = 0; k < 50; k++) {
      const a = randomPoseLookingAtOrigin(rng, 3);
      const b = randomPoseLookingAtOrigin(rng, 7);
      const X = [rng.gaussian(), rng.gaussian(), rng.gaussian()];
      const ab = poseCompose(a, b);
      expectClose(poseTransformPoint(ab, X), poseTransformPoint(a, poseTransformPoint(b, X)), 1e-12);
      const ai = poseInverse(a);
      expectClose(poseTransformPoint(ai, poseTransformPoint(a, X)), X, 1e-12);
      const I = poseCompose(a, ai);
      expectClose(I.R, [1, 0, 0, 0, 1, 0, 0, 0, 1], 1e-14);
      expectClose(I.t, [0, 0, 0], 1e-12);
      // relative pose maps camera-1 coordinates to camera-2 coordinates
      const rel = poseRelative(a, b);
      expectClose(poseTransformPoint(rel, poseTransformPoint(a, X)), poseTransformPoint(b, X), 1e-12);
    }
  });

  it('Mat34 round trip and in-place transform', () => {
    const p = randomPoseLookingAtOrigin(rng, 2);
    const M = poseToMat34(p);
    expect(M[3]).toBe(p.t[0]);
    expect(M[7]).toBe(p.t[1]);
    expect(M[11]).toBe(p.t[2]);
    const q = poseFromMat34(M);
    expectClose(q.R, p.R, 0);
    expectClose(q.t, p.t, 0);
    const X = Float64Array.of(1, 2, 3);
    const Y = poseTransformPoint(p, X);
    poseTransformPoint(p, X, X); // aliasing allowed
    expectClose(X, Y, 0);
  });
});
