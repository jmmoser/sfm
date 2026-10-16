import { describe, expect, it } from 'vitest';
import {
  evaluateLoss,
  LOSS_CAUCHY,
  LOSS_HUBER,
  LOSS_SOFT_L1,
  LOSS_TRIVIAL,
  lossKind,
  lossRho,
  lossWeight,
  type LossKind,
} from './loss';

const kinds: [string, LossKind][] = [
  ['trivial', LOSS_TRIVIAL],
  ['huber', LOSS_HUBER],
  ['cauchy', LOSS_CAUCHY],
  ['soft_l1', LOSS_SOFT_L1],
];

describe('robust losses', () => {
  it('maps names to kinds and rejects unknown names', () => {
    for (const [name, k] of kinds) expect(lossKind(name as 'trivial')).toBe(k);
    expect(() => lossKind('tukey' as 'trivial')).toThrow(RangeError);
  });

  for (const [name, k] of kinds) {
    it(`${name}: ρ(0)=0, ρ'(0)=1, derivatives match finite differences, monotone`, () => {
      const out = new Float64Array(3);
      for (const b of [0.25, 1, 9]) {
        evaluateLoss(k, 0, b, out);
        expect(out[0]).toBe(0);
        expect(out[1]).toBe(1);
        let prev = -1;
        for (const s of [1e-6, 0.01, 0.2, 0.9 * b, 1.1 * b, 3 * b, 50, 1e4]) {
          evaluateLoss(k, s, b, out);
          expect(out[0]).toBeGreaterThan(prev);
          prev = out[0];
          expect(out[0]).toBeLessThanOrEqual(s + 1e-12); // all robust losses are ≤ the squared loss
          const h = 1e-6 * Math.max(1, s);
          const d1 = (lossRho(k, s + h, b) - lossRho(k, s - h, b)) / (2 * h);
          const d2 = (lossWeight(k, s + h, b) - lossWeight(k, s - h, b)) / (2 * h);
          expect(out[1]).toBeCloseTo(d1, 6);
          expect(out[1]).toBe(lossWeight(k, s, b));
          expect(Math.abs(out[2] - d2)).toBeLessThan(1e-6 * Math.max(1, Math.abs(d2)));
          expect(out[1]).toBeGreaterThan(0);
          expect(out[1]).toBeLessThanOrEqual(1);
        }
      }
    });
  }

  it('huber is continuous with continuous slope at s = b; cauchy/soft_l1 known values', () => {
    const b = 4;
    expect(lossRho(LOSS_HUBER, b, b)).toBeCloseTo(b, 12);
    expect(lossRho(LOSS_HUBER, b * (1 + 1e-12), b)).toBeCloseTo(b, 9);
    expect(lossWeight(LOSS_HUBER, b * (1 + 1e-12), b)).toBeCloseTo(1, 9);
    expect(lossRho(LOSS_HUBER, 100, 4)).toBeCloseTo(2 * 2 * 10 - 4, 12);
    expect(lossRho(LOSS_CAUCHY, 3, 1)).toBeCloseTo(Math.log(4), 12);
    expect(lossRho(LOSS_SOFT_L1, 3, 1)).toBeCloseTo(2, 12);
    expect(lossWeight(LOSS_CAUCHY, 3, 1)).toBeCloseTo(0.25, 12);
  });
});
