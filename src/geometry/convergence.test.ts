import { describe, expect, it } from 'vitest';
import type { Pose } from '../core/types';
import { lookAt } from '../dense/synthetic';
import { opticalAxisConvergence } from './convergence';

/** Orbit of n cameras at radius r and height h around `target`, looking at it. */
function orbit(n: number, r: number, h: number, target: [number, number, number]): Pose[] {
  const out: Pose[] = [];
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n;
    out.push(lookAt([target[0] + r * Math.cos(a), target[1] + r * Math.sin(a), target[2] + h], target, [0, 0, 1]));
  }
  return out;
}

describe('opticalAxisConvergence', () => {
  it('finds the orbit centre and passes the orbit test', () => {
    const c = opticalAxisConvergence(orbit(24, 0.5, 0.2, [0.3, -0.2, 0.8]));
    expect(c.point).not.toBeNull();
    expect(Math.hypot(c.point![0] - 0.3, c.point![1] + 0.2, c.point![2] - 0.8)).toBeLessThan(1e-9);
    expect(c.converging).toBe(true);
    expect(c.inFront).toBe(24);
    expect(c.distMedian).toBeCloseTo(Math.hypot(0.5, 0.2), 9);
  });

  it('rejects parallel axes (a sideways walk)', () => {
    const walk: Pose[] = [];
    for (let i = 0; i < 10; i++) walk.push(lookAt([i, 0, 1.5], [i, 10, 1.5], [0, 0, 1]));
    const c = opticalAxisConvergence(walk);
    expect(c.point).toBeNull();
    expect(c.converging).toBe(false);
  });

  it('does not call an outward-looking ring converging', () => {
    const ring: Pose[] = [];
    for (let i = 0; i < 12; i++) {
      const a = (2 * Math.PI * i) / 12;
      ring.push(lookAt([0, 0, 1.5], [Math.cos(a), Math.sin(a), 1.5], [0, 0, 1]));
    }
    expect(opticalAxisConvergence(ring).converging).toBe(false);
  });
});
