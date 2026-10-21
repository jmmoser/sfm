import { describe, expect, it } from 'vitest';
import { grownIntersectionCapacity } from './renderer';

describe('grownIntersectionCapacity', () => {
  const LIMIT = 128 * 1024 * 1024; // WebGPU's default maxStorageBufferBindingSize: 33 554 432 intersections
  it('grows a reservation the frame overflowed, with 25 % headroom', () => {
    expect(grownIntersectionCapacity(5000, 4096, LIMIT)).toBe(6250);
    expect(grownIntersectionCapacity(4096, 4096, LIMIT)).toBeNull(); // fits
    expect(grownIntersectionCapacity(100, 4096, LIMIT)).toBeNull();
  });
  it('never grows beyond what one sort buffer can bind (4 B per intersection)', () => {
    // 30M fits the limit, 1.25 × 30M would not: clamped to the limit, still enough for the frame
    expect(grownIntersectionCapacity(30_000_000, 4096, LIMIT)).toBe(LIMIT / 4);
    expect(grownIntersectionCapacity(LIMIT / 4, 4096, LIMIT)).toBe(LIMIT / 4);
    // a frame that cannot be bound at all keeps the reservation (its farthest intersections are dropped)
    expect(grownIntersectionCapacity(LIMIT / 4 + 1, 4096, LIMIT)).toBeNull();
  });
});
