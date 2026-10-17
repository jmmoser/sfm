import { describe, expect, it } from 'vitest';
import { debugKeyframes, debugPairs } from './debug-dump';

describe('debug-dump (inline engine internals)', () => {
  const I = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const fake = {
    engine: {
      live: {
        state: {
          order: [0, 1, 2],
          records: new Map([
            [0, { frameIndex: 5 }],
            [1, { frameIndex: 8 }],
            [2, { frameIndex: 11 }],
          ]),
          db: {
            *pairs() {
              yield [0, 1, { config: 'calibrated', inliers: new Uint32Array(20), triAngle: Math.PI / 36, R: Float64Array.from(I), t: Float64Array.of(1, 0, 0) }];
              yield [1, 2, { config: 'degenerate', inliers: new Uint32Array(0) }];
            },
          },
        },
        mapper: {
          reconstruction: {
            images: new Map([
              [0, { registered: true, numPoints3D: 40, numVisiblePoints3D: 41, pose: { R: Float64Array.from(I), t: Float64Array.of(0, 0, 0) } }],
              [1, { registered: true, numPoints3D: 35, pose: { R: Float64Array.from(I), t: Float64Array.of(-1, 0, 0) } }],
              [2, { registered: false, numPoints3D: 0, pose: { R: Float64Array.from(I), t: Float64Array.of(0, 0, 0) } }],
            ]),
          },
        },
      },
    },
  };

  it('keyframes in capture order with registration state, frame index and pose', () => {
    const k = debugKeyframes(fake)!;
    expect(k.map((x) => [x.id, x.frameIndex, x.registered, x.numPoints3D])).toEqual([
      [0, 5, true, 40],
      [1, 8, true, 35],
      [2, 11, false, 0],
    ]);
    expect(k[1].pose).toEqual({ R: I, t: [-1, 0, 0] });
    expect(k[2].pose).toBeUndefined();
  });

  it('pairs with frame indices, inlier counts and geometry', () => {
    const p = debugPairs(fake)!;
    expect(p).toHaveLength(2);
    expect(p[0]).toMatchObject({ a: 0, b: 1, frameA: 5, frameB: 8, config: 'calibrated', inliers: 10, R: I, t: [1, 0, 0] });
    expect(p[0].triAngleDeg).toBeCloseTo(5, 10);
    expect(p[1]).toMatchObject({ inliers: 0, R: null, t: null, triAngleDeg: null });
  });

  it('null for engines without reachable internals (worker / mock)', () => {
    expect(debugKeyframes({})).toBeNull();
    expect(debugPairs({ engine: { live: null } })).toBeNull();
    expect(debugKeyframes(null)).toBeNull();
  });
});
