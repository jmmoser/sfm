import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics } from '../core/types';
import { createIntrinsics, projectCam } from '../geometry/camera';
import { Rng } from '../math/random';
import { rotvecToMat3 } from '../math/rotation';
import { KeyframeSelector, rotationCompensatedParallax, type KeyframeCandidate } from './keyframes';

const W = 1920;

function cand(t: number, o: Partial<KeyframeCandidate> = {}): KeyframeCandidate {
  return {
    t,
    sharpness: 100,
    angularSpeed: 0.2,
    numFeatures: 2000,
    numMatchesToLastKF: 900,
    numInliersToLastKF: 600,
    medianParallaxPx: 5,
    rotationSinceLastKFRad: 0.01,
    imageWidth: W,
    ...o,
  };
}

/** Selector with a warmed-up sharpness reference (median 100) and a first keyframe at t = 0. */
function warmed(opts: ConstructorParameters<typeof KeyframeSelector>[0] = {}): KeyframeSelector {
  const s = new KeyframeSelector(opts);
  for (let i = -6; i < -1; i++) expect(s.evaluate(cand(i * 20)).reason).toMatch(/warming/);
  expect(s.evaluate(cand(-20, { sharpness: 50 })).decision).toBe('reject_blur');
  expect(s.evaluate(cand(0))).toMatchObject({ decision: 'keyframe', trigger: 'first', guidance: 'good' });
  return s;
}

describe('KeyframeSelector', () => {
  it('first keyframe after warm-up; blur and fast rotation are rejected', () => {
    const s = warmed();
    for (let i = 1; i <= 6; i++) s.evaluate(cand(i * 30, { sharpness: 100 + i }));
    expect(s.sharpnessReference).toBeGreaterThan(99);
    const blur = s.evaluate(cand(300, { sharpness: 40 }));
    expect(blur.decision).toBe('reject_blur');
    const blurFast = s.evaluate(cand(330, { sharpness: 40, angularSpeed: 1.0 }));
    expect(blurFast).toMatchObject({ decision: 'reject_blur', guidance: 'move_slower' });
    const fast = s.evaluate(cand(360, { angularSpeed: 1.5 }));
    expect(fast).toMatchObject({ decision: 'reject_motion', guidance: 'move_slower' });
    // unknown angular speed (no IMU) is not a rejection
    expect(s.evaluate(cand(390, { angularSpeed: NaN })).decision).toBe('skip');
    // the threshold adapts when the whole scene becomes softer
    for (let i = 0; i < 40; i++) s.evaluate(cand(400 + i * 30, { sharpness: 45 }));
    expect(s.evaluate(cand(1700, { sharpness: 40, medianParallaxPx: 200 })).decision).toBe('keyframe');
  });

  it('absolute sharpness floor applies before the running statistic is warmed up', () => {
    const s = new KeyframeSelector({ minSharpness: 10, sharpnessWarmup: 2 });
    expect(s.evaluate(cand(0, { sharpness: 5 })).decision).toBe('reject_blur');
    expect(s.evaluate(cand(30, { sharpness: 50 })).decision).toBe('skip');
    expect(s.evaluate(cand(60, { sharpness: 50 })).decision).toBe('keyframe');
  });

  it('parallax trigger depends on the mode', () => {
    for (const [mode, frac] of [
      ['object', 0.03],
      ['auto', 0.04],
      ['scene', 0.05],
    ] as const) {
      const s = warmed({ mode });
      expect(s.parallaxThresholdPx(W)).toBeCloseTo(frac * W, 9);
      expect(s.evaluate(cand(400, { medianParallaxPx: frac * W - 1 })).decision).toBe('skip');
      const d = s.evaluate(cand(450, { medianParallaxPx: frac * W + 1 }));
      expect(d).toMatchObject({ decision: 'keyframe', trigger: 'parallax' });
      expect(s.lastKeyframeTime).toBe(450);
    }
  });

  it('overlap trigger relative to the best inlier count since the keyframe', () => {
    const s = warmed();
    expect(s.evaluate(cand(300, { numInliersToLastKF: 500 })).decision).toBe('skip');
    expect(s.evaluate(cand(400, { numInliersToLastKF: 610 })).decision).toBe('skip'); // best = 610
    expect(s.evaluate(cand(500, { numInliersToLastKF: 430 })).decision).toBe('skip'); // 0.705
    expect(s.evaluate(cand(600, { numInliersToLastKF: 420 }))).toMatchObject({ decision: 'keyframe', trigger: 'overlap' });
  });

  it('rotation trigger; pure rotation asks to translate', () => {
    const s = warmed();
    const d1 = s.evaluate(cand(300, { rotationSinceLastKFRad: 0.1, medianParallaxPx: 3 }));
    expect(d1).toMatchObject({ decision: 'skip', guidance: 'translate_more' });
    const d2 = s.evaluate(cand(400, { rotationSinceLastKFRad: (13 * Math.PI) / 180, medianParallaxPx: 3 }));
    expect(d2).toMatchObject({ decision: 'keyframe', trigger: 'rotation', guidance: 'translate_more' });
    // rotation with parallax is fine
    const d3 = s.evaluate(cand(600, { rotationSinceLastKFRad: 0.1, medianParallaxPx: 40 }));
    expect(d3.guidance).toBe('good');
  });

  it('tracking lost: keyframe after the minimum interval, guidance', () => {
    const s = warmed();
    const early = s.evaluate(cand(100, { numInliersToLastKF: 5 }));
    expect(early).toMatchObject({ decision: 'skip', guidance: 'tracking_lost' });
    const lost = s.evaluate(cand(200, { numInliersToLastKF: 5 }));
    expect(lost).toMatchObject({ decision: 'keyframe', trigger: 'tracking_lost', guidance: 'tracking_lost' });
  });

  it('min / max interval: no keyframe when static, keyframe after 2 s while moving slowly', () => {
    const s = warmed();
    for (let t = 200; t <= 4000; t += 100) expect(s.evaluate(cand(t, { medianParallaxPx: 2, rotationSinceLastKFRad: 0.005 })).decision).toBe('skip');
    const thr = s.parallaxThresholdPx(W);
    let kf = -1;
    for (let t = 4100; t <= 7000; t += 100) {
      const d = s.evaluate(cand(t, { medianParallaxPx: 0.3 * thr }));
      if (d.decision === 'keyframe') {
        expect(d.trigger).toBe('max_interval');
        kf = t;
        break;
      }
    }
    // the static period counted towards the interval: first moving frame already exceeds 2 s
    expect(kf).toBe(4100);
    expect(s.evaluate(cand(4200, { medianParallaxPx: 1000 })).decision).toBe('skip'); // < 150 ms
    expect(s.evaluate(cand(4260, { medianParallaxPx: 1000 })).decision).toBe('keyframe');
  });

  it('few features → more_texture', () => {
    const s = warmed();
    expect(s.evaluate(cand(500, { numFeatures: 50, numInliersToLastKF: 3 }))).toMatchObject({ decision: 'skip', guidance: 'more_texture' });
  });

  it('defers soft triggers briefly for a sharper frame', () => {
    const s = warmed();
    const thr = s.parallaxThresholdPx(W);
    const d1 = s.evaluate(cand(300, { medianParallaxPx: thr + 5, sharpness: 75 }));
    expect(d1.decision).toBe('skip');
    expect(d1.reason).toMatch(/sharper/);
    expect(s.evaluate(cand(400, { medianParallaxPx: thr + 8, sharpness: 98 }))).toMatchObject({ decision: 'keyframe', trigger: 'parallax' });
    // … but not forever
    const d2 = s.evaluate(cand(600, { medianParallaxPx: thr + 5, sharpness: 75 }));
    expect(d2.decision).toBe('skip');
    expect(s.evaluate(cand(700, { medianParallaxPx: thr + 5, sharpness: 75 })).decision).toBe('skip');
    expect(s.evaluate(cand(860, { medianParallaxPx: thr + 5, sharpness: 75 })).decision).toBe('keyframe');
    // hard overlap loss is never deferred
    expect(s.evaluate(cand(1000, { numInliersToLastKF: 600 })).decision).toBe('skip');
    expect(s.evaluate(cand(1100, { numInliersToLastKF: 200, sharpness: 75 }))).toMatchObject({ decision: 'keyframe', trigger: 'overlap' });
  });

  it('revert / reset', () => {
    const s = warmed();
    s.evaluate(cand(500, { medianParallaxPx: 500 }));
    expect(s.lastKeyframeTime).toBe(500);
    expect(s.numKeyframes).toBe(2);
    s.revertLastKeyframe();
    expect(s.lastKeyframeTime).toBe(0);
    expect(s.numKeyframes).toBe(1);
    s.resetTracking();
    expect(s.evaluate(cand(700)).trigger).toBe('first'); // statistic kept: no warm-up
    s.reset();
    expect(s.numKeyframes).toBe(0);
    expect(Number.isNaN(s.sharpnessReference)).toBe(true);
  });

  it('continuous capture simulation: keyframe spacing follows parallax and rotation', () => {
    // camera translating sideways at 0.3 m/s in front of a scene at 2 m, rotating 10°/s; 10 Hz frames
    const rng = new Rng(4);
    const s = new KeyframeSelector({ mode: 'auto' });
    const f = 1500;
    let lastT = 0;
    const kfT: number[] = [];
    for (let i = 0; i < 300; i++) {
      const t = i * 100;
      const since = (t - lastT) / 1000;
      const parallax = (f * 0.3 * since) / 2; // px
      const blurry = rng.next() < 0.1;
      const d = s.evaluate(
        cand(t, {
          sharpness: blurry ? 30 : 100 + 10 * rng.gaussian(),
          angularSpeed: 0.17,
          medianParallaxPx: parallax,
          rotationSinceLastKFRad: 0.17 * since,
          numInliersToLastKF: Math.max(0, Math.round(800 * (1 - parallax / 600))),
        }),
      );
      if (blurry && i >= 5) expect(d.decision).toBe('reject_blur');
      if (d.decision === 'keyframe') {
        lastT = t;
        kfT.push(t);
      }
    }
    const gaps = kfT.slice(1).map((v, i) => v - kfT[i]);
    // parallax threshold 76.8 px ⇒ ≈ 0.34 s between keyframes (+ blur rejections / deferral)
    expect(Math.min(...gaps)).toBeGreaterThanOrEqual(300);
    expect(Math.max(...gaps)).toBeLessThanOrEqual(700);
    expect(kfT.length).toBeGreaterThan(50);
  });
});

describe('rotationCompensatedParallax', () => {
  function scene(rng: Rng, cam: CameraIntrinsics, n: number, depth: [number, number]) {
    const P: number[][] = [];
    const tmp = new Float64Array(2);
    while (P.length < n) {
      const X = [rng.uniform(-1, 1), rng.uniform(-0.75, 0.75), 1];
      const z = rng.uniform(depth[0], depth[1]);
      const Xc = [X[0] * z, X[1] * z, z];
      if (projectCam(cam, Xc, tmp)) P.push(Xc);
    }
    return P;
  }
  function observe(cam: CameraIntrinsics, P: number[][], R: Float64Array, t: number[]) {
    const kpA = new Float32Array(4 * P.length),
      kpB = new Float32Array(4 * P.length);
    const m: number[] = [];
    const a = new Float64Array(2),
      b = new Float64Array(2);
    P.forEach((X, i) => {
      const Y = [0, 1, 2].map((r) => R[3 * r] * X[0] + R[3 * r + 1] * X[1] + R[3 * r + 2] * X[2] + t[r]);
      if (!projectCam(cam, X, a) || !projectCam(cam, Y, b)) return;
      kpA.set([a[0], a[1], 2, 0], 4 * i);
      kpB.set([b[0], b[1], 2, 0], 4 * i);
      m.push(i, i);
    });
    return { kpA, kpB, matches: Uint32Array.from(m) };
  }

  it('≈ 0 for pure rotation (distortion handled with intrinsics), > 0 with translation', () => {
    const rng = new Rng(8);
    const cam = createIntrinsics(1920, 1440, (68 * Math.PI) / 180, -0.08, 0.01);
    const P = scene(rng, cam, 400, [2, 30]);
    const R = rotvecToMat3([0.05, -0.12, 0.03]);
    const rot = observe(cam, P, R, [0, 0, 0]);
    expect(rotationCompensatedParallax(rot.kpA, rot.kpB, rot.matches, cam, R)).toBeLessThan(1e-3);
    // raw-pixel homography ignores distortion → small but non-zero
    const K = Float64Array.of(cam.f, 0, cam.cx, 0, cam.f, cam.cy, 0, 0, 1);
    const withK = rotationCompensatedParallax(rot.kpA, rot.kpB, rot.matches, K, R);
    expect(withK).toBeGreaterThan(0.1);
    // translation sideways 0.2 m: median parallax ≈ f·0.2/median depth (distortion-free check)
    const pin = createIntrinsics(1920, 1440, (68 * Math.PI) / 180);
    const P2 = scene(rng, pin, 2000, [4, 4]);
    const tr = observe(pin, P2, R, [0.2, 0, 0]);
    const par = rotationCompensatedParallax(tr.kpA, tr.kpB, tr.matches, pin, R);
    const Kp = Float64Array.of(pin.f, 0, pin.cx, 0, pin.f, pin.cy, 0, 0, 1);
    expect(par).toBeGreaterThan((pin.f * 0.2) / 4 * 0.85);
    expect(par).toBeLessThan((pin.f * 0.2) / 4 * 1.15);
    expect(rotationCompensatedParallax(tr.kpA, tr.kpB, tr.matches, Kp, R)).toBeCloseTo(par, 6);
    // without rotation compensation the value would be much larger
    const I = rotvecToMat3([0, 0, 0]);
    expect(rotationCompensatedParallax(tr.kpA, tr.kpB, tr.matches, pin, I)).toBeGreaterThan(2 * par);
    expect(Number.isNaN(rotationCompensatedParallax(tr.kpA, tr.kpB, new Uint32Array(0), pin, R))).toBe(true);
  });
});
