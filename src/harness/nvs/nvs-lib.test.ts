/**
 * Tests of the novel-view evaluation helpers (e2e/lib/nvs-metrics.mjs, nvs-views.mjs,
 * nvs-args.mjs; plain JS typed via .d.mts): image metrics against brute-force references,
 * the affine colour fit, coverage / compositing, the alignment shift, held-out view selection and
 * the pose / focal mapping into the run frame.
 */
import { describe, expect, it } from 'vitest';
import {
  AFFINE_IDENTITY,
  aggregateScores,
  alignmentShift,
  applyAffineColour,
  fitAffineColour,
  gaussianKernel,
  prepareGroundTruth,
  psnr,
  scoreCommon,
  scoreView,
  ssim,
  triptych,
} from '../../../e2e/lib/nvs-metrics.mjs';
import {
  cameraCenter,
  cameraNu,
  cameraOmega,
  denseImageSize,
  gtRollingShutter,
  gyroAt,
  heldOutFrames,
  mapPoseRelative,
  nearestInSpace,
  mapPoseSim3,
  runRollingShutter,
  sceneMismatch,
  selectEvenly,
  testViews,
  type PoseJSON,
} from '../../../e2e/lib/nvs-views.mjs';
import { NvsArgError, parseNvsArgs } from '../../../e2e/lib/nvs-args.mjs';
import { axisAngle, mat3Mul, mat3T, mat3Vec } from '../../../e2e/lib/linalg.mjs';
import { pinholePerspective } from './pinhole';

function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** Smooth random RGBA8 texture (sum of sinusoids), alpha 255. */
function texture(w: number, h: number, seed: number, shift = [0, 0]): Uint8ClampedArray {
  const r = rng(seed);
  const waves = Array.from({ length: 6 }, () => ({ fx: 0.02 + 0.2 * r(), fy: 0.02 + 0.2 * r(), p: 6.28 * r(), a: 20 + 30 * r(), c: Math.floor(3 * r()) }));
  const out = new Uint8ClampedArray(4 * w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = 4 * (y * w + x);
      const v = [128, 128, 128];
      for (const wv of waves) v[wv.c] += wv.a * Math.sin(wv.fx * (x + shift[0]) + wv.fy * (y + shift[1]) + wv.p);
      out[i] = v[0];
      out[i + 1] = v[1];
      out[i + 2] = v[2];
      out[i + 3] = 255;
    }
  return out;
}

/** Brute-force masked SSIM: direct 11 × 11 Gaussian windows over masked pixels, per channel mean. */
function ssimReference(a: ArrayLike<number>, b: ArrayLike<number>, w: number, h: number, mask: Uint8Array | null): number {
  const k = gaussianKernel();
  const C1 = (0.01 * 255) ** 2,
    C2 = (0.03 * 255) ** 2;
  let total = 0;
  for (let c = 0; c < 3; c++) {
    let s = 0,
      n = 0;
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        if (mask && !mask[y * w + x]) continue;
        let W = 0,
          mx = 0,
          my = 0,
          xx = 0,
          yy = 0,
          xy = 0;
        for (let dy = -5; dy <= 5; dy++)
          for (let dx = -5; dx <= 5; dx++) {
            const X = x + dx,
              Y = y + dy;
            if (X < 0 || Y < 0 || X >= w || Y >= h) continue;
            const j = Y * w + X;
            if (mask && !mask[j]) continue;
            const g = k[dx + 5] * k[dy + 5];
            const p = a[4 * j + c],
              q = b[4 * j + c];
            W += g;
            mx += g * p;
            my += g * q;
            xx += g * p * p;
            yy += g * q * q;
            xy += g * p * q;
          }
        mx /= W;
        my /= W;
        const vx = Math.max(0, xx / W - mx * mx),
          vy = Math.max(0, yy / W - my * my),
          cv = xy / W - mx * my;
        s += ((2 * mx * my + C1) * (2 * cv + C2)) / ((mx * mx + my * my + C1) * (vx + vy + C2));
        n++;
      }
    total += s / n;
  }
  return total / 3;
}

describe('nvs metrics', () => {
  const w = 40,
    h = 30,
    n = w * h;

  it('PSNR: identical → ∞, constant offset → 20·log10(255/d), masks', () => {
    const a = texture(w, h, 1);
    expect(psnr(a, a, n)).toBe(Infinity);
    const b = Uint8ClampedArray.from(a);
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) b[4 * i + c] = a[4 * i + c] >= 128 ? a[4 * i + c] - 5 : a[4 * i + c] + 5;
    expect(psnr(a, b, n)).toBeCloseTo(20 * Math.log10(255 / 5), 9);
    const mask = new Uint8Array(n);
    mask[3] = 1;
    const c = Uint8ClampedArray.from(a);
    c[0] = 255 - c[0]; // pixel 0 is outside the mask
    expect(psnr(a, c, n, mask)).toBe(Infinity);
  });

  it('SSIM matches a brute-force masked reference, = 1 for identical images, ignores masked-out pixels', () => {
    const a = texture(w, h, 2);
    const b = texture(w, h, 2, [1.5, -0.7]);
    const r = rng(9);
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) b[4 * i + c] += 12 * (r() - 0.5);
    expect(ssim(a, a, w, h)).toBeCloseTo(1, 6);
    expect(ssim(a, b, w, h)).toBeCloseTo(ssimReference(a, b, w, h, null), 5);
    const mask = new Uint8Array(n);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) mask[y * w + x] = (x - 18) ** 2 + (y - 14) ** 2 < 140 ? 1 : 0;
    const ref = ssimReference(a, b, w, h, mask);
    expect(ssim(a, b, w, h, mask)).toBeCloseTo(ref, 5);
    // pixels outside the mask do not matter (neither as centres nor inside windows)
    const a2 = Uint8ClampedArray.from(a);
    for (let i = 0; i < n; i++) if (!mask[i]) a2[4 * i] = 255 - a2[4 * i];
    expect(ssim(a2, b, w, h, mask)).toBeCloseTo(ref, 5);
    // a reused GT cache gives the same values
    const cache = new Map();
    expect(ssim(a, b, w, h, mask, cache)).toBeCloseTo(ref, 5);
    expect(ssim(a2, b, w, h, mask, cache)).toBeCloseTo(ref, 5);
    expect(ssim(b, b, w, h, mask, cache)).toBeCloseTo(1, 6);
  });

  it('affine colour fit recovers a 3 × 4 transform (cross-channel terms included)', () => {
    const r = rng(3);
    const src = new Uint8ClampedArray(4 * n);
    for (let i = 0; i < 4 * n; i++) src[i] = 40 + 150 * r();
    const M = [0.9, 0.05, -0.02, 6, 0.03, 1.1, 0.01, -4, -0.04, 0.02, 0.8, 10];
    const dst = new Float64Array(4 * n); // unrounded target
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) dst[4 * i + c] = M[4 * c] * src[4 * i] + M[4 * c + 1] * src[4 * i + 1] + M[4 * c + 2] * src[4 * i + 2] + M[4 * c + 3];
    const fit = fitAffineColour(src, dst as unknown as Uint8Array, n);
    for (let k = 0; k < 12; k++) expect(fit[k]).toBeCloseTo(M[k], 4);
    // degenerate input (constant channels): stays finite, falls back near the identity
    const flat = new Uint8ClampedArray(4 * n).fill(100);
    const f2 = fitAffineColour(flat, flat, n);
    expect(f2.every(Number.isFinite)).toBe(true);
    expect(fitAffineColour(src, src, 10)).toEqual(AFFINE_IDENTITY);
    const img = Uint8ClampedArray.from([10, 20, 30, 255]);
    applyAffineColour(img, 1, [2, 0, 0, 0, 0, 1, 0, -30, 0, 0, 10, 0]);
    expect(Array.from(img)).toEqual([20, 0, 255, 255]);
  });

  it('scoreView: coverage, full frame vs covered, exposure compensation', () => {
    const gt = texture(w, h, 4);
    const same = scoreView(gt, gt, w, h);
    expect(same.coverage).toBe(1);
    expect(same.full.psnr).toBe(Infinity);
    expect(same.full.ssim).toBeCloseTo(1, 6);
    expect(same.shift).toEqual({ dx: 0, dy: 0 });
    // left half rendered (premultiplied, alpha 255), right half empty
    const half = new Uint8ClampedArray(4 * n);
    for (let y = 0; y < h; y++) for (let x = 0; x < w / 2; x++) for (let c = 0; c < 4; c++) half[4 * (y * w + x) + c] = gt[4 * (y * w + x) + c];
    const s = scoreView(half, gt, w, h, { shift: false });
    expect(s.coverage).toBeCloseTo(0.5, 9);
    expect(s.covered.psnr).toBe(Infinity);
    expect(s.covered.ssim).toBeCloseTo(1, 6);
    expect(s.full.psnr).toBeLessThan(15);
    // a GT pixel that is not valid is excluded everywhere
    const gtHole = Uint8ClampedArray.from(gt);
    gtHole[3] = 0;
    expect(scoreView(gt, gtHole, w, h).validPixels).toBe(n - 1);
    // exposure drift: the render is 0.8 × GT + 5 → poor raw score, near-perfect after the fit
    const dim = Uint8ClampedArray.from(gt);
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) dim[4 * i + c] = 0.8 * gt[4 * i + c] + 5;
    const e = scoreView(dim, gt, w, h);
    expect(e.covered.psnr).toBeLessThan(25);
    expect(e.cc.covered.psnr).toBeGreaterThan(45);
    expect(e.cc.affine[0]).toBeCloseTo(1.25, 1);
    // partial alpha (MSAA edge): straight colour recovered, counted as covered at ≥ 50 %
    const edge = Uint8ClampedArray.from(gt);
    for (let c = 0; c < 4; c++) edge[c] = Math.round(gt[c] * 0.6);
    const pe = scoreView(edge, gt, w, h, { ssim: false, shift: false });
    expect(pe.coverage).toBe(1);
    expect(pe.covered.psnr).toBeGreaterThan(45);
    expect(pe.full.psnr).toBeLessThan(pe.covered.psnr);
    // the prepared GT gives the same result as the raw image
    const g = prepareGroundTruth(gt, w, h);
    expect(scoreView(half, g, w, h, { shift: false })).toEqual(s);
    const agg = aggregateScores([same, s]);
    expect(agg.views).toBe(2);
    expect(agg.coverage.mean).toBeCloseTo(0.75, 9);
    expect(agg.covered.psnr.n).toBe(0); // ∞ is not averaged
    expect(triptych(gt, gt, w, h).length).toBe(4 * 3 * n);
  });

  it('scoreCommon: every renderer on the pixels all of them cover', () => {
    const gt = texture(w, h, 5);
    // A: exact on the left half only; B: everywhere, but 0.8 × GT + 5 on the right half and +3 on the left
    const a = new Uint8ClampedArray(4 * n);
    const b = Uint8ClampedArray.from(gt);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = 4 * (y * w + x);
        if (x < w / 2) for (let c = 0; c < 4; c++) a[i + c] = gt[i + c];
        for (let c = 0; c < 3; c++) b[i + c] = x < w / 2 ? gt[i + c] + 3 : 0.8 * gt[i + c] + 5;
      }
    const own = scoreView(b, gt, w, h, { shift: false });
    const c = scoreCommon({ a, b }, prepareGroundTruth(gt, w, h), w, h);
    expect(c.coverage).toBeCloseTo(0.5, 9);
    expect(c.pixels).toBe(n / 2);
    expect(c.scores.a.psnr).toBe(Infinity);
    expect(c.scores.a.ssim).toBeCloseTo(1, 6);
    // B on the common (left) pixels: a constant +3 offset → 20·log10(255/3), better than on all its own pixels
    expect(c.scores.b.psnr).toBeCloseTo(20 * Math.log10(255 / 3), 1);
    expect(c.scores.b.psnr).toBeGreaterThan(own.covered.psnr);
    expect(c.scores.b.ccPsnr).toBeGreaterThan(45);
    expect(scoreCommon({ a, b }, gt, w, h, { ssim: false }).scores.b.ssim).toBeNull();
    const none = scoreCommon({ a, e: new Uint8ClampedArray(4 * n) }, gt, w, h);
    expect(none.pixels).toBe(0);
    expect(none.scores.a.psnr).toBeNaN();
    const agg = aggregateScores([{ ...own, common: { coverage: c.coverage, ...c.scores.b } }]);
    expect(agg.common.psnr.mean).toBeCloseTo(c.scores.b.psnr, 9);
    expect(agg.common.coverage.mean).toBeCloseTo(0.5, 9);
    expect(aggregateScores([own]).common.psnr.n).toBe(0);
  });

  it('alignmentShift finds an image translation (render(x) = gt(x + d))', () => {
    const W = 120,
      H = 90;
    const gt = texture(W, H, 5);
    const render = texture(W, H, 5, [4, -2]);
    const d = alignmentShift(render, null, gt, null, W, H)!;
    expect(d.dx).toBeCloseTo(4, 0);
    expect(d.dy).toBeCloseTo(-2, 0);
    const d0 = alignmentShift(gt, null, gt, null, W, H)!;
    expect(Math.hypot(d0.dx, d0.dy)).toBeLessThan(1e-9);
  });
});

function randomRotation(r: () => number): number[] {
  const ax = [r() - 0.5, r() - 0.5, r() - 0.5];
  const l = Math.hypot(ax[0], ax[1], ax[2]);
  return axisAngle([ax[0] / l, ax[1] / l, ax[2] / l], 3 * r());
}

function expectPose(a: PoseJSON, b: PoseJSON, tol = 1e-9): void {
  for (let i = 0; i < 9; i++) expect(a.R[i]).toBeCloseTo(b.R[i], -Math.log10(tol));
  for (let i = 0; i < 3; i++) expect(a.t[i]).toBeCloseTo(b.t[i], -Math.log10(tol));
}

describe('nvs views', () => {
  // GT trajectory and a run frame related by an exact Sim3 (X_gt = s·R·X_run + t)
  const r = rng(11);
  const frames = Array.from({ length: 30 }, (_, i) => {
    const R = randomRotation(r);
    const C = [Math.cos(i / 5), Math.sin(i / 5), 0.3 * r()];
    const t = mat3Vec(R, C).map((v) => -v);
    return { index: i, file: `frames/${String(i).padStart(6, '0')}.jpg`, tTrue: 1000 + 125 * i, pose: { R, t }, f: 750 + i * 0.1 };
  });
  const sim = { s: 2.5, R: randomRotation(r), t: [0.3, -1.2, 0.7] };
  const runPose = (p: PoseJSON) => mapPoseSim3(p, sim);
  const keys = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27];
  const result = {
    intrinsics: { f: 760, cx: 359, cy: 481, k1: -0.03, k2: 0.005, width: 720, height: 960 },
    images: keys.map((k) => ({ index: k, pose: runPose(frames[k].pose), f: 760 * (1 + (k - 13) * 1e-4) })),
    live: { keyframeIndices: [...keys, 28] },
    frames: [{ index: 5, decision: 'reject_blur' }],
  };
  const meta = { name: 't', width: 720, height: 960, intrinsics: { f: 750 }, R_cam_device: [1, 0, 0, 0, -1, 0, 0, 0, -1] };

  it('maps GT poses into the run frame (global Sim3 and keyframe-relative agree on exact data)', () => {
    const j = 7,
      k = 6;
    const viaSim = mapPoseSim3(frames[j].pose, sim);
    const viaRel = mapPoseRelative(frames[j].pose, frames[k].pose, runPose(frames[k].pose), sim.s);
    expectPose(viaRel, viaSim);
    // the camera centre maps through the inverse Sim3
    const C = cameraCenter(frames[j].pose);
    const Cr = cameraCenter(viaSim);
    const back = mat3Vec(sim.R, Cr).map((v, i) => sim.s * v + sim.t[i]);
    for (let i = 0; i < 3; i++) expect(back[i]).toBeCloseTo(C[i], 9);
    // a keyframe's own pose: relative mapping is the identity
    expectPose(mapPoseRelative(frames[k].pose, frames[k].pose, result.images[2].pose, sim.s), result.images[2].pose);
  });

  it('a keyframe pose error carries over to the relative mapping only', () => {
    const k = 6;
    const err = axisAngle([0, 0, 1], 0.01);
    const kf = runPose(frames[k].pose);
    const noisy = { R: mat3Mul(err, kf.R), t: kf.t };
    const rel = mapPoseRelative(frames[k + 1].pose, frames[k].pose, noisy, sim.s);
    // relative motion keyframe → frame is preserved exactly
    const exact = mapPoseSim3(frames[k + 1].pose, sim);
    const relMotion = (a: PoseJSON, b: PoseJSON) => mat3Mul(a.R, mat3T(b.R));
    const m1 = relMotion(rel, noisy),
      m2 = relMotion(exact, kf);
    for (let i = 0; i < 9; i++) expect(m1[i]).toBeCloseTo(m2[i], 9);
  });

  it('held-out frames, even selection and test views', () => {
    const held = heldOutFrames(frames, result);
    expect(held.map((f) => f.index)).not.toContain(5); // rejected as blurred
    expect(held.map((f) => f.index)).not.toContain(28); // live keyframe without a pose
    expect(held.some((f) => keys.includes(f.index))).toBe(false);
    expect(heldOutFrames(frames, result, { keepBlurred: true }).map((f) => f.index)).toContain(5);
    expect(selectEvenly([1, 2, 3, 4, 5, 6, 7, 8], 4)).toEqual([2, 4, 6, 8]);
    expect(selectEvenly([1, 2], 5)).toEqual([1, 2]);

    const tv = testViews({ meta, frames }, result, { views: 5 });
    expect(tv.sim3.s).toBeCloseTo(sim.s, 9);
    expect(tv.candidates).toBe(held.length);
    expect(tv.views.length).toBe(5);
    for (const v of tv.views) {
      expectPose(v.pose, mapPoseSim3(frames[v.index].pose, sim), 1e-8);
      const k = v.nearestKeyframe;
      expect(v.gap).toBe(Math.abs(k - v.index));
      expect(keys).toContain(k);
      // relative focal: keyframe's refined focal (median-normalised) × GT focal ratio
      const im = result.images.find((x) => x.index === k)!;
      const fs = result.images.map((x) => x.f).sort((a, b) => a - b);
      const kf = (760 * im.f) / fs[fs.length >> 1];
      expect(v.camera.f).toBeCloseTo((kf * frames[v.index].f) / frames[k].f, 9);
    }
    // the per-image focal comes from sparse.images when the metrics-input images have none
    const split = { ...result, images: result.images.map(({ index, pose }) => ({ index, pose })), sparse: { images: result.images } };
    const tk = testViews({ meta, frames }, split, { frames: [7], focal: 'keyframe' }).views[0];
    const fs7 = result.images.map((x) => x.f).sort((a, b) => a - b);
    expect(tk.camera.f).toBeCloseTo((760 * result.images.find((x) => x.index === 6)!.f) / fs7[fs7.length >> 1], 9);
    expect(tk.camera.f).not.toBe(760);
    const shared = testViews({ meta, frames }, result, { frames: [7], focal: 'shared', pose: 'sim3' });
    expect(shared.views[0].camera.f).toBe(760);
    expect(shared.views[0].index).toBe(7);
  });

  it('test dataset: its frames through the Sim3, the keyframe nearest in space, its own ω / ν', () => {
    // the same world, a wider and higher path; frames indexed from 0 again
    const tFrames = Array.from({ length: 12 }, (_, i) => {
      const R = randomRotation(r);
      const C = [1.3 * Math.cos(i / 2), 1.3 * Math.sin(i / 2), 0.5];
      const t = mat3Vec(R, C).map((v) => -v);
      return { index: i, file: `frames/${String(i).padStart(6, '0')}.jpg`, tTrue: 1000 + 125 * i, pose: { R, t }, f: 752 + i * 0.2, focusDistance: 1 };
    });
    const tImu = [
      { t: 0, gyro: [0.1, 0.2, 0.3] },
      { t: 1e5, gyro: [0.1, 0.2, 0.3] },
    ];
    const test = { meta: { ...meta, name: 'test' }, frames: tFrames, imu: tImu };
    const fs = result.images.map((x) => x.f).sort((a, b) => a - b);
    for (const pose of ['sim3', 'relative'] as const) {
      const tv = testViews({ meta, frames }, result, { views: 4, pose, test });
      expect(tv.candidates).toBe(12); // every test frame, keyframe indices and blur decisions do not apply
      expect(tv.views.map((v) => v.index)).toEqual(selectEvenly(tFrames, 4).map((f) => f.index));
      for (const v of tv.views) {
        const f = tFrames[v.index];
        expectPose(v.pose, mapPoseSim3(f.pose, sim), 1e-8); // exact data: both modes agree
        expect(v.gap).toBeNull();
        // nearest registered keyframe by GT camera centre
        const c = cameraCenter(f.pose);
        const d = keys.map((k) => Math.hypot(...cameraCenter(frames[k].pose).map((x, i) => x - c[i])));
        const best = keys[d.indexOf(Math.min(...d))];
        expect(v.nearestKeyframe).toBe(best);
        expect(v.nearest!.keyframe).toBe(best);
        expect(v.nearest!.distM).toBeCloseTo(Math.min(...d), 12);
        const im = result.images.find((x) => x.index === best)!;
        expect(v.camera.f).toBeCloseTo((((760 * im.f) / fs[fs.length >> 1]) * f.f) / frames[best].f, 9);
        expect(v.omega).toEqual([0.1, -0.2, -0.3]);
        expect(v.nu).not.toBeNull(); // central difference of the TEST frames
        expect(v.gt.f).toBe(f.f);
      }
    }
    // held-out views also report the keyframe nearest in space (a keyframe itself: distance 0, angle 0)
    const own = testViews({ meta, frames }, result, { frames: [6] }).views[0];
    expect(own.nearest).toMatchObject({ keyframe: 6 });
    expect(own.nearest!.distM).toBe(0);
    expect(own.nearest!.axisDeg).toBeCloseTo(0, 6);
    expect(nearestInSpace([], frames[0].pose)).toBeNull();
    expect(() => testViews({ meta, frames }, result, { test: { ...test, meta: { ...meta, width: 960, height: 720 } } })).toThrow(/test dataset frames/);
  });

  it('sceneMismatch: same seeded scene and camera, yard loop unchanged', () => {
    const m = { scene: 'yard', seed: 1, intrinsics: { width: 960, height: 720, f: 753, cx: 478.5, cy: 361, k1: -0.03, k2: 0.005 }, config: { textureScale: 1, anisotropy: 8, trajectory: { shake: {} } } };
    const moved = { ...m, config: { ...m.config, trajectory: { height: 2.25, lateralOffset: 1.5 } } };
    expect(sceneMismatch(m, moved)).toEqual([]);
    expect(sceneMismatch(m, { ...m, seed: 2 })).toEqual(['seed: 1 vs 2']);
    expect(sceneMismatch(m, { ...moved, config: { ...moved.config, trajectory: { semiAxes: [8, 5] } } })).toEqual(['trajectory.semiAxes: null vs [8,5]']);
    expect(sceneMismatch(m, { ...m, scene: 'object' })[0]).toMatch(/^scene/);
    expect(sceneMismatch(m, { ...m, intrinsics: { ...m.intrinsics, f: 700 } })).toEqual(['intrinsics.f: 753 vs 700']);
    // the object scene does not depend on the trajectory
    const o = { ...m, scene: 'object', config: { ...m.config, trajectory: { semiAxes: [1, 1] } } };
    expect(sceneMismatch({ ...m, scene: 'object' }, o)).toEqual([]);
  });

  it('gyro interpolation, camera-frame ω, rolling-shutter models, dense size', () => {
    const imu = [
      { t: 0, gyro: [0, 0, 0] },
      { t: 10, gyro: [1, 2, 3] },
    ];
    expect(gyroAt(imu, 5)).toEqual([0.5, 1, 1.5]);
    expect(gyroAt(imu, 11)).toBeNull();
    expect(cameraOmega(meta, imu, { tTrue: 10 })).toEqual([1, -2, -3]);
    // ν: camera moving along its own +x at 0.5 m/s, focus distance 2 m → (0.25, 0, 0) / s
    const Rc = [0, 1, 0, 0, 0, -1, -1, 0, 0]; // camera x = world y
    const fr = [0, 1, 2].map((i) => ({ index: i, file: '', tTrue: 100 * i, focusDistance: 2, pose: { R: Rc, t: mat3Vec(Rc, [0, 0.05 * i, 0]).map((v) => -v) } })); // centre (0, 0.05·i, 0)
    const nu = cameraNu(new Map(fr.map((f) => [f.index, f])), fr[1])!;
    for (let k = 0; k < 3; k++) expect(nu[k]).toBeCloseTo([0.25, 0, 0][k], 12);
    expect(cameraNu(new Map([[1, fr[1]]]), fr[1])).toBeNull();
    expect(cameraNu(new Map(fr.map((f) => [f.index, f])), { ...fr[1], focusDistance: undefined })).toBeNull();
    expect(runRollingShutter({ sparse: { stats: { rollingShutter: 'cols -21.5 ms' } } })).toEqual({ axis: 'cols', readoutMs: -21.5 });
    expect(runRollingShutter({ sparse: { stats: { rollingShutter: 'none' } } })).toBeNull();
    expect(gtRollingShutter({ captureArtefacts: { rollingShutter: { axis: 'cols', readoutMs: 22, firstLine: 'right column (u = W-1)' } } })).toEqual({ axis: 'cols', readoutMs: -22 });
    expect(gtRollingShutter({ captureArtefacts: { rollingShutter: { axis: 'rows', readoutMs: 22, firstLine: 'top row (v = 0)' } } })).toEqual({ axis: 'rows', readoutMs: 22 });
    expect(gtRollingShutter({ captureArtefacts: null })).toBeNull();
    expect(denseImageSize({ logs: [{ message: 'dense memory plan (40 views 600×800, budget 1200 MB' }] })).toBe(800);
    expect(denseImageSize({ config: { quality: 'balanced' } })).toBe(1280);
    expect(denseImageSize({})).toBeNull();
  });

  it('pinhole → three perspective: fov from fy, principal point as view offset', () => {
    const p = pinholePerspective({ width: 600, height: 800, fx: 628, fy: 628, cx: 300.4, cy: 398.9 });
    expect(Math.tan((p.fovDeg * Math.PI) / 360)).toBeCloseTo(400 / 628, 12);
    expect(p.aspect).toBeCloseTo(600 / 800, 12);
    expect(p.offsetX).toBeCloseTo(299.5 - 300.4, 12);
    expect(p.offsetY).toBeCloseTo(399.5 - 398.9, 12);
  });
});

describe('parseNvsArgs', () => {
  it('parses the documented options', () => {
    const o = parseNvsArgs(['object-real', 'out/run', '--renderers', 'points,mesh-unlit', '--views', '8', '--max-size', '480', '--pose', 'sim3', '--focal', 'shared', '--rs', 'gt', '--save-images', '--no-ssim', '--background', '11,12,16']);
    expect(o).toMatchObject({ dataset: 'object-real', runDir: 'out/run', renderers: ['points', 'mesh-unlit'], views: 8, maxSize: 480, pose: 'sim3', focal: 'shared', rs: 'gt', saveImages: true, ssim: false, background: [11, 12, 16] });
    const d = parseNvsArgs(['yard-real', 'r']);
    expect(d).toMatchObject({ renderers: null, views: 24, maxSize: 'dense', pose: 'relative', focal: 'relative', rs: 'auto', saveImages: false, ssim: true });
    expect(parseNvsArgs(['a', 'b', '--frames', '3,9']).frames).toEqual([3, 9]);
    // an empty item (Number('') is 0) or a repeated frame would change the scored view set silently
    for (const bad of ['12,37,', ',', '12,12', 'a,1', '-1,2']) expect(() => parseNvsArgs(['a', 'b', '--frames', bad])).toThrow(NvsArgError);
    expect(parseNvsArgs(['a', 'b', '--test-dataset', 'a-up'])).toMatchObject({ testDataset: 'a-up', pose: 'sim3' });
    expect(parseNvsArgs(['a', 'b', '--test-dataset', 'a-up', '--pose', 'relative']).pose).toBe('relative');
    expect(d.testDataset).toBeNull();
    expect(d.common).toBe(false);
    expect(parseNvsArgs(['a', 'b', '--common']).common).toBe(true);
    expect(parseNvsArgs(['--help']).help).toBe(true);
  });

  it('rejects bad input', () => {
    expect(() => parseNvsArgs(['a'])).toThrow(NvsArgError);
    expect(() => parseNvsArgs(['a', 'b', '--renderers', 'nerf'])).toThrow(/unknown renderer/);
    expect(() => parseNvsArgs(['a', 'b', '--views', '0'])).toThrow(/integer/);
    expect(() => parseNvsArgs(['a', 'b', '--max-size', 'big'])).toThrow(/max-size/);
    expect(() => parseNvsArgs(['a', 'b', '--background', '1,2'])).toThrow(/background/);
    expect(() => parseNvsArgs(['a', 'b', '--bogus'])).toThrow(/unknown option/);
  });
});
