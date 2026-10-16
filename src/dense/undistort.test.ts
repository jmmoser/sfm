import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics, RGBAImage } from '../core/types';
import { lookAt, renderSynthetic, type SynthScene } from './synthetic';
import { areaResampleRGBA, radialFoldRadiusSq, undistortImage, undistortedCamera } from './undistort';

const log = (...a: unknown[]): void => console.log('[undistort]', ...a);

/** Normalised undistortion by Newton on r (test reference). */
function undistortNorm(k1: number, k2: number, xd: number, yd: number): [number, number] {
  const rd = Math.hypot(xd, yd);
  if (rd === 0) return [0, 0];
  let r = rd;
  for (let i = 0; i < 50; i++) {
    const r2 = r * r;
    const g = r * (1 + k1 * r2 + k2 * r2 * r2) - rd;
    const dg = 1 + 3 * k1 * r2 + 5 * k2 * r2 * r2;
    r -= g / dg;
  }
  return [(xd * r) / rd, (yd * r) / rd];
}

function bilinear(img: RGBAImage, u: number, v: number, c: number): number {
  const x0 = Math.max(0, Math.min(img.width - 1, Math.floor(u)));
  const y0 = Math.max(0, Math.min(img.height - 1, Math.floor(v)));
  const x1 = Math.min(img.width - 1, x0 + 1);
  const y1 = Math.min(img.height - 1, y0 + 1);
  const ax = Math.max(0, Math.min(1, u - x0));
  const ay = Math.max(0, Math.min(1, v - y0));
  const d = img.data;
  const at = (x: number, y: number): number => d[4 * (y * img.width + x) + c];
  return (1 - ax) * (1 - ay) * at(x0, y0) + ax * (1 - ay) * at(x1, y0) + (1 - ax) * ay * at(x0, y1) + ax * ay * at(x1, y1);
}

const scene: SynthScene = {
  primitives: [{ kind: 'quad', center: [0, 4, 0], axisU: [1, 0, 0], axisV: [0, 0, 1], halfU: 30, halfV: 30, material: { scale: 0.6, octaves: 5, contrast: 1.6 } }],
};
const pose = lookAt([0, 0, 0], [0, 1, 0]);

/** Pinhole render of the wall on a canvas `pad`× larger than cam (same f, centred). */
function wideRender(cam: CameraIntrinsics, pad: number, ss = 2): { img: RGBAImage; ox: number; oy: number } {
  const W = Math.round(cam.width * pad);
  const H = Math.round(cam.height * pad);
  const ox = (W - cam.width) / 2;
  const oy = (H - cam.height) / 2;
  const big: CameraIntrinsics = { width: W, height: H, f: cam.f, cx: cam.cx + ox, cy: cam.cy + oy, k1: 0, k2: 0 };
  return { img: renderSynthetic(scene, big, pose, { supersample: ss }).image, ox, oy };
}

/** Distorted image of the wall for a RADIAL camera (sampled from a wide pinhole render). */
function distortedImage(cam: CameraIntrinsics): RGBAImage {
  const { img: wide, ox, oy } = wideRender({ ...cam, k1: 0, k2: 0 }, 2, 2);
  const out = new Uint8ClampedArray(cam.width * cam.height * 4);
  for (let y = 0; y < cam.height; y++) {
    for (let x = 0; x < cam.width; x++) {
      const [xn, yn] = undistortNorm(cam.k1, cam.k2, (x - cam.cx) / cam.f, (y - cam.cy) / cam.f);
      const u = cam.f * xn + cam.cx + ox;
      const v = cam.f * yn + cam.cy + oy;
      for (let c = 0; c < 3; c++) out[4 * (y * cam.width + x) + c] = Math.round(bilinear(wide, u, v, c));
      out[4 * (y * cam.width + x) + 3] = 255;
    }
  }
  return { width: cam.width, height: cam.height, data: out };
}

describe('undistortImage', () => {
  it('photometric round trip (barrel distortion): matches a direct pinhole render', () => {
    const cam: CameraIntrinsics = { width: 320, height: 240, f: 260, cx: 161.3, cy: 118.6, k1: -0.12, k2: 0.03 };
    const D = distortedImage(cam);
    const t0 = performance.now();
    const { image: U, cam: oc } = undistortImage(D, cam, 320);
    const t1 = performance.now();
    expect(oc).toEqual({ ...cam, k1: 0, k2: 0 });
    const P = renderSynthetic(scene, oc, pose, { supersample: 2 }).image;
    let sum = 0;
    let n = 0;
    let transparent = 0;
    for (let i = 0; i < 320 * 240; i++) {
      if (U.data[4 * i + 3] !== 255) {
        transparent++;
        continue;
      }
      for (let c = 0; c < 3; c++) sum += Math.abs(U.data[4 * i + c] - P.data[4 * i + c]);
      n += 3;
    }
    log(`barrel 320×240: mean |Δ| ${(sum / n).toFixed(2)} levels, ${transparent} transparent px, ${(t1 - t0).toFixed(1)} ms`);
    expect(sum / n).toBeLessThan(2.5);
    // barrel: the undistorted pinhole view is fully covered by the source
    expect(transparent).toBe(0);
  });

  it('pincushion distortion: pixels mapping outside the source get alpha 0', () => {
    const cam: CameraIntrinsics = { width: 200, height: 150, f: 160, cx: 99.5, cy: 74.5, k1: 0.15, k2: 0 };
    const D = distortedImage(cam);
    const { image: U } = undistortImage(D, cam, 1000);
    let mismatch = 0;
    let outside = 0;
    for (let y = 0; y < 150; y++) {
      for (let x = 0; x < 200; x++) {
        const xn = (x - cam.cx) / cam.f;
        const yn = (y - cam.cy) / cam.f;
        const r2 = xn * xn + yn * yn;
        const d = 1 + cam.k1 * r2;
        const u = cam.f * d * xn + cam.cx;
        const v = cam.f * d * yn + cam.cy;
        const inside = u >= -0.5 && v >= -0.5 && u <= 199.5 && v <= 149.5;
        if (!inside) outside++;
        const a = U.data[4 * (y * 200 + x) + 3];
        if ((a === 255) !== inside) mismatch++;
      }
    }
    log(`pincushion: ${outside} px outside the source, ${mismatch} alpha mismatches`);
    expect(outside).toBeGreaterThan(100);
    expect(mismatch).toBe(0);
  });

  it('geometry: a dot lands at its pinhole projection (full size and downscaled)', () => {
    const cam: CameraIntrinsics = { width: 400, height: 300, f: 330, cx: 203.7, cy: 146.2, k1: -0.2, k2: 0.05 };
    // dot at a normalised (undistorted) direction near a corner
    const xn = 0.45;
    const yn = -0.31;
    const r2 = xn * xn + yn * yn;
    const dd = 1 + cam.k1 * r2 + cam.k2 * r2 * r2;
    const ud = cam.f * dd * xn + cam.cx;
    const vd = cam.f * dd * yn + cam.cy;
    const sigma = 2.2;
    const data = new Uint8ClampedArray(400 * 300 * 4);
    for (let y = 0; y < 300; y++) {
      for (let x = 0; x < 400; x++) {
        const g = 230 * Math.exp(-((x - ud) ** 2 + (y - vd) ** 2) / (2 * sigma * sigma));
        data.set([g + 10, g + 10, g + 10, 255], 4 * (y * 400 + x));
      }
    }
    const src: RGBAImage = { width: 400, height: 300, data };
    for (const maxSize of [400, 200, 133]) {
      const { image: U, cam: oc } = undistortImage(src, cam, maxSize);
      expect(oc).toEqual(undistortedCamera(cam, maxSize));
      const ex = oc.f * xn + oc.cx;
      const ey = oc.f * yn + oc.cy;
      let sw = 0;
      let sx = 0;
      let sy = 0;
      const R = Math.ceil(5 * sigma * (U.width / 400));
      for (let y = Math.floor(ey) - R; y <= Math.ceil(ey) + R; y++) {
        for (let x = Math.floor(ex) - R; x <= Math.ceil(ex) + R; x++) {
          const w = Math.max(0, U.data[4 * (y * U.width + x)] - 10);
          sw += w;
          sx += w * x;
          sy += w * y;
        }
      }
      const err = Math.hypot(sx / sw - ex, sy / sw - ey);
      log(`dot at maxSize ${maxSize} (${U.width}×${U.height}): centroid error ${err.toFixed(4)} px`);
      expect(err).toBeLessThan(0.05);
    }
  });

  it('downscaling is anti-aliased and matches a direct render at the target size', () => {
    const cam: CameraIntrinsics = { width: 480, height: 360, f: 400, cx: 239.5, cy: 179.5, k1: -0.08, k2: 0 };
    const D = distortedImage(cam);
    const t0 = performance.now();
    const { image: U, cam: oc } = undistortImage(D, cam, 160);
    const t1 = performance.now();
    expect(U.width).toBe(160);
    expect(U.height).toBe(120);
    const P = renderSynthetic(scene, oc, pose, { supersample: 4 }).image;
    let sum = 0;
    let n = 0;
    for (let i = 0; i < 160 * 120; i++) {
      if (U.data[4 * i + 3] !== 255) continue;
      for (let c = 0; c < 3; c++) sum += Math.abs(U.data[4 * i + c] - P.data[4 * i + c]);
      n += 3;
    }
    log(`480×360 → 160×120: mean |Δ| ${(sum / n).toFixed(2)} levels vs. a 4×4-supersampled render, ${(t1 - t0).toFixed(1)} ms`);
    expect(sum / n).toBeLessThan(4);
  });

  it('no distortion, no resize: exact copy; area resampling preserves the mean', () => {
    const cam: CameraIntrinsics = { width: 64, height: 48, f: 50, cx: 31.5, cy: 23.5, k1: 0, k2: 0 };
    const data = new Uint8ClampedArray(64 * 48 * 4);
    for (let i = 0; i < data.length; i++) data[i] = (i * 37) % 256;
    const src = { width: 64, height: 48, data };
    const { image } = undistortImage(src, cam, 64);
    expect(Array.from(image.data)).toEqual(Array.from(data));
    const r = areaResampleRGBA(src, 23, 17);
    let m0 = 0;
    let m1 = 0;
    for (let i = 0; i < 64 * 48; i++) m0 += data[4 * i];
    for (let i = 0; i < 23 * 17; i++) m1 += r[4 * i];
    expect(m1 / (23 * 17)).toBeCloseTo(m0 / (64 * 48), 4);
  });

  it('fold radius of the RADIAL model', () => {
    expect(radialFoldRadiusSq(0.1, 0)).toBe(Infinity);
    expect(radialFoldRadiusSq(-0.3, 0)).toBeCloseTo(1 / 0.9, 12);
    const s = radialFoldRadiusSq(-0.5, 0.05);
    expect(1 + 3 * -0.5 * s + 5 * 0.05 * s * s).toBeCloseTo(0, 10);
  });
});
