/**
 * Dense stage on a rolling-shutter corrected sparse model (`SparseModel.rollingShutter`): the
 * PatchMatch views AND the colour thumbnails that fusion / TSDF sample (with the frame-stamp
 * poses) must be the corrected images — in-session (`runDensify`) and after a reload
 * (`rebuildColourViews`) alike.
 */
import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics, RGBAImage } from '../core/types';
import { lookAt, pinholeIntrinsics, renderSynthetic } from '../dense/synthetic';
import { PatchMatchHolder, denseViewImage, rebuildColourViews, runDensify } from './dense';
import { resizeRGBA } from './frame-ingest';
import { MemoryEngineStore } from './persistence';
import { objectScene } from './testutil';
import type { KeyframeRecord, SparseModel } from './types';

function sameImage(a: RGBAImage, b: RGBAImage): boolean {
  if (a.width !== b.width || a.height !== b.height) return false;
  for (let i = 0; i < a.data.length; i++) if (a.data[i] !== b.data[i]) return false;
  return true;
}

async function setup(): Promise<{ store: MemoryEngineStore; model: SparseModel; frames: Map<number, RGBAImage>; cam: CameraIntrinsics }> {
  const cam = pinholeIntrinsics(120, 160, (55 * Math.PI) / 180);
  const scene = objectScene();
  const store = new MemoryEngineStore('dense-rs');
  const ids = [0, 1, 2, 3];
  const poses = ids.map((k) => {
    const a = ((-20 + 12 * k) * Math.PI) / 180;
    return lookAt([1.4 * Math.sin(a), -1.4 * Math.cos(a), 0.9], [0, 0, 0.2]);
  });
  const frames = new Map<number, RGBAImage>();
  let depth0: Float32Array | null = null;
  for (const id of ids) {
    const r = renderSynthetic(scene, cam, poses[id], { supersample: 1 });
    if (id === 0) depth0 = r.depth;
    frames.set(id, r.image);
    const rec = { id, frameIndex: id, t: 1000 + 100 * id, width: cam.width, height: cam.height } as unknown as KeyframeRecord;
    await store.putFrame(id, r.image, rec);
  }
  // sparse points: back-projected GT depth of view 0, observed by every view
  const xyz: number[] = [];
  const P = poses[0];
  for (let v = 4; v < cam.height; v += 9)
    for (let u = 4; u < cam.width; u += 9) {
      const z = depth0![v * cam.width + u];
      if (!(z > 0)) continue;
      const xc = [((u - cam.cx) / cam.f) * z - P.t[0], ((v - cam.cy) / cam.f) * z - P.t[1], z - P.t[2]];
      // X = Rᵀ (x_c − t)
      xyz.push(P.R[0] * xc[0] + P.R[3] * xc[1] + P.R[6] * xc[2], P.R[1] * xc[0] + P.R[4] * xc[1] + P.R[7] * xc[2], P.R[2] * xc[0] + P.R[5] * xc[1] + P.R[8] * xc[2]);
    }
  const n = xyz.length / 3;
  const offsets = new Uint32Array(n + 1);
  const ti: number[] = [],
    tf: number[] = [];
  for (let i = 0; i < n; i++) {
    offsets[i] = ti.length;
    for (const id of ids) {
      ti.push(id);
      tf.push(i);
    }
  }
  offsets[n] = ti.length;
  const model: SparseModel = {
    cameras: new Map([[1, { ...cam }]]),
    images: ids.map((id) => ({ id, cameraId: 1, pose: { R: Float64Array.from(poses[id].R), t: Float64Array.from(poses[id].t) } })),
    numPoints: n,
    xyz: Float64Array.from(xyz),
    colors: new Uint8Array(3 * n).fill(128),
    errors: new Float32Array(n),
    trackOffsets: offsets,
    trackImages: Uint32Array.from(ti),
    trackFeatures: Uint32Array.from(tf),
    // a fast pan with a 30 ms readout: corrections of several pixels
    rollingShutter: {
      axis: 'rows',
      readoutMs: 30,
      omega: { 0: [0.4, 1.6, 0.2], 1: [0.3, 1.5, -0.1], 2: [-0.2, 1.7, 0.1], 3: [0.1, 1.4, 0.3] },
      nu: { 0: [0.3, 0, 0.05], 1: [0.3, 0.02, 0], 2: [0.25, 0, 0.02], 3: [0.3, -0.01, 0] },
    },
  };
  return { store, model, frames, cam };
}

describe('dense views of a rolling-shutter corrected model', () => {
  it('colour thumbnails (fusion / TSDF) are made from the corrected images, in-session and after a reload', async () => {
    const { store, model, frames, cam } = await setup();
    const out = await runDensify(
      {
        store,
        depthBackend: null,
        model,
        quality: 'fast',
        mode: 'object',
        config: {},
        holder: new PatchMatchHolder(false),
        waitVisible: async () => undefined,
        gpuLock: (fn) => fn(),
        log: () => undefined,
        memoryBudgetMB: 1000,
      },
      { maxImageSize: 96, maxViews: 4 },
    );
    expect(out.views.length).toBeGreaterThanOrEqual(2);
    const maxSize = out.meta.maxImageSize;
    for (const v of out.views) {
      const rgba = frames.get(v.id)!;
      const corrected = denseViewImage(rgba, cam, maxSize, model, v.id).image;
      const uncorrected = denseViewImage(rgba, cam, maxSize, { ...model, rollingShutter: null }, v.id).image;
      const thumb = v.image;
      // meaningful: the correction changes the thumbnail
      expect(sameImage(resizeRGBA(uncorrected, thumb.width, thumb.height), resizeRGBA(corrected, thumb.width, thumb.height))).toBe(false);
      expect(sameImage(thumb, resizeRGBA(corrected, thumb.width, thumb.height))).toBe(true);
    }
    // after a reload: the same (corrected) images
    const rebuilt = await rebuildColourViews(store, model, out.meta, maxSize);
    expect(rebuilt.length).toBe(out.views.length);
    for (const v of rebuilt) expect(sameImage(v.image, denseViewImage(frames.get(v.id)!, cam, maxSize, model, v.id).image)).toBe(true);
  }, 120_000);
});
