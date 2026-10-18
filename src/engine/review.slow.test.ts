/**
 * Integration-review regressions (engine runs on synthetic captures): a raw-buffer session that
 * starts in a landscape-270 UI, resume after an R_cam_device switch with a store that clones
 * records like IndexedDB, resume of keyframes whose mapping never completed, finishCapture with
 * a frame in flight and a cancelled re-densify.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import type { Pose } from '../core/types';
import type { FeaturePipeline } from '../features/extractor';
import { umeyama } from '../math/sim3';
import { cameraFromDevice } from '../sensors/frame-orientation';
import { createCpuFeaturePipeline } from './cpu-pipeline';
import { PhotogrammetryEngine } from './engine';
import { MemoryEngineStore, type StoredImage } from './persistence';
import { centre, frameInputs, makeCapture, type SynthCapture } from './testutil';
import type { EngineConfig, KeyframeRecord, SparseResultEx } from './types';

const DEG = Math.PI / 180;

let pipeline: FeaturePipeline;
beforeAll(() => {
  pipeline = createCpuFeaturePipeline({ maxFeatures: 2048, maxImageSize: 1024 });
});

const cfg = (extra: EngineConfig = {}): EngineConfig => ({ pipeline, mapping: 'inline', ...extra });

/** MemoryEngineStore that copies records / values on write and read, as IndexedDB does. */
class CloningStore extends MemoryEngineStore {
  override async putFrame(id: number, image: StoredImage, record: KeyframeRecord): Promise<void> {
    return super.putFrame(id, image, structuredClone(record));
  }
  override async getFrameRecords(): Promise<KeyframeRecord[]> {
    return structuredClone(await super.getFrameRecords());
  }
  override async putValue(key: string, value: unknown): Promise<void> {
    return super.putValue(key, structuredClone(value));
  }
  override async getValue<T>(key: string): Promise<T | null> {
    return structuredClone(await super.getValue<T>(key));
  }
}

/** Max angle (deg) between the result's and the GT's world +Z, over all cameras. */
function gravityErrorDeg(cap: SynthCapture, res: SparseResultEx): number {
  let g = 0;
  for (const c of res.cameras) {
    const gt = cap.frames[c.frameIndex!].pose;
    const d = c.pose.R[2] * gt.R[2] + c.pose.R[5] * gt.R[5] + c.pose.R[8] * gt.R[8];
    g = Math.max(g, Math.acos(Math.min(1, d)) / DEG);
  }
  return g;
}

/** ATE (m) after a Sim3 fit of the camera centres to GT. */
function ate(cap: SynthCapture, res: SparseResultEx): number {
  const n = res.cameras.length;
  const src = new Float64Array(3 * n);
  const dst = new Float64Array(3 * n);
  res.cameras.forEach((c, i) => {
    src.set(centre(c.pose), 3 * i);
    dst.set(centre(cap.frames[c.frameIndex!].pose as Pose), 3 * i);
  });
  const s = umeyama(src, dst, n, true)!;
  let e = 0;
  for (let i = 0; i < n; i++)
    for (let r = 0; r < 3; r++) {
      const p = s.s * (s.R[3 * r] * src[3 * i] + s.R[3 * r + 1] * src[3 * i + 1] + s.R[3 * r + 2] * src[3 * i + 2]) + s.t[r];
      e += (p - dst[3 * i + r]) ** 2;
    }
  return Math.sqrt(e / n);
}

describe('raw sensor delivery is a property of the session', () => {
  it('engine: raw buffers from a landscape-270 start, then portrait — one canonical frame, θ₀ = 90, gravity-aligned refine', async () => {
    // images upright for θ = 90 (raw sensor buffers); UI at 270 for the first 1.2 s, then portrait
    const cap = makeCapture({ frames: 48, arcDeg: 60, width: 320, height: 240, theta: 90 });
    const e = new PhotogrammetryEngine({ mode: 'object', quality: 'fast', hfovPriorDeg: 60 }, cfg());
    const logs: string[] = [];
    e.on('log', (l, m) => logs.push(`${l}: ${m}`));
    await e.init();
    for (const f of [...frameInputs(cap, { screenAngle: 270, to: 12 }), ...frameInputs(cap, { screenAngle: 0, from: 12 })]) await e.processFrame(f);
    const st = e.liveSession!.state;
    expect(st.thetaSession).toBe(90);
    expect(st.cameraTheta).toBe(90);
    expect(st.rawDelivery).toBe(true);
    const recs = st.order.map((id) => st.records.get(id)!);
    expect(recs.length).toBeGreaterThan(8);
    expect(recs.some((r) => r.screenAngle === 270) && recs.some((r) => r.screenAngle === 0)).toBe(true);
    // every keyframe unrotated (raw buffers in both UI angles) with the θ = 90 mapping
    for (const r of recs) {
      expect(r.quarterTurns).toBe(0);
      expect(Array.from(r.info.R_cam_device)).toEqual(Array.from(cameraFromDevice(90)));
    }
    await e.finishCapture();
    const res = await e.refine();
    expect(res.cameras.length).toBe(st.order.length);
    expect(res.metric.gravityAligned).toBe(true);
    expect(gravityErrorDeg(cap, res)).toBeLessThan(1);
    expect(ate(cap, res)).toBeLessThan(0.01);
    e.dispose();
  }, 300_000);
});

describe('resume (store that clones records, like IndexedDB)', () => {
  it('re-applies a hand-eye R_cam_device switch to keyframes stored before it', async () => {
    // frames upright for θ = 0 reported at θ = 180: the gyro check switches the mapping to θ = 0
    const cap = makeCapture({ frames: 44, arcDeg: 55 });
    const store = new CloningStore('handeye');
    const opts = { mode: 'object' as const, quality: 'fast' as const, sessionId: 'handeye', hfovPriorDeg: 60 };
    const a = new PhotogrammetryEngine(opts, cfg({ store }));
    await a.init();
    for (const f of frameInputs(cap, { screenAngle: 180, to: 32 })) await a.processFrame(f);
    await a.flush();
    expect(a.liveSession!.state.cameraTheta).toBe(0);
    const firstId = a.liveSession!.state.order[0];
    // the stored record of the first keyframe predates the switch
    const stored = (await store.getFrameRecords()).find((r) => r.id === firstId)!;
    expect(Array.from(stored.info.R_cam_device)).toEqual(Array.from(cameraFromDevice(180)));
    a.dispose();

    const b = new PhotogrammetryEngine({ ...opts, resume: true }, cfg({ store }));
    await b.init();
    const st = b.liveSession!.state;
    expect(st.cameraTheta).toBe(0);
    for (const id of st.order) {
      const r = st.records.get(id)!;
      expect(Array.from(r.info.R_cam_device)).toEqual(Array.from(cameraFromDevice(0)));
      expect(st.db.frames.get(id)!.R_cam_device).toBe(r.info.R_cam_device);
    }
    for (const f of frameInputs(cap, { screenAngle: 180, from: 32 })) await b.processFrame(f);
    await b.finishCapture();
    const res = await b.refine();
    expect(res.metric.gravityAligned).toBe(true);
    expect(gravityErrorDeg(cap, res)).toBeLessThan(1);
    b.dispose();
  }, 300_000);

  it('matches and maps again the keyframes whose mapping never completed before the respawn', async () => {
    const cap = makeCapture({ frames: 30, arcDeg: 40 });
    const store = new CloningStore('unmapped');
    const opts = { mode: 'object' as const, quality: 'fast' as const, sessionId: 'unmapped', hfovPriorDeg: 60 };
    const a = new PhotogrammetryEngine(opts, cfg({ store }));
    await a.init();
    for (const f of frameInputs(cap)) await a.processFrame(f);
    await a.flush();
    const order = a.liveSession!.state.order.slice();
    expect(order.length).toBeGreaterThan(6);
    a.dispose();
    // the worker died while the last two keyframes were still in the mapping queue: their
    // images / features are stored, their verified pairs are not
    const lost = order.slice(-2);
    for (const id of lost) await store.putValue(`kf/${id}/pairs`, null);

    const b = new PhotogrammetryEngine({ ...opts, resume: true }, cfg({ store }));
    await b.init();
    const live = b.liveSession!;
    expect(live.state.order).toEqual(order);
    const rec = live.mapper.reconstruction;
    for (const id of lost) {
      expect(rec.images.get(id)?.registered).toBe(true);
      expect(live.state.db.hasPair(id, order[order.indexOf(id) - 1])).toBe(true);
      expect(await store.getValue(`kf/${id}/pairs`)).not.toBeNull();
    }
    expect(rec.numRegisteredImages).toBe(order.length);
    b.dispose();
  }, 300_000);
});

describe('finishCapture while a frame is being analysed', () => {
  it('the frame does not become a keyframe after the capture was finished', async () => {
    const cap = makeCapture({ frames: 24, arcDeg: 30 });
    const opts = { mode: 'object' as const, quality: 'fast' as const, hfovPriorDeg: 60 };
    // reference run: which frame becomes the 3rd keyframe
    const ref = new PhotogrammetryEngine(opts, cfg());
    await ref.init();
    const decisions: string[] = [];
    for (const f of frameInputs(cap)) decisions.push((await ref.processFrame(f)).decision);
    ref.dispose();
    const kfIdx = decisions.map((d, i) => (d === 'keyframe' ? i : -1)).filter((i) => i >= 0);
    expect(kfIdx.length).toBeGreaterThanOrEqual(3);
    const k = kfIdx[2];

    const e = new PhotogrammetryEngine(opts, cfg());
    await e.init();
    const inputs = frameInputs(cap);
    for (let i = 0; i < k; i++) await e.processFrame(inputs[i]);
    const before = e.liveSession!.state.order.length;
    // the frame is in flight (feature extraction) when finishCapture starts
    const pending = e.processFrame(inputs[k]);
    await e.finishCapture();
    const u = await pending;
    expect(u.decision).toBe('skip');
    expect(e.liveSession!.state.order.length).toBe(before);
    expect(e.liveSession!.pendingMapping).toBe(0);
    e.dispose();
  }, 300_000);
});

describe('cancelled re-densify (stored depth maps)', () => {
  it('mesh after a cancelled densify does not pair the previous views with the cleared / rewritten maps', async () => {
    const cap = makeCapture({ frames: 30, arcDeg: 40 });
    const store = new MemoryEngineStore('redense');
    // depth maps go to the session store like with IndexedDB (one shared 'depth/' prefix)
    Object.defineProperty(store, 'persistent', { value: true });
    const e = new PhotogrammetryEngine(
      { mode: 'object', quality: 'fast', sessionId: 'redense', hfovPriorDeg: 60 },
      cfg({ store, dense: { maxImageSize: 64, cpuMaxImageSize: 64, maxViews: 6, iterations: 2, geometricPasses: 0 } }),
    );
    await e.init();
    for (const f of frameInputs(cap)) await e.processFrame(f);
    await e.finishCapture();
    await e.refine();
    const first = await e.densify();
    expect(first.count).toBeGreaterThan(100);
    const mesh1 = await e.mesh();
    expect(mesh1.triangleCount).toBeGreaterThan(500);
    // a second run at another size, cancelled after its first depth map
    const token = { cancelled: false };
    await expect(
      e.densify({ maxImageSize: 80 }, (p) => {
        if (p.stage === 'depth_maps' && p.progress > 0) token.cancelled = true;
      }, token),
    ).rejects.toMatchObject({ name: 'CancelledError' });
    // the earlier cloud is still the result
    expect(e.denseCloudResult?.count ?? (await e.exportModel('ply-points')).size).toBeGreaterThan(0);
    // the mesh is rebuilt from complete, consistent maps (not from what the cancelled run left)
    const mesh = await e.mesh();
    expect(mesh.triangleCount).toBeGreaterThan(0.8 * mesh1.triangleCount);
    e.dispose();
  }, 300_000);
});
