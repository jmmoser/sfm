/**
 * Orientation of the dense halo filter's sky test (`HaloFilterOptions.skyUp`): `runDensify` passes
 * each view's up direction — world +Z through the pose when the sparse model is gravity-aligned,
 * else −gravityCam of the keyframe, else nothing (upright images assumed) — and the engine reads
 * what it needs from the session (in memory or, after a reload, from the store).
 */
import { describe, expect, it, vi } from 'vitest';
import type { RGBAImage } from '../core/types';
import { worldUpInCamera } from '../dense/depth-filter';
import { lookAt, pinholeIntrinsics, renderSynthetic } from '../dense/synthetic';
import { cameraFromDevice } from '../sensors/frame-orientation';
import { PatchMatchHolder, denseSkyUp, runDensify, type DenseDeps } from './dense';
import { PhotogrammetryEngine } from './engine';
import { MemoryEngineStore } from './persistence';
import { objectScene } from './testutil';
import type { KeyframeRecord, SparseModel } from './types';

// every halo-filter call of the engine (src/dense/depth-filter `removeDepthHalos`) with its options
const haloCalls = vi.hoisted(() => [] as { skyUp: ArrayLike<number> | null }[]);
vi.mock('../dense/depth-filter', async (importOriginal) => {
  const mod = await importOriginal<typeof import('../dense/depth-filter')>();
  return {
    ...mod,
    removeDepthHalos: (...args: Parameters<typeof mod.removeDepthHalos>) => {
      haloCalls.push({ skyUp: args[3]?.skyUp ?? null });
      return mod.removeDepthHalos(...args);
    },
  };
});

async function setup(): Promise<{ store: MemoryEngineStore; model: SparseModel }> {
  const cam = pinholeIntrinsics(96, 128, (55 * Math.PI) / 180);
  const scene = objectScene();
  const store = new MemoryEngineStore('dense-sky');
  const ids = [0, 1, 2];
  // views rolled differently (one upside down): the up direction differs per view
  const poses = ids.map((k) => {
    const a = ((-15 + 12 * k) * Math.PI) / 180;
    const p = lookAt([1.4 * Math.sin(a), -1.4 * Math.cos(a), 0.9], [0, 0, 0.2]);
    if (k === 2) for (let r = 0; r < 2; r++) for (let c = 0; c < 3; c++) p.R[3 * r + c] = -p.R[3 * r + c]; // 180° roll
    if (k === 2) {
      p.t[0] = -p.t[0];
      p.t[1] = -p.t[1];
    }
    return p;
  });
  for (const id of ids) {
    const r = renderSynthetic(scene, cam, poses[id], { supersample: 1 });
    await store.putFrame(id, r.image, { id, frameIndex: id, t: 1000 + 100 * id, width: cam.width, height: cam.height } as unknown as KeyframeRecord);
  }
  // sparse points on the object, seen by every view
  const xyz: number[] = [];
  for (let i = 0; i < 60; i++) xyz.push(0.2 * Math.cos(i), 0.2 * Math.sin(i), 0.1 + 0.004 * i);
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
  };
  return { store, model };
}

function deps(store: MemoryEngineStore, model: SparseModel, extra: Partial<DenseDeps>): DenseDeps {
  return {
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
    ...extra,
  };
}

const close = (a: ArrayLike<number> | null, b: ArrayLike<number>): boolean => !!a && Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < 1e-9;

describe('denseSkyUp', () => {
  it('pose of a gravity-aligned model first, then −gravityCam (normalised), else null', () => {
    const R = lookAt([1, -1, 0.5], [0, 0, 0]).R;
    expect(close(denseSkyUp({ R, t: new Float64Array(3) }, true, [0, 9.81, 0]), worldUpInCamera(R))).toBe(true);
    expect(denseSkyUp({ R, t: new Float64Array(3) }, false, [0, 2, 0])).toEqual([-0, -1, -0]);
    // an upside-down frame: gravity points up in the image ⇒ sky at the bottom border
    expect(denseSkyUp({ R, t: new Float64Array(3) }, false, [0, -1, 0])).toEqual([-0, 1, -0]);
    expect(denseSkyUp({ R, t: new Float64Array(3) }, false, null)).toBeNull();
    expect(denseSkyUp({ R, t: new Float64Array(3) }, false, [0, 0, 0])).toBeNull();
    expect(denseSkyUp({ R, t: new Float64Array(3) }, false, [NaN, 1, 0])).toBeNull();
  });
});

describe('runDensify: sky-test orientation per view', () => {
  it('gravity-aligned model ⇒ world up through each pose; else keyframe gravity; else none', async () => {
    const { store, model } = await setup();
    const run = async (extra: Partial<DenseDeps>) => {
      haloCalls.length = 0;
      const out = await runDensify(deps(store, model, extra), { maxImageSize: 64, maxViews: 3 });
      return { out, calls: haloCalls.slice() };
    };
    // 1. gravity-aligned: R · (0, 0, 1) of each view (halo calls follow the fused view order)
    const a = await run({ gravityAligned: true, gravityCam: new Map([[0, [1, 0, 0]]]) });
    expect(a.out.stats.haloSkyUp).toBe('gravity-aligned model');
    expect(a.calls.length).toBe(a.out.meta.views.length);
    expect(a.calls.length).toBeGreaterThanOrEqual(2);
    a.out.meta.views.forEach((v, k) => expect(close(a.calls[k].skyUp, worldUpInCamera(v.pose.R))).toBe(true));
    // the upside-down view gets an up direction pointing down the image (+y)
    const k2 = a.out.meta.views.findIndex((v) => v.id === 2);
    if (k2 >= 0) expect(a.calls[k2].skyUp![1]).toBeGreaterThan(0.5);
    // 2. not gravity-aligned: −gravityCam of the keyframe
    const g = new Map<number, number[]>([
      [0, [0, 9.8, 0.5]],
      [1, [0.3, 9.7, 0]],
      [2, [0, -9.8, 0.2]],
    ]);
    const b = await run({ gravityAligned: false, gravityCam: g });
    expect(b.out.stats.haloSkyUp).toBe('keyframe gravity');
    b.out.meta.views.forEach((v, k) => {
      const gv = g.get(v.id)!;
      const n = Math.hypot(gv[0], gv[1], gv[2]);
      expect(close(b.calls[k].skyUp, [-gv[0] / n, -gv[1] / n, -gv[2] / n])).toBe(true);
    });
    // 3. neither: the filter's default (top row)
    const c = await run({});
    expect(c.out.stats.haloSkyUp).toBe('top row');
    expect(c.calls.every((x) => x.skyUp === null)).toBe(true);
  }, 120_000);
});

describe('engine: inputs of the sky-test orientation', () => {
  it('reads gravity alignment from the stored sparse meta, else the stored keyframe gravity re-rotated with the stored camera←device mapping', async () => {
    const store = new MemoryEngineStore('sky-inputs');
    const img: RGBAImage = { width: 2, height: 2, data: new Uint8ClampedArray(16) };
    const Rold = cameraFromDevice(0);
    const gDev: [number, number, number] = [0.1, -0.98, 0.15];
    for (const id of [3, 4]) {
      const rec = {
        id,
        frameIndex: id,
        t: id,
        width: 2,
        height: 2,
        // created under θ = 0, the session's mapping was switched to θ = 90 later
        info: { id, t: id, width: 2, height: 2, sharpness: 1, R_cam_device: Rold, gravityCam: Float64Array.of(1, 2, 3), heading: null, geo: null },
        gravityDevice: id === 3 ? gDev : null,
        headingAccuracy: null,
        segment: 0,
        screenAngle: 0,
        quarterTurns: 0,
      } as unknown as KeyframeRecord;
      await store.putFrame(id, img, rec);
    }
    await store.putValue('engine/state', { thetaSession: 0, cameraTheta: 90, cameraThetaVerified: true });
    const metric = { metric: true, source: 'imu', relativeUncertainty: 0.01, gravityAligned: false, northAligned: false, originLatLonAlt: null };
    await store.putValue('sparse/meta', { cameras: [], metric, stats: {} });
    const e = new PhotogrammetryEngine({ mode: 'object', quality: 'fast' });
    (e as unknown as { store: MemoryEngineStore }).store = store;
    const inputs = () => (e as unknown as { denseSkyInputs(): Promise<{ gravityAligned: boolean; gravityCam: Map<number, ArrayLike<number>> }> }).denseSkyInputs();
    const r = await inputs();
    expect(r.gravityAligned).toBe(false);
    const R90 = cameraFromDevice(90);
    const expected = [0, 1, 2].map((i) => R90[3 * i] * gDev[0] + R90[3 * i + 1] * gDev[1] + R90[3 * i + 2] * gDev[2]);
    expect(close(r.gravityCam.get(3)!, expected)).toBe(true);
    // no device gravity stored: the record's own gravityCam
    expect(Array.from(r.gravityCam.get(4)!)).toEqual([1, 2, 3]);
    await store.putValue('sparse/meta', { cameras: [], metric: { ...metric, gravityAligned: true }, stats: {} });
    const r2 = await inputs();
    expect(r2.gravityAligned).toBe(true);
    expect(r2.gravityCam.size).toBe(0);
  });
});
