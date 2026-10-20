/**
 * PatchMatchGPU's buffers (~140 MB at 1600 px) are freed when the depth maps end, whether
 * `runDensify` finishes them or leaves them on a cancel / failure (src/engine/dense.ts).
 */
import { describe, expect, it, vi } from 'vitest';
import { patchMatchCPU } from '../dense/patchmatch-cpu';
import type { PatchMatchGPU } from '../dense/patchmatch-gpu';
import { lookAt, pinholeIntrinsics, renderSynthetic } from '../dense/synthetic';
import { runDensify, type DenseDeps, type PatchMatchHolder } from './dense';
import { MemoryEngineStore } from './persistence';
import { objectScene } from './testutil';
import type { KeyframeRecord, SparseModel } from './types';

async function setup(): Promise<{ store: MemoryEngineStore; model: SparseModel }> {
  const cam = pinholeIntrinsics(96, 128, (55 * Math.PI) / 180);
  const scene = objectScene();
  const store = new MemoryEngineStore('dense-cancel');
  const ids = [0, 1, 2];
  const poses = ids.map((k) => {
    const a = ((-15 + 12 * k) * Math.PI) / 180;
    return lookAt([1.4 * Math.sin(a), -1.4 * Math.cos(a), 0.9], [0, 0, 0.2]);
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

/** A holder whose "GPU" PatchMatch runs the CPU one; `onCompute` runs after each depth map. */
function fakeHolder(onCompute: () => void): { holder: PatchMatchHolder; reset: ReturnType<typeof vi.fn> } {
  const pm = {
    compute: async (...args: Parameters<typeof patchMatchCPU>) => {
      const dm = patchMatchCPU(...args);
      onCompute();
      return dm;
    },
  } as unknown as PatchMatchGPU;
  const reset = vi.fn();
  const holder = { reason: null, get: async () => pm, reset, destroy() {} } as unknown as PatchMatchHolder;
  return { holder, reset };
}

function deps(store: MemoryEngineStore, model: SparseModel, holder: PatchMatchHolder): DenseDeps {
  return {
    store,
    depthBackend: null,
    model,
    quality: 'fast',
    mode: 'object',
    config: {},
    holder,
    waitVisible: async () => undefined,
    gpuLock: (fn) => fn(),
    log: () => undefined,
    memoryBudgetMB: 1000,
  };
}

describe('runDensify: PatchMatchGPU release', () => {
  it('resets the holder after the depth maps, also when cancelled or failing in between', async () => {
    const { store, model } = await setup();
    const opts = { maxImageSize: 64, maxViews: 3 };
    // finished: released once
    const ok = fakeHolder(() => undefined);
    await runDensify(deps(store, model, ok.holder), opts);
    expect(ok.reset).toHaveBeenCalledTimes(1);
    // cancelled after the first depth map
    const cancel = { cancelled: false };
    const cancelled = fakeHolder(() => (cancel.cancelled = true));
    await expect(runDensify(deps(store, model, cancelled.holder), opts, undefined, cancel)).rejects.toMatchObject({ name: 'CancelledError' });
    expect(cancelled.reset).toHaveBeenCalledTimes(1);
    // a GPU error that is not a device loss
    let n = 0;
    const failing = fakeHolder(() => {
      if (++n === 1) throw new Error('compute failed');
    });
    await expect(runDensify(deps(store, model, failing.holder), opts)).rejects.toThrow(/compute failed/);
    expect(failing.reset).toHaveBeenCalledTimes(1);
  }, 120_000);
});
