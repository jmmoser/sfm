/**
 * Multi-scale dense passes (`DensePresetConfig.multiScale`): `runDensify` runs a ½-resolution
 * photometric pass, a prior-initialised full-resolution photometric pass and the geometric pass,
 * with the same output as the single-scale schedule (CPU PatchMatch, tiny synthetic object).
 */
import { describe, expect, it } from 'vitest';
import { lookAt, pinholeIntrinsics, renderSynthetic } from '../dense/synthetic';
import { PatchMatchHolder, runDensify, type DenseDeps } from './dense';
import { MemoryEngineStore } from './persistence';
import { objectScene } from './testutil';
import type { KeyframeRecord, SparseModel } from './types';

async function setup(): Promise<{ store: MemoryEngineStore; model: SparseModel }> {
  const cam = pinholeIntrinsics(96, 128, (55 * Math.PI) / 180);
  const scene = objectScene();
  const store = new MemoryEngineStore('dense-ms');
  const ids = [0, 1, 2, 3];
  const poses = ids.map((k) => {
    const a = ((-18 + 12 * k) * Math.PI) / 180;
    return lookAt([1.4 * Math.sin(a), -1.4 * Math.cos(a), 0.9], [0, 0, 0.2]);
  });
  for (const id of ids) {
    const r = renderSynthetic(scene, cam, poses[id], { supersample: 1 });
    await store.putFrame(id, r.image, { id, frameIndex: id, t: 1000 + 100 * id, width: cam.width, height: cam.height } as unknown as KeyframeRecord);
  }
  const xyz: number[] = [];
  for (let i = 0; i < 60; i++) xyz.push(0.2 * Math.cos(i), 0.2 * Math.sin(i), 0.1 + 0.004 * i);
  const n = xyz.length / 3;
  const offsets = new Uint32Array(n + 1);
  const ti: number[] = [];
  const tf: number[] = [];
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

function deps(store: MemoryEngineStore, model: SparseModel, dense: DenseDeps['config']['dense']): DenseDeps {
  return {
    store,
    depthBackend: null,
    model,
    quality: 'fast',
    mode: 'object',
    config: { dense },
    holder: new PatchMatchHolder(false),
    waitVisible: async () => undefined,
    gpuLock: (fn) => fn(),
    log: () => undefined,
    memoryBudgetMB: 1000,
  };
}

describe('runDensify: multi-scale passes', () => {
  it('½-resolution photometric → prior-initialised photometric → geometric; same cloud quality', async () => {
    const { store, model } = await setup();
    const run = async (multiScale: boolean) => {
      const messages: string[] = [];
      const out = await runDensify(deps(store, model, { multiScale, fullResIterations: 2, cpuMaxImageSize: 128 }), { maxImageSize: 96, maxViews: 4 }, (p) => {
        if (p.message) messages.push(p.message);
      });
      return { out, messages };
    };
    const single = await run(false);
    const multi = await run(true);
    expect(single.out.stats.denseMultiScale).toBeUndefined();
    expect(multi.out.stats.denseMultiScale).toBe(1);
    expect(single.messages.some((m) => m.startsWith('photometric ½'))).toBe(false);
    const last = (ms: string[], label: string): number => Number(/(\d+)\/(\d+)/.exec(ms.filter((m) => m.startsWith(label)).pop() ?? '')?.[2] ?? 0);
    // three passes over every view with sources
    expect(multi.messages.some((m) => m.startsWith('photometric ½'))).toBe(true);
    expect(last(multi.messages, 'geometric')).toBe(3 * Number(multi.out.stats.viewsWithSources));
    expect(last(single.messages, 'geometric')).toBe(2 * Number(single.out.stats.viewsWithSources));
    expect(single.out.cloud.count).toBeGreaterThan(500);
    expect(multi.out.cloud.count).toBeGreaterThan(0.8 * single.out.cloud.count);
  }, 300_000);
});
