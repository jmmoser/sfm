import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { getGPU, type GPUContext } from '../gpu/device';
import { PatchMatchGPU } from './patchmatch-gpu';
import { patchMatchCPU, lastPatchMatchCPUStats } from './patchmatch-cpu';
import { evaluateDepthMap, formatDepthStats, makeSyntheticDataset, surfaceMask } from './synthetic';
import type { DepthMap } from './types';

const log = (...a: unknown[]): void => console.log('[patchmatch-gpu]', ...a);

let ctx: GPUContext | null = null;
let pm: PatchMatchGPU;

beforeAll(async () => {
  ctx = await getGPU();
  if (!ctx) throw new Error('WebGPU unavailable in the test browser');
  const t0 = performance.now();
  pm = await PatchMatchGPU.create(ctx);
  log(`adapter ${ctx.info.vendor}/${ctx.info.architecture} software=${ctx.isSoftware}; create (pipeline compile) ${(performance.now() - t0).toFixed(0)} ms`);
}, 600_000); // pipeline compilation on SwiftShader can exceed the 30 s default hook timeout while other GPU files run

afterAll(() => {
  pm?.destroy();
});

function fmtTimings(): string {
  const t = pm.lastTimings!;
  return `total ${t.totalMs.toFixed(0)} ms (upload ${t.uploadMs.toFixed(0)}, init ${t.initMs.toFixed(0)}, iters [${t.iterationsMs
    .map((v) => v.toFixed(0))
    .join(', ')}], readback ${t.readbackMs.toFixed(0)}; ${t.submits} submits, band ${t.bandRows} rows)`;
}

describe('PatchMatchGPU', () => {
  it('self-test agrees with the CPU reference', async () => {
    const r = await pm.selfTest();
    log(`selfTest: ${JSON.stringify(r)}`);
    expect(r.ok).toBe(true);
  });

  it('near scene 256×192: accurate, and statistically equal to the CPU reference', async () => {
    const W = 256;
    const H = 192;
    const ds = makeSyntheticDataset('near', { width: W, height: H });
    const dm = await pm.compute(ds.views[0], ds.views.slice(1), ds.range);
    log(`near ${W}x${H} GPU: ${fmtTimings()}`);
    const t0 = performance.now();
    const dc = patchMatchCPU(ds.views[0], ds.views.slice(1), ds.range);
    log(`near ${W}x${H} CPU: ${(performance.now() - t0).toFixed(0)} ms`);
    const mask = surfaceMask(ds.views[0].cam, ds.gtDepth[0], ds.gtNormal[0]);
    const g = evaluateDepthMap(dm, ds.gtDepth[0], ds.gtNormal[0], { mask });
    const c = evaluateDepthMap(dc, ds.gtDepth[0], ds.gtNormal[0], { mask });
    const gAll = evaluateDepthMap(dm, ds.gtDepth[0], ds.gtNormal[0]);
    const cAll = evaluateDepthMap(dc, ds.gtDepth[0], ds.gtNormal[0]);
    log(`  GPU mask: ${formatDepthStats(g)}`);
    log(`  CPU mask: ${formatDepthStats(c)}`);
    log(`  GPU all : ${formatDepthStats(gAll)}`);
    log(`  CPU all : ${formatDepthStats(cAll)}`);
    let same = 0;
    let n = 0;
    for (let i = 0; i < dm.depth.length; i++) {
      if (!(ds.gtDepth[0][i] > 0)) continue;
      n++;
      if (Math.abs(dm.depth[i] - dc.depth[i]) <= 1e-4 * Math.max(dc.depth[i], 1e-6)) same++;
    }
    log(`  pixels with identical GPU/CPU depth (1e-4 rel): ${((100 * same) / n).toFixed(1)}%`);
    expect(g.validFraction).toBeGreaterThanOrEqual(0.7);
    expect(g.accurateFraction).toBeGreaterThanOrEqual(0.85);
    expect(g.medianNormalErrorDeg).toBeLessThan(10);
    expect(Math.abs(g.validFraction - c.validFraction)).toBeLessThan(0.03);
    expect(Math.abs(g.accurateFraction - c.accurateFraction)).toBeLessThan(0.03);
    expect(Math.abs(gAll.accurateFraction - cAll.accurateFraction)).toBeLessThan(0.03);
    expect(Math.abs(g.medianNormalErrorDeg - c.medianNormalErrorDeg)).toBeLessThan(1.5);
    // well-formed output
    for (let i = 0; i < dm.depth.length; i++) {
      if (dm.depth[i] > 0) {
        expect(Math.abs(Math.hypot(dm.normal[3 * i], dm.normal[3 * i + 1], dm.normal[3 * i + 2]) - 1)).toBeLessThan(1e-4);
        expect(dm.cost[i]).toBeLessThanOrEqual(0.5);
      }
    }
  });

  it('affine 13 px window (preset kernel) on the GPU is statistically equal to the CPU reference', async () => {
    const W = 256;
    const H = 192;
    const ds = makeSyntheticDataset('near', { width: W, height: H });
    const opts = { patchRadius: 6, patchStep: 3, affineWindow: true, numSamplesPerIter: 3 };
    const dm = await pm.compute(ds.views[0], ds.views.slice(1), ds.range, opts);
    log(`near ${W}x${H} GPU affine r6/s3/3 proposals: ${fmtTimings()}`);
    const dc = patchMatchCPU(ds.views[0], ds.views.slice(1), ds.range, opts);
    const mask = surfaceMask(ds.views[0].cam, ds.gtDepth[0], ds.gtNormal[0]);
    const g = evaluateDepthMap(dm, ds.gtDepth[0], ds.gtNormal[0], { mask });
    const c = evaluateDepthMap(dc, ds.gtDepth[0], ds.gtNormal[0], { mask });
    log(`  GPU mask: ${formatDepthStats(g)}`);
    log(`  CPU mask: ${formatDepthStats(c)}`);
    expect(g.validFraction).toBeGreaterThanOrEqual(0.7);
    expect(g.accurateFraction).toBeGreaterThanOrEqual(0.8);
    expect(Math.abs(g.validFraction - c.validFraction)).toBeLessThan(0.03);
    expect(Math.abs(g.accurateFraction - c.accurateFraction)).toBeLessThan(0.03);
    expect(Math.abs(g.medianNormalErrorDeg - c.medianNormalErrorDeg)).toBeLessThan(1.5);
  });

  it('far scene (5–60 m) 256×192 works with inverse-depth sampling', async () => {
    const ds = makeSyntheticDataset('far', { width: 256, height: 192 });
    const dm = await pm.compute(ds.views[0], ds.views.slice(1), ds.range);
    log(`far 256x192 GPU: ${fmtTimings()}`);
    const mask = surfaceMask(ds.views[0].cam, ds.gtDepth[0], ds.gtNormal[0]);
    const s1 = evaluateDepthMap(dm, ds.gtDepth[0], ds.gtNormal[0], { mask });
    const s2 = evaluateDepthMap(dm, ds.gtDepth[0], ds.gtNormal[0], { mask, relTol: 0.02 });
    log(`  all : ${formatDepthStats(evaluateDepthMap(dm, ds.gtDepth[0], ds.gtNormal[0]))}`);
    log(`  mask: ${formatDepthStats(s1)} | within 2%: ${(100 * s2.accurateFraction).toFixed(1)}%`);
    expect(s1.validFraction).toBeGreaterThanOrEqual(0.7);
    expect(s2.accurateFraction).toBeGreaterThanOrEqual(0.85);
    expect(s1.medianRelError).toBeLessThan(0.01);
    expect(s1.medianNormalErrorDeg).toBeLessThan(10);
  });

  it('geometric-consistency pass reduces the error (all views on the GPU)', async () => {
    const W = 192;
    const H = 144;
    const ds = makeSyntheticDataset('near', { width: W, height: H });
    const others = (i: number) => ds.views.filter((_, j) => j !== i);
    const t0 = performance.now();
    const photo: DepthMap[] = [];
    for (let i = 0; i < ds.views.length; i++) photo.push(await pm.compute(ds.views[i], others(i), ds.range));
    const t1 = performance.now();
    const geo = await pm.compute(ds.views[0], others(0), ds.range, {}, { prior: photo[0], sourceDepths: photo.slice(1) });
    log(`geometric ${W}x${H}: photometric ×${ds.views.length} ${(t1 - t0).toFixed(0)} ms; geometric ${fmtTimings()}`);
    // CPU geometric on the same inputs, for comparison
    const gc = patchMatchCPU(ds.views[0], others(0), ds.range, {}, { prior: photo[0], sourceDepths: photo.slice(1) });
    log(`  CPU geometric: ${lastPatchMatchCPUStats().timeMs.toFixed(0)} ms`);
    const mask = surfaceMask(ds.views[0].cam, ds.gtDepth[0], ds.gtNormal[0]);
    const pAll = evaluateDepthMap(photo[0], ds.gtDepth[0], ds.gtNormal[0]);
    const gAll = evaluateDepthMap(geo, ds.gtDepth[0], ds.gtNormal[0]);
    const cAll = evaluateDepthMap(gc, ds.gtDepth[0], ds.gtNormal[0]);
    const pS = evaluateDepthMap(photo[0], ds.gtDepth[0], ds.gtNormal[0], { mask });
    const gS = evaluateDepthMap(geo, ds.gtDepth[0], ds.gtNormal[0], { mask });
    log(`  photometric all : ${formatDepthStats(pAll)}`);
    log(`  geometric   all : ${formatDepthStats(gAll)}`);
    log(`  CPU geom.   all : ${formatDepthStats(cAll)}`);
    log(`  photometric mask: ${formatDepthStats(pS)}`);
    log(`  geometric   mask: ${formatDepthStats(gS)}`);
    expect(gAll.completeness).toBeGreaterThan(pAll.completeness + 0.01);
    expect(gAll.meanRelError).toBeLessThan(pAll.meanRelError);
    expect(gS.accurateFraction).toBeGreaterThanOrEqual(pS.accurateFraction - 0.005);
    expect(Math.abs(gAll.accurateFraction - cAll.accurateFraction)).toBeLessThan(0.03);
  });

  it('row-band tiling does not change the result (bit-identical)', async () => {
    const ds = makeSyntheticDataset('near', { width: 96, height: 72, numSources: 3 });
    const a = await pm.compute(ds.views[0], ds.views.slice(1), ds.range, { iterations: 2 });
    const ta = pm.lastTimings!;
    const b = await pm.compute(ds.views[0], ds.views.slice(1), ds.range, { iterations: 2, maxPixelViewsPerSubmit: 2000 });
    const tb = pm.lastTimings!;
    log(`banding: ${ta.submits} submits (band ${ta.bandRows}) vs ${tb.submits} submits (band ${tb.bandRows})`);
    expect(tb.submits).toBeGreaterThan(ta.submits);
    expect(Array.from(b.depth)).toEqual(Array.from(a.depth));
    expect(Array.from(b.cost)).toEqual(Array.from(a.cost));
    // adaptive band controller (normally disabled on software adapters): same result
    PatchMatchGPU.adaptiveOnSoftware = true;
    let e: DepthMap;
    try {
      e = await pm.compute(ds.views[0], ds.views.slice(1), ds.range, { iterations: 2, targetSubmitMs: 5 });
    } finally {
      PatchMatchGPU.adaptiveOnSoftware = false;
    }
    const te = pm.lastTimings!;
    log(`adaptive bands (target 5 ms): ${te.submits} submits, last band ${te.bandRows} rows, max submit ${te.maxSubmitMs.toFixed(1)} ms`);
    expect(te.submits).toBeGreaterThan(ta.submits);
    expect(Array.from(e.depth)).toEqual(Array.from(a.depth));
    // chunked readback through a tiny staging buffer gives the same output (fresh
    // instances on both sides: texture-array sizes affect sampling at the ulp level)
    const saved = PatchMatchGPU.stagingBytes;
    PatchMatchGPU.stagingBytes = 4096;
    const pm2 = await PatchMatchGPU.create(ctx!);
    PatchMatchGPU.stagingBytes = saved;
    const pm3 = await PatchMatchGPU.create(ctx!);
    try {
      const c = await pm2.compute(ds.views[0], ds.views.slice(1), ds.range, { iterations: 2 });
      const d = await pm3.compute(ds.views[0], ds.views.slice(1), ds.range, { iterations: 2 });
      expect(Array.from(c.depth)).toEqual(Array.from(d.depth));
      expect(Array.from(c.normal)).toEqual(Array.from(d.normal));
      expect(Array.from(c.cost)).toEqual(Array.from(d.cost));
    } finally {
      pm2.destroy();
      pm3.destroy();
    }
  });

  it('upload caches: a repeated source image / depth map is not uploaded again, results are unchanged', async () => {
    const ds = makeSyntheticDataset('near', { width: 96, height: 72, numSources: 4, supersample: 1 });
    const fresh = await PatchMatchGPU.create(ctx!);
    try {
      const a = await fresh.compute(ds.views[0], ds.views.slice(1), ds.range, { iterations: 2 });
      const s0 = fresh.cacheStats;
      expect(s0.imageMisses).toBe(4);
      // the same sources again (another order) + the reference as a source: 4 hits, 1 upload
      const b = await fresh.compute(ds.views[1], [ds.views[0], ds.views[4], ds.views[3], ds.views[2]], ds.range, { iterations: 2 });
      const s1 = fresh.cacheStats;
      expect(s1.imageHits - s0.imageHits).toBe(3);
      expect(s1.imageMisses - s0.imageMisses).toBe(1);
      expect(b.depth.some((z) => z > 0)).toBe(true);
      // a geometric pass twice with the same depth maps: the second one uploads none of them
      const geo = { prior: a, sourceDepths: [a, a, a, a].map((m) => ({ ...m, depth: m.depth.slice() })) };
      const g1 = await fresh.compute(ds.views[0], ds.views.slice(1), ds.range, { geometricIterations: 1 }, geo);
      const s2 = fresh.cacheStats;
      const g2 = await fresh.compute(ds.views[0], ds.views.slice(1), ds.range, { geometricIterations: 1 }, geo);
      const s3 = fresh.cacheStats;
      expect(s3.depthMisses).toBe(s2.depthMisses);
      expect(s3.depthHits - s2.depthHits).toBe(4);
      expect(Array.from(g2.depth)).toEqual(Array.from(g1.depth));
      // the first result again from cached sources equals a fresh instance's
      const again = await fresh.compute(ds.views[0], ds.views.slice(1), ds.range, { iterations: 2 });
      expect(Array.from(again.depth)).toEqual(Array.from(a.depth));
    } finally {
      fresh.destroy();
    }
  });

  /**
   * Simulated dropped work (iOS discards queue work while the page is hidden): `hidden(true)`
   * drops source uploads (writeTexture into the 'pm-src' cache) and every command buffer until
   * the first one was dropped; `dropReadbacks` drops that many 'pm-readback' command buffers.
   */
  const withDroppedWork = async <T>(fn: (sim: { hidden: boolean; dropReadbacks: number; dropped: number }) => Promise<T>): Promise<T> => {
    const device = ctx!.device;
    const q = device.queue;
    const sim = { hidden: false, dropReadbacks: 0, dropped: 0 };
    const drop = new WeakSet<GPUCommandBuffer>();
    const origSubmit = q.submit;
    const origWriteTexture = q.writeTexture;
    const origEncoder = device.createCommandEncoder;
    q.submit = (bufs: Iterable<GPUCommandBuffer>): undefined => {
      const keep = [...bufs].filter((b) => !drop.has(b));
      if (sim.hidden) {
        sim.hidden = false; // visible again after this submission
        sim.dropped++;
        return undefined;
      }
      sim.dropped += [...bufs].length - keep.length;
      return origSubmit.call(q, keep);
    };
    q.writeTexture = ((dst: GPUTexelCopyTextureInfo, ...rest: unknown[]) => {
      if (sim.hidden && dst.texture.label === 'pm-src') return undefined;
      return (origWriteTexture as (...a: unknown[]) => undefined).call(q, dst, ...rest);
    }) as typeof q.writeTexture;
    device.createCommandEncoder = (d?: GPUCommandEncoderDescriptor): GPUCommandEncoder => {
      const enc = origEncoder.call(device, d);
      if (d?.label === 'pm-readback' && sim.dropReadbacks > 0) {
        sim.dropReadbacks--;
        const finish = enc.finish.bind(enc);
        enc.finish = (fd?: GPUCommandBufferDescriptor): GPUCommandBuffer => {
          const cb = finish(fd);
          drop.add(cb);
          return cb;
        };
      }
      return enc;
    };
    try {
      return await fn(sim);
    } finally {
      delete (q as unknown as Record<string, unknown>).submit;
      delete (q as unknown as Record<string, unknown>).writeTexture;
      delete (device as unknown as Record<string, unknown>).createCommandEncoder;
    }
  };

  it('a dropped source upload is redone by the retry (upload caches are invalidated)', async () => {
    const ds = makeSyntheticDataset('near', { width: 96, height: 72, numSources: 4, supersample: 1 });
    const opts = { iterations: 2, targetSubmitMs: 0 };
    const ref = await PatchMatchGPU.create(ctx!);
    const sim = await PatchMatchGPU.create(ctx!);
    const warn = console.warn;
    console.warn = () => undefined;
    try {
      const want = await ref.compute(ds.views[0], ds.views.slice(1), ds.range, opts);
      const got = await withDroppedWork(async (s) => {
        s.hidden = true;
        const out = await sim.compute(ds.views[0], ds.views.slice(1), ds.range, opts);
        expect(s.dropped).toBe(1);
        return out;
      });
      expect(Array.from(got.depth)).toEqual(Array.from(want.depth));
      expect(Array.from(got.cost)).toEqual(Array.from(want.cost));
      expect(sim.lastTimings!.retries).toBe(1);
      expect(sim.cacheStats.imageMisses).toBe(8);
    } finally {
      console.warn = warn;
      ref.destroy();
      sim.destroy();
    }
  });

  it('a dropped readback never returns stale staging contents (previous call or chunk)', async () => {
    const ds = makeSyntheticDataset('near', { width: 96, height: 72, numSources: 4, supersample: 1 });
    // fixed bands: both references take the same number of submissions (equal sentinels)
    const opts = { iterations: 2, targetSubmitMs: 0 };
    const srcB = [ds.views[0], ds.views[2], ds.views[3], ds.views[4]];
    const saved = PatchMatchGPU.stagingBytes;
    const warn = console.warn;
    console.warn = () => undefined;
    try {
      for (const stagingBytes of [saved, 4096]) {
        PatchMatchGPU.stagingBytes = stagingBytes; // 4096: several row chunks per readback
        const ref = await PatchMatchGPU.create(ctx!);
        const sim = await PatchMatchGPU.create(ctx!);
        try {
          await ref.compute(ds.views[0], ds.views.slice(1), ds.range, opts);
          const want = await ref.compute(ds.views[1], srcB, ds.range, opts);
          const a = await sim.compute(ds.views[0], ds.views.slice(1), ds.range, opts);
          expect(Array.from(want.depth)).not.toEqual(Array.from(a.depth));
          // the first readback chunk of B dropped: the staging buffer still holds A's last chunk
          const got = await withDroppedWork(async (s) => {
            s.dropReadbacks = 1;
            const out = await sim.compute(ds.views[1], srcB, ds.range, opts);
            expect(s.dropped).toBe(1);
            return out;
          });
          expect(Array.from(got.depth)).toEqual(Array.from(want.depth));
          expect(Array.from(got.cost)).toEqual(Array.from(want.cost));
          expect(sim.lastTimings!.retries).toBe(1);
          if (stagingBytes !== saved) {
            // a later chunk dropped: the staging buffer holds the previous chunk of this call
            const got2 = await withDroppedWork(async (s) => {
              const enc = ctx!.device.createCommandEncoder;
              let seen = 0;
              ctx!.device.createCommandEncoder = (d?: GPUCommandEncoderDescriptor): GPUCommandEncoder => {
                if (d?.label === 'pm-readback' && ++seen === 2) s.dropReadbacks = 1;
                return enc.call(ctx!.device, d);
              };
              const out = await sim.compute(ds.views[1], srcB, ds.range, opts);
              expect(s.dropped).toBe(1);
              return out;
            });
            expect(Array.from(got2.depth)).toEqual(Array.from(want.depth));
            expect(sim.lastTimings!.retries).toBe(1);
          }
        } finally {
          ref.destroy();
          sim.destroy();
        }
      }
    } finally {
      PatchMatchGPU.stagingBytes = saved;
      console.warn = warn;
    }
  });

  it('supports 16 source views, truncates more, and reuses / reallocates resources across sizes', async () => {
    const ds = makeSyntheticDataset('near', { width: 80, height: 60, numSources: 17 });
    const warn = console.warn;
    let warned = 0;
    console.warn = () => {
      warned++;
    };
    let d17: DepthMap;
    try {
      d17 = await pm.compute(ds.views[0], ds.views.slice(1), ds.range, { iterations: 3 });
    } finally {
      console.warn = warn;
    }
    expect(warned).toBe(1);
    const d16 = await pm.compute(ds.views[0], ds.views.slice(1, 17), ds.range, { iterations: 3 });
    expect(Array.from(d17.depth)).toEqual(Array.from(d16.depth));
    const s = evaluateDepthMap(d16, ds.gtDepth[0], ds.gtNormal[0], { relTol: 0.02 });
    log(`16 sources 80x60: ${formatDepthStats(s, 0.02)}; ${fmtTimings()}`);
    expect(s.validFraction).toBeGreaterThan(0.7);
    expect(s.medianRelError).toBeLessThan(0.02);
    // different sizes back to back
    const small = makeSyntheticDataset('near', { width: 64, height: 48, numSources: 2 });
    const e = await pm.compute(small.views[0], small.views.slice(1), small.range, { iterations: 1 });
    expect(e.width).toBe(64);
    expect(e.depth.length).toBe(64 * 48);
    const empty = await pm.compute(small.views[0], [], small.range);
    expect(empty.depth.every((z) => z === 0)).toBe(true);
    await expect(pm.compute(small.views[0], small.views.slice(1), { min: 1, max: 0.5 })).rejects.toThrow();
    // a destroyed instance refuses work
    const other = await PatchMatchGPU.create(ctx!);
    other.destroy();
    await expect(other.compute(small.views[0], small.views.slice(1), small.range)).rejects.toThrow(/destroyed/);
  });

  it('timing at 640×480 with 8 sources (2 iterations; SwiftShader is a CPU emulation)', async () => {
    const ds = makeSyntheticDataset('near', { width: 640, height: 480, numSources: 8, supersample: 1 });
    const dm = await pm.compute(ds.views[0], ds.views.slice(1), ds.range, { iterations: 2 });
    const t = pm.lastTimings!;
    const mpv = (640 * 480 * 8) / 1e6;
    const perIter = t.iterationsMs.reduce((a, b) => a + b, 0) / t.iterationsMs.length;
    log(`640x480, 8 sources, 2 iterations: ${fmtTimings()}; ${(mpv / (perIter / 1000)).toFixed(2)} Mpixel·views/s per iteration`);
    const s = evaluateDepthMap(dm, ds.gtDepth[0], ds.gtNormal[0]);
    log(`  quality after 2 iterations: ${formatDepthStats(s)}`);
    expect(s.validFraction).toBeGreaterThan(0.7);
    expect(s.medianRelError).toBeLessThan(0.01);
  });
});
