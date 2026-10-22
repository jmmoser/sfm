/**
 * CPU/GPU overlap of the dense depth-map stage (src/engine/dense.ts): with CPU helper jobs that
 * finish late and out of order (fake `DenseHelpers` running the worker code on structured-cloned
 * inputs after seeded random delays) and slow, delayed store writes, every PatchMatch pass map,
 * every fusion-filtered map and the fused cloud are bit-identical to the serial run
 * (`denseOverlap: false`); a failing helper falls back to the engine thread with the same output,
 * decoding there at most the plan's decodes at a time; a cancel leaves no write behind, and a
 * failed store write (quota) deletes no input of an unstored view: both resumes complete to the
 * same maps and cloud. CPU PatchMatch on the tiny synthetic object (`denseFixture`), persistent
 * depth backend (float16 store copies).
 */
import { describe, expect, it } from 'vitest';
import type { PointCloud, ProgressEvent, RGBAImage } from '../core/types';
import { Rng } from '../math/random';
import { PatchMatchHolder, runDensify, type DenseDeps } from './dense';
import { decodeDenseView, denseColourThumbnail, haloStoreJob, type DenseHelpers } from './dense-jobs';
import { resizeRGBA } from './frame-ingest';
import { MemoryEngineStore } from './persistence';
import { denseFixture } from './testutil';
import type { KeyframeRecord, SparseModel } from './types';

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

async function frameStore(n: number): Promise<{ store: MemoryEngineStore; model: SparseModel }> {
  const { cam, images, model } = denseFixture(n);
  const store = new MemoryEngineStore('dense-overlap');
  for (let id = 0; id < n; id++) {
    await store.putFrame(id, images[id], { id, frameIndex: id, t: 1000 + 100 * id, width: cam.width, height: cam.height } as unknown as KeyframeRecord);
  }
  return { store, model };
}

/**
 * A "persistent" in-memory depth backend (float16 store copies, resume) whose writes take
 * `writeMs` (± random; `slowMs` for some keys) and are recorded. The first write of a key in
 * `failOnce` rejects (as IndexedDB's QuotaExceededError) and stores nothing.
 */
function slowBackend(
  rng: Rng,
  writeMs: number,
  slowMs: Record<string, number> = {},
  failOnce: Iterable<string> = [],
): { backend: MemoryEngineStore; writes: Map<string, Uint8Array>; order: string[]; count: () => number } {
  const backend = new MemoryEngineStore('depth');
  Object.defineProperty(backend, 'persistent', { value: true });
  const writes = new Map<string, Uint8Array>();
  const order: string[] = [];
  const failing = new Set(failOnce);
  let n = 0;
  const put = backend.putBinary.bind(backend);
  backend.putBinary = async (key, data) => {
    const bytes = data instanceof ArrayBuffer ? new Uint8Array(data.slice(0)) : new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
    if (/^pm\d+\//.test(key)) order.push(key);
    await sleep(slowMs[key] ?? Math.floor(writeMs * (0.5 + rng.next())));
    if (failing.delete(key)) throw Object.assign(new Error(`QuotaExceededError: writing ${key}`), { name: 'QuotaExceededError' });
    n++;
    if (/^(pm\d+|depth)\//.test(key)) writes.set(key, bytes);
    return put(key, data);
  };
  return { backend, writes, order, count: () => n };
}

/** Helper jobs as the CpuPool runs them: inputs copied at the call, the same job code, seeded random latency. */
function fakeHelpers(
  rng: Rng,
  opts: { fail?: boolean; maxDelayMs?: number; haloDelaysMs?: number[] } = {},
): DenseHelpers & { calls: { denseView: number; haloStore: number }; maskedPx: number } {
  const delay = (): Promise<void> => sleep(Math.floor(rng.next() * (opts.maxDelayMs ?? 8)));
  const h = {
    size: 2,
    busyMs: { denseView: 0, haloStore: 0 },
    calls: { denseView: 0, haloStore: 0 },
    maskedPx: 0,
    async denseView(...args: Parameters<DenseHelpers['denseView']>) {
      h.calls.denseView++;
      const [stored, distorted, maxSize, rs, thumbBytes] = structuredClone(args);
      await delay();
      if (opts.fail) throw new Error('cpu worker: error');
      const v = await decodeDenseView(stored, distorted, maxSize, rs);
      const thumb = thumbBytes > 0 ? denseColourThumbnail(v.image, thumbBytes) : null;
      return structuredClone({ image: v.image, cam: v.cam, thumb });
    },
    async haloStore(...args: Parameters<DenseHelpers['haloStore']>) {
      const k = h.calls.haloStore++;
      const [map, halo, serialize] = structuredClone(args);
      await (opts.haloDelaysMs ? sleep(opts.haloDelaysMs[k % opts.haloDelaysMs.length]) : delay());
      if (opts.fail) throw new Error('cpu worker: error');
      const r = haloStoreJob(map, halo, serialize);
      if (r.mask) for (const w of r.mask) for (let b = w; b; b &= b - 1) h.maskedPx++;
      return structuredClone(r);
    },
    dispose() {},
  };
  return h;
}

function deps(store: MemoryEngineStore, backend: MemoryEngineStore, model: SparseModel, overlap: DenseHelpers | null | false, logs: string[] = []): DenseDeps {
  return {
    store,
    depthBackend: backend,
    model,
    quality: 'fast',
    mode: 'object',
    // fast preset: ½-resolution photometric → photometric → geometric, halo filter on the final maps
    config: { dense: { cpuMaxImageSize: 128 }, denseOverlap: overlap !== false },
    holder: new PatchMatchHolder(false),
    waitVisible: async () => undefined,
    gpuLock: (fn) => fn(),
    log: (level, m) => {
      if (level === 'warn') logs.push(m);
    },
    memoryBudgetMB: 1000,
    ...(overlap ? { cpuWorkers: 2, createHelpers: () => overlap } : {}),
  };
}

const opts = { maxImageSize: 96, maxViews: 6 };

function expectSameCloud(a: PointCloud, b: PointCloud): void {
  expect(b.count).toBe(a.count);
  expect(Buffer.from(b.positions.buffer, b.positions.byteOffset, b.positions.byteLength).equals(Buffer.from(a.positions.buffer, a.positions.byteOffset, a.positions.byteLength))).toBe(true);
  expect(Array.from(b.colors)).toEqual(Array.from(a.colors));
  for (const k of ['normals', 'radii', 'confidence'] as const) {
    expect(!!b[k]).toBe(!!a[k]);
    if (a[k]) expect(Array.from(b[k]!)).toEqual(Array.from(a[k]!));
  }
}

function expectSameWrites(a: Map<string, Uint8Array>, b: Map<string, Uint8Array>): void {
  expect([...b.keys()].sort()).toEqual([...a.keys()].sort());
  for (const [k, v] of a) expect(Buffer.from(b.get(k)!).equals(Buffer.from(v)), k).toBe(true);
}

describe('runDensify: CPU/GPU overlap', () => {
  it('late, out-of-order helper jobs and delayed writes give bit-identical maps and cloud; failing helpers fall back', async () => {
    const { store, model } = await frameStore(6);
    // serial reference
    const ref = slowBackend(new Rng(1), 2);
    const serial = await runDensify(deps(store, ref.backend, model, false), opts);
    expect(serial.stats.denseHelperWorkers).toBe(0);
    const passMaps = [...ref.writes.keys()].filter((k) => k.startsWith('pm'));
    // ½-res photometric, photometric, geometric × every view with sources, + the filtered maps
    expect(passMaps.length).toBe(3 * Number(serial.stats.viewsWithSources));
    expect([...ref.writes.keys()].filter((k) => k.startsWith('depth/')).length).toBe(Number(serial.stats.viewsWithSources));

    // overlap: decodes + halo filters on (fake) helpers, writes in flight; every 3rd halo job is
    // slower than the next view's PatchMatch, so the next map is ready first; the maps the next
    // pass reads second (serpentine order) are still being written when their pass ends
    const ov = slowBackend(new Rng(2), 6, { 'pm0/4': 1500, 'pm1/1': 1500 });
    const helpers = fakeHelpers(new Rng(3), { haloDelaysMs: [1500, 0, 5] });
    const logs: string[] = [];
    const overlapped = await runDensify(deps(store, ov.backend, model, helpers, logs), opts);
    expect(overlapped.stats.denseHelperWorkers).toBe(2);
    // every view decoded on a helper at ½ and full size, every final map filtered there
    expect(helpers.calls.denseView).toBe(2 * 6);
    expect(helpers.calls.haloStore).toBe(Number(serial.stats.viewsWithSources));
    expect(helpers.maskedPx).toBeGreaterThan(0);
    expect(logs).toEqual([]);
    expectSameWrites(ref.writes, ov.writes);
    // maps are put in view order, whenever their helper jobs finish (the store's resident map is
    // the serial path's)
    expect(ov.order).toEqual(ref.order);
    expectSameCloud(serial.cloud, overlapped.cloud);
    expect(overlapped.stats.fusedPoints).toBe(serial.stats.fusedPoints);
    expect(serial.cloud.count).toBeGreaterThan(500);

    // DenseOptions.overlap: false switches it off for one run (no helpers created)
    let created = 0;
    const off = await runDensify({ ...deps(store, slowBackend(new Rng(6), 1).backend, model, helpers), createHelpers: () => (created++, helpers) }, { ...opts, overlap: false });
    expect(created).toBe(0);
    expect(off.stats.denseHelperWorkers).toBe(0);
    expectSameCloud(serial.cloud, off.cloud);

    // helpers whose jobs all fail: the engine thread decodes / filters instead, same output
    const fb = slowBackend(new Rng(4), 2);
    const failing = fakeHelpers(new Rng(5), { fail: true });
    const fbLogs: string[] = [];
    const fallback = await runDensify(deps(store, fb.backend, model, failing, fbLogs), opts);
    expect(failing.calls.denseView).toBeGreaterThan(0);
    // (the maps already in flight when the first job fails were posted too)
    expect(failing.calls.haloStore).toBeGreaterThanOrEqual(1);
    expect(failing.calls.haloStore).toBeLessThanOrEqual(3);
    expect(fbLogs.some((m) => /decoding on a CPU helper worker failed/.test(m))).toBe(true);
    expect(fbLogs.some((m) => /halo filter on a CPU helper worker failed/.test(m))).toBe(true);
    expectSameWrites(ref.writes, fb.writes);
    expectSameCloud(serial.cloud, fallback.cloud);
  }, 300_000);

  it('a cancel waits for the writes in flight; the resume completes to the same cloud', async () => {
    const { store, model } = await frameStore(6);
    const ref = slowBackend(new Rng(11), 1);
    const serial = await runDensify(deps(store, ref.backend, model, false), opts);

    const ov = slowBackend(new Rng(12), 25);
    const cancel = { cancelled: false };
    const onProgress = (p: ProgressEvent): void => {
      // in the final (geometric) pass, with halo jobs and writes in flight
      if (p.stage === 'depth_maps' && p.progress > 0.75) cancel.cancelled = true;
    };
    await expect(runDensify(deps(store, ov.backend, model, fakeHelpers(new Rng(13), { maxDelayMs: 40 })), opts, onProgress, cancel)).rejects.toMatchObject({ name: 'CancelledError' });
    const written = ov.count();
    const stored = (await ov.backend.listBinaryKeys('pm2/')).length;
    expect(stored).toBeGreaterThan(0);
    await sleep(150);
    expect(ov.count()).toBe(written);
    expect((await ov.backend.listBinaryKeys('pm2/')).length).toBe(stored);

    const resumed = await runDensify(deps(store, ov.backend, model, fakeHelpers(new Rng(14))), opts);
    expect(resumed.stats.denseResumedPass).toBe(2);
    expect(resumed.stats.denseResumedMaps).toBe(stored);
    expectSameCloud(serial.cloud, resumed.cloud);
    for (const [k, v] of ref.writes) if (k.startsWith('pm2/') || k.startsWith('depth/')) expect(Buffer.from(ov.writes.get(k)!).equals(Buffer.from(v)), k).toBe(true);
  }, 300_000);

  it('a failed store write (quota) mid-pass keeps the inputs of every unstored view; the resume gives the same maps and cloud', async () => {
    const { store, model } = await frameStore(6);
    const ref = slowBackend(new Rng(21), 1);
    const serial = await runDensify(deps(store, ref.backend, model, false), opts);
    const ids = serial.views.map((v) => v.id);
    // the third map of the photometric pass (maps before it landed, later ones still to come), and
    // the last map of the final pass, its halo job slow on a helper (the view's prior and sources
    // are last read by it: the maps a resume needs to recompute it)
    const cases = [
      { pass: 1, nth: 2, helpers: fakeHelpers(new Rng(22)) },
      { pass: 2, nth: -1, helpers: fakeHelpers(new Rng(23), { haloDelaysMs: [5, 0, 400] }) },
    ];
    for (const { pass, nth, helpers } of cases) {
      const failKey = ref.order.filter((k) => k.startsWith(`pm${pass}/`)).at(nth)!;
      const ov = slowBackend(new Rng(24 + pass), 6, {}, [failKey]);
      await expect(runDensify(deps(store, ov.backend, model, helpers), opts), failKey).rejects.toMatchObject({ name: 'QuotaExceededError' });
      // every view without a map of the failed pass still has its previous-pass map (its prior)
      const stored = new Set(await ov.backend.listBinaryKeys(`pm${pass}/`));
      const prevStored = new Set(await ov.backend.listBinaryKeys(`pm${pass - 1}/`));
      expect(stored.has(failKey), failKey).toBe(false);
      for (const id of ids) if (!stored.has(`pm${pass}/${id}`)) expect(prevStored.has(`pm${pass - 1}/${id}`), `pm${pass - 1}/${id}`).toBe(true);

      const resumed = await runDensify(deps(store, ov.backend, model, fakeHelpers(new Rng(30 + pass))), opts);
      expect(resumed.stats.denseResumedPass).toBe(pass);
      expect(resumed.stats.denseResumedMaps).toBe(stored.size);
      expectSameCloud(serial.cloud, resumed.cloud);
      // the failed pass's maps (stored before the failure or by the resume), later passes', the filtered maps
      const fromPass = (k: string): boolean => k.startsWith('depth/') || Number(/^pm(\d+)\//.exec(k)?.[1] ?? -1) >= pass;
      for (const [k, v] of ref.writes) if (fromPass(k)) expect(Buffer.from(ov.writes.get(k)!).equals(Buffer.from(v)), k).toBe(true);
    }
  }, 300_000);

  it("after a failed helper decode, the engine thread decodes at most the plan's decodes at a time", async () => {
    const { cam, images, model } = denseFixture(6);
    // keyframes stored as Blobs (raw RGBA behind a size header, standing in for JPEGs) and a
    // createImageBitmap + OffscreenCanvas stand-in that counts the decoded bitmaps alive at once
    const store = new MemoryEngineStore('dense-overlap-blobs');
    for (let id = 0; id < images.length; id++) {
      const im = images[id];
      const record = { id, frameIndex: id, t: 1000 + 100 * id, width: cam.width, height: cam.height } as unknown as KeyframeRecord;
      await store.putFrame(id, new Blob([new Uint32Array([im.width, im.height]), new Uint8Array(im.data)]), record);
    }
    let live = 0;
    let maxLive = 0;
    const g = globalThis as Record<string, unknown>;
    const saved = { createImageBitmap: g.createImageBitmap, OffscreenCanvas: g.OffscreenCanvas };
    g.createImageBitmap = async (blob: Blob) => {
      const buf = await blob.arrayBuffer();
      const [width, height] = new Uint32Array(buf, 0, 2);
      maxLive = Math.max(maxLive, ++live);
      await sleep(4);
      return { width, height, rgba: { width, height, data: new Uint8ClampedArray(buf, 8) }, close: () => void live-- };
    };
    g.OffscreenCanvas = class {
      private img: RGBAImage | null = null;
      constructor(
        public width: number,
        public height: number,
      ) {}
      getContext() {
        return {
          drawImage: (bmp: { rgba: RGBAImage }, _x: number, _y: number, w: number, h: number) => void (this.img = resizeRGBA(bmp.rgba, w, h)),
          getImageData: () => ({ width: this.img!.width, height: this.img!.height, data: this.img!.data }),
        };
      }
    };
    try {
      const serial = await runDensify(deps(store, slowBackend(new Rng(41), 1).backend, model, false), opts);
      expect(maxLive).toBe(1);
      maxLive = 0;
      const failing = fakeHelpers(new Rng(42), { fail: true, maxDelayMs: 20 });
      const logs: string[] = [];
      const d = deps(store, slowBackend(new Rng(43), 1).backend, model, failing);
      d.log = (_level, m) => void logs.push(m);
      const fallback = await runDensify(d, opts);
      const decodes = Number(/(\d+) decodes in flight/.exec(logs.join('\n'))?.[1]);
      expect(decodes).toBeGreaterThanOrEqual(1);
      expect(failing.calls.denseView).toBeGreaterThan(0);
      expect(logs.some((m) => /decoding on a CPU helper worker failed/.test(m))).toBe(true);
      // (a view's 1 + sources images were requested at once when the first job failed)
      expect(maxLive).toBeGreaterThanOrEqual(1);
      expect(maxLive).toBeLessThanOrEqual(decodes);
      expectSameCloud(serial.cloud, fallback.cloud);
    } finally {
      for (const [k, v] of Object.entries(saved)) {
        if (v === undefined) delete g[k];
        else g[k] = v;
      }
    }
  }, 300_000);
});
