/**
 * CPU/GPU overlap of the dense depth-map stage with REAL CPU helper workers (Chromium): the
 * workers' keyframe decode (JPEG Blob → createImageBitmap + OffscreenCanvas → undistort →
 * rolling-shutter warp → colour thumbnail) and halo filter + float16 serialisation are
 * byte-identical to the engine thread's, and `runDensify` with the pool, IndexedDB depth maps and
 * PatchMatch on the CPU and on WebGPU gives the same fusion-filtered maps and fused cloud as the
 * serial run (`denseOverlap: false`). Tiny synthetic object (`denseFixture`, 96×128 JPEGs).
 */
import { describe, expect, it } from 'vitest';
import type { PointCloud } from '../core/types';
import { removeDepthHalos } from '../dense/depth-filter';
import { patchMatchCPU } from '../dense/patchmatch-cpu';
import { serializeDepthMap } from '../io/serialize';
import { CpuPool } from './cpu-pool';
import { PatchMatchHolder, runDensify, type DenseDeps } from './dense';
import { decodeDenseView, denseColourThumbnail, type DenseViewRS } from './dense-jobs';
import { encodeJpegRGBA } from './frame-ingest';
import { MemoryEngineStore, SessionEngineStore } from './persistence';
import { denseFixture } from './testutil';
import type { KeyframeRecord, SparseModel } from './types';

const DB = 'sfm-test-dense-overlap';

const same = (a: ArrayBufferView, b: ArrayBufferView): boolean => {
  if (a.byteLength !== b.byteLength) return false;
  const x = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  const y = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
  for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
  return true;
};

function sameCloud(a: PointCloud, b: PointCloud): boolean {
  if (a.count !== b.count || !same(a.positions, b.positions) || !same(a.colors, b.colors)) return false;
  for (const k of ['normals', 'radii', 'confidence'] as const) if (!!a[k] !== !!b[k] || (a[k] && !same(a[k]!, b[k]!))) return false;
  return true;
}

async function jpegFrames(n: number): Promise<{ store: MemoryEngineStore; model: SparseModel }> {
  const { cam, images, model } = denseFixture(n);
  const store = new MemoryEngineStore('dense-overlap-gpu');
  for (let id = 0; id < n; id++) {
    const jpeg = await encodeJpegRGBA(images[id], 0.9);
    expect(jpeg).not.toBeNull();
    await store.putFrame(id, jpeg!, { id, frameIndex: id, t: 1000 + 100 * id, width: cam.width, height: cam.height } as unknown as KeyframeRecord);
  }
  // a rolling-shutter model, so the workers run the RS warp too
  const omega: Record<number, [number, number, number]> = {};
  const nu: Record<number, [number, number, number]> = {};
  for (let id = 0; id < n; id++) {
    omega[id] = [0.4 - 0.1 * id, 0.2, -0.1];
    nu[id] = [0.05, 0, 0.01 * id];
  }
  model.rollingShutter = { axis: 'rows', readoutMs: 12, omega, nu };
  return { store, model };
}

describe('dense CPU/GPU overlap with CPU helper workers', () => {
  it('decodes and halo-filters on the workers exactly like the engine thread', async () => {
    const { store, model } = await jpegFrames(3);
    const pool = CpuPool.create(2);
    expect(pool).not.toBeNull();
    try {
      const cam = model.cameras.get(1)!;
      const rs: DenseViewRS = { axis: 'rows', readoutMs: 12, omega: model.rollingShutter!.omega[1], nu: model.rollingShutter!.nu![1] };
      const jpeg = (await store.getFrameImage(1))!;
      for (const [size, r, thumbBytes] of [
        [96, rs, 0],
        [64, rs, 4 * 32 * 24],
        [96, null, 1e9],
      ] as const) {
        const here = await decodeDenseView(jpeg, cam, size, r);
        const there = await pool!.denseView(jpeg, cam, size, r, thumbBytes);
        expect(there.cam).toEqual(here.cam);
        expect([there.image.width, there.image.height]).toEqual([here.image.width, here.image.height]);
        expect(same(there.image.data, here.image.data)).toBe(true);
        if (thumbBytes > 0) {
          const t = denseColourThumbnail(here.image, thumbBytes);
          expect([there.thumb!.width, there.thumb!.height]).toEqual([t.width, t.height]);
          expect(same(there.thumb!.data, t.data)).toBe(true);
        } else expect(there.thumb).toBeNull();
      }
      expect(pool!.busyMs.denseView).toBeGreaterThan(0);

      // halo filter + serialisation of a real PatchMatch map (the caller's map is untouched)
      const views = await Promise.all(
        [0, 1, 2].map(async (id) => {
          const v = await decodeDenseView((await store.getFrameImage(id))!, cam, 96, null);
          return { id, cam: v.cam, pose: model.images[id].pose, image: v.image };
        }),
      );
      const dm = patchMatchCPU(views[1], [views[0], views[2]], { min: 0.6, max: 3 }, { iterations: 2 });
      const before = dm.depth.slice();
      const halo = { image: views[1].image, cam: views[1].cam, options: { skyUp: [0, -1, 0] as [number, number, number] } };
      const r = await pool!.haloStore(dm, halo, true);
      expect(same(dm.depth, before)).toBe(true);
      const filtered = removeDepthHalos(dm, halo.image, halo.cam, halo.options);
      expect(same(new Uint8Array(r.stored!), new Uint8Array(serializeDepthMap(filtered, { float16: true })))).toBe(true);
      let masked = 0;
      for (let i = 0; i < before.length; i++) {
        const bit = r.mask ? (r.mask[i >>> 5] >>> (i & 31)) & 1 : 0;
        expect(bit ? filtered.depth[i] === 0 : filtered.depth[i] === before[i]).toBe(true);
        masked += bit;
      }
      expect(masked).toBeGreaterThan(0);
    } finally {
      pool!.dispose();
    }
  }, 300_000);

  for (const gpu of [false, true]) {
    it(`runDensify with the pool and IndexedDB maps = the serial run (${gpu ? 'WebGPU' : 'CPU'} PatchMatch)`, async () => {
      const { store, model } = await jpegFrames(6);
      const holder = new PatchMatchHolder(gpu);
      const run = async (overlap: boolean) => {
        const id = `overlap-${gpu ? 'gpu' : 'cpu'}-${overlap ? 'on' : 'off'}-${Date.now()}`;
        const backend = await SessionEngineStore.open(id, DB);
        await backend.ensureSession({ mode: 'object', quality: 'fast' });
        const logs: string[] = [];
        const info: string[] = [];
        let pools = 0;
        try {
          const deps: DenseDeps = {
            store,
            depthBackend: backend,
            model,
            quality: 'fast',
            mode: 'object',
            config: { dense: { cpuMaxImageSize: 128 }, denseOverlap: overlap },
            holder,
            waitVisible: async () => undefined,
            gpuLock: (fn) => fn(),
            log: (level, m) => {
              if (level === 'warn' || level === 'error') logs.push(m);
              else info.push(m);
            },
            memoryBudgetMB: 1000,
            cpuWorkers: 2,
            createHelpers: (n) => {
              pools++;
              return CpuPool.create(n);
            },
          };
          const out = await runDensify(deps, { maxImageSize: 96, maxViews: 6 });
          const maps = new Map<number, ArrayBuffer>();
          for (const v of out.views) maps.set(v.id, (await backend.getBinary(`depth/${v.id}`))!);
          return { out, maps, logs, info, pools };
        } finally {
          await backend.deleteSession().catch(() => undefined);
          backend.close();
        }
      };
      const serial = await run(false);
      const overlapped = await run(true);
      expect(serial.pools).toBe(0);
      expect(overlapped.pools).toBe(1);
      expect(serial.out.stats.patchMatch).toBe(gpu ? 'gpu' : 'cpu');
      expect(overlapped.out.stats.denseHelperWorkers).toBe(2);
      expect(overlapped.logs).toEqual(serial.logs);
      // the workers did decode and filter (no fallback: no warning above)
      const line = overlapped.info.find((m) => m.startsWith('dense depth maps'))!;
      const busy = /CPU helper workers \(2\): decodes ([\d.]+) s, halo filter \+ serialise ([\d.]+) s/.exec(line);
      expect(busy, line).not.toBeNull();
      expect(serial.info.find((m) => m.startsWith('dense depth maps'))).not.toMatch(/helper/);
      expect(serial.out.cloud.count).toBeGreaterThan(300);
      expect([...overlapped.maps.keys()]).toEqual([...serial.maps.keys()]);
      for (const [id, buf] of serial.maps) expect(same(new Uint8Array(overlapped.maps.get(id)!), new Uint8Array(buf)), `depth/${id}`).toBe(true);
      expect(sameCloud(serial.out.cloud, overlapped.out.cloud)).toBe(true);
      holder.destroy();
    }, 600_000);
  }
});
