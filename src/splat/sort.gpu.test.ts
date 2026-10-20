import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { getGPU, type GPUContext } from '../gpu/device';
import { pipelineCompileLimits } from '../features/gpu-common';
import { Rng } from '../math/random';
import { RadixSorter, compileSortPipelines, type SortPipelines } from './gpu-util';

let ctx: GPUContext;
let pipes: SortPipelines;

beforeAll(async () => {
  const c = await getGPU();
  if (!c) throw new Error('WebGPU unavailable in the test browser');
  // SwiftShader compiles these pipelines for minutes: no async → sync fallback (it compiles twice)
  if (c.isSoftware) pipelineCompileLimits.timeoutMs = 600_000;
  ctx = c;
  pipes = await compileSortPipelines(ctx.device, ctx.device.lost);
}, 600_000);

async function readU32(buf: GPUBuffer, n: number): Promise<Uint32Array> {
  const d = ctx.device;
  const st = d.createBuffer({ size: Math.max(16, 4 * n), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = d.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, st, 0, Math.max(16, 4 * n));
  d.queue.submit([enc.finish()]);
  await st.mapAsync(GPUMapMode.READ);
  const out = new Uint32Array(st.getMappedRange().slice(0, 4 * n));
  st.unmap();
  st.destroy();
  return out;
}

async function sortOnGPU(keys: Uint32Array, bits: number, capacity = keys.length): Promise<{ k: Uint32Array; v: Uint32Array }> {
  const d = ctx.device;
  const s = new RadixSorter(d, pipes, Math.max(1, capacity), 'test-sort');
  const n = keys.length;
  if (n) {
    d.queue.writeBuffer(s.keys[0], 0, keys);
    d.queue.writeBuffer(s.vals[0], 0, Uint32Array.from({ length: n }, (_, i) => i));
  }
  s.setCount(n);
  const enc = d.createCommandEncoder();
  const pass = enc.beginComputePass();
  const r = s.encode(pass, bits);
  pass.end();
  d.queue.submit([enc.finish()]);
  const k = await readU32(s.keys[r], n);
  const v = await readU32(s.vals[r], n);
  s.destroy();
  return { k, v };
}

function check(keys: Uint32Array, bits: number, got: { k: Uint32Array; v: Uint32Array }): void {
  const mask = bits >= 32 ? 0xffffffff : 2 ** bits - 1;
  const idx = Array.from({ length: keys.length }, (_, i) => i);
  idx.sort((a, b) => ((keys[a] & mask) >>> 0) - ((keys[b] & mask) >>> 0) || a - b); // stable
  for (let i = 0; i < idx.length; i++) {
    if (got.v[i] !== idx[i] || got.k[i] !== keys[idx[i]]) throw new Error(`mismatch at ${i}: got (${got.k[i]}, ${got.v[i]}), want (${keys[idx[i]]}, ${idx[i]})`);
  }
}

describe('splat radix sort (GPU)', () => {
  it('sorts 32-bit keys stably across many blocks (with ties)', async () => {
    const rng = new Rng(3);
    for (const n of [1, 1000, 1024, 1025, 70_001]) {
      const keys = Uint32Array.from({ length: n }, () => (rng.next() < 0.3 ? 12345 : rng.nextU32()));
      check(keys, 32, await sortOnGPU(keys, 32));
    }
  });

  it('sorts the low bits only (tile ids), odd pass count, capacity larger than n', async () => {
    const rng = new Rng(4);
    const n = 50_000;
    const keys = Uint32Array.from({ length: n }, () => rng.int(3000));
    check(keys, 12, await sortOnGPU(keys, 12, 3 * n));
  });

  it('handles n = 0', async () => {
    const r = await sortOnGPU(new Uint32Array(0), 8, 16);
    expect(r.k.length).toBe(0);
  });

  it('sorts f32 depth bit patterns in depth order', async () => {
    const rng = new Rng(5);
    const depths = Float32Array.from({ length: 5000 }, () => rng.uniform(0.01, 300));
    const keys = new Uint32Array(depths.buffer);
    const r = await sortOnGPU(keys, 32);
    for (let i = 1; i < r.v.length; i++) expect(depths[r.v[i]]).toBeGreaterThanOrEqual(depths[r.v[i - 1]]);
  });
});
