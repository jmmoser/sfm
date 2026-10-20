/**
 * Small WebGPU helpers for src/splat: pipeline compilation (shared helper of src/features),
 * bind groups, a radix sorter over the WGSL in ./wgsl/sort.ts, and an optional per-pass GPU
 * timestamp profiler.
 */
import { compilePipeline } from '../features/gpu-common';
import {
  CTL,
  RADIX_COUNT_WGSL,
  RADIX_SCATTER_WGSL,
  SCAN_DOWN_WGSL,
  SCAN_REDUCE_WGSL,
  SCAN_TOP_WGSL,
  SORT_SETUP_WGSL,
  SORT_UNIFORM_STRIDE,
} from './wgsl/sort';

export const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

export function makeBuffer(device: GPUDevice, bytes: number, label: string, usage: GPUBufferUsageFlags = STORAGE): GPUBuffer {
  return device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage, label });
}

export function bindGroup(device: GPUDevice, pipeline: GPUComputePipeline, buffers: (GPUBuffer | GPUBufferBinding)[], label?: string): GPUBindGroup {
  return device.createBindGroup({
    label,
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((b, i) => ({ binding: i, resource: 'buffer' in b ? b : { buffer: b } })),
  });
}

/** Compiles named WGSL modules in parallel. */
export async function compileAll<K extends string>(
  device: GPUDevice,
  lost: Promise<unknown>,
  sources: Record<K, string>,
  prefix: string,
): Promise<Record<K, GPUComputePipeline>> {
  const keys = Object.keys(sources) as K[];
  const out = await Promise.all(keys.map((k) => compilePipeline(device, `${prefix}-${k}`, sources[k], lost)));
  const r = {} as Record<K, GPUComputePipeline>;
  keys.forEach((k, i) => (r[k] = out[i]));
  return r;
}

// ------------------------------------------------------------------ radix sort

export interface SortPipelines {
  setup: GPUComputePipeline;
  count: GPUComputePipeline;
  scatter: GPUComputePipeline;
  scanReduce: GPUComputePipeline;
  scanTop: GPUComputePipeline;
  scanDown: GPUComputePipeline;
}

export function compileSortPipelines(device: GPUDevice, lost: Promise<unknown>): Promise<SortPipelines> {
  return compileAll(
    device,
    lost,
    {
      setup: SORT_SETUP_WGSL,
      count: RADIX_COUNT_WGSL,
      scatter: RADIX_SCATTER_WGSL,
      scanReduce: SCAN_REDUCE_WGSL,
      scanTop: SCAN_TOP_WGSL,
      scanDown: SCAN_DOWN_WGSL,
    },
    'splat-sort',
  );
}

/** ScanCfg uniform: (nSlot, totalSlot). */
export function scanCfgBuffer(device: GPUDevice, nSlot: number, totalSlot: number, label: string): GPUBuffer {
  const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label });
  device.queue.writeBuffer(b, 0, new Uint32Array([nSlot, totalSlot >>> 0, 0, 0]));
  return b;
}

/**
 * 4-bit LSD radix sort of (u32 key, u32 value) pairs, up to `capacity` elements, n in
 * `ctl[0]` (written by the host or a kernel). `keys[0]` / `vals[0]` are the input; after
 * `encode(pass, bits)` the result is in index `bits / 4 % 2` of the ping-pong pair.
 */
export class RadixSorter {
  readonly keys: [GPUBuffer, GPUBuffer];
  readonly vals: [GPUBuffer, GPUBuffer];
  readonly ctl: GPUBuffer;
  private readonly hist: GPUBuffer;
  private readonly blockSums: GPUBuffer;
  private readonly passUniform: GPUBuffer;
  private readonly histScanCfg: GPUBuffer;
  private readonly bgSetup: GPUBindGroup;
  private readonly bgCount: GPUBindGroup[] = [];
  private readonly bgScatter: GPUBindGroup[] = [];
  private readonly bgScanReduce: GPUBindGroup;
  private readonly bgScanTop: GPUBindGroup;
  private readonly bgScanDown: GPUBindGroup;

  constructor(
    private readonly device: GPUDevice,
    private readonly pipes: SortPipelines,
    readonly capacity: number,
    label: string,
  ) {
    const nb = Math.max(1, Math.ceil(capacity / 1024));
    if (16 * nb > 1024 * 1024) throw new Error(`RadixSorter: capacity ${capacity} too large`);
    this.keys = [makeBuffer(device, capacity * 4, `${label}-keys0`), makeBuffer(device, capacity * 4, `${label}-keys1`)];
    this.vals = [makeBuffer(device, capacity * 4, `${label}-vals0`), makeBuffer(device, capacity * 4, `${label}-vals1`)];
    this.ctl = makeBuffer(device, CTL.words * 4, `${label}-ctl`, STORAGE | GPUBufferUsage.INDIRECT);
    this.hist = makeBuffer(device, 16 * nb * 4, `${label}-hist`);
    this.blockSums = makeBuffer(device, 1024 * 4, `${label}-blocksums`);
    this.passUniform = device.createBuffer({ size: 8 * SORT_UNIFORM_STRIDE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: `${label}-passes` });
    const u = new Uint32Array((8 * SORT_UNIFORM_STRIDE) / 4);
    for (let p = 0; p < 8; p++) u[(p * SORT_UNIFORM_STRIDE) / 4] = 4 * p;
    device.queue.writeBuffer(this.passUniform, 0, u);
    this.histScanCfg = scanCfgBuffer(device, CTL.histLen, 0xffffffff, `${label}-histscan`);
    this.bgSetup = bindGroup(device, pipes.setup, [this.ctl]);
    for (let p = 0; p < 8; p++) {
      const src = p & 1;
      const dst = src ^ 1;
      const uni = { buffer: this.passUniform, offset: p * SORT_UNIFORM_STRIDE, size: 16 };
      this.bgCount.push(bindGroup(device, pipes.count, [uni, this.ctl, this.keys[src], this.hist]));
      this.bgScatter.push(bindGroup(device, pipes.scatter, [uni, this.ctl, this.keys[src], this.vals[src], this.keys[dst], this.vals[dst], this.hist]));
    }
    this.bgScanReduce = bindGroup(device, pipes.scanReduce, [this.histScanCfg, this.ctl, this.hist, this.blockSums]);
    this.bgScanTop = bindGroup(device, pipes.scanTop, [this.histScanCfg, this.ctl, this.blockSums]);
    this.bgScanDown = bindGroup(device, pipes.scanDown, [this.histScanCfg, this.ctl, this.hist, this.blockSums]);
  }

  /** Host-side element count (alternative to a kernel writing ctl[0]). */
  setCount(n: number): void {
    if (n > this.capacity) throw new Error(`RadixSorter: ${n} > capacity ${this.capacity}`);
    this.device.queue.writeBuffer(this.ctl, 0, new Uint32Array([n]));
  }

  /** Records the sort of the low `bits` key bits (multiple of 4, ≤ 32); returns the result index. */
  encode(pass: GPUComputePassEncoder, bits: number): 0 | 1 {
    const passes = Math.ceil(bits / 4);
    pass.setPipeline(this.pipes.setup);
    pass.setBindGroup(0, this.bgSetup);
    pass.dispatchWorkgroups(1);
    for (let p = 0; p < passes; p++) {
      pass.setPipeline(this.pipes.count);
      pass.setBindGroup(0, this.bgCount[p]);
      pass.dispatchWorkgroupsIndirect(this.ctl, CTL.argsBlocks * 4);
      pass.setPipeline(this.pipes.scanReduce);
      pass.setBindGroup(0, this.bgScanReduce);
      pass.dispatchWorkgroupsIndirect(this.ctl, CTL.argsHistScan * 4);
      pass.setPipeline(this.pipes.scanTop);
      pass.setBindGroup(0, this.bgScanTop);
      pass.dispatchWorkgroups(1);
      pass.setPipeline(this.pipes.scanDown);
      pass.setBindGroup(0, this.bgScanDown);
      pass.dispatchWorkgroupsIndirect(this.ctl, CTL.argsHistScan * 4);
      pass.setPipeline(this.pipes.scatter);
      pass.setBindGroup(0, this.bgScatter[p]);
      pass.dispatchWorkgroupsIndirect(this.ctl, CTL.argsBlocks * 4);
    }
    return (passes & 1) as 0 | 1;
  }

  destroy(): void {
    for (const b of [...this.keys, ...this.vals, this.ctl, this.hist, this.blockSums, this.passUniform, this.histScanCfg]) b.destroy();
  }
}

/** Bits needed to sort keys < n, rounded up to the 4-bit digit. */
export function radixBits(n: number): number {
  let b = 0;
  while (b < 32 && 2 ** b < n) b++;
  return Math.max(4, Math.ceil(b / 4) * 4);
}

// ------------------------------------------------------------------ timestamps

/**
 * Per-pass GPU timing with timestamp queries (when the device has 'timestamp-query'):
 * `pass(enc, label)` begins a compute pass that writes its start / end times; `resolve(enc)`
 * copies them to a mappable buffer and `read()` returns ms per label (summed over passes
 * with the same label) for the last resolved command buffer.
 */
export class PassProfiler {
  private readonly querySet: GPUQuerySet;
  private readonly resolveBuf: GPUBuffer;
  private readonly readBuf: GPUBuffer;
  private labels: string[] = [];
  private pending: string[] | null = null;

  constructor(
    private readonly device: GPUDevice,
    private readonly maxPasses = 64,
  ) {
    this.querySet = device.createQuerySet({ type: 'timestamp', count: 2 * maxPasses });
    this.resolveBuf = device.createBuffer({ size: 16 * maxPasses, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    this.readBuf = device.createBuffer({ size: 16 * maxPasses, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }

  static supported(device: GPUDevice): boolean {
    return device.features.has('timestamp-query');
  }

  pass(enc: GPUCommandEncoder, label: string): GPUComputePassEncoder {
    if (this.labels.length >= this.maxPasses) return enc.beginComputePass({ label });
    const k = this.labels.length;
    this.labels.push(label);
    return enc.beginComputePass({ label, timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 } });
  }

  /** Resolves this command buffer's queries (call once at the end of encoding). */
  resolve(enc: GPUCommandEncoder): void {
    const n = this.labels.length;
    if (n === 0 || this.pending) {
      this.labels = [];
      return;
    }
    enc.resolveQuerySet(this.querySet, 0, 2 * n, this.resolveBuf, 0);
    enc.copyBufferToBuffer(this.resolveBuf, 0, this.readBuf, 0, 16 * n);
    this.pending = this.labels;
    this.labels = [];
  }

  /** ms per label of the last resolved command buffer (after it was submitted), or null. */
  async read(): Promise<Map<string, number> | null> {
    const labels = this.pending;
    if (!labels) return null;
    await this.readBuf.mapAsync(GPUMapMode.READ, 0, 16 * labels.length);
    const t = new BigUint64Array(this.readBuf.getMappedRange(0, 16 * labels.length).slice(0));
    this.readBuf.unmap();
    this.pending = null;
    const out = new Map<string, number>();
    labels.forEach((l, k) => {
      const dt = Number(t[2 * k + 1] - t[2 * k]) / 1e6;
      if (dt >= 0 && dt < 1e5) out.set(l, (out.get(l) ?? 0) + dt);
    });
    return out;
  }

  destroy(): void {
    this.querySet.destroy();
    this.resolveBuf.destroy();
    this.readBuf.destroy();
  }
}
