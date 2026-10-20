/**
 * Splat forward pass on the GPU (shared by `SplatRenderer` and `SplatTrainer`): projection,
 * depth sort of the Gaussians, intersection emission in depth order, stable tile sort, tile
 * ranges and the tile rasteriser — all GPU-resident, no readback (the intersection count
 * drives the sort indirectly; an overflow beyond the reserved capacity drops the farthest
 * intersections of the frame and is reported through `binCtl[13]`).
 *
 * Passes (one compute pass each, so a PassProfiler can time them): project, depth-sort (32-bit
 * keys = f32 depth bits, 8 radix passes over the Gaussians), bin (tile counts in depth order →
 * exclusive scan → emit), tile-sort (⌈log2 tiles⌉ bits), raster (ranges + rasteriser).
 */
import { CTL } from './wgsl/sort';
import { FRAME_BYTES, type SplatShaderConfig } from './wgsl/common';
import { BIN_SETUP_WGSL, EMIT_WGSL, GATHER_WGSL, RANGES_WGSL, projectWGSL, rasterForwardWGSL } from './wgsl/raster';
import { PassProfiler, RadixSorter, bindGroup, compileAll, compileSortPipelines, makeBuffer, radixBits, scanCfgBuffer, type SortPipelines } from './gpu-util';
import { DEFAULT_NEAR, FRUSTUM_GUARD, TILE_SIZE, cameraCentre, type SplatCamera } from './types';

export interface ForwardPipelines {
  cfg: SplatShaderConfig;
  sort: SortPipelines;
  project: GPUComputePipeline;
  gather: GPUComputePipeline;
  binSetup: GPUComputePipeline;
  emit: GPUComputePipeline;
  ranges: GPUComputePipeline;
  raster: GPUComputePipeline;
}

export async function compileForwardPipelines(device: GPUDevice, lost: Promise<unknown>, cfg: SplatShaderConfig, sort?: SortPipelines): Promise<ForwardPipelines> {
  const [s, p] = await Promise.all([
    sort ? Promise.resolve(sort) : compileSortPipelines(device, lost),
    compileAll(
      device,
      lost,
      { project: projectWGSL(cfg), gather: GATHER_WGSL, binSetup: BIN_SETUP_WGSL, emit: EMIT_WGSL, ranges: RANGES_WGSL, raster: rasterForwardWGSL(cfg) },
      'splat-fwd',
    ),
  ]);
  return { cfg, sort: s, ...p };
}

const forwardCache = new WeakMap<GPUDevice, Map<string, Promise<ForwardPipelines>>>();

/** Forward pipelines for a device and shader configuration, compiled once and cached. */
export function getForwardPipelines(device: GPUDevice, cfg: SplatShaderConfig): Promise<ForwardPipelines> {
  let perDevice = forwardCache.get(device);
  if (!perDevice) forwardCache.set(device, (perDevice = new Map()));
  const key = `${cfg.shDegree}-${cfg.antialiased}`;
  let p = perDevice.get(key);
  if (!p) {
    p = compileForwardPipelines(device, device.lost, cfg);
    perDevice.set(key, p);
    p.catch(() => perDevice!.delete(key));
  }
  return p;
}

export interface SplatForwardOptions {
  /** Gaussian capacity of the parameter layout. */
  capacity: number;
  maxWidth: number;
  maxHeight: number;
  /** Tile intersections reserved (default 12 per Gaussian, at least 2^16). */
  isectCapacity?: number;
  /** Parameter buffer (wgsl/common.ts layout for `capacity`). */
  params: GPUBuffer;
  label?: string;
}

/** Frame-uniform contents for a camera (see FRAME_WGSL). */
export function frameData(cam: SplatCamera, count: number, capacity: number, isectCapacity: number, background: ArrayLike<number> = [0, 0, 0]): ArrayBuffer {
  const buf = new ArrayBuffer(FRAME_BYTES);
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  const R = cam.R;
  const t = cam.t;
  f.set([R[0], R[1], R[2], t[0], R[3], R[4], R[5], t[1], R[6], R[7], R[8], t[2]], 0);
  f.set([cam.fx, cam.fy, cam.cx, cam.cy], 12);
  const C = cameraCentre(cam);
  f.set([C[0], C[1], C[2], cam.near ?? DEFAULT_NEAR], 16);
  f.set([background[0], background[1], background[2], 0], 20);
  const gx = (FRUSTUM_GUARD * 0.5 * cam.width) / cam.fx;
  const gy = (FRUSTUM_GUARD * 0.5 * cam.height) / cam.fy;
  f.set([(cam.width - cam.cx) / cam.fx + gx, cam.cx / cam.fx + gx, (cam.height - cam.cy) / cam.fy + gy, cam.cy / cam.fy + gy], 24);
  u.set([cam.width, cam.height, Math.ceil(cam.width / TILE_SIZE), Math.ceil(cam.height / TILE_SIZE)], 28);
  u.set([count, capacity, isectCapacity, 0], 32);
  return buf;
}

export class SplatForward {
  readonly frame: GPUBuffer;
  readonly proj: GPUBuffer;
  readonly rects: GPUBuffer;
  readonly tileCount: GPUBuffer;
  readonly offsets: GPUBuffer;
  readonly binCtl: GPUBuffer;
  readonly ranges: GPUBuffer;
  readonly outColor: GPUBuffer;
  readonly outN: GPUBuffer;
  readonly depthSort: RadixSorter;
  isectSort: RadixSorter;
  readonly capacity: number;
  readonly maxWidth: number;
  readonly maxHeight: number;
  private readonly blockSums: GPUBuffer;
  private readonly binScanCfg: GPUBuffer;
  private readonly bgProject: GPUBindGroup;
  private readonly bgGather: GPUBindGroup;
  private readonly bgScanReduce: GPUBindGroup;
  private readonly bgScanTop: GPUBindGroup;
  private readonly bgScanDown: GPUBindGroup;
  private bgBinSetup!: GPUBindGroup;
  private bgEmit!: GPUBindGroup;
  private bgRanges!: GPUBindGroup[];
  private bgRaster!: GPUBindGroup[];
  /** Current frame (set by setFrame). */
  width = 0;
  height = 0;
  tilesX = 0;
  tilesY = 0;
  count = 0;
  /** Ping-pong index holding the sorted intersections after `encode`. */
  isectResult: 0 | 1 = 0;
  private tileBits = 4;

  constructor(
    private readonly device: GPUDevice,
    readonly pipes: ForwardPipelines,
    opts: SplatForwardOptions,
  ) {
    const label = opts.label ?? 'splat';
    const N = opts.capacity;
    this.capacity = N;
    this.maxWidth = opts.maxWidth;
    this.maxHeight = opts.maxHeight;
    const maxPx = opts.maxWidth * opts.maxHeight;
    const maxTiles = Math.ceil(opts.maxWidth / TILE_SIZE) * Math.ceil(opts.maxHeight / TILE_SIZE);
    this.frame = device.createBuffer({ size: FRAME_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: `${label}-frame` });
    this.proj = makeBuffer(device, N * 48, `${label}-proj`);
    this.rects = makeBuffer(device, N * 8, `${label}-rects`);
    this.tileCount = makeBuffer(device, N * 4, `${label}-tilecount`);
    this.offsets = makeBuffer(device, N * 4, `${label}-offsets`);
    this.binCtl = makeBuffer(device, CTL.words * 4, `${label}-binctl`);
    this.blockSums = makeBuffer(device, 1024 * 4, `${label}-binsums`);
    this.binScanCfg = scanCfgBuffer(device, 0, CTL.total, `${label}-binscan`);
    this.ranges = makeBuffer(device, maxTiles * 8, `${label}-ranges`);
    this.outColor = makeBuffer(device, maxPx * 16, `${label}-color`);
    this.outN = makeBuffer(device, maxPx * 4, `${label}-n`);
    if (N > 1024 * 1024) throw new Error('SplatForward: capacity above 1M Gaussians');
    this.depthSort = new RadixSorter(device, pipes.sort, N, `${label}-depthsort`);
    this.isectSort = new RadixSorter(device, pipes.sort, Math.max(1 << 16, opts.isectCapacity ?? 12 * N), `${label}-isectsort`);
    const P = pipes;
    this.bgProject = bindGroup(device, P.project, [this.frame, opts.params, this.proj, this.rects, this.tileCount, this.depthSort.keys[0], this.depthSort.vals[0]]);
    this.bgGather = bindGroup(device, P.gather, [this.binCtl, this.depthSort.vals[0], this.tileCount, this.offsets]);
    this.bgScanReduce = bindGroup(device, P.sort.scanReduce, [this.binScanCfg, this.binCtl, this.offsets, this.blockSums]);
    this.bgScanTop = bindGroup(device, P.sort.scanTop, [this.binScanCfg, this.binCtl, this.blockSums]);
    this.bgScanDown = bindGroup(device, P.sort.scanDown, [this.binScanCfg, this.binCtl, this.offsets, this.blockSums]);
    this.makeIsectBindGroups();
  }

  get isectCapacity(): number {
    return this.isectSort.capacity;
  }

  private makeIsectBindGroups(): void {
    const d = this.device;
    const P = this.pipes;
    const S = this.isectSort;
    this.bgBinSetup = bindGroup(d, P.binSetup, [this.frame, this.binCtl, S.ctl]);
    this.bgEmit = bindGroup(d, P.emit, [this.frame, this.depthSort.vals[0], this.tileCount, this.rects, this.offsets, S.keys[0], S.vals[0], this.proj]);
    this.bgRanges = [0, 1].map((k) => bindGroup(d, P.ranges, [S.ctl, S.keys[k], this.ranges]));
    this.bgRaster = [0, 1].map((k) => bindGroup(d, P.raster, [this.frame, this.ranges, S.vals[k], this.proj, this.outColor, this.outN]));
  }

  /** Reallocates the intersection buffers (between frames). */
  resizeIntersections(capacity: number): void {
    this.isectSort.destroy();
    this.isectSort = new RadixSorter(this.device, this.pipes.sort, capacity, 'splat-isectsort');
    this.makeIsectBindGroups();
  }

  /** Sorted intersection values (Gaussian ids) of the last encoded frame. */
  get sortedVals(): GPUBuffer {
    return this.isectSort.vals[this.isectResult];
  }

  /** Writes the frame uniform and the element counts for rendering `count` Gaussians with `cam`. */
  setFrame(cam: SplatCamera, count: number, background: ArrayLike<number> = [0, 0, 0]): void {
    if (cam.width > this.maxWidth || cam.height > this.maxHeight) throw new Error(`SplatForward: ${cam.width}×${cam.height} exceeds ${this.maxWidth}×${this.maxHeight}`);
    if (count > this.capacity) throw new Error(`SplatForward: ${count} Gaussians > capacity ${this.capacity}`);
    this.width = cam.width;
    this.height = cam.height;
    this.tilesX = Math.ceil(cam.width / TILE_SIZE);
    this.tilesY = Math.ceil(cam.height / TILE_SIZE);
    this.count = count;
    this.tileBits = radixBits(this.tilesX * this.tilesY);
    this.device.queue.writeBuffer(this.frame, 0, frameData(cam, count, this.capacity, this.isectCapacity, background));
    this.depthSort.setCount(count);
    this.device.queue.writeBuffer(this.binCtl, 0, new Uint32Array([count]));
  }

  private begin(enc: GPUCommandEncoder, label: string, prof?: PassProfiler | null): GPUComputePassEncoder {
    return prof ? prof.pass(enc, label) : enc.beginComputePass({ label });
  }

  /** Records the forward pass of the frame set by `setFrame`. */
  encode(enc: GPUCommandEncoder, prof?: PassProfiler | null): void {
    const P = this.pipes;
    const N = this.count;
    const g256 = Math.ceil(N / 256);
    const g1024 = Math.ceil(N / 1024);
    let pass = this.begin(enc, 'project', prof);
    pass.setPipeline(P.project);
    pass.setBindGroup(0, this.bgProject);
    pass.dispatchWorkgroups(g256);
    pass.end();
    pass = this.begin(enc, 'depth-sort', prof);
    this.depthSort.encode(pass, 32);
    pass.end();
    pass = this.begin(enc, 'bin', prof);
    pass.setPipeline(P.gather);
    pass.setBindGroup(0, this.bgGather);
    pass.dispatchWorkgroups(g256);
    pass.setPipeline(P.sort.scanReduce);
    pass.setBindGroup(0, this.bgScanReduce);
    pass.dispatchWorkgroups(g1024);
    pass.setPipeline(P.sort.scanTop);
    pass.setBindGroup(0, this.bgScanTop);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(P.sort.scanDown);
    pass.setBindGroup(0, this.bgScanDown);
    pass.dispatchWorkgroups(g1024);
    pass.setPipeline(P.binSetup);
    pass.setBindGroup(0, this.bgBinSetup);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(P.emit);
    pass.setBindGroup(0, this.bgEmit);
    pass.dispatchWorkgroups(g256);
    pass.end();
    pass = this.begin(enc, 'tile-sort', prof);
    this.isectResult = this.isectSort.encode(pass, this.tileBits);
    pass.end();
    enc.clearBuffer(this.ranges, 0, this.tilesX * this.tilesY * 8);
    pass = this.begin(enc, 'raster', prof);
    pass.setPipeline(P.ranges);
    pass.setBindGroup(0, this.bgRanges[this.isectResult]);
    pass.dispatchWorkgroupsIndirect(this.isectSort.ctl, CTL.argsBlocks * 4);
    pass.setPipeline(P.raster);
    pass.setBindGroup(0, this.bgRaster[this.isectResult]);
    pass.dispatchWorkgroups(this.tilesX, this.tilesY);
    pass.end();
  }

  destroy(): void {
    for (const b of [this.frame, this.proj, this.rects, this.tileCount, this.offsets, this.binCtl, this.blockSums, this.binScanCfg, this.ranges, this.outColor, this.outN]) b.destroy();
    this.depthSort.destroy();
    this.isectSort.destroy();
  }
}

