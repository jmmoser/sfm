/**
 * Small WebGPU helpers private to the feature module (SIFT + matching on the GPU):
 * error type used to trigger the CPU fallback, uniform parameter blocks, a reusable
 * readback staging buffer, async pipeline compilation raced against device loss,
 * 2-D dispatch for large workgroup counts and a work-budgeted command encoder that
 * splits long GPU work into several submissions (iOS GPU watchdog, one heavy
 * command buffer in flight).
 */

/**
 * Error raised for GPU-side failures (device loss, validation / out-of-memory errors,
 * failed readbacks, unsupported sizes). `createFeaturePipeline` falls back to the CPU
 * implementation when it sees one; input-validation errors are plain `Error`s.
 */
export class GPUFeatureError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message);
    this.name = 'GPUFeatureError';
    if (options && 'cause' in options) (this as { cause?: unknown }).cause = options.cause;
  }
}

/**
 * Invalid caller input detected inside a GPU operation (e.g. an unreadable image source,
 * a FeatureSet whose arrays are shorter than `count`). Plain Error subclass, NOT a
 * GPUFeatureError: it must not switch the feature pipeline to the CPU.
 */
export class FeatureInputError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'FeatureInputError';
  }
}

/** Uniform-buffer slot size; >= minUniformBufferOffsetAlignment on every implementation (256). */
export const PARAM_SLOT = 256;

/**
 * Accumulates per-dispatch uniform parameter structs into 256-byte slots of one
 * buffer. Each struct is written as `u32` fields followed by `f32` fields, which
 * matches the WGSL structs (all members 4-byte scalars, u32s first).
 */
export class ParamBlock {
  private data: ArrayBuffer;
  private u32: Uint32Array;
  private f32: Float32Array;
  count = 0;

  constructor(initialSlots = 16) {
    this.data = new ArrayBuffer(initialSlots * PARAM_SLOT);
    this.u32 = new Uint32Array(this.data);
    this.f32 = new Float32Array(this.data);
  }

  reset(): void {
    this.count = 0;
  }

  /** Appends a struct and returns its byte offset. */
  add(u: ArrayLike<number>, f: ArrayLike<number> = []): number {
    if (u.length + f.length > PARAM_SLOT / 4) throw new Error('ParamBlock: struct too large');
    if ((this.count + 1) * PARAM_SLOT > this.data.byteLength) {
      const d = new ArrayBuffer(this.data.byteLength * 2);
      new Uint8Array(d).set(new Uint8Array(this.data));
      this.data = d;
      this.u32 = new Uint32Array(d);
      this.f32 = new Float32Array(d);
    }
    const base = (this.count * PARAM_SLOT) >> 2;
    this.u32.fill(0, base, base + PARAM_SLOT / 4);
    for (let i = 0; i < u.length; i++) this.u32[base + i] = u[i] >>> 0;
    for (let i = 0; i < f.length; i++) this.f32[base + u.length + i] = f[i];
    this.count++;
    return (this.count - 1) * PARAM_SLOT;
  }

  get byteLength(): number {
    return this.count * PARAM_SLOT;
  }

  bytes(): Uint8Array {
    return new Uint8Array(this.data, 0, this.count * PARAM_SLOT);
  }
}

/** Grow-only GPU buffer holder (e.g. per-call uploads). */
export class GrowBuffer {
  buffer: GPUBuffer | null = null;
  size = 0;
  constructor(
    private readonly device: GPUDevice,
    private readonly usage: GPUBufferUsageFlags,
    private readonly label: string,
  ) {}

  /** Ensures capacity >= bytes; returns true if the buffer was (re)created. */
  ensure(bytes: number): boolean {
    const need = Math.max(16, Math.ceil(bytes / 16) * 16);
    if (this.buffer && this.size >= need) return false;
    this.buffer?.destroy();
    const size = Math.max(need, Math.ceil(this.size * 1.5 / 16) * 16);
    this.buffer = this.device.createBuffer({ size, usage: this.usage, label: this.label });
    this.size = size;
    return true;
  }

  destroy(): void {
    this.buffer?.destroy();
    this.buffer = null;
    this.size = 0;
  }
}

/**
 * Reusable MAP_READ staging buffer. Callers copy results into it (`copyFrom`) in their
 * command encoder, submit, then call `read` (one mapAsync + one getMappedRange per
 * readback, as recommended for iOS).
 */
export class Staging {
  private buf: GrowBuffer;
  constructor(
    private readonly device: GPUDevice,
    label: string,
    private readonly lost: Promise<unknown>,
  ) {
    this.buf = new GrowBuffer(device, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label);
  }

  get buffer(): GPUBuffer {
    if (!this.buf.buffer) throw new Error('Staging: ensure() first');
    return this.buf.buffer;
  }

  ensure(bytes: number): void {
    this.buf.ensure(bytes);
  }

  /** Maps the first `bytes` and returns a copy. Rejects with GPUFeatureError on device loss. */
  async read(bytes: number): Promise<ArrayBuffer> {
    const b = this.buffer;
    const size = Math.max(4, Math.ceil(bytes / 4) * 4);
    try {
      const r = await Promise.race([b.mapAsync(GPUMapMode.READ, 0, size).then(() => 'ok' as const), this.lost.then(() => 'lost' as const)]);
      if (r === 'lost') throw new GPUFeatureError('GPU device lost during readback');
      // Always unmap (a buffer left mapped would make every later copy into it invalid).
      try {
        return b.getMappedRange(0, size).slice(0);
      } finally {
        b.unmap();
      }
    } catch (e) {
      if (e instanceof GPUFeatureError) throw e;
      throw new GPUFeatureError(`readback failed (device lost?): ${(e as Error)?.message ?? e}`, { cause: e });
    }
  }

  destroy(): void {
    this.buf.destroy();
  }
}

/**
 * Upper bound (ms) for one `createComputePipelineAsync`. On Safari 26.x/27.0 the async variant
 * compiles synchronously in the GPU process, and Safari Technology Preview fixed async creations
 * that stalled or never settled (docs/research/ios-webgpu.md §4.6). After this long, a device that
 * is still alive compiles the pipeline with the synchronous `createComputePipeline` instead (as
 * PatchMatchGPU does). Mutable for tests.
 */
export const pipelineCompileLimits = { timeoutMs: 30_000 };

/** Compiles a compute pipeline asynchronously; rejects with GPUFeatureError on errors / device loss. */
export async function compilePipeline(
  device: GPUDevice,
  label: string,
  code: string,
  lost: Promise<unknown>,
  entryPoint = 'main',
): Promise<GPUComputePipeline> {
  // The error scope only wraps the synchronous createShaderModule call: the scope stack is
  // shared by every user of the device, so holding a scope across the await below would
  // pop another compilation's scope when several pipelines compile concurrently (wrong
  // error attribution). createComputePipelineAsync reports its failures by rejecting.
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ code, label });
  const moduleError = device.popErrorScope();
  const desc: GPUComputePipelineDescriptor = { layout: 'auto', compute: { module, entryPoint }, label };
  let pipeline: GPUComputePipeline | null = null;
  let failure: unknown = null;
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const timeout = new Promise<'timeout'>((resolve) => {
      timer = setTimeout(() => resolve('timeout'), pipelineCompileLimits.timeoutMs);
    });
    const r = await Promise.race([device.createComputePipelineAsync(desc), lost.then(() => 'lost' as const), timeout]);
    if (r === 'lost') failure = new GPUFeatureError(`device lost while compiling ${label}`);
    else if (r === 'timeout') {
      console.warn(`[gpu] async compilation of ${label} timed out; compiling synchronously`);
      // push / create / pop with no await in between: the scope is this call's own
      device.pushErrorScope('validation');
      const p = device.createComputePipeline(desc);
      const e = await device.popErrorScope();
      if (e) failure = new GPUFeatureError(e.message);
      else pipeline = p;
    } else pipeline = r;
  } catch (e) {
    failure = e;
  } finally {
    clearTimeout(timer);
  }
  let err: GPUError | null = null;
  try {
    err = await moduleError;
  } catch (e) {
    failure ??= e;
  }
  if (failure || err || !pipeline) {
    let msg = err?.message ?? (failure as Error)?.message ?? String(failure);
    if (!err) {
      // Pull the compiler messages for diagnostics.
      try {
        const info = await module.getCompilationInfo();
        const m = info.messages.filter((x) => x.type === 'error').map((x) => `${x.lineNum}:${x.linePos} ${x.message}`);
        if (m.length) msg += ` | ${m.join(' | ')}`;
      } catch {
        // ignore
      }
    }
    throw new GPUFeatureError(`failed to compile ${label}: ${msg}`, { cause: failure ?? err });
  }
  return pipeline;
}

/** Splits a 1-D workgroup count into (x, y) with x <= 32768 (maxComputeWorkgroupsPerDimension >= 65535). */
export function dispatch2D(n: number): [number, number] {
  if (n <= 32768) return [Math.max(1, n), 1];
  return [32768, Math.ceil(n / 32768)];
}

/**
 * Error for GPU work that did not execute (iOS may silently drop command buffers
 * submitted while the page is hidden). Transient: the operation can be retried.
 */
export class GPUWorkDroppedError extends GPUFeatureError {
  readonly transient = true;
  constructor(message: string) {
    super(message);
    this.name = 'GPUWorkDroppedError';
  }
}

/** One-invocation kernel recorded at the start of every command buffer (see WorkStamp). */
export const STAMP_WGSL = /* wgsl */ `
@group(0) @binding(0) var<uniform> attempt: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> s: array<u32>;
@compute @workgroup_size(1)
fn main() {
  let o = attempt.y;
  if (s[o] == attempt.x) { s[o + 1u] = s[o + 1u] + 1u; } else { s[o] = attempt.x; s[o + 1u] = 1u; }
}
`;

/**
 * Detects dropped GPU work. Every command buffer of an attempt starts with a stamp
 * dispatch that counts the executed command buffers of the attempt (fresh attempt id
 * per operation) in two u32 words of `target` at `wordOffset`; readbacks copy those
 * words and `check` compares them with the number of command buffers submitted.
 */
export class WorkStamp {
  private readonly uniform: GPUBuffer;
  private readonly bg: GPUBindGroup;
  private attemptId = 0;

  constructor(
    private readonly device: GPUDevice,
    private readonly pipe: GPUComputePipeline,
    target: GPUBuffer,
    readonly wordOffset: number,
  ) {
    this.uniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'work-stamp' });
    this.bg = device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniform } },
        { binding: 1, resource: { buffer: target } },
      ],
    });
  }

  /** Starts a new attempt (new id written to the uniform before any of its submissions). */
  begin(): void {
    this.attemptId = (this.attemptId + 1) >>> 0 || 1;
    this.device.queue.writeBuffer(this.uniform, 0, new Uint32Array([this.attemptId, this.wordOffset, 0, 0]));
  }

  record(pass: GPUComputePassEncoder): void {
    pass.setPipeline(this.pipe);
    pass.setBindGroup(0, this.bg);
    pass.dispatchWorkgroups(1);
  }

  /** Throws GPUWorkDroppedError unless the read words show this attempt with `submits` command buffers. */
  check(words: Uint32Array, submits: number, where: string): void {
    if (words[0] !== this.attemptId || words[1] !== submits) {
      throw new GPUWorkDroppedError(
        `GPU work was not executed (${where}: stamp ${words[0]}/${words[1]}, expected ${this.attemptId}/${submits}; page hidden?)`,
      );
    }
  }

  destroy(): void {
    this.uniform.destroy();
  }
}

/**
 * Command encoding with a work budget: dispatches are recorded into compute passes;
 * once the accumulated cost exceeds `budget`, the command buffer is submitted and
 * the GPU is awaited before encoding continues (at most one heavy command buffer
 * in flight; each sized to a few ms on a phone GPU). With a WorkStamp, every command
 * buffer starts with a stamp dispatch (see WorkStamp).
 */
export class BudgetEncoder {
  /** Current command encoder; created lazily (none is left dangling after the last submit). */
  private enc: GPUCommandEncoder | null = null;
  private pass: GPUComputePassEncoder | null = null;
  private cost = 0;
  /** Command buffers submitted so far. */
  submits = 0;

  /**
   * @param profile when given, every dispatch is submitted and awaited on its own and
   *   its wall time (ms) is accumulated under the pipeline label (diagnostics only).
   * @param dropSubmissions testing hook: 0-based indices of command buffers to discard
   *   instead of submitting (simulates work dropped by the OS).
   */
  constructor(
    private readonly device: GPUDevice,
    private readonly budget: number,
    private readonly lost: Promise<unknown>,
    private readonly label: string,
    private readonly profile: Map<string, number> | null = null,
    private readonly stamp: WorkStamp | null = null,
    private readonly dropSubmissions: ReadonlySet<number> | null = null,
  ) {}

  /** The current encoder; a new one starts with the work-stamp dispatch. */
  private ensureEncoder(): GPUCommandEncoder {
    if (!this.enc) {
      const enc = this.device.createCommandEncoder({ label: this.label });
      this.enc = enc;
      if (this.stamp) {
        this.pass = enc.beginComputePass({ label: this.label });
        this.stamp.record(this.pass);
      }
    }
    return this.enc;
  }

  private computePass(): GPUComputePassEncoder {
    const enc = this.ensureEncoder();
    if (!this.pass) this.pass = enc.beginComputePass({ label: this.label });
    return this.pass;
  }

  private endPass(): void {
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
  }

  /** Records one dispatch; may flush (submit + await) BEFORE recording it if over budget. */
  async dispatch(pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, x: number, y = 1, z = 1, cost = 0): Promise<void> {
    if (this.profile) await this.flush();
    else if (this.cost > 0 && this.cost + cost > this.budget) await this.flush();
    const t0 = performance.now();
    const p = this.computePass();
    p.setPipeline(pipeline);
    p.setBindGroup(0, bindGroup);
    p.dispatchWorkgroups(x, y, z);
    this.cost += cost;
    if (this.profile) {
      await this.flush();
      const k = pipeline.label || 'dispatch';
      this.profile.set(k, (this.profile.get(k) ?? 0) + performance.now() - t0);
    }
  }

  /** Raw access to the command encoder for copies / clears (ends the current pass). */
  encoder(): GPUCommandEncoder {
    const enc = this.ensureEncoder();
    this.endPass();
    return enc;
  }

  private submitCurrent(): void {
    // A submitted command buffer always carries a stamp (the stamp counts command buffers).
    const enc = this.ensureEncoder();
    this.endPass();
    this.enc = null;
    const cb = enc.finish();
    if (!this.dropSubmissions?.has(this.submits)) this.device.queue.submit([cb]);
    this.submits++;
    this.cost = 0;
  }

  /** Submits what has been recorded (if anything) and waits for the GPU. */
  async flush(): Promise<void> {
    if (!this.enc) return;
    this.submitCurrent();
    const r = await Promise.race([this.device.queue.onSubmittedWorkDone().then(() => 'ok' as const), this.lost.then(() => 'lost' as const)]);
    if (r === 'lost') throw new GPUFeatureError('GPU device lost');
  }

  /** Submits the remaining work without waiting (follow with a staging read). */
  submit(): void {
    this.submitCurrent();
  }
}

/** Pushes validation + OOM error scopes; the returned function pops them and throws GPUFeatureError. */
export function errorScope(device: GPUDevice, label: string): () => Promise<void> {
  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('validation');
  return async () => {
    // Pop both scopes synchronously: with an await between them, another module could push
    // its scopes in between and the second pop would remove (and swallow) one of those.
    const pv = device.popErrorScope();
    const po = device.popErrorScope();
    const [v, o] = await Promise.all([pv, po]);
    if (v) throw new GPUFeatureError(`[${label}] validation error: ${v.message}`);
    if (o) throw new GPUFeatureError(`[${label}] out of memory: ${o.message}`);
  };
}
