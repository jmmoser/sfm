// Adversarial review tests (Node) of the WebGPU feature module's host-side logic.
import { describe, expect, it } from 'vitest';
import type { FeatureSet } from '../core/types';
import { DESCRIPTOR_SIZE } from '../core/types';
import { BudgetEncoder, FeatureInputError, GPUFeatureError, PARAM_SLOT, ParamBlock, WorkStamp, dispatch2D } from './gpu-common';
import { validateFeatureSet } from './match-gpu-host';
import { CANDIDATE_WORDS, descriptorFixedPoint } from './shaders/sift-wgsl';
import { planSift, resolveSiftOptions } from './sift-common';
import { decodeCandidates } from './sift-gpu-host';

const D = DESCRIPTOR_SIZE;

// ---------------------------------------------------------------------------
// Minimal fake WebGPU device recording what BudgetEncoder / WorkStamp do.
// ---------------------------------------------------------------------------

interface FakeLog {
  encoders: number;
  /** Per finished command buffer: the recorded ops ('stamp', 'dispatch:<label>', 'copy', 'clear'). */
  finished: string[][];
  submitted: string[][];
  waits: number;
}

function fakeDevice(): { device: GPUDevice; log: FakeLog; stampPipe: GPUComputePipeline } {
  const log: FakeLog = { encoders: 0, finished: [], submitted: [], waits: 0 };
  const stampPipe = { label: 'stamp', getBindGroupLayout: () => ({}) } as unknown as GPUComputePipeline;
  const device = {
    createBuffer: (d: GPUBufferDescriptor) => ({ size: d.size, destroy() {} }),
    createBindGroup: () => ({}),
    createCommandEncoder: () => {
      log.encoders++;
      const ops: string[] = [];
      let open = false;
      let current = '';
      const enc = {
        beginComputePass: () => {
          if (open) throw new Error('pass already open');
          open = true;
          return {
            setPipeline: (p: GPUComputePipeline) => {
              current = p.label;
            },
            setBindGroup: () => {},
            dispatchWorkgroups: () => {
              ops.push(current === 'stamp' ? 'stamp' : `dispatch:${current}`);
            },
            end: () => {
              open = false;
            },
          };
        },
        copyBufferToBuffer: () => {
          if (open) throw new Error('copy inside a pass');
          ops.push('copy');
        },
        clearBuffer: () => {
          if (open) throw new Error('clear inside a pass');
          ops.push('clear');
        },
        finish: () => {
          if (open) throw new Error('finish with an open pass');
          log.finished.push(ops);
          return { ops } as unknown as GPUCommandBuffer;
        },
      };
      return enc;
    },
    queue: {
      writeBuffer: () => {},
      submit: (cbs: GPUCommandBuffer[]) => {
        for (const cb of cbs) log.submitted.push((cb as unknown as { ops: string[] }).ops);
      },
      onSubmittedWorkDone: () => {
        log.waits++;
        return Promise.resolve();
      },
    },
  } as unknown as GPUDevice;
  return { device, log, stampPipe };
}

function pipe(label: string): GPUComputePipeline {
  return { label } as unknown as GPUComputePipeline;
}

describe('BudgetEncoder + WorkStamp (fake device)', () => {
  it('every submitted command buffer starts with exactly one stamp; nothing dangles after the last submit', async () => {
    const { device, log, stampPipe } = fakeDevice();
    const g = globalThis as Record<string, unknown>;
    const hadUsage = 'GPUBufferUsage' in g;
    if (!hadUsage) g.GPUBufferUsage = { UNIFORM: 64, COPY_DST: 8, STORAGE: 128, COPY_SRC: 4, MAP_READ: 1 };
    try {
      const stamp = new WorkStamp(device, stampPipe, { size: 16 } as GPUBuffer, 1);
      stamp.begin();
      const be = new BudgetEncoder(device, 100, new Promise(() => {}), 't', null, stamp);
      expect(log.encoders).toBe(0); // lazy
      await be.dispatch(pipe('a'), {} as GPUBindGroup, 1, 1, 1, 60);
      await be.dispatch(pipe('b'), {} as GPUBindGroup, 1, 1, 1, 60); // over budget: flush before
      be.encoder().clearBuffer({} as GPUBuffer, 0, 4);
      await be.dispatch(pipe('c'), {} as GPUBindGroup, 1, 1, 1, 10);
      be.encoder().copyBufferToBuffer({} as GPUBuffer, 0, {} as GPUBuffer, 0, 4);
      be.submit();
      // Readback, then more work: a new command buffer, again starting with a stamp.
      await be.dispatch(pipe('d'), {} as GPUBindGroup, 1, 1, 1, 10);
      be.submit();
      await be.flush(); // nothing recorded: no empty submission
      expect(be.submits).toBe(3);
      expect(log.submitted).toEqual([
        ['stamp', 'dispatch:a'],
        ['stamp', 'dispatch:b', 'clear', 'dispatch:c', 'copy'],
        ['stamp', 'dispatch:d'],
      ]);
      expect(log.encoders).toBe(3); // the pre-fix encoder began a 4th (stamp-only, never finished) one
      expect(log.finished.length).toBe(3);
    } finally {
      if (!hadUsage) delete g.GPUBufferUsage;
    }
  });

  it('dropped submissions are counted (so the stamp check detects them) and the profile mode waits per dispatch', async () => {
    const { device, log } = fakeDevice();
    const profile = new Map<string, number>();
    const be = new BudgetEncoder(device, 1e9, new Promise(() => {}), 't', profile, null, new Set([1]));
    await be.dispatch(pipe('x'), {} as GPUBindGroup, 1);
    await be.dispatch(pipe('y'), {} as GPUBindGroup, 1);
    await be.dispatch(pipe('z'), {} as GPUBindGroup, 1);
    expect(be.submits).toBe(3);
    expect(log.submitted).toEqual([['dispatch:x'], ['dispatch:z']]);
    expect([...profile.keys()].sort()).toEqual(['x', 'y', 'z']);
  });

  it('a lost device rejects a flush with GPUFeatureError', async () => {
    const { device } = fakeDevice();
    (device.queue as unknown as { onSubmittedWorkDone: () => Promise<void> }).onSubmittedWorkDone = () => new Promise(() => {});
    const be = new BudgetEncoder(device, 1, Promise.resolve(), 't');
    await be.dispatch(pipe('x'), {} as GPUBindGroup, 1, 1, 1, 5);
    await expect(be.dispatch(pipe('y'), {} as GPUBindGroup, 1, 1, 1, 5)).rejects.toBeInstanceOf(GPUFeatureError);
  });
});

describe('ParamBlock / dispatch2D', () => {
  it('writes u32 fields then f32 fields into zero-filled 256-byte slots and keeps earlier slots when growing', () => {
    const pb = new ParamBlock(1);
    const offs: number[] = [];
    for (let i = 0; i < 5; i++) offs.push(pb.add([i, 0xffffffff, 2 ** 31 + i], [0.5 + i, -1e-3]));
    expect(offs).toEqual([0, 256, 512, 768, 1024]);
    expect(pb.byteLength).toBe(5 * PARAM_SLOT);
    const b = pb.bytes();
    const u = new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4);
    const f = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
    for (let i = 0; i < 5; i++) {
      const o = (i * PARAM_SLOT) / 4;
      expect([u[o], u[o + 1], u[o + 2]]).toEqual([i, 0xffffffff, 2 ** 31 + i]);
      expect(f[o + 3]).toBe(Math.fround(0.5 + i));
      expect(f[o + 4]).toBe(Math.fround(-1e-3));
      for (let k = 5; k < PARAM_SLOT / 4; k++) expect(u[o + k]).toBe(0);
    }
    pb.reset();
    pb.add([7]);
    expect(new Uint32Array(pb.bytes().buffer, 0, 2)[1]).toBe(0); // slot re-zeroed
    expect(() => pb.add(new Array(60).fill(1), new Array(5).fill(1))).toThrow();
  });

  it('dispatch2D covers every 1-D index with x <= 32768 and y <= 65535', () => {
    for (const n of [0, 1, 32767, 32768, 32769, 65536 * 3 + 5, 2e6]) {
      const [x, y] = dispatch2D(n);
      expect(x).toBeLessThanOrEqual(32768);
      expect(y).toBeLessThanOrEqual(65535);
      expect(x * y).toBeGreaterThanOrEqual(Math.max(1, n));
      expect(x * (y - 1)).toBeLessThan(Math.max(1, n));
    }
  });
});

describe('descriptor fixed point', () => {
  it('cannot overflow a u32 bin for 0..1 images, keeps >= 2^12 resolution, 2^22 at the defaults', () => {
    expect(descriptorFixedPoint(1.6, 3)).toBe(2 ** 22);
    for (const S of [1, 2, 3, 4, 6, 8]) {
      for (const sigma of [0.8, 1.2, 1.6, 2.0, 3.2]) {
        const fix = descriptorFixedPoint(sigma, S);
        // Largest keypoint scale: level S + 0.6 (refinement offsets <= 0.6).
        const sbp = 3 * sigma * 2 ** ((S + 0.6) / S);
        const pixels = (2 * sbp + 2) ** 2; // samples within one bin's 2x2-bin support
        const worst = pixels * Math.SQRT2 * fix + pixels; // + 0.5 rounding per trilinear contribution (<= 1)
        if (fix > 2 ** 12) expect(worst, `S=${S} sigma=${sigma}`).toBeLessThan(2 ** 32);
        expect(fix).toBeGreaterThanOrEqual(2 ** 12);
        expect(Math.log2(fix) % 1).toBe(0);
      }
    }
  });
});

describe('decodeCandidates', () => {
  it('keeps identical integer locations in different octaves and refuses records beyond the buffer', () => {
    const o = resolveSiftOptions();
    const plan = planSift(640, 480, o);
    const raw = new Uint32Array(3 * CANDIDATE_WORDS);
    const f = new Float32Array(raw.buffer);
    for (let k = 0; k < 3; k++) {
      const oct = k === 2 ? 1 : 0;
      const s0 = k === 1 ? 2 : 1;
      f[k * 8] = 10.1;
      f[k * 8 + 1] = 20.2;
      f[k * 8 + 2] = 1.2;
      f[k * 8 + 3] = 0.05;
      raw[k * 8 + 4] = 10 | (20 << 16);
      raw[k * 8 + 5] = 1 | (s0 << 8) | (oct << 16);
      raw[k * 8 + 6] = 10 | (20 << 16);
    }
    const c = decodeCandidates(raw, 3, plan, 3, 1.6);
    // k=0 and k=1 converge to the same (octave 0) location: one kept; k=2 is octave 1.
    expect(c.n).toBe(2);
    expect(Array.from(c.octIdx.subarray(0, 2))).toEqual([0, 1]);
    // count larger than the records read: decodes only what is there
    expect(decodeCandidates(raw, 1000, plan, 3, 1.6).n).toBe(2);
  });
});

describe('validateFeatureSet', () => {
  const mk = (n: number): FeatureSet => ({
    count: n,
    keypoints: new Float32Array(4 * n),
    descriptors: new Uint8Array(D * n),
    scores: new Float32Array(n),
    width: 10,
    height: 10,
  });
  it('accepts well-formed (and over-long) sets, rejects short arrays / bad counts with a plain input error', () => {
    expect(() => validateFeatureSet(mk(5), 'a', true)).not.toThrow();
    expect(() => validateFeatureSet({ ...mk(5), count: 3 }, 'a', true)).not.toThrow();
    expect(() => validateFeatureSet(mk(0), 'a', true)).not.toThrow();
    const bad: FeatureSet[] = [
      { ...mk(5), descriptors: new Uint8Array(D * 5 - 1) },
      { ...mk(5), count: 6 },
      { ...mk(5), count: -1 },
      { ...mk(5), count: NaN },
      { ...mk(5), count: 2.5 },
      { ...mk(5), descriptors: new Float32Array(D * 5) as unknown as Uint8Array },
    ];
    for (const fs of bad) {
      let err: unknown = null;
      try {
        validateFeatureSet(fs, 'a', false);
      } catch (e) {
        err = e;
      }
      expect(err).toBeInstanceOf(FeatureInputError);
      expect(err).not.toBeInstanceOf(GPUFeatureError);
    }
    expect(() => validateFeatureSet({ ...mk(5), keypoints: new Float32Array(19) }, 'a', false)).not.toThrow();
    expect(() => validateFeatureSet({ ...mk(5), keypoints: new Float32Array(19) }, 'a', true)).toThrow(FeatureInputError);
  });
});
