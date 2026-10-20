/**
 * Host ⇄ GPU parameter layout (see wgsl/common.ts): one f32 array with SoA sections for a
 * capacity of `cap` Gaussians — means (vec4-padded), log-scales (vec4-padded), quaternions,
 * opacity logits, then SH (3·K floats per Gaussian).
 */
import { allocSplatModel, paramFloatsPerGaussian, shCoeffCount, type ShDegree, type SplatModel } from './types';

export interface ParamOffsets {
  mean: number;
  scale: number;
  quat: number;
  opacity: number;
  sh: number;
  /** Total floats. */
  total: number;
}

export function paramOffsets(cap: number, deg: ShDegree): ParamOffsets {
  return { mean: 0, scale: 4 * cap, quat: 8 * cap, opacity: 12 * cap, sh: 13 * cap, total: cap * paramFloatsPerGaussian(deg) };
}

/** Packs a model into the GPU layout for capacity `cap` (≥ model.count). */
export function packParams(model: SplatModel, cap: number, out?: Float32Array): Float32Array {
  if (model.count > cap) throw new Error(`packParams: ${model.count} Gaussians > capacity ${cap}`);
  const o = paramOffsets(cap, model.shDegree);
  const d = out ?? new Float32Array(o.total);
  const n = model.count;
  const SN = 3 * shCoeffCount(model.shDegree);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      d[o.mean + 4 * i + k] = model.means[3 * i + k];
      d[o.scale + 4 * i + k] = model.logScales[3 * i + k];
    }
    for (let k = 0; k < 4; k++) d[o.quat + 4 * i + k] = model.quats[4 * i + k];
    d[o.opacity + i] = model.opacities[i];
  }
  d.set(model.sh.subarray(0, n * SN), o.sh);
  return d;
}

/** Unpacks the first `count` Gaussians of a GPU parameter array. */
export function unpackParams(d: Float32Array, count: number, cap: number, deg: ShDegree): SplatModel {
  const o = paramOffsets(cap, deg);
  const m = allocSplatModel(count, deg);
  const SN = 3 * shCoeffCount(deg);
  for (let i = 0; i < count; i++) {
    for (let k = 0; k < 3; k++) {
      m.means[3 * i + k] = d[o.mean + 4 * i + k];
      m.logScales[3 * i + k] = d[o.scale + 4 * i + k];
    }
    for (let k = 0; k < 4; k++) m.quats[4 * i + k] = d[o.quat + 4 * i + k];
    m.opacities[i] = d[o.opacity + i];
  }
  m.sh.set(d.subarray(o.sh, o.sh + count * SN));
  return m;
}
