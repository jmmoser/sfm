/**
 * Gaussian-splat PLY in the INRIA 3DGS attribute layout, which SuperSplat, Brush, Spark,
 * gsplat and most splat viewers read: binary little-endian float32 vertex properties
 *   x y z nx ny nz f_dc_0..2 f_rest_0..(3(K−1)−1) opacity scale_0..2 rot_0..3
 * with f_rest channel-major (all R coefficients of bands ≥ 1, then G, then B), opacity as a
 * logit, log-scales, and the quaternion (w, x, y, z) unnormalised. Normals are written as 0.
 * The world frame is the model's (Z-up, metres when the reconstruction is metric).
 *
 * A model trained with the Mip-Splatting opacity compensation (`antialiased`, the default) is
 * flagged with the header comment `antialiased`; viewers without that filter show small
 * splats slightly more opaque.
 */
import { allocSplatModel, shCoeffCount, type ShDegree, type SplatModel } from './types';

export interface SplatPLYOptions {
  antialiased?: boolean;
  comments?: string[];
}

export function encodeSplatPLY(m: SplatModel, o: SplatPLYOptions = {}): Uint8Array {
  const K = shCoeffCount(m.shDegree);
  const nRest = 3 * (K - 1);
  const props = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2'];
  for (let i = 0; i < nRest; i++) props.push(`f_rest_${i}`);
  props.push('opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3');
  const lines = ['ply', 'format binary_little_endian 1.0'];
  for (const c of o.comments ?? []) lines.push(`comment ${c.replace(/[\r\n]/g, ' ')}`);
  if (o.antialiased ?? true) lines.push('comment antialiased');
  lines.push(`element vertex ${m.count}`);
  for (const p of props) lines.push(`property float ${p}`);
  lines.push('end_header', '');
  const header = new TextEncoder().encode(lines.join('\n'));
  const P = props.length;
  const body = new Float32Array(m.count * P);
  for (let i = 0; i < m.count; i++) {
    const b = i * P;
    body[b] = m.means[3 * i];
    body[b + 1] = m.means[3 * i + 1];
    body[b + 2] = m.means[3 * i + 2];
    const sh = 3 * K * i;
    body[b + 6] = m.sh[sh];
    body[b + 7] = m.sh[sh + 1];
    body[b + 8] = m.sh[sh + 2];
    for (let c = 0; c < 3; c++) for (let k = 1; k < K; k++) body[b + 9 + c * (K - 1) + (k - 1)] = m.sh[sh + 3 * k + c];
    const r = 9 + nRest;
    body[b + r] = m.opacities[i];
    body[b + r + 1] = m.logScales[3 * i];
    body[b + r + 2] = m.logScales[3 * i + 1];
    body[b + r + 3] = m.logScales[3 * i + 2];
    for (let k = 0; k < 4; k++) body[b + r + 4 + k] = m.quats[4 * i + k];
  }
  const out = new Uint8Array(header.length + body.byteLength);
  out.set(header, 0);
  out.set(new Uint8Array(body.buffer), header.length);
  return out;
}

const TYPE_SIZE: Record<string, number> = { char: 1, uchar: 1, int8: 1, uint8: 1, short: 2, ushort: 2, int16: 2, uint16: 2, int: 4, uint: 4, int32: 4, uint32: 4, float: 4, float32: 4, double: 8, float64: 8 };

/**
 * Reads an INRIA-layout splat PLY (binary little-endian; float or double vertex properties,
 * other fixed-size properties skipped). SH bands above `maxDegree` (default 1) are dropped.
 */
export function decodeSplatPLY(data: Uint8Array, maxDegree: ShDegree = 1): { model: SplatModel; antialiased: boolean } {
  const probe = new TextDecoder().decode(data.subarray(0, Math.min(data.length, 1 << 16)));
  const end = probe.indexOf('end_header');
  if (!probe.startsWith('ply') || end < 0) throw new Error('splat PLY: not a PLY file');
  const headerLen = new TextEncoder().encode(probe.slice(0, end + 'end_header'.length)).length + 1 + (probe[end + 10] === '\r' ? 1 : 0);
  const lines = probe.slice(0, end).split(/\r?\n/);
  if (!lines.some((l) => l.startsWith('format binary_little_endian'))) throw new Error('splat PLY: only binary_little_endian is supported');
  let count = 0;
  let inVertex = false;
  let stride = 0;
  let antialiased = false;
  const props = new Map<string, { off: number; type: string }>();
  for (const l of lines) {
    const p = l.trim().split(/\s+/);
    if (p[0] === 'comment' && p[1] === 'antialiased') antialiased = true;
    if (p[0] === 'element') {
      if (inVertex) break;
      inVertex = p[1] === 'vertex';
      if (inVertex) count = Number(p[2]);
    } else if (p[0] === 'property' && inVertex) {
      if (p[1] === 'list') throw new Error('splat PLY: list property in vertex element');
      const size = TYPE_SIZE[p[1]];
      if (!size) throw new Error(`splat PLY: unknown type ${p[1]}`);
      props.set(p[2], { off: stride, type: p[1] });
      stride += size;
    }
  }
  for (const k of ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'])
    if (!props.has(k)) throw new Error(`splat PLY: missing property ${k}`);
  let nRest = 0;
  while (props.has(`f_rest_${nRest}`)) nRest++;
  const fileK = 1 + nRest / 3;
  const deg: ShDegree = maxDegree >= 1 && fileK >= 4 ? 1 : 0;
  const K = shCoeffCount(deg);
  if (data.length < headerLen + count * stride) throw new Error('splat PLY: truncated');
  const dv = new DataView(data.buffer, data.byteOffset + headerLen, count * stride);
  const get = (i: number, name: string): number => {
    const p = props.get(name)!;
    const o = i * stride + p.off;
    switch (p.type) {
      case 'float':
      case 'float32':
        return dv.getFloat32(o, true);
      case 'double':
      case 'float64':
        return dv.getFloat64(o, true);
      default:
        throw new Error(`splat PLY: property ${name} must be float`);
    }
  };
  const m = allocSplatModel(count, deg);
  for (let i = 0; i < count; i++) {
    m.means[3 * i] = get(i, 'x');
    m.means[3 * i + 1] = get(i, 'y');
    m.means[3 * i + 2] = get(i, 'z');
    for (let c = 0; c < 3; c++) m.sh[3 * K * i + c] = get(i, `f_dc_${c}`);
    for (let c = 0; c < 3; c++) for (let k = 1; k < K; k++) m.sh[3 * K * i + 3 * k + c] = get(i, `f_rest_${c * (fileK - 1) + (k - 1)}`);
    m.opacities[i] = get(i, 'opacity');
    for (let c = 0; c < 3; c++) m.logScales[3 * i + c] = get(i, `scale_${c}`);
    for (let c = 0; c < 4; c++) m.quats[4 * i + c] = get(i, `rot_${c}`);
  }
  return { model: m, antialiased };
}
