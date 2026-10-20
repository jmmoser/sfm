/**
 * WGSL for the splat forward pass: projection, binning (tile-intersection emission in depth
 * order), tile ranges and the tile rasteriser. See ../cpu-reference.ts for the math and
 * ../forward.ts for the buffers and the dispatch sequence.
 *
 * proj (3 vec4 per Gaussian): (mean2d.xy, depth, opacity), (conic A, B, C, ln(o / α_min)),
 * (colour rgb, 0) with conic = [[A, B], [B, C]] (inverse of the filtered screen covariance); with
 * the geometry channels a 4th row (camera-frame normal, 0).
 * geo (geometry channels, 2 vec4 per pixel): [2o] = the rendered Σ w·(1/z, n) (forward), [2o + 1] =
 * ∂loss/∂(that) (the geometry loss kernel, wgsl/loss.ts), read by the backward.
 * rects (vec2u per Gaussian): (x0 | y0 << 16, x1 | y1 << 16), inclusive tile rectangle.
 */
import { ALPHA_MIN, TILE_SIZE } from '../types';
import { ATOMIC_ADD_F32_WGSL, FRAME_WGSL, PARAMS_WGSL, PROJECT_WGSL, ROT_WGSL, TILE_TEST_WGSL, constantsWGSL, grad2dStride, projStride, type SplatShaderConfig } from './common';

/** Per-Gaussian projection → proj, rect, tile count, depth-sort key / value. */
export function projectWGSL(cfg: SplatShaderConfig): string {
  return /* wgsl */ `
${constantsWGSL(cfg)}
${FRAME_WGSL}
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read_write> proj: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> rects: array<vec2u>;
@group(0) @binding(4) var<storage, read_write> tileCount: array<u32>;
@group(0) @binding(5) var<storage, read_write> depthKeys: array<u32>;
@group(0) @binding(6) var<storage, read_write> depthVals: array<u32>;
${PARAMS_WGSL}
${ROT_WGSL}
${PROJECT_WGSL}
${TILE_TEST_WGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= frame.info.x) { return; }
  let pr = projectGaussian(i);
  depthVals[i] = i;
  if (!pr.visible) {
    tileCount[i] = 0u;
    depthKeys[i] = 0xffffffffu;
    return;
  }
  proj[PS * i] = vec4f(pr.mean2d, pr.depth, pr.opacity);
  proj[PS * i + 1u] = vec4f(pr.conic, log(pr.opacity / ALPHA_MIN));
  proj[PS * i + 2u] = vec4f(pr.colour, 0.0);
  if (GEOMETRY) { proj[4u * i + 3u] = vec4f(pr.normal, 0.0); }
  rects[i] = vec2u(pr.rect.x | (pr.rect.y << 16u), pr.rect.z | (pr.rect.w << 16u));
  let tau = 2.0 * log(pr.opacity / ALPHA_MIN);
  var n = 0u;
  for (var ty = pr.rect.y; ty <= pr.rect.w; ty++) {
    for (var tx = pr.rect.x; tx <= pr.rect.z; tx++) {
      if (tileTouched(pr.mean2d, pr.conic, tau, tx, ty)) { n++; }
    }
  }
  tileCount[i] = n;
  depthKeys[i] = select(0xffffffffu, bitcast<u32>(pr.depth), n > 0u);
}
`;
}

/** Tile counts in depth order (input of the intersection-offset scan). */
export const GATHER_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> ctl: array<u32>;
@group(0) @binding(1) var<storage, read> depthVals: array<u32>;
@group(0) @binding(2) var<storage, read> tileCount: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let j = gid.x;
  if (j >= ctl[0]) { return; }
  counts[j] = tileCount[depthVals[j]];
}
`;

/**
 * Number of intersections to sort = min(total, capacity) → the intersection sorter's ctl[0];
 * binCtl[13] keeps the uncapped total (overflow check on the host).
 */
export const BIN_SETUP_WGSL = /* wgsl */ `
${FRAME_WGSL}
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read_write> binCtl: array<u32>;
@group(0) @binding(2) var<storage, read_write> isectCtl: array<u32>;
@compute @workgroup_size(1)
fn main() {
  let total = binCtl[12];
  binCtl[13] = total;
  isectCtl[0] = min(total, frame.info.z);
}
`;

/** Emits (tile id, Gaussian) pairs in depth order at the scanned offsets (touched tiles only). */
export const emitWGSL = (cfg: Pick<SplatShaderConfig, 'geometry' | 'wideProj'>): string => /* wgsl */ `
const TILE: u32 = ${TILE_SIZE}u;
const ALPHA_MIN: f32 = ${ALPHA_MIN};
const PS: u32 = ${projStride(cfg)}u;
${FRAME_WGSL}
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> depthVals: array<u32>;
@group(0) @binding(2) var<storage, read> tileCount: array<u32>;
@group(0) @binding(3) var<storage, read> rects: array<vec2u>;
@group(0) @binding(4) var<storage, read> offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> keys: array<u32>;
@group(0) @binding(6) var<storage, read_write> vals: array<u32>;
@group(0) @binding(7) var<storage, read> proj: array<vec4f>;
${TILE_TEST_WGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let j = gid.x;
  if (j >= frame.info.x) { return; }
  let g = depthVals[j];
  if (tileCount[g] == 0u) { return; }
  let r = rects[g];
  let x0 = r.x & 0xffffu;
  let y0 = r.x >> 16u;
  let x1 = r.y & 0xffffu;
  let y1 = r.y >> 16u;
  let cap = frame.info.z;
  let pa = proj[PS * g];
  let conic = proj[PS * g + 1u].xyz;
  let tau = 2.0 * log(pa.w / ALPHA_MIN);
  var o = offsets[j];
  for (var ty = y0; ty <= y1; ty++) {
    for (var tx = x0; tx <= x1; tx++) {
      if (tileTouched(pa.xy, conic, tau, tx, ty)) {
        if (o < cap) {
          keys[o] = ty * frame.dims.z + tx;
          vals[o] = g;
        }
        o++;
      }
    }
  }
}
`;

/** Start / end of each tile's run in the sorted intersections (ranges cleared to 0 before). */
export const RANGES_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> ctl: array<u32>;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> ranges: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let n = ctl[0];
  for (var k = 0u; k < 4u; k++) {
    let i = wg.x * 1024u + k * 256u + li;
    if (i < n) {
      let t = keys[i];
      if (i == 0u || keys[i - 1u] != t) { ranges[2u * t] = i; }
      if (i + 1u == n || keys[i + 1u] != t) { ranges[2u * t + 1u] = i + 1u; }
    }
  }
}
`;

/**
 * Tile rasteriser: one 16×16 workgroup per tile; splats are loaded 256 at a time into
 * workgroup memory and composited front to back. Writes (colour + T·background, T) and the
 * number of tile-list entries up to the last contributor (for the backward pass); with the
 * geometry channels also Σ w·(1/z, n) to geo[2o] (the normal staged as sC.w + sD: 14 KB).
 */
export function rasterForwardWGSL(cfg: SplatShaderConfig): string {
  const geo = !!cfg.geometry;
  return /* wgsl */ `
${constantsWGSL(cfg)}
${FRAME_WGSL}
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> ranges: array<u32>;
@group(0) @binding(2) var<storage, read> isectVals: array<u32>;
@group(0) @binding(3) var<storage, read> proj: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> outColor: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> outN: array<u32>;
${geo ? `@group(0) @binding(6) var<storage, read_write> geo: array<vec4f>;
var<workgroup> sD: array<vec2f, 256>;` : ''}
var<workgroup> sA: array<vec4f, 256>;
var<workgroup> sB: array<vec4f, 256>;
var<workgroup> sC: array<vec4f, 256>;
var<workgroup> doneCount: atomic<u32>;
var<workgroup> doneSnap: u32;
var<workgroup> rangeSnap: vec2u;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) li: u32) {
  let W = frame.dims.x;
  let H = frame.dims.y;
  let tile = wg.y * frame.dims.z + wg.x;
  let px = wg.x * TILE + lid.x;
  let py = wg.y * TILE + lid.y;
  let inside = px < W && py < H;
  if (li == 0u) {
    atomicStore(&doneCount, 0u);
    rangeSnap = vec2u(ranges[2u * tile], ranges[2u * tile + 1u]);
  }
  workgroupBarrier();
  let range = workgroupUniformLoad(&rangeSnap);
  let n = select(0u, range.y - range.x, range.y > range.x);
  let pf = vec2f(f32(px), f32(py));
  var T = 1.0;
  var C = vec3f(0.0);
  var Gd = vec4f(0.0);
  var last = 0u;
  var done = !inside;
  if (done) { atomicAdd(&doneCount, 1u); }
  for (var b = 0u; b < n; b += 256u) {
    workgroupBarrier();
    if (li == 0u) { doneSnap = atomicLoad(&doneCount); }
    if (workgroupUniformLoad(&doneSnap) >= 256u) { break; }
    let idx = range.x + b + li;
    if (idx < range.y) {
      let g = isectVals[idx];
      sA[li] = proj[PS * g];
      sB[li] = proj[PS * g + 1u];
      ${geo ? `let nrm = proj[4u * g + 3u].xyz;
      sC[li] = vec4f(proj[4u * g + 2u].xyz, nrm.x);
      sD[li] = nrm.yz;` : 'sC[li] = proj[PS * g + 2u];'}
    }
    workgroupBarrier();
    if (!done) {
      let cnt = min(256u, n - b);
      for (var k = 0u; k < cnt; k++) {
        let s = sA[k];
        let d = pf - s.xy;
        let cn = sB[k];
        let sigma = 0.5 * (cn.x * d.x * d.x + cn.z * d.y * d.y) + cn.y * d.x * d.y;
        if (sigma < 0.0 || sigma > cn.w) { continue; }
        let alpha = min(ALPHA_MAX, s.w * exp(-sigma));
        if (alpha < ALPHA_MIN) { continue; }
        let Tn = T * (1.0 - alpha);
        if (Tn < T_MIN) {
          done = true;
          break;
        }
        C += sC[k].xyz * (alpha * T);
        ${geo ? 'Gd += (alpha * T) * vec4f(1.0 / s.z, sC[k].w, sD[k]);' : ''}
        T = Tn;
        last = b + k + 1u;
      }
      if (done) { atomicAdd(&doneCount, 1u); }
    }
  }
  if (inside) {
    let o = py * W + px;
    outColor[o] = vec4f(C + T * frame.bg.xyz, T);
    outN[o] = last;
    ${geo ? 'geo[2u * o] = Gd;' : ''}
  }
}
`;
}

/**
 * Rasteriser backward without per-pixel atomics.
 *
 * A tile's 256 pixels form 8 groups of 32 (two tile rows each); a workgroup takes `groups` of
 * them (32·groups threads; the dispatch has 8 / groups workgroups per tile in z). The tile
 * list is cut into slots of `perStep` consecutive splats; thread (group g, lane j) owns the
 * slots m ≡ j (mod 32). Diagonal schedule: at step s, thread j processes pixel k = (s − j) mod 32
 * of its group with the splats of slot m = s − k, so every pixel meets its splats in
 * front-to-back order (its state — transmittance T and r = gᵀ·colour still to come — is handed
 * from lane j to lane j + 1 through workgroup memory, one barrier per step) while each thread
 * accumulates its slot's 9 gradients per splat over the 32 pixels of its group in registers.
 * When a slot is complete (k = 31) the thread stores its partial sums; every 32 steps the
 * workgroup's group partials of 32 slots are added and each non-zero sum goes to the global
 * accumulators with ONE compare-exchange float add per (splat, workgroup, component). Splats are
 * staged through a ring of two blocks of 32 slots. A workgroup runs ⌈max(n) / perStep⌉ + 32
 * steps over its own pixels (n = last contributor + 1 from the forward pass). Front to back
 * with the forward's final colour: ∂C/∂α_i = T_i c_i − (C − C_{≤i}) / (1 − α_i), no division
 * by the transmittance.
 */
export function rasterBackwardWGSL(cfg: SplatShaderConfig, groups = 2, perStep = 2): string {
  if (![1, 2, 4, 8].includes(groups)) throw new Error(`rasterBackwardWGSL: groups ${groups}`);
  if (![1, 2, 4].includes(perStep)) throw new Error(`rasterBackwardWGSL: perStep ${perStep}`);
  const NT = 32 * groups;
  const Q = perStep;
  const RING = 64 * Q;
  const geo = !!cfg.geometry;
  const NA = grad2dStride(cfg);
  const qs = Array.from({ length: Q }, (_, q) => q);
  const accDecl = qs.map((q) => `  var acc${q}: array<f32, ${NA}>;\n  for (var c = 0u; c < ${NA}u; c++) { acc${q}[c] = 0.0; }`).join('\n');
  const evalSplat = (q: number): string => /* wgsl */ `
        {
          let i = m * ${Q}u + ${q}u;
          if (i < n) {
            let slot = i % ${RING}u;
            let sa = ringA[slot];
            let cn = ringB[slot];
            let d = pf - sa.xy;
            let sigma = 0.5 * (cn.x * d.x * d.x + cn.z * d.y * d.y) + cn.y * d.x * d.y;
            if (sigma >= 0.0 && sigma <= cn.w) {
              let G = exp(-sigma);
              let a0 = sa.w * G;
              if (a0 >= ALPHA_MIN) {
                let alpha = min(ALPHA_MAX, a0);
                let w = alpha * T;
                ${
                  geo
                    ? `let rc = ringC[slot];
                let gc = dot(gp.xyz, rc.xyz) + gp.w + gG.x / sa.z + dot(gG.yzw, vec3f(rc.w, ringD[slot]));
                acc${q}[9] += w * gG.x;
                acc${q}[10] += w * gG.y;
                acc${q}[11] += w * gG.z;
                acc${q}[12] += w * gG.w;`
                    : 'let gc = dot(gp, ringC[slot].xyz);'
                }
                r = r - w * gc;
                let dLda = T * gc - r / (1.0 - alpha);
                acc${q}[6] += w * gp.x;
                acc${q}[7] += w * gp.y;
                acc${q}[8] += w * gp.z;
                if (a0 < ALPHA_MAX) {
                  acc${q}[5] += G * dLda;
                  let dLds = -alpha * dLda;
                  acc${q}[2] += dLds * 0.5 * d.x * d.x;
                  acc${q}[3] += dLds * d.x * d.y;
                  acc${q}[4] += dLds * 0.5 * d.y * d.y;
                  acc${q}[0] -= dLds * (cn.x * d.x + cn.y * d.y);
                  acc${q}[1] -= dLds * (cn.y * d.x + cn.z * d.y);
                }
                T = T * (1.0 - alpha);
              }
            }
          }
        }`;
  const storePartial = qs
    .map((q) => `        for (var c = 0u; c < ${NA}u; c++) {\n          partial[((j * ${Q}u + ${q}u) * NG + g) * ${NA}u + c] = acc${q}[c];\n          acc${q}[c] = 0.0;\n        }`)
    .join('\n');
  return /* wgsl */ `
${constantsWGSL(cfg)}
${FRAME_WGSL}
const NG: u32 = ${groups}u;
const NT: u32 = ${NT}u;
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> ranges: array<u32>;
@group(0) @binding(2) var<storage, read> isectVals: array<u32>;
@group(0) @binding(3) var<storage, read> proj: array<vec4f>;
@group(0) @binding(4) var<storage, read> outColor: array<vec4f>;
@group(0) @binding(5) var<storage, read> outN: array<u32>;
@group(0) @binding(6) var<storage, read> dLdC: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> grad2d: array<atomic<u32>>;
${geo ? `@group(0) @binding(8) var<storage, read> geo: array<vec4f>;
var<workgroup> pixGG: array<vec4f, ${NT}>;
var<workgroup> ringD: array<vec2f, ${RING}>;` : ''}
var<workgroup> pixT: array<f32, ${NT}>;
var<workgroup> pixR: array<f32, ${NT}>;
var<workgroup> pixN: array<u32, ${NT}>;
var<workgroup> pixG: array<vec4f, ${NT}>;
var<workgroup> ringA: array<vec4f, ${RING}>;
var<workgroup> ringB: array<vec4f, ${RING}>;
var<workgroup> ringC: array<vec4f, ${RING}>;
var<workgroup> ringId: array<u32, ${RING}>;
var<workgroup> partial: array<f32, ${32 * Q * groups * NA}>;
var<workgroup> kmax: atomic<u32>;
var<workgroup> kSnap: vec2u;
${ATOMIC_ADD_F32_WGSL('grad2d')}
@compute @workgroup_size(${NT})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let W = frame.dims.x;
  let H = frame.dims.y;
  let tile = wg.y * frame.dims.z + wg.x;
  let g = li >> 5u;
  let j = li & 31u;
  let ox = wg.x * TILE;
  let oy = wg.y * TILE;
  // first tile pixel of this workgroup (groups are two tile rows each)
  let p0 = wg.z * NT;
  // thread li initialises pixel p0 + li
  let ipx = ox + ((p0 + li) & 15u);
  let ipy = oy + ((p0 + li) >> 4u);
  if (li == 0u) { atomicStore(&kmax, 0u); }
  for (var e = li; e < ${RING}u; e += NT) { ringId[e] = 0xffffffffu; }
  workgroupBarrier();
  pixT[li] = 1.0;
  if (ipx < W && ipy < H) {
    let o = ipy * W + ipx;
    let nn = outN[o];
    let gp = dLdC[o];
    pixR[li] = dot(gp.xyz, outColor[o].xyz);
    ${geo ? `let gg = geo[2u * o + 1u];
    pixR[li] += dot(gg, geo[2u * o]) + gp.w * (1.0 - outColor[o].w);
    pixGG[li] = gg;` : ''}
    pixN[li] = nn;
    pixG[li] = gp;
    atomicMax(&kmax, nn);
  } else {
    pixR[li] = 0.0;
    pixN[li] = 0u;
    pixG[li] = vec4f(0.0);
    ${geo ? 'pixGG[li] = vec4f(0.0);' : ''}
  }
  workgroupBarrier();
  if (li == 0u) { kSnap = vec2u(atomicLoad(&kmax), ranges[2u * tile]); }
  let ks = workgroupUniformLoad(&kSnap);
  let K = ks.x;
  let start = ks.y;
  let nb = (K + ${32 * Q - 1}u) / ${32 * Q}u;
${accDecl}
  for (var B = 0u; B <= nb; B++) {
    for (var e = li; e < ${32 * Q}u; e += NT) {
      let i = B * ${32 * Q}u + e;
      let slot = i % ${RING}u;
      if (i < K) {
        let gid = isectVals[start + i];
        ringA[slot] = proj[PS * gid];
        ringB[slot] = proj[PS * gid + 1u];
        ${geo ? `let nrm = proj[4u * gid + 3u].xyz;
        ringC[slot] = vec4f(proj[4u * gid + 2u].xyz, nrm.x);
        ringD[slot] = nrm.yz;` : 'ringC[slot] = proj[PS * gid + 2u];'}
        ringId[slot] = gid;
      } else {
        ringId[slot] = 0xffffffffu;
      }
    }
    workgroupBarrier();
    for (var u = 0u; u < 32u; u++) {
      let k = (u + 32u - j) & 31u;
      let lp = g * 32u + k;
      let mS = i32(B * 32u + u) - i32(k);
      let n = pixN[lp];
      if (mS >= 0 && u32(mS) * ${Q}u < n) {
        let m = u32(mS);
        let lx = (p0 + lp) & 15u;
        let ly = (p0 + lp) >> 4u;
        let pf = vec2f(f32(ox + lx), f32(oy + ly));
        let gp = pixG[lp]${geo ? '' : '.xyz'};
        ${geo ? 'let gG = pixGG[lp];' : ''}
        var T = pixT[lp];
        var r = pixR[lp];
${qs.map(evalSplat).join('\n')}
        pixT[lp] = T;
        pixR[lp] = r;
      }
      if (k == 31u) {
${storePartial}
      }
      workgroupBarrier();
    }
    // lane 0's slot of this block and lanes 1..31's slots of the previous block are complete
    for (var t = li; t < ${32 * Q * NA}u; t += NT) {
      let jq = t / ${NA}u;
      let c = t - ${NA}u * jq;
      let jj = jq / ${Q}u;
      let q = jq - ${Q}u * jj;
      var s = 0.0;
      for (var gg = 0u; gg < NG; gg++) { s += partial[(jq * NG + gg) * ${NA}u + c]; }
      if (s != 0.0) {
        let m = select(32u * (B - 1u) + jj, 32u * B, jj == 0u);
        let gid = ringId[(m * ${Q}u + q) % ${RING}u];
        if (gid != 0xffffffffu) { atomicAddF32(gid * ${NA}u + c, s); }
      }
    }
    workgroupBarrier();
  }
}
`;
}
