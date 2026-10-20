/**
 * WGSL for the splat forward pass: projection, binning (tile-intersection emission in depth
 * order), tile ranges and the tile rasteriser. See ../cpu-reference.ts for the math and
 * ../forward.ts for the buffers and the dispatch sequence.
 *
 * proj (3 vec4 per Gaussian): (mean2d.xy, depth, opacity), (conic A, B, C, 0), (colour rgb, 0)
 * with conic = [[A, B], [B, C]] (inverse of the filtered screen covariance).
 * rects (vec2u per Gaussian): (x0 | y0 << 16, x1 | y1 << 16), inclusive tile rectangle.
 */
import { ATOMIC_ADD_F32_WGSL, FRAME_WGSL, PARAMS_WGSL, PROJECT_WGSL, ROT_WGSL, constantsWGSL, type SplatShaderConfig } from './common';

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
  proj[3u * i] = vec4f(pr.mean2d, pr.depth, pr.opacity);
  proj[3u * i + 1u] = vec4f(pr.conic, 0.0);
  proj[3u * i + 2u] = vec4f(pr.colour, 0.0);
  rects[i] = vec2u(pr.rect.x | (pr.rect.y << 16u), pr.rect.z | (pr.rect.w << 16u));
  tileCount[i] = (pr.rect.z - pr.rect.x + 1u) * (pr.rect.w - pr.rect.y + 1u);
  depthKeys[i] = bitcast<u32>(pr.depth);
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

/** Emits (tile id, Gaussian) pairs in depth order at the scanned offsets. */
export const EMIT_WGSL = /* wgsl */ `
${FRAME_WGSL}
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> depthVals: array<u32>;
@group(0) @binding(2) var<storage, read> tileCount: array<u32>;
@group(0) @binding(3) var<storage, read> rects: array<vec2u>;
@group(0) @binding(4) var<storage, read> offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> keys: array<u32>;
@group(0) @binding(6) var<storage, read_write> vals: array<u32>;
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
  var o = offsets[j];
  for (var ty = y0; ty <= y1; ty++) {
    for (var tx = x0; tx <= x1; tx++) {
      if (o < cap) {
        keys[o] = ty * frame.dims.z + tx;
        vals[o] = g;
      }
      o++;
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
 * number of tile-list entries up to the last contributor (for the backward pass).
 */
export function rasterForwardWGSL(cfg: SplatShaderConfig): string {
  return /* wgsl */ `
${constantsWGSL(cfg)}
${FRAME_WGSL}
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> ranges: array<u32>;
@group(0) @binding(2) var<storage, read> isectVals: array<u32>;
@group(0) @binding(3) var<storage, read> proj: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> outColor: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> outN: array<u32>;
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
      sA[li] = proj[3u * g];
      sB[li] = proj[3u * g + 1u];
      sC[li] = proj[3u * g + 2u];
    }
    workgroupBarrier();
    if (!done) {
      let cnt = min(256u, n - b);
      for (var k = 0u; k < cnt; k++) {
        let s = sA[k];
        let d = pf - s.xy;
        let cn = sB[k];
        let sigma = 0.5 * (cn.x * d.x * d.x + cn.z * d.y * d.y) + cn.y * d.x * d.y;
        if (sigma < 0.0) { continue; }
        let alpha = min(ALPHA_MAX, s.w * exp(-sigma));
        if (alpha < ALPHA_MIN) { continue; }
        let Tn = T * (1.0 - alpha);
        if (Tn < T_MIN) {
          done = true;
          break;
        }
        C += sC[k].xyz * (alpha * T);
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
  }
}
`;
}

/**
 * Rasteriser backward, one 256-thread workgroup per tile, no per-pixel atomics.
 *
 * The 256 pixels form 8 groups of 32 (two tile rows each); thread (group g, lane j) owns the
 * splats i ≡ j (mod 32) of the tile list. Diagonal schedule: at step s, thread j processes
 * pixel k = (s − j) mod 32 of its group with splat i = s − k, so every pixel meets its splats
 * in front-to-back order (its state — transmittance T and r = gᵀ·colour still to come — is
 * handed from lane j to lane j + 1 through workgroup memory, one barrier per step) while each
 * thread accumulates one splat's 9 gradients over the 32 pixels of its group in registers.
 * When a splat is complete (k = 31) the thread stores its partial sums; every 32 steps the
 * 8 groups' partials of 32 splats are added and each non-zero sum goes to the global
 * accumulators with ONE compare-exchange float add per (splat, tile, component). Splats are
 * staged through a 64-entry ring (two blocks of 32). Front to back with the forward's final
 * colour: ∂C/∂α_i = T_i c_i − (C − C_{≤i}) / (1 − α_i), no division by transmittance.
 */
export function rasterBackwardWGSL(cfg: SplatShaderConfig): string {
  return /* wgsl */ `
${constantsWGSL(cfg)}
${FRAME_WGSL}
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> ranges: array<u32>;
@group(0) @binding(2) var<storage, read> isectVals: array<u32>;
@group(0) @binding(3) var<storage, read> proj: array<vec4f>;
@group(0) @binding(4) var<storage, read> outColor: array<vec4f>;
@group(0) @binding(5) var<storage, read> outN: array<u32>;
@group(0) @binding(6) var<storage, read> dLdC: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> grad2d: array<atomic<u32>>;
var<workgroup> pixT: array<f32, 256>;
var<workgroup> pixR: array<f32, 256>;
var<workgroup> pixN: array<u32, 256>;
var<workgroup> ringA: array<vec4f, 64>;
var<workgroup> ringB: array<vec4f, 64>;
var<workgroup> ringC: array<vec4f, 64>;
var<workgroup> ringId: array<u32, 64>;
var<workgroup> partial: array<f32, 2304>;
var<workgroup> kmax: atomic<u32>;
var<workgroup> kSnap: vec2u;
${ATOMIC_ADD_F32_WGSL('grad2d')}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let W = frame.dims.x;
  let H = frame.dims.y;
  let tile = wg.y * frame.dims.z + wg.x;
  let g = li >> 5u;
  let j = li & 31u;
  let ox = wg.x * TILE;
  let oy = wg.y * TILE;
  // pixel li initialises its state
  let ipx = ox + (li & 15u);
  let ipy = oy + (li >> 4u);
  if (li == 0u) { atomicStore(&kmax, 0u); }
  if (li < 64u) { ringId[li] = 0xffffffffu; }
  workgroupBarrier();
  pixT[li] = 1.0;
  if (ipx < W && ipy < H) {
    let o = ipy * W + ipx;
    let nn = outN[o];
    pixR[li] = dot(dLdC[o].xyz, outColor[o].xyz);
    pixN[li] = nn;
    atomicMax(&kmax, nn);
  } else {
    pixR[li] = 0.0;
    pixN[li] = 0u;
  }
  workgroupBarrier();
  if (li == 0u) { kSnap = vec2u(atomicLoad(&kmax), ranges[2u * tile]); }
  let ks = workgroupUniformLoad(&kSnap);
  let K = ks.x;
  let start = ks.y;
  let nb = (K + 31u) / 32u;
  var aMx = 0.0;
  var aMy = 0.0;
  var aA = 0.0;
  var aB = 0.0;
  var aC = 0.0;
  var aO = 0.0;
  var aR = 0.0;
  var aG = 0.0;
  var aBl = 0.0;
  for (var B = 0u; B <= nb; B++) {
    if (li < 32u) {
      let i = B * 32u + li;
      let slot = (B & 1u) * 32u + li;
      if (i < K) {
        let gid = isectVals[start + i];
        ringA[slot] = proj[3u * gid];
        ringB[slot] = proj[3u * gid + 1u];
        ringC[slot] = proj[3u * gid + 2u];
        ringId[slot] = gid;
      } else {
        ringId[slot] = 0xffffffffu;
      }
    }
    workgroupBarrier();
    for (var u = 0u; u < 32u; u++) {
      let k = (u + 32u - j) & 31u;
      let lp = g * 32u + k;
      let iS = i32(B * 32u + u) - i32(k);
      if (iS >= 0 && u32(iS) < pixN[lp]) {
        let slot = u32(iS) & 63u;
        let lx = lp & 15u;
        let ly = lp >> 4u;
        let sa = ringA[slot];
        let cn = ringB[slot];
        let d = vec2f(f32(ox + lx), f32(oy + ly)) - sa.xy;
        let sigma = 0.5 * (cn.x * d.x * d.x + cn.z * d.y * d.y) + cn.y * d.x * d.y;
        if (sigma >= 0.0) {
          let G = exp(-sigma);
          let a0 = sa.w * G;
          if (a0 >= ALPHA_MIN) {
            let alpha = min(ALPHA_MAX, a0);
            let T = pixT[lp];
            let w = alpha * T;
            let gp = dLdC[(oy + ly) * W + ox + lx].xyz;
            let col = ringC[slot].xyz;
            let gc = dot(gp, col);
            let r = pixR[lp] - w * gc;
            let dLda = T * gc - r / (1.0 - alpha);
            aR += w * gp.x;
            aG += w * gp.y;
            aBl += w * gp.z;
            if (a0 < ALPHA_MAX) {
              aO += G * dLda;
              let dLds = -alpha * dLda;
              aA += dLds * 0.5 * d.x * d.x;
              aB += dLds * d.x * d.y;
              aC += dLds * 0.5 * d.y * d.y;
              aMx -= dLds * (cn.x * d.x + cn.y * d.y);
              aMy -= dLds * (cn.y * d.x + cn.z * d.y);
            }
            pixT[lp] = T * (1.0 - alpha);
            pixR[lp] = r;
          }
        }
      }
      if (k == 31u) {
        let pb = (j * 8u + g) * 9u;
        partial[pb] = aMx;
        partial[pb + 1u] = aMy;
        partial[pb + 2u] = aA;
        partial[pb + 3u] = aB;
        partial[pb + 4u] = aC;
        partial[pb + 5u] = aO;
        partial[pb + 6u] = aR;
        partial[pb + 7u] = aG;
        partial[pb + 8u] = aBl;
        aMx = 0.0;
        aMy = 0.0;
        aA = 0.0;
        aB = 0.0;
        aC = 0.0;
        aO = 0.0;
        aR = 0.0;
        aG = 0.0;
        aBl = 0.0;
      }
      workgroupBarrier();
    }
    // lane 0's splat of this block and lanes 1..31's splats of the previous block are complete
    for (var t = li; t < 288u; t += 256u) {
      let jj = t / 9u;
      let c = t - 9u * jj;
      var s = 0.0;
      for (var gg = 0u; gg < 8u; gg++) { s += partial[(jj * 8u + gg) * 9u + c]; }
      if (s != 0.0) {
        let slot = select(((B + 1u) & 1u) * 32u + jj, (B & 1u) * 32u, jj == 0u);
        let gid = ringId[slot];
        if (gid != 0xffffffffu) { atomicAddF32(gid * 9u + c, s); }
      }
    }
    workgroupBarrier();
  }
}
`;
}
