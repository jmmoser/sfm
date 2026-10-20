/**
 * WGSL for the GPU prefix sum and the 4-bit LSD radix sort (key, value) used by the splat
 * renderer (depth sort of the Gaussians, then a stable sort of the tile intersections by tile
 * id). Both are reduce-then-scan: no decoupled look-back, so they need no forward-progress
 * guarantee between workgroups (Apple GPUs give none). No subgroups, no float atomics.
 *
 * Blocks are 1024 elements (256 threads × 4 consecutive elements). Element counts live in a
 * small `ctl` storage buffer so a count produced on the GPU (the number of tile
 * intersections) drives the dispatches indirectly, without a readback:
 *   ctl[0] n (elements)         ctl[1] nb = ⌈n / 1024⌉      ctl[2] 16·nb (histogram length)
 *   ctl[4..6] indirect args (nb, 1, 1)   ctl[8..10] indirect args (⌈16·nb / 1024⌉, 1, 1)
 *   ctl[12] scan total (written by the scan's top level when cfg.totalSlot = 12)
 * The histogram is digit-major ([digit][block]) so that one exclusive scan over it yields the
 * global output offset of every (digit, block) pair; the scatter ranks elements stably inside
 * the block with a workgroup scan of per-thread digit counts (two 16-bit counts per u32).
 */

export const SORT_BLOCK = 1024;
/** Byte stride of the per-pass uniform slots. */
export const SORT_UNIFORM_STRIDE = 256;
/** ctl layout (u32 indices). */
export const CTL = { n: 0, nb: 1, histLen: 2, argsBlocks: 4, argsHistScan: 8, total: 12, words: 16 } as const;

const WG_SCAN = /* wgsl */ `
var<workgroup> scanTmp: array<u32, 256>;
// Inclusive scan of one value per thread over the 256 threads (Hillis–Steele).
fn wgScanInclusive(li: u32, v: u32) -> u32 {
  scanTmp[li] = v;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off = off << 1u) {
    var t = scanTmp[li];
    if (li >= off) { t = t + scanTmp[li - off]; }
    workgroupBarrier();
    scanTmp[li] = t;
    workgroupBarrier();
  }
  return scanTmp[li];
}
`;

/** Scan uniforms: which ctl slot holds n, where to write the total (0xffffffff = nowhere). */
const SCAN_CFG = /* wgsl */ `
struct ScanCfg { nSlot: u32, totalSlot: u32, pad0: u32, pad1: u32 }
`;

/** Level 1: per 1024-block sums. */
export const SCAN_REDUCE_WGSL = /* wgsl */ `
${SCAN_CFG}
@group(0) @binding(0) var<uniform> cfg: ScanCfg;
@group(0) @binding(1) var<storage, read> ctl: array<u32>;
@group(0) @binding(2) var<storage, read> data: array<u32>;
@group(0) @binding(3) var<storage, read_write> blockSums: array<u32>;
var<workgroup> red: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let n = ctl[cfg.nSlot];
  let base = wg.x * 1024u + li * 4u;
  var s = 0u;
  for (var k = 0u; k < 4u; k++) {
    if (base + k < n) { s = s + data[base + k]; }
  }
  red[li] = s;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st = st >> 1u) {
    if (li < st) { red[li] = red[li] + red[li + st]; }
    workgroupBarrier();
  }
  if (li == 0u) { blockSums[wg.x] = red[0]; }
}
`;

/** Level 2: one workgroup scans ≤ 1024 block sums in place (exclusive) and writes the total. */
export const SCAN_TOP_WGSL = /* wgsl */ `
${SCAN_CFG}
@group(0) @binding(0) var<uniform> cfg: ScanCfg;
@group(0) @binding(1) var<storage, read_write> ctl: array<u32>;
@group(0) @binding(2) var<storage, read_write> blockSums: array<u32>;
${WG_SCAN}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  let n = ctl[cfg.nSlot];
  let nb = (n + 1023u) / 1024u;
  var v: array<u32, 4>;
  var s = 0u;
  for (var k = 0u; k < 4u; k++) {
    let i = li * 4u + k;
    v[k] = select(0u, blockSums[i], i < nb);
    s = s + v[k];
  }
  let incl = wgScanInclusive(li, s);
  var run = incl - s;
  for (var k = 0u; k < 4u; k++) {
    let i = li * 4u + k;
    if (i < nb) { blockSums[i] = run; }
    run = run + v[k];
  }
  if (li == 255u && cfg.totalSlot != 0xffffffffu) { ctl[cfg.totalSlot] = incl; }
}
`;

/** Level 3: exclusive scan inside each block plus the block's prefix. */
export const SCAN_DOWN_WGSL = /* wgsl */ `
${SCAN_CFG}
@group(0) @binding(0) var<uniform> cfg: ScanCfg;
@group(0) @binding(1) var<storage, read> ctl: array<u32>;
@group(0) @binding(2) var<storage, read_write> data: array<u32>;
@group(0) @binding(3) var<storage, read> blockSums: array<u32>;
${WG_SCAN}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let n = ctl[cfg.nSlot];
  let base = wg.x * 1024u + li * 4u;
  var v: array<u32, 4>;
  var s = 0u;
  for (var k = 0u; k < 4u; k++) {
    v[k] = select(0u, data[base + k], base + k < n);
    s = s + v[k];
  }
  let incl = wgScanInclusive(li, s);
  var run = incl - s + blockSums[wg.x];
  for (var k = 0u; k < 4u; k++) {
    if (base + k < n) { data[base + k] = run; }
    run = run + v[k];
  }
}
`;

/** Computes nb and the indirect dispatch arguments from ctl[0]. */
export const SORT_SETUP_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> ctl: array<u32>;
@compute @workgroup_size(1)
fn main() {
  let n = ctl[0];
  let nb = (n + 1023u) / 1024u;
  ctl[1] = nb;
  ctl[2] = 16u * nb;
  ctl[4] = nb;
  ctl[5] = 1u;
  ctl[6] = 1u;
  ctl[8] = (16u * nb + 1023u) / 1024u;
  ctl[9] = 1u;
  ctl[10] = 1u;
}
`;

const PASS_CFG = /* wgsl */ `
struct PassCfg { shift: u32, pad0: u32, pad1: u32, pad2: u32 }
`;

/** Per-block digit histogram, written digit-major. */
export const RADIX_COUNT_WGSL = /* wgsl */ `
${PASS_CFG}
@group(0) @binding(0) var<uniform> cfg: PassCfg;
@group(0) @binding(1) var<storage, read> ctl: array<u32>;
@group(0) @binding(2) var<storage, read> keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> hist: array<u32>;
var<workgroup> lh: array<atomic<u32>, 16>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li < 16u) { atomicStore(&lh[li], 0u); }
  workgroupBarrier();
  let n = ctl[0];
  let nb = ctl[1];
  let base = wg.x * 1024u + li * 4u;
  for (var k = 0u; k < 4u; k++) {
    if (base + k < n) {
      atomicAdd(&lh[(keys[base + k] >> cfg.shift) & 15u], 1u);
    }
  }
  workgroupBarrier();
  if (li < 16u) { hist[li * nb + wg.x] = atomicLoad(&lh[li]); }
}
`;

/** Stable scatter of a block to its scanned (digit, block) offsets. */
export const RADIX_SCATTER_WGSL = /* wgsl */ `
${PASS_CFG}
@group(0) @binding(0) var<uniform> cfg: PassCfg;
@group(0) @binding(1) var<storage, read> ctl: array<u32>;
@group(0) @binding(2) var<storage, read> keysIn: array<u32>;
@group(0) @binding(3) var<storage, read> valsIn: array<u32>;
@group(0) @binding(4) var<storage, read_write> keysOut: array<u32>;
@group(0) @binding(5) var<storage, read_write> valsOut: array<u32>;
@group(0) @binding(6) var<storage, read> hist: array<u32>;
var<workgroup> sc: array<u32, 2048>;
var<workgroup> digitBase: array<u32, 16>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let n = ctl[0];
  let nb = ctl[1];
  if (li < 16u) { digitBase[li] = hist[li * nb + wg.x]; }
  let base = wg.x * 1024u + li * 4u;
  var key: array<u32, 4>;
  var val: array<u32, 4>;
  var rank: array<u32, 4>;
  var cnt: array<u32, 8>;
  for (var w = 0u; w < 8u; w++) { cnt[w] = 0u; }
  for (var k = 0u; k < 4u; k++) {
    let i = base + k;
    if (i < n) {
      key[k] = keysIn[i];
      val[k] = valsIn[i];
      let d = (key[k] >> cfg.shift) & 15u;
      let sh = (d & 1u) * 16u;
      rank[k] = (cnt[d >> 1u] >> sh) & 0xffffu;
      cnt[d >> 1u] = cnt[d >> 1u] + (1u << sh);
    }
  }
  for (var w = 0u; w < 8u; w++) { sc[li * 8u + w] = cnt[w]; }
  workgroupBarrier();
  for (var off = 1u; off < 256u; off = off << 1u) {
    var t: array<u32, 8>;
    for (var w = 0u; w < 8u; w++) {
      t[w] = sc[li * 8u + w];
      if (li >= off) { t[w] = t[w] + sc[(li - off) * 8u + w]; }
    }
    workgroupBarrier();
    for (var w = 0u; w < 8u; w++) { sc[li * 8u + w] = t[w]; }
    workgroupBarrier();
  }
  for (var k = 0u; k < 4u; k++) {
    let i = base + k;
    if (i < n) {
      let d = (key[k] >> cfg.shift) & 15u;
      let sh = (d & 1u) * 16u;
      let excl = ((sc[li * 8u + (d >> 1u)] - cnt[d >> 1u]) >> sh) & 0xffffu;
      let dst = digitBase[d] + excl + rank[k];
      keysOut[dst] = key[k];
      valsOut[dst] = val[k];
    }
  }
}
`;
