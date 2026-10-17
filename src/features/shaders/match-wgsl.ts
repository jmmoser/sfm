/**
 * WGSL kernel of the WebGPU descriptor matcher (src/features/match-gpu.ts).
 *
 * The brute-force distance matrix is computed GEMM-style: a workgroup of 16x16
 * invocations owns a tile of 64 "row" descriptors and walks tiles of 64 "column"
 * descriptors; 16-dimension chunks of both tiles are unpacked from the packed
 * 4-bytes-per-u32 layout (manual shifts) into workgroup memory as f32 and every
 * invocation accumulates a 4x4 block of exact integer dot products (products of
 * bytes <= 65025, sums <= 8.3e6 < 2^24 are exact in f32). Each invocation keeps the
 * best / second-best column (score = dot * invNorm(col), ties -> smaller index) of
 * its 4 rows; the 16 partial lists of a row are merged in workgroup memory and
 * (index, dot) of the best two are written per (row, column chunk). The host turns
 * those into exact f64 similarities and applies ratio / mutual / maxDistance exactly
 * as matchCPU.
 *
 * Guided variants evaluate the guide constraint per (row, column) pair and skip whole
 * column tiles that cannot satisfy it for any row of the tile (rows and columns are
 * spatially sorted by the host; tile boxes are precomputed):
 *   mode 'pos': a-record (px, py, flag, 0) with flag 0 = normal, 1 = unconstrained,
 *               2 = never; pass if |pos_b - (px, py)|^2 <= r^2 (raw pixel coordinates).
 *   mode 'epi': a-record (l0, l1, l2, flag) = unit-normalised epipolar line in b with
 *               b coordinates centred at the image centre; pass if |l . (x, y, 1)| <= tol.
 * `reverse` swaps the roles (rows = b, columns = a) for the mutual check.
 */

export type MatchKernelMode = 'plain' | 'pos' | 'epi';

export const MATCH_TILE = 64;
export const MATCH_INVALID = 0xffffffff;

const COMP = ['x', 'y', 'z', 'w'];

export function matchShader(mode: MatchKernelMode, reverse: boolean): string {
  const guided = mode !== 'plain';
  const lines: string[] = [];
  lines.push(/* wgsl */ `
const INVALID: u32 = 0xffffffffu;
struct Params {
  nRows: u32, nCols: u32, rowTile0: u32, tilesPerChunk: u32,
  nColTiles: u32, nChunks: u32, outOff: u32, aOff: u32,
  bOff: u32, boxOff: u32,
  r2: f32, radius: f32, tol: f32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> rowDesc: array<u32>;
@group(0) @binding(2) var<storage, read> colDesc: array<u32>;
@group(0) @binding(3) var<storage, read> colInv: array<f32>;
@group(0) @binding(4) var<storage, read_write> outBuf: array<vec4<u32>>;
var<workgroup> sh: array<vec4<u32>, 512>;
`);
  if (guided) {
    lines.push(/* wgsl */ `
@group(0) @binding(5) var<storage, read> rowPerm: array<u32>;
@group(0) @binding(6) var<storage, read> colPerm: array<u32>;
@group(0) @binding(7) var<storage, read> aux: array<vec4f>;
var<workgroup> hitA: atomic<u32>;
var<workgroup> hitU: u32;
fn rowIndex(sp: u32) -> u32 { if (sp >= p.nRows) { return INVALID; } return rowPerm[sp]; }
fn colIndex(sp: u32) -> u32 { if (sp >= p.nCols) { return INVALID; } return colPerm[sp]; }
`);
    if (mode === 'pos') {
      lines.push(/* wgsl */ `
fn pred(rec: vec4f, pos: vec4f) -> bool {
  if (rec.z == 1.0) { return true; }
  if (rec.z != 0.0) { return false; }
  let dx = pos.x - rec.x;
  let dy = pos.y - rec.y;
  return dx * dx + dy * dy <= p.r2;
}
fn tileTest(rec: vec4f, box: vec4f) -> bool {
  if (rec.z == 1.0) { return true; }
  if (rec.z != 0.0) { return false; }
  let R = p.radius;
  return rec.x + R >= box.x && rec.x - R <= box.z && rec.y + R >= box.y && rec.y - R <= box.w;
}
`);
    } else {
      lines.push(/* wgsl */ `
fn pred(rec: vec4f, pos: vec4f) -> bool {
  if (rec.w != 0.0) { return false; }
  return abs(rec.x * pos.z + rec.y * pos.w + rec.z) <= p.tol;
}
fn tileTest(rec: vec4f, box: vec4f) -> bool {
  if (rec.w != 0.0) { return false; }
  let d0 = rec.x * box.x + rec.y * box.y + rec.z;
  let d1 = rec.x * box.z + rec.y * box.y + rec.z;
  let d2 = rec.x * box.x + rec.y * box.w + rec.z;
  let d3 = rec.x * box.z + rec.y * box.w + rec.z;
  return min(min(d0, d1), min(d2, d3)) <= p.tol && max(max(d0, d1), max(d2, d3)) >= -p.tol;
}
`);
    }
  } else {
    lines.push(/* wgsl */ `
fn rowIndex(sp: u32) -> u32 { if (sp >= p.nRows) { return INVALID; } return sp; }
fn colIndex(sp: u32) -> u32 { if (sp >= p.nCols) { return INVALID; } return sp; }
`);
  }

  // Unrolled top-2 insertion for row r with candidate (score s, index j, dot d).
  const insert = (r: number, s: string, j: string, d: string) => `
    if (${s} > s1_${r} || (${s} == s1_${r} && ${j} < i1_${r})) {
      s2_${r} = s1_${r}; i2_${r} = i1_${r}; d2_${r} = d1_${r};
      s1_${r} = ${s}; i1_${r} = ${j}; d1_${r} = ${d};
    } else if (${s} > s2_${r} || (${s} == s2_${r} && ${j} < i2_${r})) {
      s2_${r} = ${s}; i2_${r} = ${j}; d2_${r} = ${d};
    }`;

  const body: string[] = [];
  body.push(/* wgsl */ `
@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let tx = lid.x;
  let ty = lid.y;
  let li = ty * 16u + tx;
  let rowTile = p.rowTile0 + wid.x;
  let chunk = wid.y;
  let rowBase = rowTile * 64u;
`);
  for (let r = 0; r < 4; r++) {
    body.push(`  let rn${r} = rowIndex(rowBase + ty * 4u + ${r}u);`);
    body.push(`  var s1_${r} = -1.0; var i1_${r} = INVALID; var d1_${r} = 0.0;`);
    body.push(`  var s2_${r} = -1.0; var i2_${r} = INVALID; var d2_${r} = 0.0;`);
  }
  if (guided) {
    // Per-row guide data held in registers: a-records (forward) or b-positions (reverse).
    for (let r = 0; r < 4; r++) {
      body.push(`  var rr${r} = vec4f(0.0);`);
      body.push(`  if (rn${r} != INVALID) { rr${r} = aux[${reverse ? 'p.bOff' : 'p.aOff'} + rn${r}]; }`);
    }
    if (reverse) {
      body.push(`  let rowBox = aux[p.boxOff + rowTile];`);
    } else {
      body.push(`  var tRec = vec4f(0.0);`);
      body.push(`  var tValid = false;`);
      body.push(`  if (li < 64u) { let n = rowIndex(rowBase + li); if (n != INVALID) { tRec = aux[p.aOff + n]; tValid = true; } }`);
    }
  }
  body.push(/* wgsl */ `
  // Loader roles: threads 0..63 load the row tile, 64..127 the column tile
  // (4 descriptors x 1 word each; writes whole vec4s of 4 consecutive descriptors).
  let ldQuad = (li & 63u) >> 2u;
  let ldWord = li & 3u;
  var ldN0 = INVALID; var ldN1 = INVALID; var ldN2 = INVALID; var ldN3 = INVALID;
  if (li < 64u) {
    ldN0 = rowIndex(rowBase + ldQuad * 4u);
    ldN1 = rowIndex(rowBase + ldQuad * 4u + 1u);
    ldN2 = rowIndex(rowBase + ldQuad * 4u + 2u);
    ldN3 = rowIndex(rowBase + ldQuad * 4u + 3u);
  }
  let ct0 = chunk * p.tilesPerChunk;
  let ct1 = min(ct0 + p.tilesPerChunk, p.nColTiles);
  for (var ct = ct0; ct < ct1; ct++) {
    let colBase = ct * 64u;
`);
  if (guided) {
    body.push(`    if (li == 0u) { atomicStore(&hitA, 0u); }`);
    body.push(`    workgroupBarrier();`);
    if (reverse) {
      body.push(`    if (li < 64u) { let n = colIndex(colBase + li); if (n != INVALID) { if (tileTest(aux[p.aOff + n], rowBox)) { atomicOr(&hitA, 1u); } } }`);
    } else {
      body.push(`    if (tValid) { if (tileTest(tRec, aux[p.boxOff + ct])) { atomicOr(&hitA, 1u); } }`);
    }
    body.push(`    workgroupBarrier();`);
    body.push(`    if (li == 0u) { hitU = atomicLoad(&hitA); }`);
    body.push(`    if (workgroupUniformLoad(&hitU) == 0u) { continue; }`);
  }
  body.push(/* wgsl */ `
    if (li >= 64u && li < 128u) {
      ldN0 = colIndex(colBase + ldQuad * 4u);
      ldN1 = colIndex(colBase + ldQuad * 4u + 1u);
      ldN2 = colIndex(colBase + ldQuad * 4u + 2u);
      ldN3 = colIndex(colBase + ldQuad * 4u + 3u);
    }
    var acc0 = vec4f(0.0); var acc1 = vec4f(0.0); var acc2 = vec4f(0.0); var acc3 = vec4f(0.0);
    for (var kc = 0u; kc < 8u; kc++) {
      if (li < 128u) {
        let wo = kc * 4u + ldWord;
        var w0 = 0u; var w1 = 0u; var w2 = 0u; var w3 = 0u;
        if (li < 64u) {
          if (ldN0 != INVALID) { w0 = rowDesc[ldN0 * 32u + wo]; }
          if (ldN1 != INVALID) { w1 = rowDesc[ldN1 * 32u + wo]; }
          if (ldN2 != INVALID) { w2 = rowDesc[ldN2 * 32u + wo]; }
          if (ldN3 != INVALID) { w3 = rowDesc[ldN3 * 32u + wo]; }
        } else {
          if (ldN0 != INVALID) { w0 = colDesc[ldN0 * 32u + wo]; }
          if (ldN1 != INVALID) { w1 = colDesc[ldN1 * 32u + wo]; }
          if (ldN2 != INVALID) { w2 = colDesc[ldN2 * 32u + wo]; }
          if (ldN3 != INVALID) { w3 = colDesc[ldN3 * 32u + wo]; }
        }
        let base = (li >> 6u) * 256u + ldQuad;
        for (var b = 0u; b < 4u; b++) {
          let sft = 8u * b;
          let v = vec4f(f32((w0 >> sft) & 0xffu), f32((w1 >> sft) & 0xffu), f32((w2 >> sft) & 0xffu), f32((w3 >> sft) & 0xffu));
          sh[base + (ldWord * 4u + b) * 16u] = bitcast<vec4<u32>>(v);
        }
      }
      workgroupBarrier();
      for (var k = 0u; k < 16u; k++) {
        let a = bitcast<vec4f>(sh[k * 16u + ty]);
        let bv = bitcast<vec4f>(sh[256u + k * 16u + tx]);
        acc0 += a.x * bv;
        acc1 += a.y * bv;
        acc2 += a.z * bv;
        acc3 += a.w * bv;
      }
      workgroupBarrier();
    }
`);
  for (let c = 0; c < 4; c++) {
    body.push(`    let cn${c} = colIndex(colBase + tx * 4u + ${c}u);`);
    body.push(`    if (cn${c} != INVALID) {`);
    body.push(`      let inv${c} = colInv[cn${c}];`);
    if (guided) body.push(`      let cr${c} = aux[${reverse ? 'p.aOff' : 'p.bOff'} + cn${c}];`);
    for (let r = 0; r < 4; r++) {
      const dot = `acc${r}.${COMP[c]}`;
      const cond = guided ? (reverse ? `pred(cr${c}, rr${r})` : `pred(rr${r}, cr${c})`) : 'true';
      body.push(`      if (${cond}) {`);
      body.push(`        let sc = ${dot} * inv${c};`);
      body.push(insert(r, 'sc', `cn${c}`, dot));
      body.push(`      }`);
    }
    body.push(`    }`);
  }
  body.push(`  }`); // end column tile loop

  // Merge the 16 partial lists of each row (over tx) in 4 halving rounds.
  body.push(/* wgsl */ `
  workgroupBarrier();
  for (var hh = 8u; hh > 0u; hh = hh >> 1u) {
    if (tx >= hh && tx < 2u * hh) {
      let e = (ty * 4u) * hh + (tx - hh);`);
  for (let r = 0; r < 4; r++) {
    body.push(`      sh[e + ${r}u * hh] = vec4<u32>(i1_${r}, u32(d1_${r}), i2_${r}, u32(d2_${r}));`);
  }
  body.push(`    }`);
  body.push(`    workgroupBarrier();`);
  body.push(`    if (tx < hh) {`);
  body.push(`      let e = (ty * 4u) * hh + tx;`);
  for (let r = 0; r < 4; r++) {
    body.push(`      {`);
    body.push(`        let m = sh[e + ${r}u * hh];`);
    body.push(`        if (m.x != INVALID) { let d = f32(m.y); let sc = d * colInv[m.x]; ${insert(r, 'sc', 'm.x', 'd')} }`);
    body.push(`        if (m.z != INVALID) { let d = f32(m.w); let sc = d * colInv[m.z]; ${insert(r, 'sc', 'm.z', 'd')} }`);
    body.push(`      }`);
  }
  body.push(`    }`);
  body.push(`    workgroupBarrier();`);
  body.push(`  }`);
  body.push(`  if (tx == 0u) {`);
  for (let r = 0; r < 4; r++) {
    body.push(`    if (rn${r} != INVALID) { outBuf[p.outOff + rn${r} * p.nChunks + chunk] = vec4<u32>(i1_${r}, u32(d1_${r}), i2_${r}, u32(d2_${r})); }`);
  }
  body.push(`  }`);
  body.push(`}`);
  return lines.join('\n') + body.join('\n');
}
