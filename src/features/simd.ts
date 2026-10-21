/**
 * Tiny hand-assembled WebAssembly SIMD kernels (no toolchain needed) used to
 * accelerate the CPU fallback paths: separable Gaussian blur passes (f32x4) and
 * brute-force descriptor dot products (i32x4.dot_i16x8_s, exact integer sums).
 *
 * WASM SIMD is supported by Safari >= 16.4, Chrome >= 91, Node >= 16. When it is
 * unavailable (or SFM_DISABLE_SIMD is set on globalThis) every caller falls back
 * to plain JS that produces identical (matching) or numerically equivalent (blur)
 * results.
 *
 * Kernels (all pointers are byte offsets into the instance's memory):
 *  blurV(src, dst, w, h, kv, r)       dst = vertical Gaussian of src; kv -> f32 k[0..r]
 *  blurH(src, dst, w, h, kv, r, pad)  dst = horizontal Gaussian (replicated borders);
 *                                     pad = scratch of (w + 2r + 8) floats
 *  dot4(a, b, nb, out)                out[4j + q] = Σ_k a_q[k] b_j[k] for 4 consecutive
 *                                     i16 descriptors a_0..a_3 (256 B each) and nb
 *                                     descriptors at b; out is i32.
 *  grad(src, mag, ang, w, h)          central-difference gradient magnitude and angle
 *                                     ([0, 2π), polynomial atan2, |err| < 1e-5 rad) of
 *                                     the interior pixels; borders are left to the caller.
 *  desc(mag, ang, wx, rows, nRows, dnx, dny, theta, P)
 *                                     scalar inner loop of the SIFT descriptor
 *                                     (trilinear accumulation, see descBody).
 * Rows are processed in 4-float blocks: planes need >= 16 bytes of slack after
 * their end (reads/writes past the row end only touch lanes that are discarded).
 *
 * The module is compiled synchronously (new WebAssembly.Module): keep it < 4 KB, the
 * size Chrome allows for synchronous compilation on the main thread (≈ 3.4 KB now).
 */

// ---------------------------------------------------------------------------
// Minimal WASM encoder
// ---------------------------------------------------------------------------

function uleb(n: number): number[] {
  const out: number[] = [];
  do {
    let b = n & 0x7f;
    n >>>= 7;
    if (n !== 0) b |= 0x80;
    out.push(b);
  } while (n !== 0);
  return out;
}

function sleb(n: number): number[] {
  const out: number[] = [];
  for (;;) {
    const b = n & 0x7f;
    n >>= 7;
    if ((n === 0 && (b & 0x40) === 0) || (n === -1 && (b & 0x40) !== 0)) {
      out.push(b);
      return out;
    }
    out.push(b | 0x80);
  }
}

const I32 = 0x7f;
const F32 = 0x7d;
const V128 = 0x7b;

const get = (i: number) => [0x20, ...uleb(i)];
const set = (i: number) => [0x21, ...uleb(i)];
const tee = (i: number) => [0x22, ...uleb(i)];
const i32c = (v: number) => [0x41, ...sleb(v)];
const I32_ADD = [0x6a];
const I32_SUB = [0x6b];
const I32_MUL = [0x6c];
const I32_SHL = [0x74];
const I32_LT_S = [0x48];
const I32_GT_S = [0x4a];
const I32_GE_S = [0x4e];
const SELECT = [0x1b];
const BLOCK = [0x02, 0x40];
const LOOP = [0x03, 0x40];
const END = [0x0b];
const br = (d: number) => [0x0c, ...uleb(d)];
const brIf = (d: number) => [0x0d, ...uleb(d)];
const f32Load = (off = 0) => [0x2a, 2, ...uleb(off)];
const f32Store = (off = 0) => [0x38, 2, ...uleb(off)];
const i32Store = (off = 0) => [0x36, 2, ...uleb(off)];
const simd = (op: number) => [0xfd, ...uleb(op)];
const v128Load = (off = 0) => [...simd(0), 0, ...uleb(off)];
const v128Load32Splat = (off = 0) => [...simd(9), 0, ...uleb(off)];
const v128Store = (off = 0) => [...simd(11), 0, ...uleb(off)];
const I32X4_EXTRACT = (lane: number) => [...simd(27), lane];
const F32X4_LT = simd(67);
const F32X4_GT = simd(68);
const F32X4_GE = simd(70);
const V128_BITSELECT = simd(82);
const F32X4_ABS = simd(224);
const F32X4_SQRT = simd(227);
const F32X4_SUB = simd(229);
const F32X4_DIV = simd(231);
const F32X4_MIN = simd(232);
const F32X4_MAX = simd(233);
const F32_ADD = [0x92];
const F32_SUB = [0x93];
const F32_MUL = [0x94];
const F32_LT = [0x5d];
const F32_GT = [0x5e];
const I32_AND = [0x71];
const I32_LE_S = [0x4c];
const I32_TRUNC_SAT_F32_S = [0xfc, 0x00];
const F32_CONVERT_I32_S = [0xb2];
const IF = [0x04, 0x40];
const i32Load = (off = 0) => [0x28, 2, ...uleb(off)];
function f32c(v: number): number[] {
  return [0x43, ...Array.from(new Uint8Array(new Float32Array([v]).buffer))];
}
/** v128.const with four equal f32 lanes. */
function f32x4Const(v: number): number[] {
  const b = new Uint8Array(new Float32Array([v, v, v, v]).buffer);
  return [...simd(12), ...Array.from(b)];
}
const I32X4_ADD = simd(174);
const I32X4_DOT = simd(186);
const F32X4_ADD = simd(228);
const F32X4_MUL = simd(230);

/** `for (; i < limit; i += step) body` with i a local; limit/step pushed by code. */
function forLoop(i: number, init: number[], cond: number[], step: number[], body: number[]): number[] {
  return [
    ...init, ...set(i),
    ...BLOCK, ...LOOP,
    ...get(i), ...cond, ...brIf(1),
    ...body,
    ...get(i), ...step, ...I32_ADD, ...set(i),
    ...br(0),
    ...END, ...END,
  ];
}

function funcBody(locals: [number, number][], code: number[]): number[] {
  const l: number[] = [...uleb(locals.length)];
  for (const [n, t] of locals) l.push(...uleb(n), t);
  const body = [...l, ...code, ...END];
  return [...uleb(body.length), ...body];
}

function section(id: number, payload: number[]): number[] {
  return [id, ...uleb(payload.length), ...payload];
}

function str(s: string): number[] {
  const b = Array.from(s, (c) => c.charCodeAt(0));
  return [...uleb(b.length), ...b];
}

// --- blurV(src0, dst1, w2, h3, kv4, r5) -------------------------------------
function blurVBody(): number[] {
  // locals: 6 y, 7 x, 8 rowBytes, 9 j, 10 rowOff, 11 yaOff, 12 ybOff, 13 hm1, 14 t, 15 kspl(v128)
  const [src, dst, w, h, kv, r] = [0, 1, 2, 3, 4, 5];
  const [y, x, rowBytes, j, rowOff, yaOff, ybOff, hm1, t, kspl] = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
  const pass0 = forLoop(x, i32c(0), [...get(rowBytes), ...I32_GE_S], i32c(16), [
    ...get(dst), ...get(rowOff), ...I32_ADD, ...get(x), ...I32_ADD,
    ...get(kspl),
    ...get(src), ...get(rowOff), ...I32_ADD, ...get(x), ...I32_ADD, ...v128Load(),
    ...F32X4_MUL,
    ...v128Store(),
  ]);
  const tapX = forLoop(x, i32c(0), [...get(rowBytes), ...I32_GE_S], i32c(16), [
    ...get(dst), ...get(rowOff), ...I32_ADD, ...get(x), ...I32_ADD, ...tee(t),
    ...get(t), ...v128Load(),
    ...get(kspl),
    ...get(yaOff), ...get(x), ...I32_ADD, ...v128Load(),
    ...get(ybOff), ...get(x), ...I32_ADD, ...v128Load(),
    ...F32X4_ADD, ...F32X4_MUL, ...F32X4_ADD,
    ...v128Store(),
  ]);
  const taps = forLoop(j, i32c(1), [...get(r), ...I32_GT_S], i32c(1), [
    // yaOff = src + max(y - j, 0) * rowBytes
    ...get(y), ...get(j), ...I32_SUB, ...tee(t), ...i32c(0), ...get(t), ...i32c(0), ...I32_GT_S, ...SELECT,
    ...get(rowBytes), ...I32_MUL, ...get(src), ...I32_ADD, ...set(yaOff),
    // ybOff = src + min(y + j, h - 1) * rowBytes
    ...get(y), ...get(j), ...I32_ADD, ...tee(t), ...get(hm1), ...get(t), ...get(hm1), ...I32_LT_S, ...SELECT,
    ...get(rowBytes), ...I32_MUL, ...get(src), ...I32_ADD, ...set(ybOff),
    ...get(kv), ...get(j), ...i32c(2), ...I32_SHL, ...I32_ADD, ...v128Load32Splat(), ...set(kspl),
    ...tapX,
  ]);
  const code = [
    ...get(w), ...i32c(2), ...I32_SHL, ...set(rowBytes),
    ...get(h), ...i32c(1), ...I32_SUB, ...set(hm1),
    ...forLoop(y, i32c(0), [...get(h), ...I32_GE_S], i32c(1), [
      ...get(y), ...get(rowBytes), ...I32_MUL, ...set(rowOff),
      ...get(kv), ...v128Load32Splat(), ...set(kspl),
      ...pass0,
      ...taps,
    ]),
  ];
  return funcBody([[9, I32], [1, V128]], code);
}

// --- blurH(src0, dst1, w2, h3, kv4, r5, pad6) --------------------------------
function blurHBody(): number[] {
  // locals: 7 y, 8 x, 9 rowBytes, 10 j, 11 rowSrc, 12 c, 13 i, 14 t, 15 rowDst, 16 r4,
  //         17 first(f32), 18 last(f32), 19 acc(v128), 20 k0(v128)
  const [src, dst, , h, kv, r, pad] = [0, 1, 2, 3, 4, 5, 6];
  const [y, x, rowBytes, j, rowSrc, c, i, t, rowDst, r4, first, last, acc, k0] = [
    7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
  ];
  const left = forLoop(i, i32c(0), [...get(r4), ...I32_GE_S], i32c(4), [
    ...get(pad), ...get(i), ...I32_ADD, ...get(first), ...f32Store(),
  ]);
  const middle = forLoop(i, i32c(0), [...get(rowBytes), ...I32_GE_S], i32c(16), [
    ...get(pad), ...get(r4), ...I32_ADD, ...get(i), ...I32_ADD,
    ...get(rowSrc), ...get(i), ...I32_ADD, ...v128Load(),
    ...v128Store(),
  ]);
  const right = [
    ...get(pad), ...get(r4), ...I32_ADD, ...get(rowBytes), ...I32_ADD, ...set(t),
    ...forLoop(i, i32c(0), [...get(r4), ...i32c(32), ...I32_ADD, ...I32_GE_S], i32c(4), [
      ...get(t), ...get(i), ...I32_ADD, ...get(last), ...f32Store(),
    ]),
  ];
  const inner = forLoop(j, i32c(1), [...get(r), ...I32_GT_S], i32c(1), [
    ...get(j), ...i32c(2), ...I32_SHL, ...set(t),
    ...get(acc),
    ...get(kv), ...get(t), ...I32_ADD, ...v128Load32Splat(),
    ...get(c), ...get(t), ...I32_SUB, ...v128Load(),
    ...get(c), ...get(t), ...I32_ADD, ...v128Load(),
    ...F32X4_ADD, ...F32X4_MUL, ...F32X4_ADD, ...set(acc),
  ]);
  const conv = forLoop(x, i32c(0), [...get(rowBytes), ...I32_GE_S], i32c(16), [
    ...get(pad), ...get(r4), ...I32_ADD, ...get(x), ...I32_ADD, ...set(c),
    ...get(k0), ...get(c), ...v128Load(), ...F32X4_MUL, ...set(acc),
    ...inner,
    ...get(rowDst), ...get(x), ...I32_ADD, ...get(acc), ...v128Store(),
  ]);
  const code = [
    ...get(2), ...i32c(2), ...I32_SHL, ...set(rowBytes),
    ...get(r), ...i32c(2), ...I32_SHL, ...set(r4),
    ...get(kv), ...v128Load32Splat(), ...set(k0),
    ...forLoop(y, i32c(0), [...get(h), ...I32_GE_S], i32c(1), [
      ...get(src), ...get(y), ...get(rowBytes), ...I32_MUL, ...I32_ADD, ...set(rowSrc),
      ...get(dst), ...get(y), ...get(rowBytes), ...I32_MUL, ...I32_ADD, ...set(rowDst),
      ...get(rowSrc), ...f32Load(), ...set(first),
      ...get(rowSrc), ...get(rowBytes), ...I32_ADD, ...i32c(4), ...I32_SUB, ...f32Load(), ...set(last),
      ...left,
      ...middle,
      ...right,
      ...conv,
    ]),
  ];
  return funcBody([[10, I32], [2, F32], [2, V128]], code);
}

// --- dot4(a0, b1, nb2, out3) --------------------------------------------------
function dot4Body(): number[] {
  // locals: 4 j, 5 bp, 6 op, 7 vb, 8..11 acc0..3
  const [a, b, nb, out] = [0, 1, 2, 3];
  const [j, bp, op, vb] = [4, 5, 6, 7];
  const accs = [8, 9, 10, 11];
  const body: number[] = [];
  for (let k = 0; k < 16; k++) {
    body.push(...get(bp), ...v128Load(16 * k), ...set(vb));
    for (let q = 0; q < 4; q++) {
      if (k === 0) {
        body.push(...get(a), ...v128Load(256 * q), ...get(vb), ...I32X4_DOT, ...set(accs[q]));
      } else {
        body.push(...get(accs[q]), ...get(a), ...v128Load(256 * q + 16 * k), ...get(vb), ...I32X4_DOT, ...I32X4_ADD, ...set(accs[q]));
      }
    }
  }
  for (let q = 0; q < 4; q++) {
    const s = accs[q];
    body.push(
      ...get(op),
      ...get(s), ...I32X4_EXTRACT(0), ...get(s), ...I32X4_EXTRACT(1), ...I32_ADD,
      ...get(s), ...I32X4_EXTRACT(2), ...I32_ADD, ...get(s), ...I32X4_EXTRACT(3), ...I32_ADD,
      ...i32Store(4 * q),
    );
  }
  body.push(...get(bp), ...i32c(256), ...I32_ADD, ...set(bp));
  body.push(...get(op), ...i32c(16), ...I32_ADD, ...set(op));
  const code = [
    ...get(b), ...set(bp),
    ...get(out), ...set(op),
    ...forLoop(j, i32c(0), [...get(nb), ...I32_GE_S], i32c(1), body),
  ];
  return funcBody([[3, I32], [5, V128]], code);
}

// --- grad(src0, mag1, ang2, w3, h4) --------------------------------------------
// Interior pixels (1 <= x <= w-2, 1 <= y <= h-2) only, in 4-pixel blocks; the
// caller fixes the border pixels afterwards (blocks overrun into them).
function gradBody(): number[] {
  // locals: 5 y, 6 x, 7 rowBytes, 8 p, 9 xend, 10 rowOff, 11 hm1, v128: 12 gx, 13 gy, 14 ax, 15 ay, 16 z, 17 z2, 18 r
  const [src, mag, ang, w, h] = [0, 1, 2, 3, 4];
  const [y, x, rowBytes, p, xend, rowOff, hm1] = [5, 6, 7, 8, 9, 10, 11];
  const [gx, gy, ax, ay, z, z2, r] = [12, 13, 14, 15, 16, 17, 18];
  const inner = [
    ...get(src), ...get(rowOff), ...I32_ADD, ...get(x), ...I32_ADD, ...set(p),
    // gx = 0.5 (L[x+1] - L[x-1]), gy = 0.5 (L[y+1] - L[y-1])
    ...get(p), ...v128Load(4), ...get(p), ...i32c(4), ...I32_SUB, ...v128Load(), ...F32X4_SUB, ...f32x4Const(0.5), ...F32X4_MUL, ...set(gx),
    ...get(p), ...get(rowBytes), ...I32_ADD, ...v128Load(), ...get(p), ...get(rowBytes), ...I32_SUB, ...v128Load(), ...F32X4_SUB,
    ...f32x4Const(0.5), ...F32X4_MUL, ...set(gy),
    // magnitude
    ...get(mag), ...get(rowOff), ...I32_ADD, ...get(x), ...I32_ADD,
    ...get(gx), ...get(gx), ...F32X4_MUL, ...get(gy), ...get(gy), ...F32X4_MUL, ...F32X4_ADD, ...F32X4_SQRT,
    ...v128Store(),
    // z = min(|gx|,|gy|) / max(max(|gx|,|gy|), tiny)
    ...get(gx), ...F32X4_ABS, ...set(ax),
    ...get(gy), ...F32X4_ABS, ...set(ay),
    ...get(ax), ...get(ay), ...F32X4_MIN,
    ...get(ax), ...get(ay), ...F32X4_MAX, ...f32x4Const(1e-30), ...F32X4_MAX,
    ...F32X4_DIV, ...set(z),
    ...get(z), ...get(z), ...F32X4_MUL, ...set(z2),
    // r = z (c1 + z2 (c3 + z2 (c5 + z2 (c7 + z2 c9))))
    ...f32x4Const(0.0208351), ...get(z2), ...F32X4_MUL, ...f32x4Const(-0.085133), ...F32X4_ADD,
    ...get(z2), ...F32X4_MUL, ...f32x4Const(0.180141), ...F32X4_ADD,
    ...get(z2), ...F32X4_MUL, ...f32x4Const(-0.3302995), ...F32X4_ADD,
    ...get(z2), ...F32X4_MUL, ...f32x4Const(0.999866), ...F32X4_ADD,
    ...get(z), ...F32X4_MUL, ...set(r),
    // octant / quadrant corrections
    ...f32x4Const(Math.PI / 2), ...get(r), ...F32X4_SUB, ...get(r), ...get(ay), ...get(ax), ...F32X4_GT, ...V128_BITSELECT, ...set(r),
    ...f32x4Const(Math.PI), ...get(r), ...F32X4_SUB, ...get(r), ...get(gx), ...f32x4Const(0), ...F32X4_LT, ...V128_BITSELECT, ...set(r),
    ...f32x4Const(2 * Math.PI), ...get(r), ...F32X4_SUB, ...get(r), ...get(gy), ...f32x4Const(0), ...F32X4_LT, ...V128_BITSELECT, ...set(r),
    ...get(ang), ...get(rowOff), ...I32_ADD, ...get(x), ...I32_ADD,
    ...f32x4Const(0), ...get(r), ...get(r), ...f32x4Const(2 * Math.PI), ...F32X4_GE, ...V128_BITSELECT,
    ...v128Store(),
  ];
  const code = [
    ...get(w), ...i32c(2), ...I32_SHL, ...set(rowBytes),
    ...get(rowBytes), ...i32c(4), ...I32_SUB, ...set(xend),
    ...get(h), ...i32c(1), ...I32_SUB, ...set(hm1),
    ...forLoop(y, i32c(1), [...get(hm1), ...I32_GE_S], i32c(1), [
      ...get(y), ...get(rowBytes), ...I32_MUL, ...set(rowOff),
      ...forLoop(x, i32c(4), [...get(xend), ...I32_GE_S], i32c(16), inner),
    ]),
  ];
  return funcBody([[7, I32], [7, V128]], code);
}

// --- desc(mag0, ang1, wx2, rows3, nRows4, dnx5:f32, dny6:f32, theta7:f32, P8) -----
// Inner loop of the SIFT descriptor: trilinear accumulation into the padded
// 6 x 6 x 9 f32 histogram P (spatial bins -1..4, orientation bins 0..8). Each row
// record (32 B) holds: i32 pixel byte offset (from mag), i32 count, i32 wx byte
// offset (from wx), f32 row weight, f32 nx and ny of the first sample.
function descBody(): number[] {
  const [mag, ang, wx, rows, nRows, dnx, dny, theta, P] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  const [r, rp, p, q, cnt, pb, angDelta, bx, by, bt] = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18];
  const [nx, ny, wyv, m, a, nt, fx, fy, rx, ry, rt, m0, m1, mm, rt0] = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33];
  const acc = (off: number, w: number[]) => [
    ...get(pb), ...get(pb), ...f32Load(off), ...get(mm), ...w, ...F32_MUL, ...F32_ADD, ...f32Store(off),
  ];
  const sample = [
    // m = mag[p] * wx[q] * wyv
    ...get(p), ...f32Load(), ...get(q), ...f32Load(), ...F32_MUL, ...get(wyv), ...F32_MUL, ...set(m),
    // a = ang[p] - theta, wrapped to [0, 2π)
    ...get(p), ...get(angDelta), ...I32_ADD, ...f32Load(), ...get(theta), ...F32_SUB, ...set(a),
    ...get(a), ...f32c(2 * Math.PI), ...F32_ADD, ...get(a), ...get(a), ...f32c(0), ...F32_LT, ...SELECT, ...set(a),
    ...get(a), ...f32c(8 / (2 * Math.PI)), ...F32_MUL, ...set(nt),
    // padded spatial coordinates in (0, 5)
    ...get(nx), ...f32c(2.5), ...F32_ADD, ...set(fx),
    ...get(ny), ...f32c(2.5), ...F32_ADD, ...set(fy),
    ...get(fx), ...I32_TRUNC_SAT_F32_S, ...set(bx),
    ...get(fy), ...I32_TRUNC_SAT_F32_S, ...set(by),
    ...get(nt), ...I32_TRUNC_SAT_F32_S, ...tee(bt), ...i32c(7), ...get(bt), ...i32c(7), ...I32_LE_S, ...SELECT, ...set(bt),
    ...get(fx), ...get(bx), ...F32_CONVERT_I32_S, ...F32_SUB, ...set(rx),
    ...get(fy), ...get(by), ...F32_CONVERT_I32_S, ...F32_SUB, ...set(ry),
    ...get(nt), ...get(bt), ...F32_CONVERT_I32_S, ...F32_SUB, ...set(rt),
    ...f32c(1), ...get(rt), ...F32_SUB, ...set(rt0),
    // pb = P + 4 * (by * 54 + bx * 9 + bt)
    ...get(P), ...get(by), ...i32c(54), ...I32_MUL, ...get(bx), ...i32c(9), ...I32_MUL, ...I32_ADD, ...get(bt), ...I32_ADD,
    ...i32c(2), ...I32_SHL, ...I32_ADD, ...set(pb),
    ...get(m), ...f32c(1), ...get(ry), ...F32_SUB, ...F32_MUL, ...set(m0),
    ...get(m), ...get(ry), ...F32_MUL, ...set(m1),
    ...get(m0), ...f32c(1), ...get(rx), ...F32_SUB, ...F32_MUL, ...set(mm),
    ...acc(0, get(rt0)), ...acc(4, get(rt)),
    ...get(m0), ...get(rx), ...F32_MUL, ...set(mm),
    ...acc(36, get(rt0)), ...acc(40, get(rt)),
    ...get(m1), ...f32c(1), ...get(rx), ...F32_SUB, ...F32_MUL, ...set(mm),
    ...acc(216, get(rt0)), ...acc(220, get(rt)),
    ...get(m1), ...get(rx), ...F32_MUL, ...set(mm),
    ...acc(252, get(rt0)), ...acc(256, get(rt)),
  ];
  const inner = [
    ...BLOCK, ...LOOP,
    ...get(cnt), ...i32c(0), ...I32_LE_S, ...brIf(1),
    ...get(nx), ...f32c(-2.5), ...F32_GT,
    ...get(nx), ...f32c(2.5), ...F32_LT, ...I32_AND,
    ...get(ny), ...f32c(-2.5), ...F32_GT, ...I32_AND,
    ...get(ny), ...f32c(2.5), ...F32_LT, ...I32_AND,
    ...IF, ...sample, ...END,
    ...get(nx), ...get(dnx), ...F32_ADD, ...set(nx),
    ...get(ny), ...get(dny), ...F32_ADD, ...set(ny),
    ...get(p), ...i32c(4), ...I32_ADD, ...set(p),
    ...get(q), ...i32c(4), ...I32_ADD, ...set(q),
    ...get(cnt), ...i32c(1), ...I32_SUB, ...set(cnt),
    ...br(0),
    ...END, ...END,
  ];
  const code = [
    ...get(ang), ...get(mag), ...I32_SUB, ...set(angDelta),
    ...forLoop(r, i32c(0), [...get(nRows), ...I32_GE_S], i32c(1), [
      ...get(rows), ...get(r), ...i32c(5), ...I32_SHL, ...I32_ADD, ...set(rp),
      ...get(rp), ...i32Load(0), ...get(mag), ...I32_ADD, ...set(p),
      ...get(rp), ...i32Load(4), ...set(cnt),
      ...get(rp), ...i32Load(8), ...get(wx), ...I32_ADD, ...set(q),
      ...get(rp), ...f32Load(12), ...set(wyv),
      ...get(rp), ...f32Load(16), ...set(nx),
      ...get(rp), ...f32Load(20), ...set(ny),
      ...inner,
    ]),
  ];
  return funcBody([[10, I32], [15, F32]], code);
}

function buildModule(): Uint8Array<ArrayBuffer> {
  const fnType = (n: number) => [0x60, ...uleb(n), ...new Array(n).fill(I32), 0];
  const descType = [0x60, 9, I32, I32, I32, I32, I32, F32, F32, F32, I32, 0];
  const types = section(1, [...uleb(5), ...fnType(6), ...fnType(7), ...fnType(4), ...fnType(5), ...descType]);
  const imports = section(2, [...uleb(1), ...str('e'), ...str('m'), 0x02, 0x00, ...uleb(1)]);
  const funcs = section(3, [...uleb(5), 0, 1, 2, 3, 4]);
  const exports = section(7, [
    ...uleb(5),
    ...str('blurV'), 0x00, 0,
    ...str('blurH'), 0x00, 1,
    ...str('dot4'), 0x00, 2,
    ...str('grad'), 0x00, 3,
    ...str('desc'), 0x00, 4,
  ]);
  const code = section(10, [...uleb(5), ...blurVBody(), ...blurHBody(), ...dot4Body(), ...gradBody(), ...descBody()]);
  return new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, ...types, ...imports, ...funcs, ...exports, ...code]);
}

// ---------------------------------------------------------------------------
// Instances
// ---------------------------------------------------------------------------

export interface SimdKernels {
  readonly memory: WebAssembly.Memory;
  blurV(src: number, dst: number, w: number, h: number, kv: number, r: number): void;
  blurH(src: number, dst: number, w: number, h: number, kv: number, r: number, pad: number): void;
  dot4(a: number, b: number, nb: number, out: number): void;
  /** Interior gradient magnitude / angle in [0, 2π) (border pixels must be fixed by the caller). */
  grad(src: number, mag: number, ang: number, w: number, h: number): void;
  /** SIFT descriptor accumulation (see descBody). */
  desc(mag: number, ang: number, wx: number, rows: number, nRows: number, dnx: number, dny: number, theta: number, P: number): void;
  /** Ensure the memory holds at least `bytes` bytes (grows; invalidates old views). */
  ensure(bytes: number): void;
}

let compiled: WebAssembly.Module | null | undefined;
/** Name of the kernel that failed the runtime self-test in this realm (null: passed / not run). */
let selfTestFailure: string | null = null;
type KernelExports = Pick<SimdKernels, 'blurV' | 'blurH' | 'dot4' | 'grad' | 'desc'>;
/**
 * The passing self-test's instance, zeroed and handed out by the next createSimdKernels(): no
 * extra Wasm memory (JSC reserves 4 GiB of address space per "fast" memory and allows only a
 * few of them, docs/research/ios-platform.md §1.3).
 */
let spare: { memory: WebAssembly.Memory; ex: KernelExports } | null = null;

function getModule(): WebAssembly.Module | null {
  if (compiled !== undefined) return compiled;
  compiled = null;
  try {
    const g = globalThis as { SFM_DISABLE_SIMD?: boolean; WebAssembly?: typeof WebAssembly };
    if (g.SFM_DISABLE_SIMD || typeof g.WebAssembly === 'undefined') return null;
    const bytes = buildModule();
    if (!WebAssembly.validate(bytes)) return null;
    const mod = new WebAssembly.Module(bytes);
    selfTestFailure = runSelfTest(mod, true);
    if (selfTestFailure) {
      console.warn(`[simd] WASM SIMD self-test failed (${selfTestFailure}); using the plain JS kernels`);
      return null;
    }
    compiled = mod;
  } catch {
    compiled = null;
  }
  return compiled;
}

// ---------------------------------------------------------------------------
// Runtime self-test
// ---------------------------------------------------------------------------

/**
 * Known-answer test of the v128 kernels against scalar JS references, run once per realm before
 * the module is first used. Safari 26.x shipped a JavaScriptCore regression ("RegisterSet::
 * normalizeWidths() lost vector-width information, causing v128 argument corruption in
 * WebAssembly SIMD thunks", fixed in Safari 27.0; the affected 26.x builds are unknown —
 * docs/research/ios-webgpu.md §6). Our exports take no v128 arguments and make no calls, but
 * tier-up / OSR-entry thunks still run while v128 locals are live, so every kernel is run for
 * several rounds (thousands of loop back-edges, across the tier-up thresholds) and checked after
 * each round. Any mismatch (or trap) disables SIMD for the realm: every caller has a plain-JS
 * path. `desc` is scalar f32 code (no v128) and is covered by the SIFT parity tests instead.
 * Returns the name of the failing kernel, or null. Cost: ≈ 10 ms once per realm (Node); the
 * 128 KiB instance is kept as `spare` when `keep` is set and the test passed.
 */
function runSelfTest(mod: WebAssembly.Module, keep = false): string | null {
  let stage = 'instantiate';
  try {
    const memory = new WebAssembly.Memory({ initial: 2 });
    const ex = new WebAssembly.Instance(mod, { e: { m: memory } }).exports as unknown as KernelExports;
    const mem = memory.buffer;
    // deterministic LCG (no Math.random: the result must not depend on the run)
    let seed = 0x2545f491;
    const rnd = (): number => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 4294967296;
    };
    const f = Math.fround;

    // --- dot4: exact i32 sums of i16 products (signed values exercise i32x4.dot_i16x8_s) ----
    stage = 'dot4';
    const NB = 128;
    const aOff = 0;
    const bOff = 1024;
    const oOff = bOff + NB * 256;
    const A = new Int16Array(mem, aOff, 4 * 128);
    const B = new Int16Array(mem, bOff, NB * 128);
    for (let i = 0; i < A.length; i++) A[i] = Math.floor(rnd() * 601) - 300;
    for (let i = 0; i < B.length; i++) B[i] = Math.floor(rnd() * 601) - 300;
    const dref = new Int32Array(4 * NB);
    for (let j = 0; j < NB; j++)
      for (let q = 0; q < 4; q++) {
        let s = 0;
        for (let t = 0; t < 128; t++) s += A[q * 128 + t] * B[j * 128 + t];
        dref[4 * j + q] = s;
      }
    const dout = new Int32Array(mem, oOff, 4 * NB);
    for (let round = 0; round < 32; round++) {
      dout.fill(0x55555555);
      ex.dot4(aOff, bOff, NB, oOff);
      for (let i = 0; i < dref.length; i++) if (dout[i] !== dref[i]) return 'dot4';
    }

    // --- planes: blurH / blurV (f32x4 mul/add, replicated borders) and grad ------------------
    const W = 64;
    const H = 24;
    const n = W * H;
    const PLANE = 8192; // 6 KiB plane + slack for the 16-byte block overrun
    const sOff = 36864;
    const tOff = sOff + PLANE;
    const dOff = tOff + PLANE;
    const mOff = dOff + PLANE;
    const angOff = mOff + PLANE;
    const padOff = angOff + PLANE;
    const kvOff = padOff + 1024;
    const kern = [0.3989423, 0.2419707, 0.05399097, 0.004431848, 0.0001338302];
    const R = kern.length - 1;
    const src = new Float32Array(mem, sOff, n);
    for (let i = 0; i < n; i++) src[i] = rnd();
    for (let x = 0; x < W; x++) src[7 * W + x] = 0.5; // a flat row: gx = 0 (angles ±π/2 exactly)
    new Float32Array(mem, kvOff, kern.length).set(kern);
    const k32 = Float32Array.from(kern);
    const S = Float32Array.from(src);
    const tmp = new Float32Array(mem, tOff, n);
    const dst = new Float32Array(mem, dOff, n);
    // scalar references in the kernels' exact operation order (f32 rounding at every step)
    const href = new Float32Array(n);
    const vref = new Float32Array(n);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        let acc = f(k32[0] * S[y * W + x]);
        for (let j = 1; j <= R; j++) acc = f(acc + f(k32[j] * f(S[y * W + Math.max(x - j, 0)] + S[y * W + Math.min(x + j, W - 1)])));
        href[y * W + x] = acc;
      }
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        let acc = f(k32[0] * href[y * W + x]);
        for (let j = 1; j <= R; j++) acc = f(acc + f(k32[j] * f(href[Math.max(y - j, 0) * W + x] + href[Math.min(y + j, H - 1) * W + x])));
        vref[y * W + x] = acc;
      }
    const near = (a: number, b: number): boolean => Math.abs(a - b) <= 1e-6 * (1 + Math.abs(b));
    const mag = new Float32Array(mem, mOff, n);
    const ang = new Float32Array(mem, angOff, n);
    for (let round = 0; round < 8; round++) {
      stage = 'blurH';
      tmp.fill(NaN);
      ex.blurH(sOff, tOff, W, H, kvOff, R, padOff);
      for (let i = 0; i < n; i++) if (!near(tmp[i], href[i])) return 'blurH';
      stage = 'blurV';
      dst.fill(NaN);
      ex.blurV(tOff, dOff, W, H, kvOff, R);
      for (let i = 0; i < n; i++) if (!near(dst[i], vref[i])) return 'blurV';
      stage = 'grad';
      mag.fill(NaN);
      ang.fill(NaN);
      ex.grad(sOff, mOff, angOff, W, H);
      for (let y = 1; y < H - 1; y++)
        for (let x = 1; x < W - 1; x++) {
          const i = y * W + x;
          const gx = f(0.5 * f(S[i + 1] - S[i - 1]));
          const gy = f(0.5 * f(S[i + W] - S[i - W]));
          if (!near(mag[i], f(Math.sqrt(f(f(gx * gx) + f(gy * gy)))))) return 'grad';
          const a = ang[i];
          if (!(a >= 0 && a < 2 * Math.PI)) return 'grad';
          if (gx === 0 && gy === 0) continue;
          let e = Math.atan2(gy, gx);
          if (e < 0) e += 2 * Math.PI;
          const d = Math.abs(a - e);
          if (Math.min(d, 2 * Math.PI - d) > 1e-4) return 'grad';
        }
    }
    if (keep) spare = { memory, ex };
    return null;
  } catch (e) {
    return `${stage}: ${(e as Error)?.message ?? e}`;
  }
}

/**
 * State of the WASM SIMD kernels in this realm (diagnostics): 'ok', 'disabled'
 * (SFM_DISABLE_SIMD), 'unsupported' (no WebAssembly / SIMD) or 'self-test-failed: <kernel>'.
 * Compiles (and self-tests) the module on first use.
 */
export function simdStatus(): string {
  const g = globalThis as { SFM_DISABLE_SIMD?: boolean };
  if (g.SFM_DISABLE_SIMD) return 'disabled';
  if (getModule()) return 'ok';
  return selfTestFailure ? `self-test-failed: ${selfTestFailure}` : 'unsupported';
}

/**
 * Run the known-answer self-test on `bytes` (default: this module) in a fresh instance; returns
 * the failing kernel (or 'compile' when the bytes do not validate) or null. Tests / diagnostics.
 */
export function simdSelfTest(bytes: Uint8Array<ArrayBuffer> = buildModule()): string | null {
  try {
    if (typeof WebAssembly === 'undefined' || !WebAssembly.validate(bytes)) return 'compile';
    return runSelfTest(new WebAssembly.Module(bytes));
  } catch {
    return 'compile';
  }
}

/** True when the WASM SIMD kernels can be used in this realm. */
export function simdAvailable(): boolean {
  const g = globalThis as { SFM_DISABLE_SIMD?: boolean };
  return !g.SFM_DISABLE_SIMD && getModule() !== null;
}

/** New kernel instance with its own memory of at least `initialBytes` bytes, or null. */
export function createSimdKernels(initialBytes = 1 << 16): SimdKernels | null {
  if (!simdAvailable()) return null;
  const mod = getModule();
  if (!mod) return null;
  try {
    const pages = Math.max(1, Math.ceil(initialBytes / 65536));
    let memory: WebAssembly.Memory;
    let ex: KernelExports;
    if (spare) {
      // the self-test's instance: zeroed, so it is indistinguishable from a fresh one
      ({ memory, ex } = spare);
      spare = null;
      new Uint8Array(memory.buffer).fill(0);
      const have = memory.buffer.byteLength / 65536;
      if (pages > have) memory.grow(pages - have);
    } else {
      memory = new WebAssembly.Memory({ initial: pages });
      ex = new WebAssembly.Instance(mod, { e: { m: memory } }).exports as unknown as KernelExports;
    }
    return {
      memory,
      blurV: ex.blurV,
      blurH: ex.blurH,
      dot4: ex.dot4,
      grad: ex.grad,
      desc: ex.desc,
      ensure(bytes: number): void {
        const cur = memory.buffer.byteLength;
        if (bytes > cur) memory.grow(Math.ceil((bytes - cur) / 65536));
      },
    };
  } catch {
    return null;
  }
}

/** Raw module bytes (exposed for tests / diagnostics). */
export function simdModuleBytes(): Uint8Array<ArrayBuffer> {
  return buildModule();
}
