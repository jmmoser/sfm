// Small dense linear algebra for evaluation (plain JS, no dependencies).
// Matrices are row-major arrays of 9 numbers (3x3); vectors are arrays of 3.

export function mat3Mul(a, b) {
  const o = new Array(9);
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++) o[3 * r + c] = a[3 * r] * b[c] + a[3 * r + 1] * b[3 + c] + a[3 * r + 2] * b[6 + c];
  return o;
}

export function mat3T(a) {
  return [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
}

export function mat3Vec(a, v) {
  return [a[0] * v[0] + a[1] * v[1] + a[2] * v[2], a[3] * v[0] + a[4] * v[1] + a[5] * v[2], a[6] * v[0] + a[7] * v[1] + a[8] * v[2]];
}

export function mat3Det(a) {
  return a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
}

export function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function norm(a) {
  return Math.hypot(a[0], a[1], a[2]);
}

/** Geodesic angle between two rotations (radians). */
export function rotationAngleBetween(a, b) {
  const M = mat3Mul(mat3T(a), b);
  const c = Math.min(1, Math.max(-1, (M[0] + M[4] + M[8] - 1) / 2));
  const s = 0.5 * Math.hypot(M[7] - M[5], M[2] - M[6], M[3] - M[1]);
  return Math.atan2(s, c);
}

/** Rotation about axis (unit) by angle (Rodrigues). */
export function axisAngle(axis, angle) {
  const [x, y, z] = axis;
  const c = Math.cos(angle),
    s = Math.sin(angle),
    C = 1 - c;
  return [c + x * x * C, x * y * C - z * s, x * z * C + y * s, y * x * C + z * s, c + y * y * C, y * z * C - x * s, z * x * C - y * s, z * y * C + x * s, c + z * z * C];
}

/**
 * SVD of a 3x3 matrix by one-sided Jacobi: A = U diag(S) Vᵀ, S descending, U and V
 * orthogonal (proper completion when A is rank deficient).
 */
export function svd3(A) {
  // Work on columns of B = A (3x3), accumulate V.
  const B = A.slice();
  const V = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < 2; p++)
      for (let q = p + 1; q < 3; q++) {
        let alpha = 0,
          beta = 0,
          gamma = 0;
        for (let i = 0; i < 3; i++) {
          const bp = B[3 * i + p],
            bq = B[3 * i + q];
          alpha += bp * bp;
          beta += bq * bq;
          gamma += bp * bq;
        }
        if (Math.abs(gamma) <= 1e-300 || Math.abs(gamma) <= 1e-15 * Math.sqrt(alpha * beta)) continue;
        off = Math.max(off, Math.abs(gamma) / Math.sqrt(alpha * beta));
        const zeta = (beta - alpha) / (2 * gamma);
        const t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t),
          s = c * t;
        for (let i = 0; i < 3; i++) {
          const bp = B[3 * i + p],
            bq = B[3 * i + q];
          B[3 * i + p] = c * bp - s * bq;
          B[3 * i + q] = s * bp + c * bq;
          const vp = V[3 * i + p],
            vq = V[3 * i + q];
          V[3 * i + p] = c * vp - s * vq;
          V[3 * i + q] = s * vp + c * vq;
        }
      }
    if (off < 1e-15) break;
  }
  const cols = [0, 1, 2].map((j) => [B[j], B[3 + j], B[6 + j]]);
  const S = cols.map(norm);
  const order = [0, 1, 2].sort((a, b) => S[b] - S[a]);
  const Ss = order.map((j) => S[j]);
  let Uc = order.map((j) => (S[j] > 1e-300 ? cols[j].map((x) => x / S[j]) : null));
  const Vc = order.map((j) => [V[j], V[3 + j], V[6 + j]]);
  // complete U for rank deficiency
  const tol = 1e-12 * (Ss[0] || 1);
  for (let k = 0; k < 3; k++) if (Ss[k] <= tol) Uc[k] = null;
  if (!Uc[0]) Uc = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  else {
    if (!Uc[1]) {
      const a = Uc[0];
      const e = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
      const c = cross(a, e);
      const n = norm(c);
      Uc[1] = c.map((x) => x / n);
    }
    if (!Uc[2]) Uc[2] = cross(Uc[0], Uc[1]);
  }
  const U = [Uc[0][0], Uc[1][0], Uc[2][0], Uc[0][1], Uc[1][1], Uc[2][1], Uc[0][2], Uc[1][2], Uc[2][2]];
  const Vm = [Vc[0][0], Vc[1][0], Vc[2][0], Vc[0][1], Vc[1][1], Vc[2][1], Vc[0][2], Vc[1][2], Vc[2][2]];
  return { U, S: Ss, V: Vm };
}

/** Nearest rotation (Frobenius) to a 3x3 matrix. */
export function projectToSO3(M) {
  const { U, V } = svd3(M);
  const d = mat3Det(mat3Mul(U, mat3T(V))) < 0 ? -1 : 1;
  return mat3Mul(mat3Mul(U, [1, 0, 0, 0, 1, 0, 0, 0, d]), mat3T(V));
}

/** Chordal L2 mean of rotations. */
export function chordalMean(Rs) {
  const S = new Array(9).fill(0);
  for (const R of Rs) for (let i = 0; i < 9; i++) S[i] += R[i];
  return projectToSO3(S);
}

/**
 * Umeyama (1991) similarity / rigid alignment: dst ≈ s R src + t.
 * src, dst: arrays of [x, y, z]. Returns null when < 3 points or degenerate.
 */
export function umeyama(src, dst, withScale = true) {
  const n = src.length;
  if (n < 3 || dst.length !== n) return null;
  const ms = [0, 0, 0],
    md = [0, 0, 0];
  for (let i = 0; i < n; i++)
    for (let k = 0; k < 3; k++) {
      ms[k] += src[i][k] / n;
      md[k] += dst[i][k] / n;
    }
  const C = new Array(9).fill(0);
  let varS = 0;
  for (let i = 0; i < n; i++) {
    const a = [src[i][0] - ms[0], src[i][1] - ms[1], src[i][2] - ms[2]];
    const b = [dst[i][0] - md[0], dst[i][1] - md[1], dst[i][2] - md[2]];
    varS += (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) / n;
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) C[3 * r + c] += (b[r] * a[c]) / n;
  }
  if (!(varS > 0)) return null;
  const { U, S, V } = svd3(C);
  if (!(S[1] > 1e-12 * (S[0] || 1))) return null; // collinear
  const d = mat3Det(U) * mat3Det(V) < 0 ? -1 : 1;
  const D = [1, 0, 0, 0, 1, 0, 0, 0, d];
  const R = mat3Mul(mat3Mul(U, D), mat3T(V));
  const s = withScale ? (S[0] + S[1] + d * S[2]) / varS : 1;
  const Rms = mat3Vec(R, ms);
  const t = [md[0] - s * Rms[0], md[1] - s * Rms[1], md[2] - s * Rms[2]];
  return { s, R, t };
}

export function applySim3(sim, p) {
  const q = mat3Vec(sim.R, p);
  return [sim.s * q[0] + sim.t[0], sim.s * q[1] + sim.t[1], sim.s * q[2] + sim.t[2]];
}

// ---- statistics ----
export function percentile(values, p) {
  if (values.length === 0) return NaN;
  const a = Float64Array.from(values).sort();
  const x = (p / 100) * (a.length - 1);
  const lo = Math.floor(x),
    hi = Math.ceil(x);
  if (lo === hi) return a[lo];
  const w = x - lo;
  if (!Number.isFinite(a[lo]) || !Number.isFinite(a[hi])) return w < 0.5 ? a[lo] : a[hi];
  return a[lo] + (a[hi] - a[lo]) * w;
}

export function mean(values) {
  let s = 0;
  for (const v of values) s += v;
  return values.length ? s / values.length : NaN;
}

export function rms(values) {
  let s = 0;
  for (const v of values) s += v * v;
  return values.length ? Math.sqrt(s / values.length) : NaN;
}
