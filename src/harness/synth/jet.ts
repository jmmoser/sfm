/**
 * Second-order forward-mode automatic differentiation in one variable (time).
 *
 * A `Jet` carries f(t), f'(t) and f''(t). Every trajectory is written once with jet
 * arithmetic, which yields EXACT analytic first and second derivatives of positions
 * and rotation matrices (no finite differences). Only C∞ primitives are provided
 * (sin, cos, atan2 away from the origin, sqrt of positive arguments), so all
 * trajectories built from them are smooth (C2 and beyond).
 */

export interface Jet {
  readonly v: number;
  readonly d: number;
  readonly dd: number;
}

export type JetVec3 = [Jet, Jet, Jet];
/** Row-major 3x3 matrix of jets. */
export type JetMat3 = Jet[];

export function jet(v: number, d = 0, dd = 0): Jet {
  return { v, d, dd };
}

/** The independent variable t (seconds). */
export function jvar(t: number): Jet {
  return { v: t, d: 1, dd: 0 };
}

export function jconst(v: number): Jet {
  return { v, d: 0, dd: 0 };
}

export function jadd(a: Jet, b: Jet): Jet {
  return { v: a.v + b.v, d: a.d + b.d, dd: a.dd + b.dd };
}

export function jsub(a: Jet, b: Jet): Jet {
  return { v: a.v - b.v, d: a.d - b.d, dd: a.dd - b.dd };
}

export function jneg(a: Jet): Jet {
  return { v: -a.v, d: -a.d, dd: -a.dd };
}

export function jaddc(a: Jet, c: number): Jet {
  return { v: a.v + c, d: a.d, dd: a.dd };
}

export function jscale(a: Jet, s: number): Jet {
  return { v: a.v * s, d: a.d * s, dd: a.dd * s };
}

export function jmul(a: Jet, b: Jet): Jet {
  return { v: a.v * b.v, d: a.d * b.v + a.v * b.d, dd: a.dd * b.v + 2 * a.d * b.d + a.v * b.dd };
}

/** Sum of any number of jets. */
export function jsum(...xs: Jet[]): Jet {
  let v = 0,
    d = 0,
    dd = 0;
  for (const x of xs) {
    v += x.v;
    d += x.d;
    dd += x.dd;
  }
  return { v, d, dd };
}

/** g(a) given g(a.v), g'(a.v), g''(a.v) (chain rule to second order). */
export function jchain(a: Jet, g0: number, g1: number, g2: number): Jet {
  return { v: g0, d: g1 * a.d, dd: g2 * a.d * a.d + g1 * a.dd };
}

export function jsin(a: Jet): Jet {
  const s = Math.sin(a.v),
    c = Math.cos(a.v);
  return jchain(a, s, c, -s);
}

export function jcos(a: Jet): Jet {
  const s = Math.sin(a.v),
    c = Math.cos(a.v);
  return jchain(a, c, -s, -c);
}

export function jsqrt(a: Jet): Jet {
  const s = Math.sqrt(a.v);
  return jchain(a, s, 0.5 / s, -0.25 / (s * a.v));
}

export function jrecip(a: Jet): Jet {
  const r = 1 / a.v;
  return jchain(a, r, -r * r, 2 * r * r * r);
}

export function jdiv(a: Jet, b: Jet): Jet {
  return jmul(a, jrecip(b));
}

/** atan2(y, x) (defined away from the origin). */
export function jatan2(y: Jet, x: Jet): Jet {
  const D = x.v * x.v + y.v * y.v;
  const N = x.v * y.d - y.v * x.d;
  const Nd = x.v * y.dd - y.v * x.dd; // d/dt (x y' - y x') = x y'' - y x''
  const Dd = 2 * (x.v * x.d + y.v * y.d);
  return { v: Math.atan2(y.v, x.v), d: N / D, dd: (Nd * D - N * Dd) / (D * D) };
}

/** A * sin(2π f t + φ) for the time variable t (most trajectory terms). */
export function jsinusoid(t: Jet, amplitude: number, freqHz: number, phase = 0): Jet {
  const w = 2 * Math.PI * freqHz;
  return jscale(jsin(jaddc(jscale(t, w), phase)), amplitude);
}

// ---------------------------------------------------------------------------
// Vectors / matrices of jets
// ---------------------------------------------------------------------------

export function jvec3(x: Jet, y: Jet, z: Jet): JetVec3 {
  return [x, y, z];
}

export function jvadd(a: JetVec3, b: JetVec3): JetVec3 {
  return [jadd(a[0], b[0]), jadd(a[1], b[1]), jadd(a[2], b[2])];
}

export function jvsub(a: JetVec3, b: JetVec3): JetVec3 {
  return [jsub(a[0], b[0]), jsub(a[1], b[1]), jsub(a[2], b[2])];
}

export function jvscale(a: JetVec3, s: Jet): JetVec3 {
  return [jmul(a[0], s), jmul(a[1], s), jmul(a[2], s)];
}

/** Elementary rotation about x by angle a (row-major). */
export function jrotX(a: Jet): JetMat3 {
  const c = jcos(a),
    s = jsin(a);
  const o = jconst(0),
    l = jconst(1);
  return [l, o, o, o, c, jneg(s), o, s, c];
}

export function jrotY(a: Jet): JetMat3 {
  const c = jcos(a),
    s = jsin(a);
  const o = jconst(0),
    l = jconst(1);
  return [c, o, s, o, l, o, jneg(s), o, c];
}

export function jrotZ(a: Jet): JetMat3 {
  const c = jcos(a),
    s = jsin(a);
  const o = jconst(0),
    l = jconst(1);
  return [c, jneg(s), o, s, c, o, o, o, l];
}

export function jmatConst(m: ArrayLike<number>): JetMat3 {
  const out: JetMat3 = [];
  for (let i = 0; i < 9; i++) out.push(jconst(m[i]));
  return out;
}

export function jmatMul(a: JetMat3, b: JetMat3): JetMat3 {
  const out: JetMat3 = [];
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out.push(jsum(jmul(a[3 * r], b[c]), jmul(a[3 * r + 1], b[3 + c]), jmul(a[3 * r + 2], b[6 + c])));
    }
  }
  return out;
}

/** Split a jet matrix into value / first / second derivative Float64Arrays. */
export function jmatSplit(m: JetMat3): { v: Float64Array; d: Float64Array; dd: Float64Array } {
  const v = new Float64Array(9),
    d = new Float64Array(9),
    dd = new Float64Array(9);
  for (let i = 0; i < 9; i++) {
    v[i] = m[i].v;
    d[i] = m[i].d;
    dd[i] = m[i].dd;
  }
  return { v, d, dd };
}

export function jvecSplit(a: JetVec3): { v: Float64Array; d: Float64Array; dd: Float64Array } {
  return {
    v: Float64Array.of(a[0].v, a[1].v, a[2].v),
    d: Float64Array.of(a[0].d, a[1].d, a[2].d),
    dd: Float64Array.of(a[0].dd, a[1].dd, a[2].dd),
  };
}
