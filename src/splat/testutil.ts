/**
 * Random small scenes for the splat tests (CPU and GPU).
 */
import { Rng } from '../math/random';
import { allocSplatModel, logit, shCoeffCount, type ShDegree, type SplatCamera, type SplatModel } from './types';

/** Row-major rotation that looks from `eye` at `target` (OpenCV: +z forward, +y down), world +z up. */
export function lookAt(eye: [number, number, number], target: [number, number, number], up: [number, number, number] = [0, 0, 1]): { R: number[]; t: number[] } {
  const f = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
  const fl = Math.hypot(f[0], f[1], f[2]);
  const z = f.map((v) => v / fl);
  // x = z × up (right), y = z × x (down)
  let x = [z[1] * up[2] - z[2] * up[1], z[2] * up[0] - z[0] * up[2], z[0] * up[1] - z[1] * up[0]];
  const xl = Math.hypot(x[0], x[1], x[2]) || 1;
  x = x.map((v) => v / xl);
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  const R = [...x, ...y, ...z];
  const t = [-(R[0] * eye[0] + R[1] * eye[1] + R[2] * eye[2]), -(R[3] * eye[0] + R[4] * eye[1] + R[5] * eye[2]), -(R[6] * eye[0] + R[7] * eye[1] + R[8] * eye[2])];
  return { R, t };
}

export function testCamera(width: number, height: number, eye: [number, number, number] = [0, -3, 0.4], target: [number, number, number] = [0, 0, 0], fScale = 1.1): SplatCamera {
  const { R, t } = lookAt(eye, target);
  const f = fScale * Math.max(width, height);
  return { width, height, fx: f, fy: f * 1.02, cx: (width - 1) / 2 + 0.7, cy: (height - 1) / 2 - 0.4, R, t };
}

export interface RandomSceneOptions {
  count: number;
  shDegree?: ShDegree;
  seed?: number;
  /** Half-size of the cube the means are drawn from (default 0.6). */
  extent?: number;
  /** log-scale range (default [-3.2, -1.6]). */
  logScale?: [number, number];
  /** Opacity range (default [0.3, 0.75]). */
  opacity?: [number, number];
}

/** Gaussians with random means in a cube around the origin, random rotations, scales, opacities and SH. */
export function randomScene(o: RandomSceneOptions): SplatModel {
  const rng = new Rng(o.seed ?? 1);
  const deg = o.shDegree ?? 0;
  const m = allocSplatModel(o.count, deg);
  const e = o.extent ?? 0.6;
  const [ls0, ls1] = o.logScale ?? [-3.2, -1.6];
  const [op0, op1] = o.opacity ?? [0.3, 0.75];
  const K = shCoeffCount(deg);
  for (let i = 0; i < o.count; i++) {
    for (let k = 0; k < 3; k++) {
      m.means[3 * i + k] = rng.uniform(-e, e);
      m.logScales[3 * i + k] = rng.uniform(ls0, ls1);
    }
    for (let k = 0; k < 4; k++) m.quats[4 * i + k] = rng.gaussian();
    m.opacities[i] = logit(rng.uniform(op0, op1));
    for (let k = 0; k < 3 * K; k++) m.sh[3 * K * i + k] = k < 3 ? rng.uniform(-1.2, 1.2) : rng.uniform(-0.4, 0.4);
  }
  return m;
}

export function cloneModel(m: SplatModel): SplatModel {
  return { ...m, means: m.means.slice(), logScales: m.logScales.slice(), quats: m.quats.slice(), opacities: m.opacities.slice(), sh: m.sh.slice() };
}
