import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { getGPU, type GPUContext } from '../gpu/device';
import { pipelineCompileLimits } from '../features/gpu-common';
import { Rng } from '../math/random';
import { backwardCPU, lossCPU, renderCPU } from './cpu-reference';
import { SplatRenderer } from './renderer';
import { cloneModel, lookAt, randomScene, testCamera } from './testutil';
import type { RGBAImage } from '../core/types';
import { SplatTrainer, SplatWorkDroppedError, type SplatTrainingView as TrainerView } from './trainer';
import { geometryLossCPU, packDepthTarget, type DepthMapLike } from './depth-target';

type SplatTrainingView = TrainerView & { image: RGBAImage };
import type { SplatCamera, SplatModel } from './types';

let ctx: GPUContext;
const trainers: SplatTrainer[] = [];

beforeAll(async () => {
  const c = await getGPU();
  if (!c) throw new Error('WebGPU unavailable in the test browser');
  // SwiftShader compiles these pipelines for minutes: no async → sync fallback (it compiles twice)
  if (c.isSoftware) pipelineCompileLimits.timeoutMs = 600_000;
  ctx = c;
}, 600_000);

afterAll(() => {
  for (const t of trainers) t.destroy();
});

function viewOf(cam: SplatCamera, image?: Uint8ClampedArray, id = 0): SplatTrainingView {
  return {
    id,
    cam: { width: cam.width, height: cam.height, f: cam.fx, cx: cam.cx, cy: cam.cy, k1: 0, k2: 0 },
    pose: { R: Float64Array.from(cam.R), t: Float64Array.from(cam.t) },
    image: { width: cam.width, height: cam.height, data: image ?? new Uint8ClampedArray(4 * cam.width * cam.height).fill(255) },
  };
}

async function trainer(model: SplatModel, views: SplatTrainingView[], extra: Partial<ConstructorParameters<typeof Object>[0]> = {}): Promise<SplatTrainer> {
  const t = await SplatTrainer.create(ctx, { maxGaussians: Math.max(64, 2 * model.count), iterations: 300, shDegree: model.shDegree, ...extra });
  t.init(model, views);
  trainers.push(t);
  return t;
}

/** ‖a − b‖ / ‖b‖ over a field. */
function relErr(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let num = 0;
  let den = 0;
  for (let i = 0; i < b.length; i++) {
    num += (a[i] - b[i]) ** 2;
    den += b[i] ** 2;
  }
  return Math.sqrt(num / Math.max(den, 1e-300));
}

describe('SplatTrainer resources', () => {
  /** Buffers created on the shared device while `fn` runs (the `failAt`-th creation throws) and whether each was destroyed. */
  const tracked = (fn: () => void, failAt = Infinity): { created: GPUBuffer[]; destroyed: Set<GPUBuffer> } => {
    const device = ctx.device;
    const created: GPUBuffer[] = [];
    const destroyed = new Set<GPUBuffer>();
    const createBuffer = device.createBuffer;
    device.createBuffer = (d: GPUBufferDescriptor): GPUBuffer => {
      if (created.length + 1 >= failAt) throw new Error('simulated failure');
      const b = createBuffer.call(device, d);
      created.push(b);
      const destroy = b.destroy.bind(b);
      b.destroy = (): undefined => {
        destroyed.add(b);
        return destroy();
      };
      return b;
    };
    try {
      fn();
    } finally {
      delete (device as unknown as Record<string, unknown>).createBuffer;
    }
    return { created, destroyed };
  };

  it('a failed init frees what it allocated (review)', async () => {
    const m = randomScene({ count: 50, seed: 41 });
    const views = [viewOf(testCamera(64, 48))];
    // an intersection reservation beyond one sorter (64 M): refused before anything is allocated
    const a = await SplatTrainer.create(ctx, { maxGaussians: 1000, iterations: 10, shDegree: 0, intersectionsPerGaussian: 100_000 });
    const ra = tracked(() => expect(() => a.init(m, views)).toThrow(/too large/));
    a.destroy();
    expect(ra.created.map((b) => b.label)).toEqual([]);
    // init throwing half-way (its 5th buffer, the forward pass's first): destroy() frees the first four
    const b = await SplatTrainer.create(ctx, { maxGaussians: 1000, iterations: 10, shDegree: 0 });
    const rb = tracked(() => expect(() => b.init(m, views)).toThrow(/simulated/), 5);
    b.destroy();
    expect(rb.created.map((x) => x.label)).toEqual(['splat-params', 'splat-adam-m', 'splat-adam-v', 'splat-grad2d']);
    expect(rb.created.filter((x) => !rb.destroyed.has(x)).map((x) => x.label)).toEqual([]);
  });
});

describe('SplatTrainer backward (GPU) vs CPU reference', () => {
  for (const deg of [0, 1] as const) {
    it(`parameter gradients match, SH degree ${deg}`, async () => {
      const m = randomScene({ count: 300, seed: 31 + deg, shDegree: deg });
      const cam = testCamera(120, 100);
      const t = await trainer(m, [viewOf(cam)]);
      const rng = new Rng(9);
      const w = new Float32Array(3 * cam.width * cam.height).map(() => rng.uniform(-1, 1));
      const g = await t.debugGradients(cam, w);
      const c = backwardCPU(m, cam, w);
      const e2d = relErr(g.grad2d, c.grad2d.g);
      const e = {
        grad2d: e2d,
        means: relErr(g.grads.means, c.grads.means),
        logScales: relErr(g.grads.logScales, c.grads.logScales),
        quats: relErr(g.grads.quats, c.grads.quats),
        opacities: relErr(g.grads.opacities, c.grads.opacities),
        sh: relErr(g.grads.sh, c.grads.sh),
      };
      console.log(`[splat] backward parity deg${deg}`, JSON.stringify(e));
      for (const v of Object.values(e)) expect(v).toBeLessThan(2e-3);
    });
  }

  it('loss kernels: L1 + D-SSIM and ∂loss/∂C match the CPU reference', async () => {
    const m = randomScene({ count: 200, seed: 41 });
    const cam = testCamera(90, 70);
    const rng = new Rng(10);
    const img = new Uint8ClampedArray(4 * cam.width * cam.height).map((_, i) => (i % 4 === 3 ? 255 : rng.int(256)));
    const t = await trainer(m, [viewOf(cam, img)], { colourCorrection: false });
    const r = await t.debugLoss(0, 1);
    const px = r.width * r.height;
    const x = new Float64Array(3 * px);
    const y = new Float64Array(3 * px);
    for (let i = 0; i < px; i++)
      for (let c = 0; c < 3; c++) {
        x[3 * i + c] = r.render[4 * i + c];
        y[3 * i + c] = r.target[4 * i + c] / 255;
      }
    const ref = lossCPU(x, y, r.width, r.height, 0.2);
    const n = 3 * r.valid;
    console.log(`[splat] loss GPU l1 ${(r.l1 / n).toFixed(6)} ssim ${(r.ssim / n).toFixed(6)} | CPU l1 ${ref.l1.toFixed(6)} ssim ${ref.ssim.toFixed(6)}`);
    expect(r.valid).toBe(px);
    expect(Math.abs(r.l1 / n - ref.l1)).toBeLessThan(1e-5);
    expect(Math.abs(r.ssim / n - ref.ssim)).toBeLessThan(1e-4);
    expect(relErr(r.dLdC, ref.grad)).toBeLessThan(1e-3);
  });

  it('masked target (alpha 0: undistortion borders): ∂loss/∂C normalised by the valid pixels like the CPU reference (review)', async () => {
    const m = randomScene({ count: 200, seed: 42 });
    const cam = testCamera(90, 70);
    const rng = new Rng(11);
    // a 13-pixel border on the left and top, and random holes
    const img = new Uint8ClampedArray(4 * cam.width * cam.height).map((_, i) => {
      if (i % 4 !== 3) return rng.int(256);
      const p = i >> 2;
      return p % cam.width < 13 || p < 13 * cam.width || rng.next() < 0.05 ? 0 : 255;
    });
    const t = await trainer(m, [viewOf(cam, img)], { colourCorrection: false });
    for (const factor of [1, 2]) {
      const r = await t.debugLoss(0, factor);
      const px = r.width * r.height;
      const x = new Float64Array(3 * px);
      const y = new Float64Array(3 * px);
      const mask = new Uint8Array(px);
      for (let i = 0; i < px; i++) {
        mask[i] = r.target[4 * i + 3] >= 128 ? 1 : 0;
        for (let c = 0; c < 3; c++) {
          x[3 * i + c] = r.render[4 * i + c];
          y[3 * i + c] = r.target[4 * i + c] / 255;
        }
      }
      const ref = lossCPU(x, y, r.width, r.height, 0.2, mask);
      const nv = mask.reduce((a, b) => a + b, 0);
      expect(r.valid).toBe(nv);
      expect(nv).toBeLessThan(0.85 * px);
      expect(Math.abs(r.l1 / (3 * nv) - ref.l1)).toBeLessThan(1e-5);
      expect(relErr(r.dLdC, ref.grad)).toBeLessThan(1e-3);
    }
  });
});

describe('SplatTrainer geometry channels (GPU) vs CPU reference', () => {
  for (const deg of [0, 1] as const) {
    it(`rendered inverse depth / normal and parameter gradients match, SH degree ${deg}`, async () => {
      const m = randomScene({ count: 300, seed: 131 + deg, shDegree: deg });
      const cam = testCamera(120, 100);
      const t = await trainer(m, [viewOf(cam)], { geometry: { normal: [0.1, 0.01] } });
      const rng = new Rng(19);
      const w = new Float32Array(3 * cam.width * cam.height).map(() => rng.uniform(-1, 1));
      const wG = new Float32Array(4 * cam.width * cam.height).map((_, i) => rng.uniform(-1, 1) * (i % 4 === 0 ? 3 : 1));
      const wA = new Float32Array(cam.width * cam.height).map(() => rng.uniform(-1, 1));
      const g = await t.debugGradients(cam, w, wG, wA);
      const c = backwardCPU(m, cam, w, { geometry: true }, undefined, wG, wA);
      expect(c.grad2d.stride).toBe(13);
      const e = {
        geo: relErr(g.geo!, c.fwd.geo!),
        grad2d: relErr(g.grad2d, c.grad2d.g),
        means: relErr(g.grads.means, c.grads.means),
        logScales: relErr(g.grads.logScales, c.grads.logScales),
        quats: relErr(g.grads.quats, c.grads.quats),
        opacities: relErr(g.grads.opacities, c.grads.opacities),
        sh: relErr(g.grads.sh, c.grads.sh),
      };
      console.log(`[splat] geometry parity deg${deg}`, JSON.stringify(e));
      for (const v of Object.values(e)) expect(v).toBeLessThan(2e-3);
      // the geometry part alone (colour weights 0) also matches: the depth / normal gradients reach the means and quaternions
      const z = new Float32Array(w.length);
      const g0 = await t.debugGradients(cam, z, wG, wA);
      const c0 = backwardCPU(m, cam, z, { geometry: true }, undefined, wG, wA);
      expect(relErr(g0.grads.means, c0.grads.means)).toBeLessThan(2e-3);
      expect(relErr(g0.grads.quats, c0.grads.quats)).toBeLessThan(2e-3);
      expect(relErr(g0.grads.sh, c0.grads.sh)).toBe(0);
    });
  }

  it('depth-only kernels (normal weights 0): inverse depth and gradients match, 10 accumulators', async () => {
    const m = randomScene({ count: 300, seed: 135 });
    const cam = testCamera(120, 100);
    const t = await trainer(m, [viewOf(cam)], { geometry: { normal: [0, 0] } });
    const rng = new Rng(29);
    const px = cam.width * cam.height;
    const w = new Float32Array(3 * px).map(() => rng.uniform(-1, 1));
    const wG = new Float32Array(4 * px).map((_, i) => (i % 4 === 0 ? 3 * rng.uniform(-1, 1) : 0));
    const wA = new Float32Array(px).map(() => rng.uniform(-1, 1));
    const g = await t.debugGradients(cam, w, wG, wA);
    const c = backwardCPU(m, cam, w, { geometry: true }, undefined, wG, wA);
    expect(g.grad2d.length).toBe(10 * m.count);
    const cpu2d = new Float64Array(10 * m.count);
    for (let i = 0; i < m.count; i++) cpu2d.set(c.grad2d.g.subarray(13 * i, 13 * i + 10), 10 * i);
    const invD = (a: ArrayLike<number>): Float64Array => Float64Array.from({ length: px }, (_, i) => a[4 * i]);
    const e = {
      invDepth: relErr(invD(g.geo!), invD(c.fwd.geo!)),
      grad2d: relErr(g.grad2d, cpu2d),
      means: relErr(g.grads.means, c.grads.means),
      quats: relErr(g.grads.quats, c.grads.quats),
      opacities: relErr(g.grads.opacities, c.grads.opacities),
    };
    console.log('[splat] depth-only parity', JSON.stringify(e));
    for (const v of Object.values(e)) expect(v).toBeLessThan(2e-3);
    for (let i = 0; i < px; i++) expect(g.geo![4 * i + 1]).toBe(0);
  });

  it('geometry loss kernel matches the CPU reference (levels 1, 2, 4)', async () => {
    const m = randomScene({ count: 250, seed: 141, opacity: [0.6, 0.95] });
    const cam = testCamera(96, 80);
    // target: the CPU render of a perturbed copy (depth from the blended inverse depth)
    const pert = cloneModel(m);
    const rng = new Rng(23);
    for (let i = 0; i < pert.means.length; i++) pert.means[i] += rng.uniform(-0.05, 0.05);
    const r = renderCPU(pert, cam, { geometry: true });
    const px = cam.width * cam.height;
    const map: DepthMapLike = { width: cam.width, height: cam.height, depth: new Float32Array(px), normal: new Float32Array(3 * px), cost: new Float32Array(px) };
    for (let i = 0; i < px; i++) {
      const a = 1 - r.T[i];
      if (a < 0.5) continue;
      (map.depth as Float32Array)[i] = a / r.geo![4 * i];
      const n = [r.geo![4 * i + 1], r.geo![4 * i + 2], r.geo![4 * i + 3]];
      const l = Math.hypot(n[0], n[1], n[2]);
      if (l > 0) (map.normal as Float32Array).set(n.map((v) => v / l), 3 * i);
      (map.cost as Float32Array)[i] = rng.uniform(0, 0.8);
    }
    const pin = { width: cam.width, height: cam.height, f: cam.fx, cx: cam.cx, cy: cam.cy };
    const target = packDepthTarget(map, pin, pin, { factor: 2 });
    expect(target.valid).toBeGreaterThan(200);
    const view = viewOf({ ...cam, fy: cam.fx });
    // decaying weights (depth exponentially, normal linearly: one end is 0) so the schedule check
    // below sees a decay; the kernel checks run at iteration 0, i.e. with the start weights 0.8 / 0.3
    const t = await trainer(m, [view], { geometry: { depth: [0.8, 0.08], normal: [0.3, 0], until: 0.3, factor: 2 } });
    t.setViewDepth(0, target);
    expect(t.depthViews).toBe(1);
    for (const level of [1, 2, 4]) {
      const g = await t.debugGeoLoss(0, level);
      const ref = geometryLossCPU(g.geo, g.alpha, g.width, g.height, level, target, 0.8, 0.3);
      const e = Math.max(relErr(g.grad, ref.grad), relErr(g.gradA, ref.gradA));
      console.log(`[splat] geometry loss level ${level}: GPU sums ${g.sums.map((v) => v.toFixed(4)).join(' ')} | CPU ${ref.depth.toFixed(4)} ${ref.normal.toFixed(4)} ${ref.valid} ${ref.weight.toFixed(4)}; grad rel err ${e.toExponential(2)}`);
      expect(g.sums[2]).toBe(ref.valid);
      expect(Math.abs(g.sums[0] - ref.depth)).toBeLessThan(1e-4 * Math.max(1, ref.depth));
      expect(Math.abs(g.sums[1] - ref.normal)).toBeLessThan(1e-4 * Math.max(1, ref.normal));
      expect(e).toBeLessThan(1e-5);
    }
    // weights decay and stop at `until` (0.3 of the 300 iterations: iteration 90).
    // At iteration 0 the kernel gets exactly the start weights the CPU reference above was given.
    // They reach it as f32 uniforms, and the f64 log-space schedule exp(ln a·(1−f) + ln b·f) gives
    // a only to 1 ulp at f = 0 (ln / exp round: a constant 0.3 gives 0.29999999999999993, 6e-17
    // off, 2e-9 of an f32 ulp; the earlier exact f64 check failed on that. Here 0.8 happens to
    // round-trip exactly and the normal schedule is linear), so compare the f32 values the GPU
    // receives, not the f64 bits.
    expect(t.geometryWeightsAt(0).map(Math.fround)).toEqual([Math.fround(0.8), Math.fround(0.3)]);
    const end = 0.3 * 300;
    for (let it = 0; it < end; it++) {
      const f = it / end;
      const [wd, wn] = t.geometryWeightsAt(it);
      // independent closed forms of the documented schedule. Their f64 evaluations agree to
      // < 1e-14 relative (measured max 4.4e-16 depth, 6.6e-15 normal: the linear form cancels at
      // the end, ×90 at it = 89), so 1e-12 is pure rounding margin, far below the change of the
      // weights from one iteration to the next (depth × 10^(-1/90): 2.5 %, normal ≥ 1/90: 1.1 %).
      expect(Math.abs(wd - 0.8 * (0.08 / 0.8) ** f)).toBeLessThan(1e-12 * wd);
      expect(Math.abs(wn - 0.3 * (1 - f))).toBeLessThan(1e-12 * wn);
    }
    expect(t.geometryWeightsAt(end)).toEqual([0, 0]);
    expect(t.geometryWeightsAt(300)).toEqual([0, 0]);
  });

  it('depth supervision pulls the Gaussians onto the target depth while the photometric fit still improves', async () => {
    const gt = randomScene({ count: 200, seed: 151, opacity: [0.6, 0.95], logScale: [-3, -2] });
    const W = 64;
    const H = 48;
    const views: SplatTrainingView[] = [];
    const maps: DepthMapLike[] = [];
    const cams: SplatCamera[] = [];
    for (let k = 0; k < 8; k++) {
      const a = (2 * Math.PI * k) / 8;
      const { R, t } = lookAt([2.4 * Math.sin(a), -2.4 * Math.cos(a), 0.4], [0, 0, 0]);
      const cam: SplatCamera = { width: W, height: H, fx: 70, fy: 70, cx: 31.5, cy: 23.5, R, t };
      const r = renderCPU(gt, cam, { geometry: true });
      const img = new Uint8ClampedArray(4 * W * H);
      const map: DepthMapLike = { width: W, height: H, depth: new Float32Array(W * H), normal: new Float32Array(3 * W * H), cost: new Float32Array(W * H) };
      for (let i = 0; i < W * H; i++) {
        for (let c = 0; c < 3; c++) img[4 * i + c] = Math.round(255 * r.color[3 * i + c]);
        img[4 * i + 3] = 255;
        const al = 1 - r.T[i];
        if (al > 0.5) {
          (map.depth as Float32Array)[i] = al / r.geo![4 * i];
          const n = [r.geo![4 * i + 1], r.geo![4 * i + 2], r.geo![4 * i + 3]];
          const l = Math.hypot(n[0], n[1], n[2]) || 1;
          (map.normal as Float32Array).set(n.map((v) => v / l), 3 * i);
        }
      }
      cams.push(cam);
      views.push(viewOf(cam, img, k));
      maps.push(map);
    }
    // perturb the means along the depth direction of view 0 (photometrically ambiguous)
    const init = cloneModel(gt);
    const rng = new Rng(152);
    const R0 = cams[0].R;
    for (let i = 0; i < init.count; i++) {
      const dz = rng.uniform(-0.15, 0.15);
      for (let c = 0; c < 3; c++) init.means[3 * i + c] += dz * R0[6 + c];
    }
    const opts = { iterations: 400, colourCorrection: false as const, resolution: [], mcmc: false as const, lr: { mean: 1e-3, meanFinal: 1e-4 } };
    const depthErr = async (t: SplatTrainer): Promise<number> => {
      let s = 0;
      let w = 0;
      for (let k = 0; k < 8; k++) {
        const g = await t.debugGeoLoss(k, 1, 0);
        s += g.sums[0];
        w += g.sums[3];
      }
      return s / w;
    };
    const pin = { width: W, height: H, f: 70, cx: 31.5, cy: 23.5 };
    // same run with the losses switched off (weights 0: the kernel writes zero gradients)
    const run = async (weight: number): Promise<{ e0: number; e1: number; loss: number }> => {
      const t = await trainer(init, views, { ...opts, geometry: { depth: [weight, weight / 10], normal: [0, 0], factor: 1, until: 1 } });
      maps.forEach((mp, k) => t.setViewDepth(k, packDepthTarget(mp, pin, pin, { factor: 1 })));
      const e0 = await depthErr(t);
      const r = await t.step(400);
      return { e0, e1: await depthErr(t), loss: r.loss };
    };
    const off = await run(0);
    const on = await run(1);
    console.log(`[splat] depth supervision: mean |D·z_t − 1| ${on.e0.toFixed(4)} → ${on.e1.toFixed(4)} (without: → ${off.e1.toFixed(4)}), photometric loss ${on.loss.toFixed(4)} (without ${off.loss.toFixed(4)})`);
    expect(on.e1).toBeLessThan(0.75 * off.e1);
    expect(on.e1).toBeLessThan(0.75 * on.e0);
    expect(on.loss).toBeLessThan(0.1);
  });

  for (const normal of [[0.1, 0.01], [0, 0]] as [number, number][]) it(`geometry losses that end early (\`until\`) switch to the plain kernels of the same layout${normal[0] ? '' : ' (depth only)'}`, async () => {
    const gt = randomScene({ count: 150, seed: 171, opacity: [0.5, 0.9] });
    const W = 48;
    const H = 40;
    const views: SplatTrainingView[] = [];
    for (let k = 0; k < 4; k++) {
      const { R, t } = lookAt([2.4 * Math.sin(k), -2.4 * Math.cos(k), 0.5], [0, 0, 0]);
      const cam: SplatCamera = { width: W, height: H, fx: 50, fy: 50, cx: 23.5, cy: 19.5, R, t };
      const r = renderCPU(gt, cam, { geometry: true });
      const img = new Uint8ClampedArray(4 * W * H);
      const map: DepthMapLike = { width: W, height: H, depth: new Float32Array(W * H), normal: new Float32Array(3 * W * H), cost: new Float32Array(W * H) };
      for (let i = 0; i < W * H; i++) {
        for (let c = 0; c < 3; c++) img[4 * i + c] = Math.round(255 * r.color[3 * i + c]);
        img[4 * i + 3] = 255;
        if (1 - r.T[i] > 0.5) {
          (map.depth as Float32Array)[i] = (1 - r.T[i]) / r.geo![4 * i];
          (map.normal as Float32Array).set([0, 0, -1], 3 * i);
        }
      }
      views.push({ ...viewOf(cam, img, k), map } as SplatTrainingView & { map: DepthMapLike });
    }
    const t = await trainer(gt, views, { iterations: 100, resolution: [], mcmc: { start: 20, stop: 60, refineEvery: 20 }, geometry: { until: 0.5, normal } });
    views.forEach((v, k) => t.setViewDepth(k, packDepthTarget((v as unknown as { map: DepthMapLike }).map, { f: 50, cx: 23.5, cy: 19.5 }, { width: W, height: H, f: 50, cx: 23.5, cy: 19.5 })));
    await t.step(40);
    expect(t.geometryActive).toBe(true);
    const r = await t.step(30);
    expect(t.geometryActive).toBe(false);
    expect(Number.isFinite(r.loss)).toBe(true);
    // the plain forward on the wide layout renders exactly what SplatRenderer renders
    const cam = { width: W, height: H, fx: 50, fy: 50, cx: 23.5, cy: 19.5, ...lookAt([0.3, -2.5, 0.6], [0, 0, 0]) };
    const a = await t.render(cam);
    const rr = await SplatRenderer.create(ctx, { maxWidth: W, maxHeight: H });
    rr.setModel(await t.readModel());
    const b = await rr.render(cam);
    rr.destroy();
    let d = 0;
    for (let i = 0; i < a.data.length; i++) d = Math.max(d, Math.abs(a.data[i] - b.data[i]));
    expect(d).toBeLessThan(1e-6);
    // a rollback before `until` brings the geometry kernels back
    const st = await t.readState();
    t.restore({ ...st, iteration: 10 }, null, await t.readModel());
    await t.step(5);
    expect(t.geometryActive).toBe(true);
  });

  it('flatten and anisotropy regularisers shrink the smallest axis and cap the elongation', async () => {
    const gt = randomScene({ count: 120, seed: 161, opacity: [0.6, 0.9], logScale: [-3, -2] });
    // needles: one long axis
    for (let i = 0; i < gt.count; i++) gt.logScales.set([-1.2, -3.2, -3.0], 3 * i);
    // views of the scene itself (the photometric loss starts at its optimum: the regularisers act alone)
    const views = Array.from({ length: 4 }, (_, k) => {
      const { R, t } = lookAt([2.4 * Math.sin(k), -2.4 * Math.cos(k), 0.5], [0, 0, 0]);
      const cam: SplatCamera = { width: 40, height: 32, fx: 45, fy: 45, cx: 19.5, cy: 15.5, R, t };
      const r = renderCPU(gt, cam);
      const img = new Uint8ClampedArray(4 * 40 * 32);
      for (let i = 0; i < 40 * 32; i++) {
        for (let c = 0; c < 3; c++) img[4 * i + c] = Math.round(255 * r.color[3 * i + c]);
        img[4 * i + 3] = 255;
      }
      return viewOf(cam, img, k);
    });
    const tiny = 1e-12;
    const t = await trainer(gt, views, { iterations: 300, colourCorrection: false, resolution: [], mcmc: false, flatten: 50, anisotropy: { maxRatio: 3, weight: 1 }, lr: { mean: tiny, meanFinal: tiny, quat: tiny, opacity: tiny, sh0: tiny } });
    await t.step(300);
    const m = await t.readModel();
    let ratio = 0;
    let flat = 0;
    for (let i = 0; i < m.count; i++) {
      const ls = Array.from(m.logScales.subarray(3 * i, 3 * i + 3)).sort((a, b) => a - b);
      ratio += Math.exp(ls[2] - ls[1]) / m.count;
      flat += (ls[0] - Math.min(...gt.logScales.subarray(3 * i, 3 * i + 3))) / m.count;
    }
    console.log(`[splat] shape regularisers: mean max/mid ratio ${ratio.toFixed(2)} (start ${Math.exp(-1.2 + 3.0).toFixed(2)}), smallest log-scale change ${flat.toFixed(3)}`);
    expect(ratio).toBeLessThan(4);
    expect(flat).toBeLessThan(-0.2);
  });
});

describe('SplatTrainer optimisation', () => {
  it('recovers a scene from perturbed Gaussians (PSNR rises)', async () => {
    const gt = randomScene({ count: 250, seed: 51, opacity: [0.5, 0.9], logScale: [-3, -2] });
    const W = 64;
    const H = 48;
    const views: SplatTrainingView[] = [];
    const cams: SplatCamera[] = [];
    const rng = new Rng(52);
    for (let k = 0; k < 10; k++) {
      const a = (2 * Math.PI * k) / 10;
      const eye: [number, number, number] = [2.4 * Math.sin(a), -2.4 * Math.cos(a), 0.4 + 0.3 * Math.sin(3 * a)];
      const { R, t } = lookAt(eye, [0, 0, 0]);
      const cam: SplatCamera = { width: W, height: H, fx: 70, fy: 70, cx: 31.5, cy: 23.5, R, t };
      const r = renderCPU(gt, cam, { background: [0, 0, 0] });
      const img = new Uint8ClampedArray(4 * W * H);
      for (let i = 0; i < W * H; i++) {
        for (let c = 0; c < 3; c++) img[4 * i + c] = Math.round(255 * r.color[3 * i + c]);
        img[4 * i + 3] = 255;
      }
      cams.push(cam);
      views.push(viewOf(cam, img, k));
    }
    const init = cloneModel(gt);
    for (let i = 0; i < init.count; i++) {
      for (let c = 0; c < 3; c++) {
        init.means[3 * i + c] += rng.uniform(-0.05, 0.05);
        init.sh[3 * i + c] = 0;
      }
      init.opacities[i] = 0;
    }
    const t = await trainer(init, views, { iterations: 600, colourCorrection: false, resolution: [], mcmc: { start: 200, stop: 400 }, lr: { mean: 1.6e-3, meanFinal: 1.6e-5 } });
    const psnr = async (): Promise<number> => {
      let s = 0;
      for (let k = 0; k < cams.length; k++) {
        const out = await t.render(cams[k]);
        let se = 0;
        for (let i = 0; i < W * H; i++) for (let c = 0; c < 3; c++) se += (out.data[4 * i + c] - views[k].image.data[4 * i + c] / 255) ** 2;
        s += 10 * Math.log10(1 / (se / (3 * W * H)));
      }
      return s / cams.length;
    };
    const p0 = await psnr();
    const r1 = await t.step(600);
    const p1 = await psnr();
    console.log(`[splat] train: PSNR ${p0.toFixed(2)} → ${p1.toFixed(2)} dB, loss ${r1.loss.toFixed(4)}, ${r1.iterationsPerSecond.toFixed(1)} it/s, count ${r1.count}, stats ${JSON.stringify(t.stats().mcmc)}`);
    expect(p1).toBeGreaterThan(p0 + 8);
    expect(p1).toBeGreaterThan(27);
    const model = await t.readModel();
    for (const v of model.means) expect(Number.isFinite(v)).toBe(true);
  });

  it('normaliseAppearance: corrected training renders unchanged, mean colour matrix → identity', async () => {
    const gt = randomScene({ count: 150, seed: 71, opacity: [0.6, 0.9] });
    const W = 48;
    const H = 40;
    const views: SplatTrainingView[] = [];
    const cams: SplatCamera[] = [];
    for (let k = 0; k < 6; k++) {
      const a = (2 * Math.PI * k) / 6;
      const { R, t } = lookAt([2.4 * Math.sin(a), -2.4 * Math.cos(a), 0.5], [0, 0, 0]);
      const cam: SplatCamera = { width: W, height: H, fx: 50, fy: 50, cx: 23.5, cy: 19.5, R, t };
      const r = renderCPU(gt, cam, { background: [0.1, 0.1, 0.1] });
      const gain = 0.8 + 0.08 * k; // exposure drift between views
      const img = new Uint8ClampedArray(4 * W * H);
      for (let i = 0; i < W * H; i++) {
        for (let c = 0; c < 3; c++) img[4 * i + c] = Math.round(255 * Math.min(1, gain * r.color[3 * i + c]));
        img[4 * i + 3] = 255;
      }
      cams.push(cam);
      views.push(viewOf(cam, img, k));
    }
    const t = await trainer(gt, views, { iterations: 200, resolution: [], mcmc: false, colourCorrection: { lr: 0.02, reg: 0 }, background: [0.1, 0.1, 0.1] });
    await t.step(200);
    const corrected = async (v: number): Promise<Float32Array> => {
      const img = await t.render(cams[v]);
      const M = (await t.readColourMatrices())[v];
      const x = new Float32Array(3 * W * H);
      for (let i = 0; i < W * H; i++)
        for (let c = 0; c < 3; c++) x[3 * i + c] = M[4 * c] * img.data[4 * i] + M[4 * c + 1] * img.data[4 * i + 1] + M[4 * c + 2] * img.data[4 * i + 2] + M[4 * c + 3];
      return x;
    };
    const before = [await corrected(0), await corrected(5)];
    const before0 = (await t.readColourMatrices()).map((M) => M[0]);
    const Mb = await t.normaliseAppearance();
    expect(Mb).not.toBeNull();
    const after = [await corrected(0), await corrected(5)];
    let d = 0;
    for (let k = 0; k < 2; k++) for (let i = 0; i < before[k].length; i++) d = Math.max(d, Math.abs(after[k][i] - before[k][i]));
    const mats = await t.readColourMatrices();
    const mean = new Float64Array(12);
    for (const M of mats) for (let k = 0; k < 12; k++) mean[k] += M[k] / mats.length;
    console.log(`[splat] normaliseAppearance: M̄ ${Array.from(Mb!, (v) => v.toFixed(3)).join(' ')}; view gains before ${before0.map((v) => v.toFixed(3)).join(' ')}; max corrected-render change ${d.toExponential(2)}`);
    expect(d).toBeLessThan(2e-3);
    for (let k = 0; k < 12; k++) expect(Math.abs(mean[k] - (k % 5 === 0 ? 1 : 0))).toBeLessThan(1e-4);
    // the learned gains (render → image) follow the exposure drift: darker views get smaller gains
    expect(before0[0]).toBeLessThan(before0[5]);
  });

  it('renders the trained model identically through SplatRenderer', async () => {
    const m = randomScene({ count: 120, seed: 61 });
    const cam = testCamera(80, 60);
    const t = await trainer(m, [viewOf(cam)]);
    const a = await t.render(cam);
    const r = await SplatRenderer.create(ctx, { maxWidth: 80, maxHeight: 60 });
    r.setModel(await t.readModel());
    const b = await r.render(cam);
    r.destroy();
    let d = 0;
    for (let i = 0; i < a.data.length; i++) d = Math.max(d, Math.abs(a.data[i] - b.data[i]));
    expect(d).toBeLessThan(1e-6);
  });
});

describe('SplatTrainer checkpoints and dropped work', () => {
  /** A small orbit of rendered views of a random scene. */
  function orbit(gt: SplatModel, n: number, W: number, H: number): SplatTrainingView[] {
    const views: SplatTrainingView[] = [];
    for (let k = 0; k < n; k++) {
      const a = (2 * Math.PI * k) / n;
      const { R, t } = lookAt([2.4 * Math.sin(a), -2.4 * Math.cos(a), 0.5], [0, 0, 0]);
      const cam: SplatCamera = { width: W, height: H, fx: 50, fy: 50, cx: (W - 1) / 2, cy: (H - 1) / 2, R, t };
      const r = renderCPU(gt, cam, { background: [0, 0, 0] });
      const img = new Uint8ClampedArray(4 * W * H);
      for (let i = 0; i < W * H; i++) {
        for (let c = 0; c < 3; c++) img[4 * i + c] = Math.round(255 * Math.min(1, (0.9 + 0.04 * k) * r.color[3 * i + c]));
        img[4 * i + 3] = 255;
      }
      views.push(viewOf(cam, img, k));
    }
    return views;
  }

  const fieldsEqual = (a: SplatModel, b: SplatModel): boolean =>
    a.count === b.count && (['means', 'logScales', 'quats', 'opacities', 'sh'] as const).every((k) => a[k].length === b[k].length && a[k].every((v, i) => v === b[k][i]));

  it('a new trainer restored from readState / readModel / readMoments continues the run', async () => {
    const gt = randomScene({ count: 160, seed: 81, opacity: [0.5, 0.9] });
    const views = orbit(gt, 6, 48, 40);
    const init = cloneModel(gt);
    for (let i = 0; i < init.count; i++) init.opacities[i] = 0;
    const opts = { iterations: 300, maxGaussians: 400, resolution: [], mcmc: { start: 50, stop: 250, refineEvery: 50 } };
    const a = await trainer(init, views, opts);
    // views uploaded one by one (the engine path): same as passing the images to init
    await a.step(120);
    const params = await a.readModel();
    const m = await a.readMoments('m');
    const v = await a.readMoments('v');
    const state = await a.readState();
    expect(state.iteration).toBe(120);
    expect(state.count).toBeGreaterThan(160); // MCMC grew it
    const b = await trainer(params, views.map((x) => ({ ...x, image: null })) as never, opts);
    views.forEach((x, i) => b.setViewImage(i, x.image));
    b.restore(state, { m, v });
    expect(fieldsEqual(await b.readModel(), params)).toBe(true);
    expect(fieldsEqual(await b.readMoments('m'), m)).toBe(true);
    expect(fieldsEqual(await b.readMoments('v'), v)).toBe(true);
    const sb = await b.readState();
    expect(sb.iteration).toBe(120);
    expect(Array.from(sb.colour)).toEqual(Array.from(state.colour));
    const ra = await a.step(60);
    const rb = await b.step(60);
    console.log(`[splat] restored run: loss ${ra.loss.toFixed(4)} (original) vs ${rb.loss.toFixed(4)} (restored), count ${ra.count} / ${rb.count}`);
    expect(rb.iteration).toBe(180);
    expect(Number.isFinite(rb.loss)).toBe(true);
    expect(rb.loss).toBeLessThan(1.5 * ra.loss + 0.01);
  });

  it('step throws SplatWorkDroppedError for dropped iterations and relocations', async () => {
    const gt = randomScene({ count: 120, seed: 91 });
    const views = orbit(gt, 4, 40, 32);
    const t = await trainer(gt, views, { iterations: 400, maxGaussians: 300, resolution: [], mcmc: { start: 100, stop: 300, refineEvery: 100 } });
    await t.step(20);
    t.simulateDroppedWork({ iterations: 3 });
    const e1 = await t.step(20).catch((e: unknown) => e);
    expect(e1).toBeInstanceOf(SplatWorkDroppedError);
    expect((e1 as SplatWorkDroppedError).iterations).toBe(3);
    expect((e1 as SplatWorkDroppedError).relocation).toBe(false);
    await t.step(30); // iteration 70: stamps fine again
    t.simulateDroppedWork({ relocations: 1 });
    const e2 = await t.step(40).catch((e: unknown) => e); // relocation at iteration 100
    expect(e2).toBeInstanceOf(SplatWorkDroppedError);
    expect((e2 as SplatWorkDroppedError).relocation).toBe(true);
    expect((await t.step(10)).iteration).toBe(120);
  });

  it('a dropped readback throws SplatWorkDroppedError instead of returning stale staging data (review)', async () => {
    const gt = randomScene({ count: 100, seed: 97 });
    const views = orbit(gt, 3, 32, 24);
    const t = await trainer(gt, views, { iterations: 200, maxGaussians: 200, resolution: [], mcmc: false });
    await t.step(10);
    const model = await t.readModel();
    const state = await t.readState();
    // the copy of the means is dropped: the staging buffer still holds the last section read (SH)
    t.simulateDroppedWork({ readbacks: 1 });
    expect(await t.readModel().catch((e: unknown) => e)).toBeInstanceOf(SplatWorkDroppedError);
    expect(fieldsEqual(await t.readModel(), model)).toBe(true);
    t.simulateDroppedWork({ readbacks: 1 });
    expect(await t.readState().catch((e: unknown) => e)).toBeInstanceOf(SplatWorkDroppedError);
    expect(Array.from((await t.readState()).colour)).toEqual(Array.from(state.colour));
    // normaliseAppearance: a dropped colour or SH read changes nothing; called again it works
    for (const drop of [1, 2]) {
      t.simulateDroppedWork({ readbacks: drop });
      if (drop === 2) await t.normaliseAppearance().catch(() => undefined); // drops the colour read
      expect(await t.normaliseAppearance().catch((e: unknown) => e)).toBeInstanceOf(SplatWorkDroppedError);
      expect(fieldsEqual(await t.readModel(), model)).toBe(true);
      expect(t.backgroundColour).toEqual([0, 0, 0]);
    }
    expect(await t.normaliseAppearance()).not.toBeNull();
  });
});

