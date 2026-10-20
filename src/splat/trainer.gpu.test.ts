import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { getGPU, type GPUContext } from '../gpu/device';
import { pipelineCompileLimits } from '../features/gpu-common';
import { Rng } from '../math/random';
import { backwardCPU, lossCPU, renderCPU } from './cpu-reference';
import { SplatRenderer } from './renderer';
import { cloneModel, lookAt, randomScene, testCamera } from './testutil';
import type { RGBAImage } from '../core/types';
import { SplatTrainer, SplatWorkDroppedError, type SplatTrainingView as TrainerView } from './trainer';

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

