#!/usr/bin/env node
/**
 * Viewer screenshots of a REAL reconstruction at iPhone size (393×852 @3x, the UI harness of
 * e2e/ui/lib.mjs): an e2e run directory (e2e/run-e2e.mjs … --splat: result.json, sparse.ply,
 * dense.ply, mesh.ply, splats.ply) is decoded in the page and handed to the MockEngine as the
 * session's models (MockEngine.injectSession), then the app's viewer screen is opened on it and
 * the SAME viewpoint is captured with the layer pill on Splats, Points and Mesh — the full-page
 * PNGs as the user sees them. Viewpoints: 'open' (what the app shows when the scan opens on its
 * splats: from the middle photo, pulled back), 'orbit' (that view orbited by --orbit degrees,
 * default 40, about the vertical through the orbit target: off the photo path) and 'overview'
 * (the whole content from above, the viewer's fit — far outside the capture). Also measures the
 * splat layer's frame time while orbiting (submit → done, full-size frames) and its GPU memory.
 *
 * Needs a hardware WebGPU adapter for the splats: run with E2E_GPU=host.
 *   E2E_GPU=host node e2e/ui/splat-shots.mjs <runDir> [--out /tmp/sfm-viewer-shots] [--name object-real] [--orbit 40]
 */
import { mkdir, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import { launch, sleep } from './lib.mjs';

const args = process.argv.slice(2);
const opt = (k, d) => {
  const i = args.indexOf(`--${k}`);
  return i >= 0 ? args[i + 1] : d;
};
const runDir = resolve(args.find((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--'))) ?? '');
if (!existsSync(join(runDir, 'result.json')) || !existsSync(join(runDir, 'splats.ply'))) {
  console.error(`usage: E2E_GPU=host node e2e/ui/splat-shots.mjs <runDir with result.json + splats.ply> [--out dir] [--name label]`);
  process.exit(2);
}
const outDir = resolve(opt('out', '/tmp/sfm-viewer-shots'));
const name = opt('name', basename(runDir));
const orbitDeg = Number(opt('orbit', '40'));
await mkdir(outDir, { recursive: true });

const { page, close, consoleErrors, base } = await launch({ verbose: args.includes('--verbose') });
const t0 = Date.now();
const step = (m) => console.log(`[shots +${((Date.now() - t0) / 1000).toFixed(1)}s] ${m}`);
const byTest = (id) => page.locator(`[data-testid="${id}"]`);
let failed = false;

// the run's files under /__run/ (streamed from disk)
await page.route('**/__run/*', (route) => {
  const file = join(runDir, basename(new URL(route.request().url()).pathname));
  if (!existsSync(file)) return route.fulfill({ status: 404, body: 'not found' });
  return route.fulfill({ path: file, contentType: 'application/octet-stream' });
});

async function shotTo(file) {
  await page.screenshot({ path: file, timeout: 60_000 });
  console.log(`[shots] ${file}`);
}

try {
  await page.goto(`${base}?mock=1`);
  await byTest('new-scan').waitFor();
  step(`loading ${runDir}`);
  const info = await page.evaluate(async (label) => {
    const { decodePLY } = await import('/src/io/ply.ts');
    const { decodeSplatPLY } = await import('/src/splat/ply.ts');
    const { SessionStore } = await import('/src/io/storage.ts');
    const get = async (f) => {
      const r = await fetch(`/__run/${f}`);
      return r.ok ? new Uint8Array(await r.arrayBuffer()) : null;
    };
    const res = JSON.parse(new TextDecoder().decode(await get('result.json')));
    const sp = res.sparse;
    const intr = { ...sp.intrinsics, k1: 0, k2: 0 };
    const cameras = sp.images.map((im) => ({ id: im.id, pose: { R: Float64Array.from(im.pose.R), t: Float64Array.from(im.pose.t) }, intrinsics: intr }));
    const sparsePly = await get('sparse.ply');
    const points = sparsePly ? decodePLY(sparsePly).points : { count: 0, positions: new Float32Array(0), colors: new Uint8Array(0) };
    const densePly = await get('dense.ply');
    const meshPly = await get('mesh.ply');
    const dense = densePly ? decodePLY(densePly).points : null;
    const mesh = meshPly ? decodePLY(meshPly).mesh : null;
    const splat = decodeSplatPLY(await get('splats.ply'), 1).model;
    const sparse = { points, cameras, metric: res.metric ?? sp.metric, stats: { registeredImages: cameras.length, images: cameras.length } };
    const mode = res.dataset?.scene === 'scene' ? 'scene' : 'object';
    const id = 'real-run';
    const store = await SessionStore.open();
    await store.createSession({ id, name: label, mode, status: mesh ? 'meshed' : 'dense', stats: { quality: 'fast', keyframes: cameras.length, splatGaussians: splat.count } });
    globalThis.__sfmApp.engine.injectSession(id, { name: label, mode, sparse, dense, mesh, splat });
    globalThis.__sfmApp.navigate({ type: 'OPEN_SESSION', sessionId: id, status: mesh ? 'meshed' : 'dense', numFrames: cameras.length });
    return { cameras: cameras.length, dense: dense?.count ?? 0, triangles: mesh?.triangleCount ?? 0, gaussians: splat.count, shDegree: splat.shDegree, intr };
  }, name);
  step(`session: ${JSON.stringify(info)}`);
  await page.waitForSelector('.viewer-screen', { timeout: 60000 });
  await page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 60000 });
  await page.waitForFunction(() => document.querySelector('[data-testid="layer-switch"]')?.dataset.layer === 'splats' && document.querySelector('.viewer-screen')?.dataset.splats === 'ready', null, {
    timeout: 120000,
  });
  step('splats shown');

  const layers = ['splats', 'points', 'mesh'].filter((l) => l !== 'mesh' || info.triangles > 0);
  const capture = async (view) => {
    for (const l of layers) {
      await page.locator(`[data-testid="layer-switch"] .segment[data-value="${l}"]`).tap();
      await page.waitForFunction((v) => document.querySelector('[data-testid="layer-switch"]')?.dataset.layer === v, l);
      await sleep(1200);
      await shotTo(join(outDir, `${name}-${view}-${l}.png`));
    }
    await page.locator('[data-testid="layer-switch"] .segment[data-value="splats"]').tap();
    await sleep(400);
  };

  step('open (the app default view of splats)');
  await sleep(800);
  await capture('open');

  step(`orbit ${orbitDeg}°`);
  await page.evaluate((deg) => {
    const v = globalThis.__sfmApp.debug.viewer;
    const c = v.camera.position;
    const t = v.controls.target;
    const a = (deg * Math.PI) / 180;
    const dx = c.x - t.x;
    const dz = c.z - t.z;
    c.x = t.x + dx * Math.cos(a) - dz * Math.sin(a);
    c.z = t.z + dx * Math.sin(a) + dz * Math.cos(a);
    v.controls.update();
    v.requestRender();
  }, orbitDeg);
  await capture('orbit');

  step('overview (whole content from above)');
  await page.evaluate(() => globalThis.__sfmApp.debug.viewer.fitToContent(false, true));
  await capture('overview');
  await page.evaluate(() => globalThis.__sfmApp.debug.viewer.viewFromCamera());

  step('frame time while orbiting (full-size frames)');
  const perf = await page.evaluate(async () => {
    const v = globalThis.__sfmApp.debug.viewer;
    const t = v.getOrbitTarget();
    const ms = [];
    for (let i = 0; i < 40; i++) {
      const before = v.getStats().splatLayer.frames;
      // a small orbit of the target: a new view each frame
      v.setOrbitTarget([t[0] + 0.002 * Math.cos(i), t[1] + 0.002 * Math.sin(i), t[2]]);
      v.renderNow();
      const tw = performance.now();
      while (v.getStats().splatLayer.frames === before && performance.now() - tw < 2000) await new Promise((r) => setTimeout(r, 1));
      await new Promise((r) => setTimeout(r, 30)); // let the frame complete (stats.fullFrameMs)
      ms.push(v.getStats().splatLayer.fullFrameMs);
    }
    ms.sort((a, b) => a - b);
    const st = v.getStats().splatLayer;
    return { medianMs: ms[ms.length >> 1], p90Ms: ms[Math.floor(ms.length * 0.9)], canvas: `${st.canvasWidth}×${st.canvasHeight}`, gpuMB: st.gpuMB, intersections: st.intersections, isectCapacity: st.isectCapacity, gaussians: st.gaussians };
  });
  step(`splat layer: ${JSON.stringify(perf)}`);
  await writeFile(join(outDir, `${name}-stats.json`), JSON.stringify({ runDir, info, perf, gpu: process.env.E2E_GPU ?? 'swiftshader' }, null, 1));
} catch (e) {
  failed = true;
  console.error(e);
  await page.screenshot({ path: join(outDir, `${name}-failure.png`) }).catch(() => {});
}
const errs = consoleErrors.filter((e) => !/favicon/i.test(e));
if (errs.length) {
  failed = true;
  console.error(`[shots] console errors: ${errs.slice(0, 5).join(' | ')}`);
}
step(failed ? 'FAILED' : 'OK');
await close();
process.exit(failed ? 1 : 0);
