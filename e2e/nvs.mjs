#!/usr/bin/env node
/**
 * Novel-view evaluation: renders a finished run (e2e/run-e2e.mjs output) at held-out dataset
 * frames — the frames that did NOT become keyframes — and scores the images against the ground
 * truth photos (PSNR / SSIM, full frame and covered pixels, coverage, exposure-compensated).
 *
 *   node e2e/nvs.mjs <dataset> <runDir> [--renderers points,points-edl,mesh-lit,mesh-unlit,splat]
 *        [--views N] [--max-size N|dense|full] [--test-dataset NAME] [--common] [--out DIR] [--save-images]
 *        … (--help for every option)
 *
 * Test cameras (e2e/lib/nvs-views.mjs): nearest registered keyframe's estimated pose ∘ GT relative
 * motion (or the global Sim3), the run's intrinsics. --test-dataset scores the frames of another
 * dataset of the same seeded scene (off-trajectory views) instead, through the global Sim3 by
 * default. Ground truth: the dataset JPEG undistorted with the run's intrinsics to the run's dense
 * view size and rolling-shutter corrected with the run's model, exactly like the engine's dense
 * views. Renderers: the app's Viewer drawing dense.ply / mesh.ply, and the splat module's renderer
 * for splats.ply when it exists (src/harness/nvs). Metrics: e2e/lib/nvs-metrics.mjs (--common adds
 * every renderer's score on the pixels all of them cover).
 * Writes <out>/nvs.json (default <runDir>/nvs/, <runDir>/nvs-<test dataset>/) and, with --save-images,
 * view-<frame>-<renderer>.png panels (GT | render | |diff|·3).
 *
 * Starts Vite (root = repo, free port) and headless Chromium (SwiftShader; E2E_GPU=host: the
 * installed Chrome on the host GPU), opens nvs.html and serves the run and dataset files to it.
 */
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { access, mkdir, readFile, writeFile } from 'node:fs/promises';
import { loadavg } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CHROMIUM_ARGS, freePort, safeJoin } from './lib/e2e-browser.mjs';
import { launchOptions } from './lib/launch.mjs';
import { NVS_USAGE, NvsArgError, parseNvsArgs } from './lib/nvs-args.mjs';
import { aggregateScores } from './lib/nvs-metrics.mjs';
import { denseImageSize, gtRollingShutter, runRollingShutter, sceneMismatch, testViews } from './lib/nvs-views.mjs';
import { table } from './lib/e2e-checks.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');

async function exists(p) {
  try {
    await access(p);
    return true;
  } catch {
    return false;
  }
}

const readJSON = async (p) => JSON.parse(await readFile(p, 'utf8'));
const f2 = (v, d = 2) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : '—');

/** Dataset directory of a name under e2e/data or a path. */
async function datasetDir(name) {
  return name.includes('/') || (await exists(join(name, 'meta.json'))) ? resolve(name) : join(ROOT, 'e2e', 'data', name);
}

async function loadDatasetJSON(dir) {
  if (!(await exists(join(dir, 'meta.json')))) throw new Error(`dataset missing: ${dir}`);
  return {
    meta: await readJSON(join(dir, 'meta.json')),
    frames: await readJSON(join(dir, 'frames.json')),
    imu: (await exists(join(dir, 'imu.json'))) ? await readJSON(join(dir, 'imu.json')) : null,
  };
}

/** Mean / median / max of the test views' distance to the nearest training camera (GT). */
function distanceSummary(views) {
  const stat = (xs) => {
    const v = xs.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
    if (!v.length) return null;
    return { mean: v.reduce((a, b) => a + b, 0) / v.length, median: v[v.length >> 1], max: v[v.length - 1] };
  };
  return { distM: stat(views.map((v) => v.nearest?.distM)), axisDeg: stat(views.map((v) => v.nearest?.axisDeg)) };
}

/** Summary table: one row per renderer, means over the views (+ common-pixel columns with --common). */
export function formatNvsTable(summary) {
  const common = Object.values(summary).some((s) => s.common?.psnr?.n > 0);
  const rows = [['renderer', 'views', 'coverage', 'PSNR', 'SSIM', 'cov PSNR', 'cov SSIM', 'cc PSNR', 'cc SSIM', 'cc cov PSNR', 'shift px', ...(common ? ['common', 'com PSNR', 'com SSIM', 'com cc PSNR'] : [])]];
  for (const [name, s] of Object.entries(summary)) {
    rows.push([
      name,
      s.views,
      f2(s.coverage.mean, 3),
      f2(s.full.psnr.mean),
      f2(s.full.ssim.mean, 3),
      f2(s.covered.psnr.mean),
      f2(s.covered.ssim.mean, 3),
      f2(s.cc.full.psnr.mean),
      f2(s.cc.full.ssim.mean, 3),
      f2(s.cc.covered.psnr.mean),
      f2(s.shiftPx.median, 2),
      ...(common ? [f2(s.common.coverage.mean, 3), f2(s.common.psnr.mean), f2(s.common.ssim.mean, 3), f2(s.common.ccPsnr.mean)] : []),
    ]);
  }
  return table(rows, [0]);
}

/**
 * Evaluate one run. o: parseNvsArgs() output (dataset = name or directory). Returns the nvs.json
 * object; throws on setup errors (missing files, no registered images).
 */
export async function runNvs(o, { log = (m) => console.log(`[nvs] ${m}`) } = {}) {
  const t0 = Date.now();
  const dsDir = await datasetDir(o.dataset);
  const runDir = resolve(o.runDir);
  if (!(await exists(join(runDir, 'result.json')))) throw new Error(`no result.json in ${runDir}`);
  const { meta, frames, imu } = await loadDatasetJSON(dsDir);
  const result = await readJSON(join(runDir, 'result.json'));
  // --test-dataset: the frames (GT images, poses, IMU) come from that dataset
  const testDir = o.testDataset ? await datasetDir(o.testDataset) : null;
  const test = testDir ? await loadDatasetJSON(testDir) : null;
  if (test) {
    const bad = sceneMismatch(meta, test.meta);
    if (bad.length) throw new Error(`test dataset ${test.meta.name} is not the scene of ${meta.name}: ${bad.join('; ')}`);
  }
  const gtDir = testDir ?? dsDir;
  const gtMeta = test?.meta ?? meta;
  const outDir = resolve(o.out ?? join(runDir, test ? `nvs-${test.meta.name}` : 'nvs'));

  const tv = testViews({ meta, frames, imu }, result, { views: o.views, frames: o.frames ?? undefined, pose: o.pose, focal: o.focal, keepBlurred: o.keepBlurred, rsTranslation: o.rsTranslation, test });
  const maxSize = o.maxSize === 'full' ? 0 : o.maxSize === 'dense' ? (denseImageSize(result) ?? 0) : o.maxSize;
  let rs = null;
  if (o.rs === 'auto' || o.rs === 'run') {
    rs = runRollingShutter(result);
    if (!rs && o.rs === 'run') log('WARNING: --rs run, but the run estimated no rolling shutter: no correction');
  } else if (o.rs === 'gt') {
    rs = gtRollingShutter(gtMeta);
    if (!rs) log('WARNING: --rs gt, but the dataset has no rolling shutter: no correction');
  }
  const files = { dense: await exists(join(runDir, 'dense.ply')), mesh: await exists(join(runDir, 'mesh.ply')), splats: await exists(join(runDir, 'splats.ply')) };
  const renderers =
    o.renderers ??
    [...(files.dense ? ['points', 'points-edl'] : []), ...(files.mesh ? ['mesh-lit', 'mesh-unlit'] : []), ...(files.splats ? ['splat'] : [])];
  if (!renderers.length) throw new Error(`nothing to render in ${runDir} (no dense.ply, mesh.ply or splats.ply)`);
  const dist = distanceSummary(tv.views);
  log(
    `${meta.name}: ${tv.views.length} of ${tv.candidates} ${test ? `frames of the test dataset ${test.meta.name}` : 'held-out frames'} (${result.images?.length ?? 0} registered), ` +
      `nearest training camera ${f2(dist.distM?.mean, 3)} m / ${f2(dist.axisDeg?.mean, 1)}° (mean), Sim3 s ${tv.sim3.s.toFixed(4)}, ` +
      `pose ${o.pose}, focal ${o.focal}, max size ${maxSize || 'full'}, RS ${rs ? `${rs.axis} ${rs.readoutMs} ms${o.rsTranslation ? ' + translation' : ' (rotation only)'}` : 'none'}; renderers ${renderers.join(', ')}`,
  );
  if (o.saveImages) await mkdir(outDir, { recursive: true });

  process.env.VITE_CONFIG_NATIVE_IGNORE_WARNING ??= 'true';
  const port = await freePort();
  const server = await createServer({
    root: ROOT,
    configFile: join(ROOT, 'vite.config.ts'),
    logLevel: o.verbose ? 'info' : 'warn',
    clearScreen: false,
    server: { port, strictPort: true, host: '127.0.0.1', hmr: false, open: false },
    optimizeDeps: { entries: ['nvs.html'] },
  });
  await server.listen();
  const base = `http://127.0.0.1:${port}`;
  const browser = await chromium.launch(launchOptions(CHROMIUM_ARGS, { headless: !o.headed }));
  let out;
  try {
    const context = await browser.newContext({ viewport: { width: 1800, height: 1400 }, deviceScaleFactor: 1 });
    const page = await context.newPage();
    page.setDefaultTimeout(0);
    page.on('console', (m) => {
      if (m.type() === 'error' || m.type() === 'warning' || o.verbose) log(`[page:${m.type()}] ${m.text()}`);
    });
    page.on('pageerror', (e) => log(`[page:exception] ${e.message}`));
    // run and dataset files, served from Node (they may live outside the repository)
    const serve = (prefix, dir) =>
      page.route(`${base}${prefix}**`, async (route) => {
        try {
          const rel = decodeURIComponent(new URL(route.request().url()).pathname.slice(prefix.length));
          const file = safeJoin(dir, rel);
          if (!(await exists(file))) return route.fulfill({ status: 404, body: '' });
          return route.fulfill({ path: file, headers: { 'content-type': 'application/octet-stream' } });
        } catch {
          return route.fulfill({ status: 400, body: '' });
        }
      });
    await serve('/__nvs/run/', runDir);
    await serve('/__nvs/data/', gtDir);
    await page.exposeFunction('__nvsLog', (m) => {
      if (o.verbose) log(String(m));
    });
    await page.exposeFunction('__nvsWrite', async (name, data, encoding) => {
      await writeFile(safeJoin(outDir, name), Buffer.from(data, encoding === 'base64' ? 'base64' : 'utf8'));
    });
    await page.goto(`${base}/nvs.html`);
    await page.waitForFunction(() => window.nvsReady === true, null, { timeout: 180_000 });
    const setup = await page.evaluate(
      (c) => window.nvsSetup(c),
      {
        runBase: '/__nvs/run/',
        dataBase: '/__nvs/data/',
        renderers,
        maxSize,
        rs,
        background: o.background,
        saveImages: o.saveImages,
        ssim: o.ssim,
        common: o.common,
        camera: tv.views[0]?.camera ?? { ...result.intrinsics, width: meta.width, height: meta.height },
      },
    );
    for (const [name, r] of Object.entries(setup.renderers)) if (!r.ok) log(`renderer ${name} skipped: ${r.reason}`);
    log(`setup: dense ${setup.points} points, mesh ${setup.triangles} triangles${setup.gl ? `; WebGL ${setup.gl}` : ''}`);
    const views = [];
    for (const [k, v] of tv.views.entries()) {
      const r = await page.evaluate((x) => window.nvsView(x), { index: v.index, file: v.file, pose: v.pose, camera: v.camera, omega: v.omega, nu: v.nu });
      views.push({ index: v.index, nearestKeyframe: v.nearestKeyframe, gap: v.gap, nearest: v.nearest, gt: v.gt, ...r });
      const s = Object.entries(r.scores).map(([n, x]) => `${n} ${f2(x.full.psnr)}/${f2(x.covered.psnr)} dB cov ${f2(x.coverage, 2)}`);
      const where = v.gap !== null ? `gap ${v.gap}` : `nearest keyframe ${v.nearestKeyframe} at ${f2(v.nearest?.distM, 3)} m, ${f2(v.nearest?.axisDeg, 1)}°`;
      log(`view ${k + 1}/${tv.views.length} frame ${v.index} (${where}): ${s.join(', ')}${Object.keys(r.errors).length ? `; errors ${JSON.stringify(r.errors)}` : ''}`);
    }
    await page.evaluate(() => window.nvsTeardown());
    await context.close();
    const summary = {};
    for (const name of renderers) {
      const sc = views.map((v) => v.scores[name]).filter(Boolean);
      if (sc.length) summary[name] = aggregateScores(sc);
    }
    out = {
      tool: 'e2e/nvs.mjs',
      dataset: meta.name,
      datasetDir: dsDir,
      testDataset: test ? { name: test.meta.name, dir: testDir, trajectory: test.meta.config?.trajectory ?? null } : null,
      runDir,
      createdAt: new Date().toISOString(),
      options: { views: o.views, frames: o.frames, testDataset: o.testDataset, renderers, maxSize, pose: o.pose, focal: o.focal, rs: o.rs, rsTranslation: o.rsTranslation, keepBlurred: o.keepBlurred, background: o.background, ssim: o.ssim, common: o.common },
      definitions: {
        psnr: '10·log10(255²/MSE), MSE over RGB of the scored pixels, 8-bit sRGB; mean over views',
        ssim: '11×11 Gaussian (σ 1.5), K1 0.01, K2 0.03, per RGB channel then averaged; windows restricted to the scored mask (normalised convolution)',
        full: 'render composited over the background vs GT on every valid GT pixel (holes score as background)',
        covered: 'un-premultiplied render vs GT on covered pixels (alpha ≥ 0.5) only',
        cc: 'after a per-view least-squares 3×4 affine colour fit render → GT on the covered pixels (applied to covered pixels only)',
        coverage: 'covered valid pixels / valid GT pixels',
        common: 'with --common: every renderer on the valid pixels that ALL renderers cover (un-premultiplied; ccPsnr after a per-renderer affine fit on them)',
        shift: 'residual image shift render → GT (2× downsampled luma, ±6 px search), full-resolution px',
      },
      run: {
        quality: result.config?.quality ?? null,
        registered: result.images?.length ?? null,
        keyframes: result.live?.keyframes ?? null,
        sim3Scale: tv.sim3.s,
        intrinsics: result.intrinsics,
        rollingShutter: rs,
        heldOutCandidates: tv.candidates,
      },
      distance: dist,
      setup,
      summary,
      views,
      host: { gpu: process.env.E2E_GPU === 'host' ? 'host' : 'swiftshader', loadavg: loadavg().map((x) => +x.toFixed(2)) },
      timing: { totalS: +((Date.now() - t0) / 1000).toFixed(1) },
    };
  } finally {
    await browser.close().catch(() => {});
    await server.close().catch(() => {});
  }
  await mkdir(outDir, { recursive: true });
  await writeFile(join(outDir, 'nvs.json'), JSON.stringify(out, null, 1));
  log(
    `${meta.name} novel views (${out.views.length} ${test ? `${test.meta.name} frames` : 'held-out frames'}, nearest training camera ${f2(dist.distM?.mean, 3)} m / ${f2(dist.axisDeg?.mean, 1)}°, ` +
      `${out.timing.totalS} s, load ${out.host.loadavg[0]}):\n${formatNvsTable(out.summary)}`,
  );
  log(`wrote ${join(outDir, 'nvs.json')}${o.saveImages ? ' + PNG panels' : ''}`);
  return out;
}

const isMain = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  let o;
  try {
    o = parseNvsArgs(process.argv.slice(2));
  } catch (e) {
    if (e instanceof NvsArgError) {
      console.error(`error: ${e.message}\n${NVS_USAGE}`);
      process.exit(2);
    }
    throw e;
  }
  if (o.help) {
    console.log(NVS_USAGE);
    process.exit(0);
  }
  try {
    await runNvs(o);
    process.exit(0);
  } catch (e) {
    console.error(`[nvs] FAILED: ${e.stack ?? e.message}`);
    process.exit(1);
  }
}
