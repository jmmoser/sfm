#!/usr/bin/env node
/**
 * Gaussian-splat training with ground-truth poses (headless), src/splat without the SfM engine.
 *
 *   node e2e/run-splat-gt.mjs [<dataset> ...] [--max-size N] [--iterations N] [--max-gaussians N]
 *        [--init cloud|gt|random] [--cloud PATH] [--init-count N] [--sparse N] [--shell N] [--shell-radius R]
 *        [--train-stride N] [--train-frames i,j,...] [--test-every N] [--eval-at i,j,...]
 *        [--sh 0|1] [--no-colour-correction] [--no-mcmc] [--profile] [--panels N]
 *        [--tag NAME] [--out DIR] [--set key=json ...] [--verbose]
 *
 * Starts Vite + headless Chromium (SwiftShader; E2E_GPU=host: the installed Chrome on the real
 * GPU) on splat-gt.html and calls window.runSplatGT(config) per dataset
 * (src/harness/splat-gt/main.ts). --cloud takes a PLY path relative to the repository root,
 * e.g. the fused cloud of a dense-gt run in the GT frame (e2e/out/dense-gt/<name>.ply).
 * `--set` sets any SplatGTConfig field by dotted path, e.g.
 * `--set trainer.lr.mean=3.2e-4 --set 'trainer.resolution=[]'`.
 * Writes to <out> (default e2e/out/splat-gt/):
 *   <name>.json         config, frames, PSNR / SSIM per evaluation point (raw + colour-fitted,
 *                       per view), training speed, GPU ms per pass (--profile), memory plan
 *   <name>.splat.ply    trained model (INRIA 3DGS layout)
 *   <name>/test-*.png   GT | render | |error| panels of a few held-out views
 */
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { launchOptions } from './lib/launch.mjs';
import { mkdir, writeFile, appendFile } from 'node:fs/promises';
import { createServer as createNetServer } from 'node:net';
import { dirname, join, resolve, isAbsolute, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CHROMIUM_ARGS } from './run-dense-gt.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');

const USAGE = `usage: node e2e/run-splat-gt.mjs [<dataset> ...] [--max-size N] [--iterations N] [--max-gaussians N]
       [--init cloud|gt|random] [--cloud PATH] [--init-count N] [--sparse N] [--shell N] [--shell-radius R]
       [--train-stride N] [--train-frames i,j,...] [--test-every N] [--eval-at i,j,...] [--sh 0|1]
       [--no-colour-correction] [--no-mcmc] [--profile] [--panels N] [--tag NAME] [--out DIR] [--set key=json ...] [--verbose]`;

export function parseArgs(argv) {
  const o = { scenes: [], cfg: { init: { kind: 'gt' } }, verbose: false, out: null, tag: null };
  const setPath = (obj, path, val) => {
    const ks = path.split('.');
    let cur = obj;
    for (const k of ks.slice(0, -1)) cur = cur[k] = cur[k] && typeof cur[k] === 'object' ? cur[k] : {};
    cur[ks[ks.length - 1]] = val;
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${a} needs a value`);
      return argv[++i];
    };
    const num = (min = 0) => {
      const v = Number(next());
      if (!Number.isFinite(v) || v < min) throw new Error(`${a} needs a number ≥ ${min}`);
      return v;
    };
    const list = () => {
      const v = next();
      const parts = v.split(',').map((s) => s.trim());
      const ids = parts.map(Number);
      // (Number('') is 0: an empty item, e.g. a trailing comma, would add frame / iteration 0)
      if (parts.some((s) => s === '') || !ids.every((x) => Number.isInteger(x) && x >= 0)) throw new Error(`${a} needs comma-separated integers ≥ 0, got ${v}`);
      if (new Set(ids).size !== ids.length) throw new Error(`${a} has duplicate values, got ${v}`);
      return ids;
    };
    if (a === '--max-size') o.cfg.maxSize = num(16);
    else if (a === '--iterations') o.cfg.iterations = num(1);
    else if (a === '--max-gaussians') o.cfg.maxGaussians = num(16);
    else if (a === '--init') {
      o.cfg.init.kind = next();
      if (!['cloud', 'gt', 'random'].includes(o.cfg.init.kind)) throw new Error('--init must be cloud | gt | random');
    } else if (a === '--cloud') {
      const p = next();
      const abs = resolve(ROOT, p);
      if (!abs.startsWith(ROOT + sep)) throw new Error('--cloud must be inside the repository');
      o.cfg.init.kind = 'cloud';
      o.cfg.init.url = '/' + relative(ROOT, abs).split(sep).join('/');
    } else if (a === '--init-count') o.cfg.init.count = num(1);
    else if (a === '--sparse') o.cfg.init.sparse = num(0);
    else if (a === '--shell') setPath(o.cfg, 'shell.count', num(0));
    else if (a === '--shell-radius') setPath(o.cfg, 'shell.radius', num(0));
    else if (a === '--train-stride') o.cfg.trainStride = num(1);
    else if (a === '--train-frames') o.cfg.trainFrames = list();
    else if (a === '--test-every') o.cfg.testEvery = num(1);
    else if (a === '--eval-at') o.cfg.evalAt = list();
    else if (a === '--sh') setPath(o.cfg, 'trainer.shDegree', num(0) >= 1 ? 1 : 0);
    else if (a === '--no-colour-correction') setPath(o.cfg, 'trainer.colourCorrection', false);
    else if (a === '--no-mcmc') setPath(o.cfg, 'trainer.mcmc', false);
    else if (a === '--profile') setPath(o.cfg, 'trainer.profile', true);
    else if (a === '--panels') o.cfg.panels = num(0);
    else if (a === '--tag') o.tag = next();
    else if (a === '--out') o.out = next();
    else if (a === '--verbose') o.verbose = true;
    else if (a === '--set') {
      const kv = next();
      const eq = kv.indexOf('=');
      if (eq <= 0) throw new Error(`--set needs key=json, got ${kv}`);
      let val;
      try {
        val = JSON.parse(kv.slice(eq + 1));
      } catch {
        throw new Error(`--set ${kv}: value is not JSON`);
      }
      setPath(o.cfg, kv.slice(0, eq), val);
    } else if (a === '--help' || a === '-h') o.help = true;
    else if (a.startsWith('--')) throw new Error(`unknown option ${a}`);
    else o.scenes.push(a);
  }
  if (o.scenes.length === 0) o.scenes = ['object-real'];
  if (o.tag !== null && !/^[\w.-]+$/.test(o.tag)) throw new Error('--tag must be a plain name');
  for (const s of o.scenes) if (!/^[\w.-]+$/.test(s)) throw new Error(`bad dataset name ${s}`);
  if (o.cfg.shell && !o.cfg.shell.radius) throw new Error('--shell needs --shell-radius');
  return o;
}

function freePort() {
  return new Promise((res, rej) => {
    const s = createNetServer();
    s.unref();
    s.on('error', rej);
    s.listen(0, '127.0.0.1', () => {
      const { port } = s.address();
      s.close(() => res(port));
    });
  });
}

async function openPage({ verbose, outDir }) {
  const port = await freePort();
  const server = await createServer({
    root: ROOT,
    configFile: join(ROOT, 'vite.config.ts'),
    logLevel: verbose ? 'info' : 'warn',
    clearScreen: false,
    server: { port, strictPort: true, host: '127.0.0.1', hmr: false, open: false },
    optimizeDeps: { entries: ['splat-gt.html'] },
  });
  await server.listen();
  const browser = await chromium.launch(launchOptions(CHROMIUM_ARGS));
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.setDefaultTimeout(0);
  page.on('console', (m) => {
    if (m.type() === 'error' || m.type() === 'warning' || verbose) console.log(`[page:${m.type()}] ${m.text()}`);
  });
  page.on('pageerror', (e) => console.error(`[page:exception] ${e.message}`));
  await page.exposeFunction('__splatWrite', async (path, data, encoding, append) => {
    if (typeof path !== 'string' || isAbsolute(path) || path.split(/[\\/]/).includes('..')) throw new Error(`bad path ${path}`);
    const file = join(outDir, path);
    if (!file.startsWith(outDir + sep)) throw new Error(`bad path ${path}`);
    await mkdir(dirname(file), { recursive: true });
    const buf = Buffer.from(data, encoding === 'base64' ? 'base64' : 'utf8');
    if (append) await appendFile(file, buf);
    else await writeFile(file, buf);
  });
  await page.exposeFunction('__splatLog', (msg) => console.log(msg));
  await page.goto(`http://127.0.0.1:${port}/splat-gt.html`);
  await page.waitForFunction(() => window.splatGTReady === true, null, { timeout: 180_000 });
  return {
    page,
    close: async () => {
      await browser.close().catch(() => {});
      await server.close().catch(() => {});
    },
  };
}

function fmt(e) {
  return `it ${String(e.iteration).padStart(5)}  PSNR ${e.psnr.toFixed(2).padStart(6)}  SSIM ${e.ssim.toFixed(4)} (no M̄ ${e.psnrRaw.toFixed(2).padStart(6)})  | colour-fit PSNR ${e.psnrCC.toFixed(2).padStart(6)}  SSIM ${e.ssimCC.toFixed(4)}  | ${String(e.count).padStart(7)} Gaussians  train ${(e.trainMs / 1000).toFixed(1).padStart(6)} s`;
}

async function main() {
  let opts;
  try {
    opts = parseArgs(process.argv.slice(2));
  } catch (e) {
    console.error(`${e.message}\n${USAGE}`);
    process.exit(2);
  }
  if (opts.help) {
    console.log(USAGE);
    return;
  }
  const outDir = resolve(opts.out ?? join(ROOT, 'e2e', 'out', 'splat-gt'));
  await mkdir(outDir, { recursive: true });
  const { page, close } = await openPage({ verbose: opts.verbose, outDir });
  let failed = false;
  try {
    for (const scene of opts.scenes) {
      const name = opts.tag ? `${scene}-${opts.tag}` : scene;
      const cfg = { ...opts.cfg, dataset: scene };
      const t0 = Date.now();
      const r = await page.evaluate((c) => window.runSplatGT(c).then((x) => window.splatGTSummary(x)), cfg);
      const written = await page.evaluate((p) => window.splatGTWrite(p), name);
      await writeFile(join(outDir, `${name}.json`), JSON.stringify({ ...r, wallMs: Date.now() - t0 }, null, 1));
      console.log(`\n${name} (${r.adapter}, ${r.size.width}×${r.size.height}, ${r.trainFrames.length} train / ${r.testFrames.length} test views, init ${r.init.kind} ${r.init.count})`);
      for (const e of r.evals) console.log('  ' + fmt(e));
      console.log(`  training: ${r.train.iterations} iterations in ${(r.train.ms / 1000).toFixed(1)} s = ${r.train.iterationsPerSecond.toFixed(1)} it/s; ${r.train.count} Gaussians; peak intersections ${r.train.maxIntersections}; GPU plan ${r.memory.totalMB.toFixed(0)} MB`);
      if (r.train.gpuMsPerIteration) console.log(`  GPU ms / iteration: ${Object.entries(r.train.gpuMsPerIteration).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(', ')}`);
      console.log(`  wrote ${join(relative(ROOT, outDir), name)}.json + ${written.length} files`);
    }
  } catch (e) {
    console.error(e);
    failed = true;
  } finally {
    await close();
  }
  if (failed) process.exit(1);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
