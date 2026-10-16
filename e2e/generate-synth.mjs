#!/usr/bin/env node
/**
 * Synthetic dataset generator (headless).
 *
 *   node e2e/generate-synth.mjs [object|yard|all] [--frames N] [--fps F] [--size WxH]
 *        [--duration S] [--out DIR] [--name NAME] [--seed N] [--blur N] [--no-noise]
 *        [--depth-every N] [--gt-points N] [--gt-focus-points N] [--time-offset MS] [--texture-scale X]
 *        [--visibility-every N] [--supersample N] [--set key=jsonValue ...]
 *
 * Starts a Vite dev server on a free port, opens synth.html in headless Chromium
 * (SwiftShader GL/WebGPU) and calls window.generateSynth(config); every output file is
 * streamed back through page.exposeFunction and written to <out>/<name>/ (default
 * e2e/data/<scene>/). Frame-count semantics: with --frames only, the frames are spread
 * over the scene's default duration (fps = N / duration); with --frames and --fps the
 * duration becomes N / fps; with --fps only, N = fps · duration.
 */
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir, writeFile, appendFile, rm, readdir, stat } from 'node:fs/promises';
import { createServer as createNetServer } from 'node:net';
import { dirname, join, resolve, relative, isAbsolute, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CliError, USAGE, parseArgs, sceneConfig } from './lib/synth-cli.mjs';

export { parseArgs, sceneConfig };

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const CHROMIUM_ARGS = [
  '--use-angle=swiftshader',
  '--enable-unsafe-swiftshader',
  '--ignore-gpu-blocklist',
  '--enable-unsafe-webgpu',
  '--use-webgpu-adapter=swiftshader',
  '--enable-features=Vulkan',
  '--use-vulkan=swiftshader',
  '--disable-vulkan-surface',
];

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

async function dirSize(dir) {
  let total = 0,
    files = 0;
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) {
      const s = await dirSize(p);
      total += s.total;
      files += s.files;
    } else {
      total += (await stat(p)).size;
      files++;
    }
  }
  return { total, files };
}

/** Start Vite + Chromium, returns { page, close }. */
export async function openSynthPage({ verbose = false } = {}) {
  const port = await freePort();
  const server = await createServer({
    root: ROOT,
    configFile: join(ROOT, 'vite.config.ts'),
    logLevel: verbose ? 'info' : 'warn',
    clearScreen: false,
    server: { port, strictPort: true, host: '127.0.0.1', hmr: false, open: false },
    optimizeDeps: { entries: ['synth.html'] },
  });
  await server.listen();
  const browser = await chromium.launch({ headless: true, args: CHROMIUM_ARGS });
  const page = await browser.newPage();
  page.on('console', (m) => {
    if (m.type() === 'error' || m.type() === 'warning' || verbose) console.log(`[page:${m.type()}] ${m.text()}`);
  });
  page.on('pageerror', (e) => console.error(`[page:exception] ${e.message}`));
  const close = async () => {
    await browser.close().catch(() => {});
    await server.close().catch(() => {});
  };
  return { page, close, url: `http://127.0.0.1:${port}/synth.html` };
}

export async function generate(scenes, opts, outRoot) {
  const { page, close, url } = await openSynthPage();
  let target = null;
  const results = [];
  try {
    await page.exposeFunction('__synthWrite', async (path, data, encoding, append) => {
      if (!target) throw new Error('no active dataset');
      if (typeof path !== 'string' || isAbsolute(path) || path.split(/[\\/]/).includes('..')) throw new Error(`bad path ${path}`);
      const file = join(target, path);
      if (!file.startsWith(target + sep)) throw new Error(`bad path ${path}`);
      await mkdir(dirname(file), { recursive: true });
      const buf = Buffer.from(data, encoding === 'base64' ? 'base64' : 'utf8');
      if (append) await appendFile(file, buf);
      else await writeFile(file, buf);
    });
    await page.exposeFunction('__synthLog', (msg) => console.log(msg));
    await page.goto(url);
    await page.waitForFunction(() => window.synthReady === true, null, { timeout: 180_000 });
    for (const scene of scenes) {
      const cfg = sceneConfig(scene, opts, (m) => console.warn(`[synth] warning: ${m}`));
      const name = cfg.name ?? scene;
      target = resolve(outRoot, name);
      // Never rm -rf anything but a direct child directory of the output root.
      if (dirname(target) !== resolve(outRoot) || name === '.' || name === '..') throw new Error(`refusing to write dataset '${name}' outside ${outRoot}`);
      await rm(target, { recursive: true, force: true });
      await mkdir(target, { recursive: true });
      console.log(`[synth] generating '${name}' → ${relative(ROOT, target) || target}  (${cfg.frames} frames @ ${cfg.fps.toFixed(3)} fps over ${cfg.duration.toFixed(1)} s)`);
      const t0 = Date.now();
      const summary = await page.evaluate((c) => window.generateSynth(c), cfg);
      const wall = (Date.now() - t0) / 1000;
      const size = await dirSize(target);
      results.push({ name, dir: target, wallSeconds: wall, bytes: size.total, files: size.files, summary });
      console.log(`[synth] '${name}': ${summary.frames} frames, ${size.files} files, ${(size.total / 1048576).toFixed(1)} MB, ${wall.toFixed(1)} s wall; timings(ms) ${JSON.stringify(Object.fromEntries(Object.entries(summary.timingsMs).map(([k, v]) => [k, Math.round(v)])))}`);
    }
  } finally {
    await close();
  }
  return results;
}

const isMain = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  let opts;
  try {
    opts = parseArgs(process.argv.slice(2));
  } catch (e) {
    if (!(e instanceof CliError)) throw e;
    console.error(`error: ${e.message}
${USAGE}`);
    process.exit(2);
  }
  if (opts.help) {
    console.log(USAGE);
    process.exit(0);
  }
  const outRoot = resolve(opts.out ?? join(ROOT, 'e2e', 'data'));
  generate(opts.scenes, opts, outRoot)
    .then(() => process.exit(0))
    .catch((e) => {
      console.error(e);
      process.exit(1);
    });
}
