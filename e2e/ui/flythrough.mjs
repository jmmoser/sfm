#!/usr/bin/env node
/**
 * Fly-through video of a REAL reconstruction in the app's viewer (iPhone size, the UI harness of
 * e2e/ui/lib.mjs). The input is an e2e run directory made with --splat (result.json, sparse.ply,
 * splats.ply; dense.ply / mesh.ply when present). As in splat-shots.mjs, the run is decoded in the
 * page and handed to the MockEngine (MockEngine.injectSession), and the viewer screen opens on it.
 * Then:
 *  1. In the page: the export's own pieces (src/app/video-export.ts) — the camera path of the
 *     sparse model, the splat layer's offscreen frames, WebCodecs H.264 and the MP4 writer. The MP4
 *     is decoded again (VideoDecoder): frame count and size, the PSNR of every 10th decoded frame
 *     against the same pose read back directly (encode loss only), wall time, peak JS heap
 *     (performance.memory) and the splat layer's GPU memory.
 *  2. The UI flow: Export → "Fly-through video (MP4)" → Prepare → Save (a download), saved as
 *     <name>-flythrough-ui.mp4 next to <name>-flythrough.mp4 (from 1) and <name>-flythrough.json.
 *     When ffprobe is installed it counts the frames of both files.
 * Checks (first-slice acceptance): 180–300 frames of the path's size (720 × 1280 for portrait
 * photos), PSNR ≥ 34 dB, and the wall time per frame reported against 30 s / 300 frames (warning
 * only: it is machine-dependent).
 *
 * Needs a hardware WebGPU adapter (the app's viewer refuses SwiftShader) and an H.264 encoder
 * (Google Chrome; Playwright's Chromium has none): run with E2E_GPU=host.
 *   E2E_GPU=host node e2e/ui/flythrough.mjs <runDir> [--out /tmp/sfm-flythrough] [--name object-real] [--verbose]
 * --probe: only the Diagnostics "Test video encoding" probe (any browser, no run directory).
 *   node e2e/ui/flythrough.mjs --probe
 */
import { mkdir, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { basename, join, resolve } from 'node:path';
import { launch } from './lib.mjs';

const args = process.argv.slice(2);
const opt = (k, d) => {
  const i = args.indexOf(`--${k}`);
  return i >= 0 ? args[i + 1] : d;
};
const probeOnly = args.includes('--probe');
// (the run directory: the first argument that is neither a flag nor the value of --out / --name)
const VALUE_OPTIONS = ['--out', '--name'];
const runDir = resolve(args.find((a, i) => !a.startsWith('--') && !(i > 0 && VALUE_OPTIONS.includes(args[i - 1]))) ?? '');
if (!probeOnly && (!existsSync(join(runDir, 'result.json')) || !existsSync(join(runDir, 'splats.ply')))) {
  console.error('usage: E2E_GPU=host node e2e/ui/flythrough.mjs <runDir with result.json + splats.ply> [--out dir] [--name label] | --probe');
  process.exit(2);
}
const outDir = resolve(opt('out', '/tmp/sfm-flythrough'));
const name = opt('name', probeOnly ? 'probe' : basename(runDir));
await mkdir(outDir, { recursive: true });

const { page, close, consoleErrors, base } = await launch({ verbose: args.includes('--verbose') });
const t0 = Date.now();
const step = (m) => console.log(`[flythrough +${((Date.now() - t0) / 1000).toFixed(1)}s] ${m}`);
const byTest = (id) => page.locator(`[data-testid="${id}"]`);
const failures = [];

function ffprobeFrames(file) {
  try {
    const out = execFileSync('ffprobe', ['-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames,width,height,codec_name,profile', '-of', 'json', file]).toString();
    return JSON.parse(out).streams?.[0] ?? null;
  } catch {
    return null;
  }
}

try {
  if (probeOnly) {
    await page.goto(`${base}?mock=1&screen=diagnostics`);
    await byTest('diag-video-probe').waitFor({ timeout: 60_000 });
    step('Diagnostics: Test video encoding');
    await byTest('diag-video-probe').tap();
    await page.waitForFunction(() => document.querySelector('[data-testid="diag-video-probe"]')?.textContent?.includes('Test video encoding'), null, { timeout: 120_000 });
    const text = await page.evaluate(() => document.querySelector('[data-testid="diag-video-probe"]')?.closest('.section')?.textContent ?? '');
    step(`probe section: ${text.replace(/\s+/g, ' ').trim()}`);
    const rep = await page.evaluate(async () => (await import('/src/app/video-export.ts')).probeVideoExport(30));
    step(`probe: ${JSON.stringify(rep)}`);
    if (!rep.framePath) failures.push('no frame path');
    if (rep.codecPortrait && (rep.error || rep.frames !== 30)) failures.push(`encode probe failed: ${rep.error}`);
    await writeFile(join(outDir, `${name}-video-probe.json`), JSON.stringify({ rep, section: text, gpu: process.env.E2E_GPU ?? 'swiftshader' }, null, 1));
  } else {
    await page.route('**/__run/*', (route) => {
      const file = join(runDir, basename(new URL(route.request().url()).pathname));
      if (!existsSync(file)) return route.fulfill({ status: 404, body: 'not found' });
      return route.fulfill({ path: file, contentType: 'application/octet-stream' });
    });
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
      const mode = res.dataset?.scene === 'scene' || res.dataset?.scene === 'yard' ? 'scene' : 'object';
      const id = 'real-run';
      const store = await SessionStore.open();
      await store.createSession({ id, name: label, mode, status: mesh ? 'meshed' : 'dense', stats: { quality: 'fast', keyframes: cameras.length, splatGaussians: splat.count } });
      globalThis.__sfmApp.engine.injectSession(id, { name: label, mode, sparse, dense, mesh, splat });
      globalThis.__sfmApp.navigate({ type: 'OPEN_SESSION', sessionId: id, status: mesh ? 'meshed' : 'dense', numFrames: cameras.length });
      return { cameras: cameras.length, gaussians: splat.count, mode };
    }, name);
    step(`session: ${JSON.stringify(info)}`);
    await page.waitForSelector('.viewer-screen', { timeout: 60_000 });
    await page.waitForFunction(() => document.querySelector('.viewer-screen')?.dataset.splats === 'ready', null, { timeout: 120_000 });
    step('splats ready');

    step('export through the API (path → frames → H.264 → MP4 → decode)');
    const api = await page.evaluate(async () => {
      const ve = await import('/src/app/video-export.ts');
      const { flythroughCamera } = await import('/src/viewer/flythrough.ts');
      const app = globalThis.__sfmApp;
      const v = app.debug.viewer;
      const path = ve.flythroughForModel(app.results.sparse, app.results.mode);
      if (!path) return { error: 'no camera path' };
      const config = await ve.chooseVideoConfig(path.width, path.height, path.fps);
      if (!config) return { error: 'no H.264 encoder in this browser (run with E2E_GPU=host: Google Chrome)' };
      const frames = v.openSplatFrames({ width: path.width, height: path.height });
      let peak = 0;
      const mem = () => {
        const m = performance.memory?.usedJSHeapSize;
        if (m) peak = Math.max(peak, m);
      };
      const timer = setInterval(mem, 100);
      const t0 = performance.now();
      try {
        const r = await ve.renderFlythroughVideo({ frames, path, config, onProgress: mem });
        const wallMs = performance.now() - t0;
        const bytes = new Uint8Array(await r.blob.arrayBuffer());
        // decoded frames (every 10th kept as RGBA, the frame closed at once)
        const kept = new Map();
        const c2 = new OffscreenCanvas(path.width, path.height).getContext('2d', { willReadFrequently: true });
        const decoded = await ve.decodeCount(bytes, (f, i) => {
          if (i % 10 !== 0) return;
          c2.drawImage(f, 0, 0, path.width, path.height);
          kept.set(i, c2.getImageData(0, 0, path.width, path.height).data);
        });
        const psnrs = [];
        for (const [i, seen] of kept) {
          const px = await frames.pixels(flythroughCamera(path, i));
          let se = 0;
          for (let k = 0; k < seen.length; k += 4) for (let c = 0; c < 3; c++) se += (seen[k + c] - px.data[k + c]) ** 2;
          const mse = se / ((3 * seen.length) / 4);
          psnrs.push(mse > 0 ? 10 * Math.log10((255 * 255) / mse) : 99);
        }
        let b64 = '';
        for (let k = 0; k < bytes.length; k += 0x8000) b64 += String.fromCharCode(...bytes.subarray(k, k + 0x8000));
        const st = v.getStats().splatLayer;
        return {
          mode: path.mode,
          coverage: path.coverage,
          depth: path.depth,
          frames: r.frames,
          width: r.width,
          height: r.height,
          codec: r.codec,
          framePath: r.framePath,
          fallbackReason: frames.fallbackReason,
          wallMs,
          msPerFrame: r.msPerFrame,
          renderMsPerFrame: r.renderMsPerFrame,
          bytes: bytes.length,
          decoded,
          psnr: psnrs,
          peakHeapMB: peak ? Math.round(peak / 1e5) / 10 : null,
          splatGpuMB: st?.gpuMB ?? null,
          mp4: btoa(b64),
        };
      } finally {
        clearInterval(timer);
        frames.close();
      }
    });
    if (api.error) throw new Error(api.error);
    const apiFile = join(outDir, `${name}-flythrough.mp4`);
    await writeFile(apiFile, Buffer.from(api.mp4, 'base64'));
    delete api.mp4;
    step(`API export: ${JSON.stringify(api)}`);
    if (!(api.frames >= 180 && api.frames <= 300)) failures.push(`${api.frames} frames (180–300 expected)`);
    if (api.decoded !== api.frames) failures.push(`decoded ${api.decoded} of ${api.frames} frames`);
    if (Math.min(...api.psnr) < 34) failures.push(`decoded vs direct PSNR ${Math.min(...api.psnr).toFixed(1)} dB < 34`);
    if (api.msPerFrame > 100) console.warn(`[flythrough] ${api.msPerFrame.toFixed(1)} ms per frame: over the 30 s / 300 frames target (machine-dependent)`);

    step('UI: Export → Fly-through video → Prepare → Save');
    await byTest('tool-export').tap();
    await byTest('export-video').waitFor({ timeout: 30_000 });
    await byTest('export-prepare-video').tap();
    await byTest('export-cancel-video').waitFor({ timeout: 30_000 });
    await byTest('export-save-video').waitFor({ timeout: 300_000 });
    const rowText = await byTest('export-video').textContent();
    const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 30_000 }), byTest('export-save-video').tap()]);
    const uiFile = join(outDir, `${name}-flythrough-ui.mp4`);
    await dl.saveAs(uiFile);
    step(`UI export: ${rowText?.replace(/\s+/g, ' ').trim()} → ${uiFile}`);
    const probe = { api: ffprobeFrames(apiFile), ui: ffprobeFrames(uiFile) };
    if (probe.api) step(`ffprobe: ${JSON.stringify(probe)}`);
    await writeFile(join(outDir, `${name}-flythrough.json`), JSON.stringify({ runDir, info, api, ui: { rowText }, ffprobe: probe, gpu: process.env.E2E_GPU ?? 'swiftshader' }, null, 1));
  }
} catch (e) {
  failures.push(String(e?.message ?? e));
  console.error(e);
  await page.screenshot({ path: join(outDir, `${name}-failure.png`) }).catch(() => {});
}
const errs = consoleErrors.filter((e) => !/favicon/i.test(e));
if (errs.length) failures.push(`console errors: ${errs.slice(0, 5).join(' | ')}`);
step(failures.length ? `FAILED: ${failures.join('; ')}` : 'OK');
await close();
process.exit(failures.length ? 1 : 0);
