#!/usr/bin/env node
/**
 * GPU back-pressure probe (docs/research/integration-2026-10.md "Unread frame grabs"): a blank page
 * with the fake camera in a <video>, createImageBitmap(video) at 10 Hz like the FramePump, each
 * bitmap closed UNREAD ('grab') or READ first through a 1×1 canvas ('grab+read', what a real
 * consumer does); optionally a WebGL canvas drawn at 2 Hz ('gl', like the capture screen's 3D
 * thumbnail) and a 2D overlay per grab ('2d'). Per mode it prints the grab rate, the GPU process's
 * RSS slope and how long a page screenshot takes right after. Headless Chromium on SwiftShader:
 * unread grabs grow the GPU process without bound and stall compositor readbacks; read grabs pace
 * themselves to the GPU. Why src/app/mock-engine.ts reads every frame ("Frame pixels").
 *
 * Usage: node e2e/ui/gpu-backpressure.mjs [--modes video,grab+read,grab] [--secs 20]   (Linux: reads /proc)
 */
// argv: --modes a,b,... (default all), --secs N per mode (default 20)
import { readFileSync } from 'node:fs';
import { launch, sleep } from './lib.mjs';

const argv = process.argv.slice(2);
const opt = (f, d) => {
  const i = argv.indexOf(f);
  return i >= 0 ? argv[i + 1] : d;
};
const SECS = Number(opt('--secs', '20'));
const MODES = opt('--modes', 'video,grab+read,grab').split(',');
const L = await launch({});
const { page } = L;
const bcdp = await L.browser.newBrowserCDPSession();
const pid = (await bcdp.send('SystemInfo.getProcessInfo')).processInfo.find((p) => p.type === 'GPU')?.id;
const rssMB = () => Number(/VmRSS:\s+(\d+)/.exec(readFileSync(`/proc/${pid}/status`, 'utf8'))?.[1] ?? NaN) / 1024;
await page.route('**/__blank.html', (r) => r.fulfill({ contentType: 'text/html', body: '<!doctype html><html><body style="margin:0;background:#000"></body></html>' }));
await page.goto(`${L.base}__blank.html`);
const info = await page.evaluate(async () => {
  const v = document.createElement('video');
  v.muted = true;
  v.playsInline = true;
  v.autoplay = true;
  v.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;object-fit:cover';
  document.body.append(v);
  v.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1920 }, height: { ideal: 1440 }, frameRate: { ideal: 30 } }, audio: false });
  await v.play();
  await new Promise((r) => setTimeout(r, 500));
  const gl = document.createElement('canvas');
  gl.width = 224;
  gl.height = 296;
  gl.style.cssText = 'position:absolute;right:12px;bottom:148px;width:112px;height:148px';
  document.body.append(gl);
  const ctx = gl.getContext('webgl2');
  const ov = document.createElement('canvas');
  ov.width = 786;
  ov.height = 1704;
  ov.style.cssText = 'position:absolute;inset:0;width:100%;height:100%';
  document.body.append(ov);
  const c2 = ov.getContext('2d');
  let timers = [];
  let busy = false;
  let grabs = 0;
  const rd = new OffscreenCanvas(1, 1).getContext('2d', { willReadFrequently: true });
  globalThis.__grabs = () => grabs;
  globalThis.__mode = (m) => {
    for (const t of timers) clearInterval(t);
    timers = [];
    gl.style.display = m.includes('gl-hidden') ? 'none' : m.includes('gl') ? '' : 'none';
    ov.style.display = m.includes('2d') ? '' : 'none';
    if (m.startsWith('grab'))
      timers.push(
        setInterval(async () => {
          if (m.includes('vf')) {
            const f = new VideoFrame(v);
            f.close();
            return;
          }
          if (busy) return;
          busy = true;
          const b = await createImageBitmap(v);
          if (m.includes('read')) {
            // like a real consumer: the pixels are read (waits for the GPU copy)
            rd.drawImage(b, 0, 0, 1, 1);
            rd.getImageData(0, 0, 1, 1);
          }
          b.close();
          grabs++;
          busy = false;
          if (m.includes('2d')) {
            c2.clearRect(0, 0, ov.width, ov.height);
            c2.beginPath();
            for (let i = 0; i < 260; i++) c2.arc((i * 397 + grabs * 31) % 786, (i * 617 + grabs * 53) % 1704, 4, 0, 7);
            c2.fill();
          }
        }, 100),
      );
    if (m.includes('gl'))
      timers.push(
        setInterval(() => {
          ctx.clearColor((grabs % 10) / 10, 0.2, 0.3, 1);
          ctx.clear(ctx.COLOR_BUFFER_BIT);
        }, 500),
      );
  };
  return { w: v.videoWidth, h: v.videoHeight };
});
console.log(`[gpu-backpressure] video ${info.w}x${info.h}, gpu pid ${pid}`);
for (const m of MODES) {
  await page.evaluate((x) => globalThis.__mode(x), m);
  await sleep(2000);
  const r0 = rssMB();
  const g0 = await page.evaluate(() => globalThis.__grabs());
  const t0 = Date.now();
  await sleep(SECS * 1000);
  const r1 = rssMB();
  const g1 = await page.evaluate(() => globalThis.__grabs());
  const dt = (Date.now() - t0) / 1000;
  await page.evaluate(() => globalThis.__mode('idle'));
  const ts = Date.now();
  let shot;
  try {
    await page.screenshot({ timeout: 240_000 });
    shot = ((Date.now() - ts) / 1000).toFixed(1);
  } catch {
    shot = 'FAILED 240';
  }
  console.log(`[gpu-backpressure] ${m}: ${((g1 - g0) / dt).toFixed(1)} grabs/s, GPU RSS ${r1.toFixed(0)} MB, ${((r1 - r0) / dt).toFixed(1)} MB/s; screenshot right after ${shot} s; load ${readFileSync('/proc/loadavg', 'utf8').split(' ')[0]}`);
  await sleep(3000);
}
await L.close();
