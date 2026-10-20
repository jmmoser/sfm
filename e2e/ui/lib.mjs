/**
 * Shared helpers for the UI end-to-end scripts: Vite dev server on a free port + headless
 * Chromium (SwiftShader WebGL/WebGPU, fake camera) emulating an iPhone (393×852 @3x, touch).
 */
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { launchOptions } from '../lib/launch.mjs';
import { createServer as createNetServer } from 'node:net';
import { mkdir } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
export const OUT = join(ROOT, 'e2e/out/ui');

export const CHROMIUM_ARGS = [
  '--use-fake-ui-for-media-stream',
  '--use-fake-device-for-media-stream',
  '--enable-unsafe-webgpu',
  '--use-webgpu-adapter=swiftshader',
  '--use-angle=swiftshader',
  '--enable-unsafe-swiftshader',
  '--ignore-gpu-blocklist',
  '--enable-features=Vulkan',
  '--use-vulkan=swiftshader',
  '--disable-vulkan-surface',
  '--autoplay-policy=no-user-gesture-required',
];

export const IPHONE = {
  viewport: { width: 393, height: 852 },
  deviceScaleFactor: 3,
  isMobile: true,
  hasTouch: true,
  userAgent:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1',
};

export function freePort() {
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

/**
 * Start Vite + Chromium. `ua: 'iphone'` pretends to be iOS Safari (Apple code paths such as the
 * install hint); the default keeps Chromium's UA so feature detection matches the real engine.
 */
export async function launch({ verbose = false, contextOptions = {}, iphoneUA = false } = {}) {
  await mkdir(OUT, { recursive: true });
  const port = await freePort();
  const server = await createServer({
    root: ROOT,
    configFile: join(ROOT, 'vite.config.ts'),
    logLevel: verbose ? 'info' : 'warn',
    clearScreen: false,
    server: { port, strictPort: true, host: '127.0.0.1', hmr: false, open: false },
    optimizeDeps: { entries: ['index.html'] },
  });
  await server.listen();
  const browser = await chromium.launch(launchOptions(CHROMIUM_ARGS));
  const consoleErrors = [];
  /** A fresh iPhone-sized context + page (isolated storage) on the same browser. */
  const newPage = async ({ iphoneUA: ua = iphoneUA, contextOptions: extra = {}, initScript = null } = {}) => {
    const { userAgent, ...iphone } = IPHONE;
    const context = await browser.newContext({
      ...iphone,
      ...(ua ? { userAgent } : {}),
      permissions: ['camera', 'geolocation'],
      geolocation: { latitude: 47.3769, longitude: 8.5417, accuracy: 6 },
      acceptDownloads: true,
      reducedMotion: 'no-preference',
      colorScheme: 'dark',
      ...contextOptions,
      ...extra,
    });
    if (initScript) await context.addInitScript(initScript);
    const page = await context.newPage();
    page.on('console', (m) => {
      if (m.type() === 'error') consoleErrors.push(m.text());
      if (verbose || m.type() === 'error' || (m.type() === 'warning' && !/GL Driver Message/.test(m.text()))) console.log(`[page:${m.type()}] ${m.text()}`);
    });
    page.on('pageerror', (e) => {
      consoleErrors.push(`pageerror: ${e.message}`);
      console.error(`[page:exception] ${e.message}`);
    });
    return { page, context };
  };
  const { page, context } = await newPage();
  const close = async () => {
    await browser.close().catch(() => {});
    await server.close().catch(() => {});
  };
  return { page, context, browser, newPage, close, consoleErrors, base: `http://127.0.0.1:${port}/` };
}

export async function shot(page, name, opts = {}) {
  const file = join(OUT, `${name}.png`);
  const t0 = Date.now();
  await page.screenshot({ path: file, timeout: 60_000, ...opts });
  console.log(`[ui] screenshot ${file} (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
  return file;
}

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * Runs in the page (`page.evaluate(layoutAudit)`): layout problems of the current screen + open
 * overlays — horizontal overflow, controls outside the viewport, touch targets < 44 px, clipped text.
 */
export function layoutAudit() {
  const problems = [];
  const vw = window.innerWidth;
  const de = document.documentElement;
  if (de.scrollWidth > vw + 1) problems.push(`page scrollWidth ${de.scrollWidth} > ${vw}`);
  const screen = document.querySelector('.screen:not(.screen-exit)');
  if (screen && screen.scrollWidth > screen.clientWidth + 1) problems.push(`screen scrollWidth ${screen.scrollWidth} > ${screen.clientWidth}`);
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return false;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || Number(cs.opacity) === 0) return false;
    return !el.closest('[hidden], [inert], .screen-exit');
  };
  const describe = (el) => `${el.tagName.toLowerCase()}${el.dataset.testid ? `[${el.dataset.testid}]` : ''}.${[...el.classList].join('.')} "${(el.getAttribute('aria-label') || el.textContent || '').trim().slice(0, 30)}"`;
  const roots = [screen, ...document.querySelectorAll('.sheet.show, .dialog.show')].filter(Boolean);
  // A modal makes the screen inert: audit only the overlays then.
  const modal = document.querySelector('.sheet.show, .dialog.show');
  for (const root of modal ? roots.slice(1) : roots) {
    for (const el of root.querySelectorAll('button, [role="button"], a[href], input, [role="switch"]')) {
      if (!visible(el)) continue;
      const target = el.matches('input[type="checkbox"]') ? (el.closest('label') ?? el) : el;
      const r = target.getBoundingClientRect();
      if (r.left < -1 || r.right > vw + 1) problems.push(`outside viewport (x ${Math.round(r.left)}…${Math.round(r.right)}): ${describe(el)}`);
      if (r.width < 43.5 || r.height < 43.5) problems.push(`touch target ${Math.round(r.width)}×${Math.round(r.height)}: ${describe(el)}`);
    }
    for (const el of root.querySelectorAll('*')) {
      if (!visible(el) || el.children.length > 0 || !el.textContent?.trim()) continue;
      const cs = getComputedStyle(el);
      const clips = cs.overflowX === 'hidden' || cs.overflowX === 'clip';
      if (clips && cs.textOverflow !== 'ellipsis' && el.scrollWidth > el.clientWidth + 1) problems.push(`clipped text: ${describe(el)}`);
    }
  }
  return problems;
}
