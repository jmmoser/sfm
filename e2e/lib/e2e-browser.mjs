// Vite dev server + headless Chromium (SwiftShader WebGPU) for the engine harness page
// (harness.html, src/harness/engine/main.ts). Used by e2e/run-e2e.mjs.
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { launchOptions } from './launch.mjs';
import { appendFile, mkdir, writeFile } from 'node:fs/promises';
import { createServer as createNetServer } from 'node:net';
import { dirname, isAbsolute, join, resolve, sep } from 'node:path';

export const CHROMIUM_ARGS = [
  '--enable-unsafe-webgpu',
  '--use-webgpu-adapter=swiftshader',
  '--enable-features=Vulkan',
  '--use-vulkan=swiftshader',
  '--disable-vulkan-surface',
  '--use-angle=swiftshader',
  '--enable-unsafe-swiftshader',
  '--ignore-gpu-blocklist',
];

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

/** Resolve `rel` inside `root`; throws for absolute paths and `..` escapes. */
export function safeJoin(root, rel) {
  if (typeof rel !== 'string' || !rel || isAbsolute(rel) || rel.split(/[\\/]/).includes('..')) throw new Error(`bad path ${rel}`);
  const file = resolve(root, rel);
  if (!file.startsWith(resolve(root) + sep)) throw new Error(`bad path ${rel}`);
  return file;
}

/**
 * Start Vite (root = repo, free port) and Chromium. Returns
 * { base, openPage({ outDir, onLog, onConsole }), close() }.
 */
export async function startHarnessBrowser({ root, verbose = false, headed = false }) {
  // vite.config.ts uses __dirname (configs are read-only here): silence the loader notice.
  process.env.VITE_CONFIG_NATIVE_IGNORE_WARNING ??= 'true';
  const port = await freePort();
  const server = await createServer({
    root,
    configFile: join(root, 'vite.config.ts'),
    logLevel: verbose ? 'info' : 'warn',
    clearScreen: false,
    server: { port, strictPort: true, host: '127.0.0.1', hmr: false, open: false },
    // Pre-scan the harness page AND the engine worker so a dependency discovered late does not
    // trigger Vite's optimiser reload in the middle of a run.
    optimizeDeps: { entries: ['harness.html', 'src/workers/engine.worker.ts'] },
  });
  await server.listen();
  const base = `http://127.0.0.1:${port}`;
  const browser = await chromium.launch(launchOptions(CHROMIUM_ARGS, { headless: !headed }));

  /**
   * New page on harness.html with `__e2eLog` / `__e2eWrite` installed.
   * onLog(msg): harness progress lines; onConsole(type, text): page console.
   */
  async function openPage({ outDir, onLog = (m) => console.log(m), onConsole }) {
    const context = await browser.newContext({ viewport: { width: 1100, height: 800 } });
    const page = await context.newPage();
    page.setDefaultTimeout(0);
    const state = { crashed: false, navigations: 0 };
    page.on('console', (m) => {
      const type = m.type();
      if (onConsole) onConsole(type, m.text());
      else if (type === 'error' || type === 'warning' || verbose) console.log(`[page:${type}] ${m.text()}`);
    });
    page.on('pageerror', (e) => console.error(`[page:exception] ${e.message}`));
    page.on('crash', () => {
      state.crashed = true;
      console.error('[page] CRASHED (renderer process died — out of memory?)');
    });
    page.on('framenavigated', (f) => {
      if (f === page.mainFrame()) state.navigations++;
    });
    await page.exposeFunction('__e2eLog', (msg) => onLog(String(msg)));
    await page.exposeFunction('__e2eWrite', async (path, data, encoding, append) => {
      const file = safeJoin(outDir, path);
      await mkdir(dirname(file), { recursive: true });
      const buf = Buffer.from(data, encoding === 'base64' ? 'base64' : 'utf8');
      if (append) await appendFile(file, buf);
      else await writeFile(file, buf);
    });
    await page.goto(`${base}/harness.html`);
    await page.waitForFunction(() => window.harnessReady === true, null, { timeout: 180_000 });
    state.navigations = 0;
    return { page, state, close: () => context.close().catch(() => {}) };
  }

  const close = async () => {
    await browser.close().catch(() => {});
    await server.close().catch(() => {});
  };
  return { base, openPage, close };
}
