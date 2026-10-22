// Vite dev server + headless Chromium (SwiftShader WebGPU) for the engine harness page
// (harness.html, src/harness/engine/main.ts). Used by e2e/run-e2e.mjs.
//   startHarnessBrowser   desktop runs: Playwright's exposeFunction bindings (__e2eLog, __e2eWrite)
//   startBridgeServer     device runs: the HTTP bridge plugin (./e2e-bridge.mjs) on a dev server a
//                         phone can reach (0.0.0.0, HTTPS), no browser
//   startBridgeBrowser    `--device local-bridge`: headless Chromium opening a bridge URL like the
//                         phone would (no bindings), to prove the phone path on a desktop
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { launchOptions } from './launch.mjs';
import { safeJoin } from './safe-path.mjs';
import { appendFile, mkdir, writeFile } from 'node:fs/promises';
import { createServer as createNetServer } from 'node:net';
import { dirname, join } from 'node:path';

export { safeJoin };

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

/** Inline Vite config of the harness servers (vite.config.ts + these overrides). */
function harnessServerConfig({ root, verbose, server, plugins = [] }) {
  // vite.config.ts uses __dirname (configs are read-only here): silence the loader notice.
  process.env.VITE_CONFIG_NATIVE_IGNORE_WARNING ??= 'true';
  return {
    root,
    configFile: join(root, 'vite.config.ts'),
    logLevel: verbose ? 'info' : 'warn',
    clearScreen: false,
    plugins,
    server: { strictPort: true, hmr: false, open: false, ...server },
    // Pre-scan the harness page AND the engine worker so a dependency discovered late does not
    // trigger Vite's optimiser reload in the middle of a run.
    optimizeDeps: { entries: ['harness.html', 'src/workers/engine.worker.ts'] },
  };
}

/**
 * Start Vite (root = repo, free port) and Chromium. Returns
 * { base, openPage({ outDir, onLog, onConsole }), close() }.
 */
export async function startHarnessBrowser({ root, verbose = false, headed = false }) {
  const port = await freePort();
  const server = await createServer(harnessServerConfig({ root, verbose, server: { port, host: '127.0.0.1' } }));
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
    const state = watchPage(page, { verbose, onConsole });
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

/** Console / crash / navigation listeners of a harness page. */
function watchPage(page, { verbose, onConsole }) {
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
  return state;
}

/**
 * Start Vite with the HTTP bridge (`bridge` from ./e2e-bridge.mjs createBridge) for device runs.
 *   host          '0.0.0.0' (phone on the LAN) or '127.0.0.1'
 *   port          null = a free port
 *   https         { cert, key } (PEM buffers) or null; HTTPS skips Vite's Host check
 *   allowedHosts  extra Host names over plain HTTP (a tunnel's hostname)
 * The engine modules are transformed up front (server.warmup) so the phone's first load does not
 * wait for them. Returns { port, protocol, close() }.
 */
export async function startBridgeServer({ root, bridge, host = '127.0.0.1', port = null, https = null, allowedHosts = [], verbose = false }) {
  const p = port ?? (await freePort());
  const server = await createServer(
    harnessServerConfig({
      root,
      verbose,
      plugins: [bridge.plugin()],
      server: {
        port: p,
        host,
        ...(https ? { https } : {}),
        ...(allowedHosts.length ? { allowedHosts } : {}),
        // the page forwards its own console through the bridge (Vite 8 enables this for agents)
        forwardConsole: false,
        warmup: { clientFiles: ['./src/harness/engine/main.ts', './src/workers/engine.worker.ts'] },
      },
    }),
  );
  await server.listen();
  return { port: p, protocol: https ? 'https' : 'http', close: () => server.close().catch(() => {}) };
}

/**
 * `--device local-bridge`: a headless Chromium that opens bridge URLs with no Playwright bindings,
 * exactly what the phone does. Returns { open(url, { onConsole, onGone }) → { page, state, close }, close() }.
 * onGone(reason) fires once when the page's renderer crashes or the page closes before close():
 * nothing will post the result then, so the caller settles the run (the bridge would only time out).
 */
export async function startBridgeBrowser({ verbose = false, headed = false }) {
  const browser = await chromium.launch(launchOptions(CHROMIUM_ARGS, { headless: !headed }));
  async function open(url, { onConsole, onGone = () => {} } = {}) {
    // self-signed / mkcert certificates of an HTTPS bridge server
    const context = await browser.newContext({ viewport: { width: 1100, height: 800 }, ignoreHTTPSErrors: true });
    const page = await context.newPage();
    page.setDefaultTimeout(0);
    const state = watchPage(page, { verbose, onConsole });
    let closing = false;
    let gone = false;
    const goneOnce = (reason) => {
      if (closing || gone) return;
      gone = true;
      onGone(reason);
    };
    page.once('crash', () => goneOnce('page crashed (renderer died; out of memory?)'));
    page.once('close', () => goneOnce('the page closed before it sent a result'));
    await page.goto(url);
    return {
      page,
      state,
      close: () => {
        closing = true;
        return context.close().catch(() => {});
      },
    };
  }
  return { open, close: () => browser.close().catch(() => {}) };
}
