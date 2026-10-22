#!/usr/bin/env node
/**
 * UI edge states at iPhone size, each in a fresh context (isolated storage), with screenshots in
 * e2e/out/ui/state-*.png:
 *   engine-missing   real EngineClient + a stub worker that answers NotImplementedError (a build
 *                    without the engine) → clear "engine unavailable" card
 *   install-hint     iOS Safari UA, not standalone → "Install to Home Screen" card
 *   cpu-mode         no WebGPU → "CPU mode — slower" banner
 *   lockdown         no WebGPU, camera, IndexedDB → Lockdown Mode explanation, New scan disabled
 *   camera-denied    getUserMedia NotAllowedError → recovery card
 *   landscape        capture UI in landscape (852×393)
 *   interrupted      page hidden during capture → paused, resume needs a tap
 *   processing-error refine fails → error banner with retry
 *   storage-full-start   IndexedDB quota exceeded → the scan is blocked with an explanation
 *   storage-full-capture keyframes cannot be stored → capture pauses, "Finish scan" works
 *   storage-write-error-capture  a non-quota write failure → capture pauses, not "Storage is full"
 *   engine-dead      EngineClient gave up (respawn cap) → error card, no toast storm
 *   export-reopen    export finishes while its sheet is closed → the reopened sheet shows Save
 *   open-other-session processing a stored scan after viewing another: no stale name / stages /
 *                    preview
 *   focus            keyboard navigation moves focus to the new screen's heading
 *   raw-overlay      a RAW sensor buffer (landscape bitmap in the portrait UI: the headless fake
 *                    camera) gets its keypoint overlay turned onto the upright preview
 *   finish-stuck     finishCapture hangs and ignores cancel → Cancel aborts it (engine restart),
 *                    "Process now" reconstructs the scan
 *   finish-stuck-home  same, "Back to home" keeps the scan (status captured)
 *   finish-stuck-discard  finishCapture hangs, X (keyboard) → Discard: the scan is deleted only after the
 *                    engine let go of it (no re-opening restart), a new scan starts right away
 *   discard-no-replay  real EngineClient + the protocol fixture worker: X → Discard during a capture
 *                    closes the session in the engine; a later engine restart replays nothing
 *   delete-open-session  deleting the scan the engine holds (opened in the viewer) on Home makes the
 *                    engine let go of it
 *   memory-limit     the engine's capture memory cap → "Memory limit reached — finish scan now?",
 *                    Continue (tracking only) adds no keyframes, Finish works
 *   dense-resume     an interrupted densify (loadSession stats.denseResumablePass) → "Resume dense
 *                    reconstruction" in the viewer + Build sheet; a densify cancelled here too (the
 *                    Stop closes its run journal entry: no "run interrupted" card after a reload)
 *   dense-resume-quality  the interrupted run used another preset than the scan (stats
 *                    denseResumableQuality): the resume uses that preset
 *   info-sheet       Model info: magnetic-north note for compass yaw, engine memory section
 *   processing-stop  Stop while the scan is still being opened, or after the running step already
 *                    finished: no further step starts; a load that fails after a Stop leaves at once
 *   open-failed      a scan that cannot be loaded after viewing another: "Back to home", not the
 *                    other scan's "Open model"
 *   viewer-loading   the viewer's tools do nothing while the opened scan loads (no sheet of the
 *                    previously loaded scan)
 *   camera-error-init  the camera fails during the engine init → its error card stays, "Reconnect
 *                    camera" starts the scan
 *   memory-limit-background  the memory cap is reached by the frame in flight when the app goes to
 *                    the background → the prompt comes after Resume
 *   splat-build      "Photoreal model" from the viewer's Build sheet: progress card with the training
 *                    preview (its pixels change; "Use it now" disabled until finishFrom), Cancel →
 *                    "Photoreal model interrupted" card → Resume continues → the preview opens larger
 *                    → "Use it now" stores the model early (no resume card; Build: Continue) →
 *                    Continue trains to the end → layer pill offers Splats, the export sheet the
 *                    Gaussian-splat PLY; after a reload the scan reopens with its splats
 *                    (E2E_GPU=host: shown by default; SwiftShader: refused, points shown)
 *   splat-no-webgpu  no WebGPU in the browser: the photoreal build is disabled with an explanation
 *   coverage         object mode: the coverage ring fills along the mock's level orbit ("Walk to the
 *                    other side" → "Raise the phone, loop looking down" with the elevation bands, or
 *                    "All sides covered" azimuth-only without motion), then its higher loop → "Done";
 *                    the Finish hint shows "Coverage N %"; auto mode shows the ring only once the
 *                    engine resolved 'object' (8 keyframes). Screenshots state-coverage-*.png
 *   killed-dense     the page is killed (reload) during depth maps → Home's "run interrupted" card names
 *                    the scan and the dense cloud; Resume → the viewer's dense-resume card → the resumed
 *                    densify runs in light mode and completes; no card afterwards
 *   killed-splat     killed during the photoreal training after a checkpoint → the card; Resume → the
 *                    viewer's splat-resume card → continues from the checkpoint in light mode (≤ 1.5 M
 *                    points shown, no mesh) → the full cloud and the mesh come back
 *   evicted-refine   the page goes to the background during refine, then is killed → the card says it
 *                    stopped in the background; Resume reconstructs the scan
 *   killed-splat-early  killed while the photoreal build prepares (no checkpoint yet) on a sparse scan →
 *                    the card promises no saved training and no lighter view; Resume → no splat-resume
 *                    card, a hint whose Build opens the Build sheet
 * Every other scenario must not show the "run interrupted" card (a watcher in each page).
 * Usage: node e2e/ui/states.mjs [name ...] [--verbose]
 */
import { launch, layoutAudit, shot, sleep } from './lib.mjs';

const SCENARIOS = [
  'engine-missing',
  'install-hint',
  'cpu-mode',
  'lockdown',
  'camera-denied',
  'landscape',
  'interrupted',
  'processing-error',
  'storage-full-start',
  'storage-full-capture',
  'storage-write-error-capture',
  'engine-dead',
  'export-reopen',
  'open-other-session',
  'focus',
  'raw-overlay',
  'finish-stuck',
  'finish-stuck-home',
  'finish-stuck-discard',
  'discard-no-replay',
  'delete-open-session',
  'memory-limit',
  'dense-resume',
  'dense-resume-quality',
  'info-sheet',
  'processing-stop',
  'open-failed',
  'viewer-loading',
  'camera-error-init',
  'camera-error-init-retry',
  'memory-limit-background',
  'empty-dense',
  'splat-build',
  'splat-no-webgpu',
  'coverage',
  'killed-dense',
  'killed-splat',
  'evicted-refine',
  'killed-splat-early',
];
/** Scenarios that kill a run on purpose: the only ones where the "run interrupted" card may appear. */
const EXPECTS_RUN_CARD = new Set(['killed-dense', 'killed-splat', 'evicted-refine', 'killed-splat-early']);
const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
// A misspelled name would run nothing and still pass.
const unknown = args.filter((a) => !SCENARIOS.includes(a));
if (unknown.length) {
  console.error(`[states] unknown scenario(s): ${unknown.join(', ')}\n[states] known: ${SCENARIOS.join(', ')}`);
  process.exit(2);
}
const want = (n) => !args.length || args.includes(n);
const ran = new Set();
const launched = await launch({ verbose: process.argv.includes('--verbose') });
const { close, consoleErrors, base } = launched;
/** The running scenario (pages report the "run interrupted" card to it). */
let current = null;
/** Scenarios whose pages showed the "run interrupted" card. */
const runCardSeen = new Set();
/** launched.newPage + a watcher that reports Home's "run interrupted" card (no false positives). */
async function newPage(opts) {
  const r = await launched.newPage(opts);
  const name = current;
  await r.context.exposeFunction('__sfmRunCardShown', () => {
    if (name) runCardSeen.add(name);
  });
  await r.context.addInitScript(() => {
    let sent = false;
    new MutationObserver((records) => {
      if (sent || !records.some((r) => r.addedNodes.length) || !document.querySelector('[data-testid="run-interrupted"]')) return;
      sent = true;
      globalThis.__sfmRunCardShown?.();
    }).observe(document, { childList: true, subtree: true });
  });
  return r;
}
const failures = [];
const check = (cond, msg) => {
  if (!cond) {
    failures.push(msg);
    console.error(`[states] FAIL: ${msg}`);
  } else console.log(`[states] ok: ${msg}`);
};
const by = (page, id) => page.locator(`[data-testid="${id}"]`);

async function startCapture(page, url) {
  await page.goto(url);
  await by(page, 'new-scan').tap();
  await by(page, 'start-capture').tap();
  await page.waitForSelector('.capture');
}

async function holdShot(page, name) {
  await page.evaluate(() => globalThis.__sfmApp?.debug.holdFrames?.(true));
  await sleep(600);
  await shot(page, name);
  await page.evaluate(() => globalThis.__sfmApp?.debug.holdFrames?.(false));
}

async function scenario(name, fn) {
  if (!SCENARIOS.includes(name)) throw new Error(`scenario ${name} is missing from SCENARIOS`);
  if (!want(name)) return;
  ran.add(name);
  console.log(`[states] — ${name}`);
  current = name;
  try {
    await fn();
  } catch (e) {
    failures.push(`${name}: ${e?.message ?? e}`);
    console.error(e);
  }
  current = null;
  if (!EXPECTS_RUN_CARD.has(name)) check(!runCardSeen.has(name), `${name}: no "run interrupted" card`);
  else check(runCardSeen.has(name), `${name}: the card watcher saw the card (it works)`);
}

/** The run journal in the page's localStorage (src/app/run-journal.ts). */
const journal = (page) => page.evaluate(() => JSON.parse(localStorage.getItem('sfm.run.v1') ?? 'null'));

/**
 * Kill the page while a run is going: a reload (no code of the run gets to end it, like an iOS
 * kill). The beforeunload prompt of a running job (desktop browsers only; iOS shows none) is accepted.
 */
async function killPage(page) {
  const accept = (d) => void d.accept().catch(() => {});
  page.on('dialog', accept);
  try {
    await page.reload();
  } finally {
    page.off('dialog', accept);
  }
}

/** Page visibility as iOS reports it for the app going to / coming back from the background. */
const setPageHidden = (page, hidden) =>
  page.evaluate((h) => {
    Object.defineProperty(document, 'hidden', { value: h, configurable: true });
    Object.defineProperty(document, 'visibilityState', { value: h ? 'hidden' : 'visible', configurable: true });
    document.dispatchEvent(new Event('visibilitychange'));
  }, hidden);

/** Poll `fn` (async, in Node) until it returns a truthy value. */
async function until(fn, timeoutMs, what) {
  const t0 = Date.now();
  for (;;) {
    const v = await fn();
    if (v) return v;
    if (Date.now() - t0 > timeoutMs) throw new Error(`timed out waiting for ${what}`);
    await sleep(200);
  }
}

/** A stored session's stats (the MockEngine keeps its resume markers there, like the engine's stored progress). */
const storedStats = (page, id) =>
  page.evaluate(async (i) => {
    const { SessionStore } = await import('/src/io/storage.ts');
    return (await (await SessionStore.open()).getSession(i))?.stats ?? null;
  }, id);

const viewerLoaded = (page) => page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 30000 });
const viewerStats = (page) =>
  page.evaluate(() => {
    const s = globalThis.__sfmApp.debug.viewer?.getStats();
    return s ? { densePoints: s.densePoints, triangles: s.triangles } : null;
  });

await scenario('engine-missing', async () => {
  const { page, context } = await newPage();
  // The real engine worker exists now: serve a stub module in its place that answers every
  // request like the old placeholder did (NotImplementedError).
  await context.route(/engine\.worker\.ts/, (route) =>
    route.fulfill({
      contentType: 'text/javascript',
      body: `self.onmessage = (ev) => { const r = ev.data; if (r && typeof r.id === 'number') self.postMessage({ id: r.id, ok: false, error: { name: 'NotImplementedError', message: 'engine not implemented yet' } }); };`,
    }),
  );
  await startCapture(page, base);
  await page.waitForSelector('[data-testid="capture-error"]', { timeout: 60000 });
  const text = await page.locator('[data-testid="capture-error"]').textContent();
  check(/engine not implemented yet/i.test(text ?? ''), 'placeholder worker error is shown');
  await shot(page, 'state-engine-missing');
  await by(page, 'capture-back').tap();
  await by(page, 'new-scan').waitFor();
  check(true, 'back to home from the error card');
  await context.close();
  // The stub worker answers with an error: expected console noise is none.
});

await scenario('install-hint', async () => {
  const { page, context } = await newPage({ iphoneUA: true });
  await page.goto(`${base}?mock=1`);
  await by(page, 'install-hint').waitFor({ timeout: 10000 });
  await sleep(400);
  await page.evaluate(() => document.querySelector('.home')?.scrollTo(0, 2000));
  await sleep(300);
  await shot(page, 'state-install-hint');
  check(true, 'install hint visible on iOS Safari (not standalone)');
  await context.close();
});

await scenario('cpu-mode', async () => {
  const { page, context } = await newPage({ initScript: () => Object.defineProperty(Navigator.prototype, 'gpu', { get: () => undefined, configurable: true }) });
  await page.goto(`${base}?mock=1`);
  await by(page, 'env-webgpu').waitFor({ timeout: 10000 });
  check(!(await by(page, 'new-scan').isDisabled()), 'scanning still allowed in CPU mode');
  await sleep(300);
  await shot(page, 'state-cpu-mode');
  await context.close();
});

await scenario('lockdown', async () => {
  const { page, context } = await newPage({
    initScript: () => {
      Object.defineProperty(Navigator.prototype, 'gpu', { get: () => undefined, configurable: true });
      Object.defineProperty(Navigator.prototype, 'mediaDevices', { get: () => undefined, configurable: true });
      Object.defineProperty(globalThis, 'indexedDB', { get: () => undefined, configurable: true });
    },
  });
  await page.goto(`${base}?mock=1`);
  await by(page, 'env-lockdown').waitFor({ timeout: 10000 });
  check(await by(page, 'new-scan').isDisabled(), 'New scan disabled under Lockdown Mode');
  await sleep(300);
  await shot(page, 'state-lockdown');
  await context.close();
});

await scenario('camera-denied', async () => {
  const { page, context } = await newPage({
    initScript: () => {
      const md = navigator.mediaDevices;
      if (md) md.getUserMedia = () => Promise.reject(new DOMException('Permission denied', 'NotAllowedError'));
    },
  });
  await startCapture(page, `${base}?mock=1`);
  await page.waitForSelector('[data-testid="capture-error"]', { timeout: 30000 });
  const text = await page.locator('[data-testid="capture-error"]').textContent();
  check(/Camera access is off/.test(text ?? ''), 'denied camera explains how to allow it');
  await shot(page, 'state-camera-denied');
  await context.close();
});

await scenario('landscape', async () => {
  const { page, context } = await newPage({ contextOptions: { viewport: { width: 852, height: 393 } } });
  await startCapture(page, `${base}?mock=1&mockSpeed=3`);
  await page.waitForFunction(() => Number(document.querySelector('[data-testid="kf-count"]')?.textContent) >= 6, null, { timeout: 90000 });
  await holdShot(page, 'state-landscape');
  const box = await by(page, 'finish').boundingBox();
  check(!!box && box.x > 852 - 150, 'controls move to the right edge in landscape');
  await context.close();
});

await scenario('interrupted', async () => {
  const { page, context } = await newPage();
  await startCapture(page, `${base}?mock=1&mockSpeed=3`);
  await page.waitForFunction(() => document.querySelector('.capture')?.getAttribute('data-phase') === 'running', null, { timeout: 45000 });
  const setHidden = (hidden) =>
    page.evaluate((h) => {
      Object.defineProperty(document, 'hidden', { value: h, configurable: true });
      Object.defineProperty(document, 'visibilityState', { value: h ? 'hidden' : 'visible', configurable: true });
      document.dispatchEvent(new Event('visibilitychange'));
    }, hidden);
  await setHidden(true);
  await page.waitForFunction(() => document.querySelector('.capture')?.getAttribute('data-phase') === 'interrupted');
  await setHidden(false);
  await by(page, 'resume').waitFor({ timeout: 10000 });
  await page.evaluate(() => document.querySelector('video')?.pause());
  await sleep(300);
  await shot(page, 'state-interrupted');
  await by(page, 'resume').tap();
  await page.waitForFunction(() => document.querySelector('.capture')?.getAttribute('data-phase') === 'running', null, { timeout: 15000 });
  check(true, 'resume from the tap after an interruption');
  await context.close();
});

await scenario('processing-error', async () => {
  const { page, context } = await newPage();
  await startCapture(page, `${base}?mock=1&mockSpeed=3&mockFail=refine:Not enough overlap between photos`);
  await page.waitForFunction(() => Number(document.querySelector('[data-testid="kf-count"]')?.textContent) >= 6, null, { timeout: 90000 });
  await by(page, 'finish').tap();
  await by(page, 'processing-error').waitFor({ timeout: 30000 });
  const text = await by(page, 'processing-error').textContent();
  check(/Not enough overlap/.test(text ?? ''), 'processing error message shown');
  await sleep(400);
  await shot(page, 'state-processing-error');
  // the reported error closed the run: the next launch shows no "run interrupted" card
  check((await journal(page))?.step === null, 'the failed run is closed in the run journal');
  await killPage(page);
  await by(page, 'session-item').first().waitFor({ timeout: 15000 });
  await sleep(500);
  check((await by(page, 'run-interrupted').count()) === 0, 'no "run interrupted" card after a reported error');
  await context.close();
});

/** Seed a finished (meshed) scan into IndexedDB so the Viewer opens without a capture. */
async function seedMeshed(page, name = 'Seeded vase') {
  await page.evaluate(async (n) => {
    const { SessionStore } = await import('/src/io/storage.ts');
    const s = await SessionStore.open();
    await s.createSession({ id: 'seed-meshed', name: n, mode: 'object', status: 'meshed', stats: { quality: 'fast', keyframes: 20 } });
  }, name);
  await page.reload();
}

await scenario('storage-full-start', async () => {
  const { page, context } = await newPage();
  await page.goto(`${base}?mock=1&mockSpeed=3`);
  await by(page, 'new-scan').waitFor();
  // From now on every IndexedDB write fails like a full origin quota.
  await page.evaluate(() => {
    IDBObjectStore.prototype.add = function () {
      throw new DOMException('The quota has been exceeded.', 'QuotaExceededError');
    };
    IDBObjectStore.prototype.put = IDBObjectStore.prototype.add;
  });
  await by(page, 'new-scan').tap();
  await by(page, 'start-capture').tap();
  await page.waitForSelector('[data-testid="capture-error"]', { timeout: 30000 });
  const text = await page.locator('[data-testid="capture-error"]').textContent();
  check(/Storage is full/.test(text ?? ''), 'a full storage blocks the scan at the start with an explanation');
  check((await page.locator('.capture').getAttribute('data-phase')) === 'error', 'no frames are processed when the scan cannot be saved');
  await shot(page, 'state-storage-full-start');
  await by(page, 'capture-back').tap();
  await by(page, 'new-scan').waitFor();
  await context.close();
});

await scenario('storage-full-capture', async () => {
  const { page, context } = await newPage();
  await startCapture(page, `${base}?mock=1&mockSpeed=3`);
  await page.waitForFunction(() => Number(document.querySelector('[data-testid="kf-count"]')?.textContent) >= 6, null, { timeout: 90000 });
  // The engine can no longer store keyframes.
  await page.evaluate(() => {
    const e = globalThis.__sfmApp.engine;
    e.processFrame = async (f) => {
      f.bitmap.close?.();
      const err = new Error('Storage quota exceeded while storing a keyframe');
      err.name = 'StorageQuotaError';
      throw err;
    };
  });
  await by(page, 'capture-storage-full').waitFor({ timeout: 15000 });
  check((await page.locator('.capture').getAttribute('data-phase')) === 'paused', 'capture pauses when keyframes cannot be stored');
  await holdShot(page, 'state-storage-full-capture');
  await page.locator('[data-testid="capture-storage-full"] button', { hasText: 'Finish scan' }).tap();
  await page.waitForSelector('.processing', { timeout: 20000 });
  check(true, 'the scan can be finished with the keyframes captured so far');
  await context.close();
});

await scenario('storage-write-error-capture', async () => {
  const { page, context } = await newPage();
  await startCapture(page, `${base}?mock=1&mockSpeed=3`);
  await page.waitForFunction(() => Number(document.querySelector('[data-testid="kf-count"]')?.textContent) >= 6, null, { timeout: 90000 });
  // A write fails for another reason than quota (iOS: the IndexedDB connection was lost).
  await page.evaluate(() => {
    const e = globalThis.__sfmApp.engine;
    e.processFrame = async (f) => {
      f.bitmap.close?.();
      const err = new Error('UnknownError: Connection to Indexed Database server lost. Refresh the page to try again');
      err.name = 'StorageWriteError';
      err.code = 'storage';
      throw err;
    };
  });
  await by(page, 'capture-storage-error').waitFor({ timeout: 15000 });
  check((await page.locator('.capture').getAttribute('data-phase')) === 'paused', 'capture pauses when a write fails for another reason than quota');
  const text = await by(page, 'capture-storage-error').textContent();
  check(!/Storage is full/.test(text ?? '') && /Connection to Indexed Database server lost/.test(text ?? ''), 'a non-quota write failure is not reported as a full storage');
  await context.close();
});

await scenario('engine-dead', async () => {
  const { page, context } = await newPage();
  await startCapture(page, `${base}?mock=1&mockSpeed=3`);
  await page.waitForFunction(() => Number(document.querySelector('[data-testid="kf-count"]')?.textContent) >= 2, null, { timeout: 90000 });
  // EngineClient after the respawn cap: state 'dead', every call rejects with EngineCrashedError.
  await page.evaluate(() => {
    const e = globalThis.__sfmApp.engine;
    Object.defineProperty(e, 'status', { get: () => ({ state: 'dead', respawns: 3, reason: 'gpu-lost' }), configurable: true });
    e.processFrame = async (f) => {
      f.bitmap.close?.();
      const err = new Error('Engine stopped after repeated failures (gpu-lost)');
      err.name = 'EngineCrashedError';
      throw err;
    };
  });
  await page.waitForSelector('[data-testid="capture-error"]', { timeout: 15000 });
  const text = await page.locator('[data-testid="capture-error"]').textContent();
  check(/Processing engine stopped/.test(text ?? ''), 'a dead engine stops the capture with an error card');
  const toasts = await page.locator('.toast').count();
  check(toasts <= 1, `no toast storm from failing frames (${toasts} toasts)`);
  await shot(page, 'state-engine-dead');
  await context.close();
});

await scenario('export-reopen', async () => {
  const { page, context } = await newPage();
  await page.goto(`${base}?mock=1`);
  await by(page, 'new-scan').waitFor();
  await seedMeshed(page);
  await by(page, 'session-item').first().tap();
  await page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 30000 });
  // A slow export (big model): close the sheet while it is being built, then open it again.
  await page.evaluate(() => {
    const e = globalThis.__sfmApp.engine;
    const orig = e.exportModel.bind(e);
    e.exportModel = async (f) => {
      await new Promise((r) => setTimeout(r, 1500));
      return orig(f);
    };
  });
  await by(page, 'tool-export').tap();
  await by(page, 'export-prepare-glb').tap();
  await page.locator('.sheet-close').last().tap();
  await page.waitForFunction(() => !document.querySelector('.sheet-wrap'), null, { timeout: 5000 });
  await by(page, 'tool-export').tap();
  await by(page, 'sheet-export').waitFor();
  await by(page, 'export-save-glb').waitFor({ timeout: 10000 });
  check(true, 'an export finished while the sheet was closed shows its Save button in the reopened sheet');
  await context.close();
});

await scenario('open-other-session', async () => {
  const { page, context } = await newPage();
  await page.goto(`${base}?mock=1&mockSpeed=4`);
  await by(page, 'new-scan').waitFor();
  await page.evaluate(async () => {
    const { SessionStore } = await import('/src/io/storage.ts');
    const s = await SessionStore.open();
    await s.createSession({ id: 'a-meshed', name: 'Alpha vase', mode: 'object', status: 'meshed', stats: { quality: 'max', keyframes: 20 } });
    await s.createSession({ id: 'b-captured', name: 'Beta yard', mode: 'scene', status: 'captured', stats: { quality: 'fast', keyframes: 20 } });
  });
  await page.reload();
  await page.locator('[data-testid="session-item"]', { hasText: 'Alpha vase' }).tap();
  await page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 30000 });
  await by(page, 'viewer-back').tap();
  // Process the other (captured) scan: the screen must not show Alpha's name / built stages.
  await page.locator('[data-testid="session-item"]', { hasText: 'Beta yard' }).tap();
  await page.waitForSelector('.processing');
  const early = await page.evaluate(() => document.querySelector('.processing .topbar-title')?.textContent);
  check(early !== 'Alpha vase', `processing never shows the previous scan's name (got "${early}")`);
  const caption = () => page.evaluate(() => document.querySelector('.proc-preview-caption')?.textContent ?? '');
  const earlyCaption = await caption();
  await page.waitForFunction(() => document.querySelector('.processing .topbar-title')?.textContent === 'Beta yard', null, { timeout: 15000 });
  // Beta has no dense cloud: a "Dense cloud" preview is Alpha's.
  const loadedCaption = await caption();
  check(![earlyCaption, loadedCaption].some((c) => c.startsWith('Dense cloud')), `the preview never shows the previous scan's model (got "${earlyCaption}", then "${loadedCaption}")`);
  const st = await page.evaluate(() => ({
    depth: document.querySelector('[data-testid="stage-depth_maps"]')?.getAttribute('data-status'),
    sub: document.querySelector('.processing .topbar-sub')?.textContent,
  }));
  check(st.depth === 'skipped', `stages of the other scan are not shown as built (depth maps: ${st.depth})`);
  check(st.sub === 'Fast quality', `quality of the opened scan (got "${st.sub}")`);
  await context.close();
});

await scenario('focus', async () => {
  const { page, context } = await newPage();
  await page.goto(`${base}?mock=1`);
  await by(page, 'new-scan').focus();
  await page.keyboard.press('Enter');
  await by(page, 'start-capture').waitFor();
  const active = await page.evaluate(() => ({ tag: document.activeElement?.tagName, text: document.activeElement?.textContent, inOld: !!document.activeElement?.closest('.screen-exit') }));
  check(active.tag === 'H1' && active.text === 'New scan' && !active.inOld, `focus moves to the new screen's heading (got ${active.tag} "${active.text}")`);
  const exiting = await page.evaluate(() => [...document.querySelectorAll('.screen-exit')].every((el) => el.inert));
  check(exiting, 'the exiting screen is inert');
  await context.close();
});

async function audit(page, name) {
  const problems = await page.evaluate(layoutAudit);
  for (const p of problems) check(false, `${name}: ${p}`);
}

const kf = (page) => page.evaluate(() => Number(document.querySelector('[data-testid="kf-count"]')?.textContent));

await scenario('raw-overlay', async () => {
  const { page, context } = await newPage();
  await startCapture(page, `${base}?mock=1&mockSpeed=3`);
  await page.waitForFunction(() => document.querySelector('.capture')?.getAttribute('data-phase') === 'running', null, { timeout: 45000 });
  // Every LiveUpdate gets two keypoints at known raw-buffer positions: (0.9, 0.5) and (0.1, 0.5).
  await page.evaluate(() => {
    const em = globalThis.__sfmApp.engine.live;
    const orig = em.emit.bind(em);
    em.emit = (u) => orig({ ...u, overlay: new Float32Array([0.9, 0.5, 0.1, 0.5]) });
  });
  const info = await page.evaluate(() => {
    const v = document.querySelector('video');
    return { w: v.videoWidth, h: v.videoHeight, angle: screen.orientation?.angle ?? 0, vw: innerWidth, vh: innerHeight };
  });
  check(info.w > info.h && info.angle === 0, `fake camera delivers landscape frames in the portrait UI (${info.w}×${info.h}, θ ${info.angle}) — raw buffers`);
  // Raw buffers are upright for θ = 90: one clockwise turn onto the portrait preview →
  // (0.9, 0.5) → (0.5, 0.9) and (0.1, 0.5) → (0.5, 0.1): bottom-centre and top-centre.
  // Unrotated (the old mapping) both would lie outside the cropped preview.
  const found = await page.waitForFunction(
    ({ vw, vh }) => {
      const c = document.querySelector('.capture-overlay');
      if (!c || !c.width) return false;
      const ctx = c.getContext('2d');
      const s = c.width / vw;
      const hit = (x, y) => {
        const d = ctx.getImageData(Math.round(x * s) - 3, Math.round(y * s) - 3, 7, 7).data;
        for (let i = 3; i < d.length; i += 4) if (d[i] > 0) return true;
        return false;
      };
      return hit(vw / 2, 0.9 * vh) && hit(vw / 2, 0.1 * vh) ? true : false;
    },
    { vw: info.vw, vh: info.vh },
    { timeout: 20000, polling: 100 },
  ).then(() => true, () => false);
  check(found, 'raw-buffer keypoints are drawn at their rotated positions (bottom-centre, top-centre)');
  await holdShot(page, 'state-raw-overlay');
  await context.close();
});

async function stuckFinish(page) {
  await startCapture(page, `${base}?mock=1&mockSpeed=3&mockStall=finishCapture`);
  await page.waitForFunction(() => Number(document.querySelector('[data-testid="kf-count"]')?.textContent) >= 6, null, { timeout: 90000 });
  await by(page, 'finish').tap();
  await by(page, 'capture-finishing').waitFor({ timeout: 10000 });
  check(await by(page, 'finish-cancel').isHidden(), 'no Cancel during a normal-length finish');
  await by(page, 'finish-cancel').waitFor({ state: 'visible', timeout: 10000 });
  await sleep(1200); // a progress message arrives (the stalled engine keeps reporting)
  const text = await by(page, 'capture-finishing').textContent();
  check(/mapping 2 keyframes|Still mapping/i.test(text ?? ''), `finishing card explains the wait ("${text?.replace(/\s+/g, ' ').trim()}")`);
  await shot(page, 'state-finish-stuck');
  await audit(page, 'finish-stuck');
  const t0 = Date.now();
  await by(page, 'finish-cancel').tap();
  await by(page, 'capture-finish-stopped').waitFor({ timeout: 15000 });
  const dt = Date.now() - t0;
  check(dt < 9000, `a stuck finishCapture is aborted within the grace period (${(dt / 1000).toFixed(1)} s)`);
  const card = await by(page, 'capture-finish-stopped').textContent();
  check(/Engine restarted/.test(card ?? ''), 'the abort restarted the engine (it ignored the cancel)');
  const toasts = await page.locator('.toast', { hasText: 'restarted after an error' }).count();
  check(toasts === 0, 'no "restarted after an error" toast for a user abort');
  await shot(page, 'state-finish-stopped');
  await audit(page, 'finish-stopped');
}

await scenario('finish-stuck', async () => {
  const { page, context } = await newPage();
  await stuckFinish(page);
  await by(page, 'finish-process').tap();
  await page.waitForSelector('.processing', { timeout: 20000 });
  await page.waitForSelector('.viewer-screen', { timeout: 120000 });
  check(true, '"Process now" after an aborted finish reconstructs the scan');
  await context.close();
});

await scenario('finish-stuck-home', async () => {
  const { page, context } = await newPage();
  await stuckFinish(page);
  await by(page, 'finish-home').tap();
  await by(page, 'session-item').first().waitFor({ timeout: 15000 });
  const list = await page.evaluate(async () => {
    const { SessionStore } = await import('/src/io/storage.ts');
    const s = await SessionStore.open();
    return (await s.listSessions()).map((m) => ({ status: m.status, stats: m.stats }));
  });
  check(list.length === 1 && list[0].status === 'captured', `the scan is kept as captured (got ${JSON.stringify(list.map((m) => m.status))})`);
  const st = list[0]?.stats ?? {};
  check(typeof st.durationMs === 'number' && typeof st.resolution === 'string', `the capture duration and resolution are kept (got ${JSON.stringify(st)})`);
  await context.close();
});

await scenario('finish-stuck-discard', async () => {
  const { page, context } = await newPage();
  await startCapture(page, `${base}?mock=1&mockSpeed=3&mockStall=finishCapture`);
  await page.waitForFunction(() => Number(document.querySelector('[data-testid="kf-count"]')?.textContent) >= 6, null, { timeout: 90000 });
  await by(page, 'finish').tap();
  await by(page, 'capture-finishing').waitFor({ timeout: 10000 });
  await sleep(500);
  // the finishing card covers the X for touch; keyboard / VoiceOver still reach it
  await by(page, 'capture-cancel').evaluate((el) => el.click());
  await by(page, 'dialog-confirm').tap();
  // the stuck step is aborted first (≤ 4 s grace): the deletion waits, the card says so
  await by(page, 'capture-discarding').waitFor({ timeout: 5000 });
  await shot(page, 'state-finish-stuck-discard');
  await by(page, 'new-scan').waitFor({ timeout: 15000 });
  const list = async () =>
    page.evaluate(async () => {
      const { SessionStore } = await import('/src/io/storage.ts');
      return (await (await SessionStore.open()).listSessions()).map((m) => m.id);
    });
  check((await list()).length === 0, 'the discarded scan is deleted');
  check((await page.evaluate(() => globalThis.__sfmApp.engine.status.state)) === 'idle', 'the restart did not re-open the discarded session');
  // a new scan right away works (no rejected init)
  await by(page, 'new-scan').tap();
  await by(page, 'start-capture').tap();
  await page.waitForFunction(() => Number(document.querySelector('[data-testid="kf-count"]')?.textContent) >= 2, null, { timeout: 60000 });
  check(true, 'a new scan starts right after the discard');
  check((await list()).length === 1, 'only the new scan is stored');
  await context.close();
});

await scenario('discard-no-replay', async () => {
  const { page, context } = await newPage();
  // the protocol fixture of engine-client.gpu.test.ts answers in place of the engine worker
  await context.route(/engine\.worker\.ts/, (route) => route.fulfill({ contentType: 'text/javascript', body: `import '/src/app/engine-client.worker-fixture.ts';` }));
  await startCapture(page, base);
  await page.waitForFunction(() => Number(document.querySelector('[data-testid="kf-count"]')?.textContent) >= 6, null, { timeout: 90000 });
  await by(page, 'capture-cancel').tap();
  await by(page, 'dialog-confirm').tap();
  await by(page, 'new-scan').waitFor({ timeout: 15000 });
  const status = () => page.evaluate(() => globalThis.__sfmApp.engine.status);
  const st = await status();
  check(st.state === 'idle' && st.reason === 'closed', `the discarded scan is closed in the engine (state ${st.state}, ${st.reason})`);
  // a later engine restart (e.g. a GPU loss while on Home) must not replay the discarded scan's init
  await page.evaluate(() => globalThis.__sfmApp.engine.restart('test'));
  await sleep(1000);
  check((await status()).state === 'idle', `the restart re-opened nothing (state ${(await status()).state})`);
  const n = await page.evaluate(async () => {
    const { SessionStore } = await import('/src/io/storage.ts');
    return (await (await SessionStore.open()).listSessions()).length;
  });
  check(n === 0, `the discarded scan is deleted (${n} stored)`);
  await context.close();
});

await scenario('delete-open-session', async () => {
  const { page, context } = await newPage();
  await page.goto(`${base}?mock=1`);
  await by(page, 'new-scan').waitFor();
  await seedMeshed(page);
  await by(page, 'session-item').first().tap();
  await page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 30000 });
  await by(page, 'viewer-back').tap();
  await by(page, 'session-more').first().tap();
  await by(page, 'session-delete').tap();
  await by(page, 'dialog-confirm').tap();
  await page.waitForSelector('.empty-state', { timeout: 5000 });
  const st = await page.evaluate(() => globalThis.__sfmApp.engine.status);
  check(st.state === 'idle' && st.reason === 'closed', `deleting the open scan makes the engine let go of it (state ${st.state}, ${st.reason})`);
  await context.close();
});

async function memoryLimit(page, name) {
  await startCapture(page, `${base}?mock=1&mockSpeed=3&mockMemoryLimit=6`);
  await by(page, 'capture-memory-limit').waitFor({ timeout: 90000 });
  check((await page.locator('.capture').getAttribute('data-phase')) === 'paused', `${name}: the capture pauses at the memory limit`);
  const text = await by(page, 'capture-memory-limit').textContent();
  check(/Memory limit reached — finish scan now\?/.test(text ?? ''), `${name}: the prompt asks to finish now`);
  await holdShot(page, `state-${name}`);
  await audit(page, name);
}

await scenario('memory-limit', async () => {
  const { page, context } = await newPage();
  await memoryLimit(page, 'memory-limit');
  await by(page, 'memory-continue').tap();
  await page.waitForFunction(() => document.querySelector('.capture')?.getAttribute('data-phase') === 'running', null, { timeout: 10000 });
  check(await by(page, 'memory-limited').isVisible(), '"Tracking only" chip after Continue');
  const before = await kf(page);
  await sleep(5000);
  const after = await kf(page);
  check(before === 6 && after === 6, `no new keyframes while tracking only (${before} → ${after})`);
  check((await page.locator('[data-testid="capture-memory-limit"]').count()) === 0, 'the prompt is shown once');
  await holdShot(page, 'state-memory-limit-tracking');
  await by(page, 'finish').tap();
  await page.waitForSelector('.processing', { timeout: 20000 });
  check(true, 'Finish works after continuing with tracking only');
  await context.close();
  // Landscape: the prompt fits and its "Finish scan" leads to processing.
  const l = await newPage({ contextOptions: { viewport: { width: 852, height: 393 } } });
  await memoryLimit(l.page, 'memory-limit-landscape');
  await by(l.page, 'memory-finish').tap();
  await l.page.waitForSelector('.processing', { timeout: 20000 });
  check(true, '"Finish scan" from the memory prompt');
  await l.context.close();
});

await scenario('dense-resume', async () => {
  const { page, context } = await newPage();
  // Slow stages: the resumed densify must still be running when Cancel is tapped.
  await page.goto(`${base}?mock=1&mockSpeed=0.3`);
  await by(page, 'new-scan').waitFor();
  await page.evaluate(async () => {
    const { SessionStore } = await import('/src/io/storage.ts');
    const s = await SessionStore.open();
    await s.createSession({ id: 'seed-resume', name: 'Garden wall', mode: 'scene', status: 'sparse', stats: { quality: 'balanced', keyframes: 30, denseResumablePass: 1 } });
  });
  await page.reload();
  await by(page, 'session-item').first().tap();
  await page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 30000 });
  await by(page, 'dense-resume').waitFor({ state: 'visible', timeout: 5000 });
  await sleep(800);
  await shot(page, 'state-dense-resume');
  await audit(page, 'dense-resume viewer');
  await by(page, 'tool-build').tap();
  await by(page, 'build-dense-resume').waitFor({ timeout: 5000 });
  const item = await by(page, 'build-dense-resume').textContent();
  check(/Resume dense reconstruction/.test(item ?? ''), 'Build sheet offers "Resume dense reconstruction"');
  await sleep(400);
  await shot(page, 'state-dense-resume-build');
  await audit(page, 'dense-resume build sheet');
  // Another quality starts over: the item says so.
  await page.locator('.segment[data-value="fast"]').tap();
  const changed = await by(page, 'build-dense-resume').textContent();
  check(/Starts over at Fast quality/.test(changed ?? ''), 'another quality explains that it starts over');
  await page.locator('.segment[data-value="balanced"]').tap();
  await by(page, 'build-dense').tap();
  await page.waitForSelector('.processing', { timeout: 20000 });
  check((await page.locator('.proc-title').textContent()) === 'Resuming dense cloud', 'processing says it resumes');
  // Cancel the resumed densify: the viewer offers to resume again.
  await page.waitForSelector('[data-testid="stage-depth_maps"][data-status="active"]', { timeout: 20000 });
  await by(page, 'processing-cancel').tap();
  await by(page, 'dialog-confirm').tap();
  await page.waitForSelector('.viewer-screen', { timeout: 20000 });
  await by(page, 'dense-resume').waitFor({ state: 'visible', timeout: 10000 });
  check(true, 'a densify cancelled in Processing is resumable from the viewer');
  // the Stop closed the run in the journal: the next launch reports no interruption
  check((await journal(page))?.step === null, 'the stopped densify is closed in the run journal');
  await page.reload();
  await by(page, 'session-item').first().waitFor({ timeout: 15000 });
  await sleep(500);
  check((await by(page, 'run-interrupted').count()) === 0, 'no "run interrupted" card after a Processing Stop');
  await by(page, 'session-item').first().tap();
  await viewerLoaded(page);
  await by(page, 'dense-resume').waitFor({ state: 'visible', timeout: 10000 });
  await by(page, 'dense-resume-button').tap();
  await page.waitForSelector('.processing', { timeout: 20000 });
  await page.waitForSelector('.viewer-screen', { timeout: 120000 });
  await page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 30000 });
  await sleep(500);
  check(await by(page, 'dense-resume').isHidden(), 'no resume offer once the dense cloud exists');
  await context.close();
});

await scenario('dense-resume-quality', async () => {
  const { page, context } = await newPage();
  await page.goto(`${base}?mock=1&mockSpeed=2`);
  await by(page, 'new-scan').waitFor();
  await page.evaluate(async () => {
    const { SessionStore } = await import('/src/io/storage.ts');
    const s = await SessionStore.open();
    // captured at Balanced; the densify that was interrupted ran at Max
    await s.createSession({ id: 'seed-resume-q', name: 'Fence', mode: 'scene', status: 'sparse', stats: { quality: 'balanced', keyframes: 30, denseResumablePass: 1, denseResumableQuality: 'max' } });
  });
  await page.reload();
  await by(page, 'session-item').first().tap();
  await page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 30000 });
  await by(page, 'dense-resume').waitFor({ state: 'visible', timeout: 5000 });
  check(/Max quality/.test((await by(page, 'dense-resume').textContent()) ?? ''), 'the resume offer names the interrupted run’s quality');
  await by(page, 'tool-build').tap();
  await by(page, 'build-dense-resume').waitFor({ timeout: 5000 });
  const item = await by(page, 'build-dense-resume').textContent();
  check(/Resume dense reconstruction/.test(item ?? '') && /continues from the saved depth maps/.test(item ?? ''), 'Build sheet: resume at the interrupted quality');
  check((await page.locator('.segmented[aria-label="Quality"] .segment[aria-checked="true"]').first().getAttribute('data-value')) === 'max', 'the quality selector starts at the interrupted run’s Max');
  await page.locator('.segment[data-value="balanced"]').tap();
  check(/Starts over at Balanced quality \(the interrupted run used Max\)/.test((await by(page, 'build-dense-resume').textContent()) ?? ''), 'the scan’s own quality starts over');
  await page.keyboard.press('Escape');
  await sleep(300);
  await by(page, 'dense-resume-button').tap();
  await page.waitForSelector('.processing', { timeout: 20000 });
  check((await page.locator('.proc-title').textContent()) === 'Resuming dense cloud', 'processing says it resumes');
  check(/Max quality/.test((await page.locator('.processing').textContent()) ?? ''), 'the resume runs at Max quality');
  await page.waitForSelector('.viewer-screen', { timeout: 120000 });
  await context.close();
});

await scenario('info-sheet', async () => {
  const { page, context } = await newPage();
  await page.goto(`${base}?mock=1`);
  await by(page, 'new-scan').waitFor();
  await seedMeshed(page);
  await by(page, 'session-item').first().tap();
  await page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 30000 });
  await by(page, 'viewer-info').tap();
  await by(page, 'sheet-info').waitFor();
  const t = await by(page, 'sheet-info').textContent();
  check(/Magnetic north/.test(t ?? '') && /declination/.test(t ?? ''), 'compass yaw is labelled magnetic north with the declination note');
  check(/Engine memory \(last step\)/.test(t ?? ''), 'engine memory (stats.memoryMB) shown');
  await sleep(400);
  await shot(page, 'state-info-sheet');
  await audit(page, 'info-sheet');
  await page.locator('.section-title', { hasText: 'Memory' }).scrollIntoViewIfNeeded();
  await sleep(300);
  await shot(page, 'state-info-sheet-memory');
  await context.close();
});

/** Store a session record (the mock engine loads its models from the status) without a capture. */
const seedSession = (page, rec) =>
  page.evaluate(async (r) => {
    const { SessionStore } = await import('/src/io/storage.ts');
    await (await SessionStore.open()).createSession(r);
  }, rec);

/** Record the processing steps the engine is asked to run in globalThis.__steps. */
const recordSteps = (page) =>
  page.evaluate(() => {
    const e = globalThis.__sfmApp.engine;
    globalThis.__steps = [];
    for (const m of ['refine', 'densify', 'mesh']) {
      const orig = e[m].bind(e);
      e[m] = (...a) => {
        globalThis.__steps.push(m);
        return orig(...a);
      };
    }
  });
const steps = (page) => page.evaluate(() => globalThis.__steps);
const phaseOf = (page) => page.locator('.capture').getAttribute('data-phase');

await scenario('processing-stop', async () => {
  // Stop while the scan is still being opened: the engine's loadSession ignores the cancel.
  const { page, context } = await newPage();
  await page.goto(`${base}?mock=1&mockSpeed=0.1`); // loadSession takes 2.5 s
  await by(page, 'new-scan').waitFor();
  await seedSession(page, { id: 'stop-captured', name: 'Stop yard', mode: 'scene', status: 'captured', stats: { quality: 'fast', keyframes: 20 } });
  await page.reload();
  await by(page, 'session-item').first().waitFor();
  await recordSteps(page);
  await by(page, 'session-item').first().tap();
  await page.waitForSelector('.processing');
  check((await page.locator('.processing .topbar-title').textContent()) === 'Opening scan…', 'the scan is still being opened');
  await by(page, 'processing-cancel').tap();
  await by(page, 'dialog-confirm').tap();
  const t0 = Date.now();
  await by(page, 'new-scan').waitFor({ timeout: 15000 });
  const dt = Date.now() - t0;
  check(dt < 6000, `a Stop during the load leaves once the load ends, not at the 8 s fallback (${(dt / 1000).toFixed(1)} s)`);
  check((await steps(page)).length === 0, `no step starts after a Stop during the load (${(await steps(page)).join(', ')})`);
  await context.close();

  // Stop that the running step no longer sees (it is writing its result): the next one must not start.
  const b = await newPage();
  await b.page.goto(`${base}?mock=1`);
  await by(b.page, 'new-scan').waitFor();
  await seedSession(b.page, { id: 'stop-sparse', name: 'Stop wall', mode: 'scene', status: 'sparse', stats: { quality: 'balanced', keyframes: 30 } });
  await b.page.reload();
  await by(b.page, 'session-item').first().tap();
  await b.page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 30000 });
  await recordSteps(b.page);
  await b.page.evaluate(() => {
    globalThis.__sfmApp.engine.cancel = () => undefined;
  });
  await by(b.page, 'tool-build').tap();
  await by(b.page, 'build-mesh').tap();
  await b.page.waitForSelector('[data-testid="stage-depth_maps"][data-status="active"]', { timeout: 20000 });
  await by(b.page, 'processing-cancel').tap();
  await by(b.page, 'dialog-confirm').tap();
  await b.page.waitForSelector('.viewer-screen', { timeout: 15000 });
  const started = (await steps(b.page)).join(', ');
  check(started === 'densify', `the mesh does not start after a Stop that the densify did not see (ran: ${started})`);
  await b.context.close();

  // Stop while a load runs that then fails: leave when it fails (it is toasted), not at the 8 s fallback.
  const c = await newPage();
  await c.page.goto(`${base}?mock=1`);
  await by(c.page, 'new-scan').waitFor();
  await seedSession(c.page, { id: 'stop-broken', name: 'Stop shed', mode: 'scene', status: 'captured', stats: { quality: 'fast', keyframes: 20 } });
  await c.page.reload();
  await by(c.page, 'session-item').first().waitFor();
  await c.page.evaluate(() => {
    globalThis.__sfmApp.engine.loadSession = async () => {
      await new Promise((r) => setTimeout(r, 2500));
      throw new Error('Stored data is damaged');
    };
  });
  await by(c.page, 'session-item').first().tap();
  await c.page.waitForSelector('.processing');
  await by(c.page, 'processing-cancel').tap();
  await by(c.page, 'dialog-confirm').tap();
  const t1 = Date.now();
  await by(c.page, 'new-scan').waitFor({ timeout: 15000 });
  const dt1 = Date.now() - t1;
  check(dt1 < 6000, `a Stop during a load that fails leaves when it fails, not at the 8 s fallback (${(dt1 / 1000).toFixed(1)} s)`);
  await c.context.close();
});

await scenario('open-failed', async () => {
  const { page, context } = await newPage();
  await page.goto(`${base}?mock=1&mockSpeed=4`);
  await by(page, 'new-scan').waitFor();
  await seedSession(page, { id: 'a-meshed', name: 'Alpha vase', mode: 'object', status: 'meshed', stats: { quality: 'max', keyframes: 20 } });
  await seedSession(page, { id: 'b-captured', name: 'Beta yard', mode: 'scene', status: 'captured', stats: { quality: 'fast', keyframes: 20 } });
  await page.reload();
  await page.locator('[data-testid="session-item"]', { hasText: 'Alpha vase' }).tap();
  await page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 30000 });
  await by(page, 'viewer-back').tap();
  // The other scan's stored data cannot be loaded.
  await page.evaluate(() => {
    globalThis.__loads = 0;
    globalThis.__sfmApp.engine.loadSession = async () => {
      globalThis.__loads++;
      throw new Error('Stored data is damaged');
    };
  });
  await page.locator('[data-testid="session-item"]', { hasText: 'Beta yard' }).tap();
  await by(page, 'processing-error').waitFor({ timeout: 15000 });
  const leave = by(page, 'processing-error').locator('button').last();
  const label = (await leave.textContent())?.trim();
  check(label === 'Back to home', `a scan that could not be opened offers "Back to home", not the previous scan's model (got "${label}")`);
  await leave.tap();
  await by(page, 'new-scan').waitFor({ timeout: 10000 });
  await sleep(500);
  const loads = await page.evaluate(() => globalThis.__loads);
  check(loads === 1, `no second attempt to open the scan in the viewer (${loads} loads)`);
  await context.close();
});

await scenario('viewer-loading', async () => {
  const { page, context } = await newPage();
  await page.goto(`${base}?mock=1&mockSpeed=0.1`); // loadSession takes 2.5 s
  await by(page, 'new-scan').waitFor();
  await seedSession(page, { id: 'a-sparse', name: 'Alpha wall', mode: 'scene', status: 'sparse', stats: { quality: 'balanced', keyframes: 30 } });
  await seedSession(page, { id: 'b-meshed', name: 'Beta vase', mode: 'object', status: 'meshed', stats: { quality: 'balanced', keyframes: 20 } });
  await page.reload();
  await page.locator('[data-testid="session-item"]', { hasText: 'Alpha wall' }).tap();
  await page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 30000 });
  await by(page, 'viewer-back').tap();
  await page.locator('[data-testid="session-item"]', { hasText: 'Beta vase' }).tap();
  await page.waitForFunction(() => document.querySelector('.viewer-screen') && !document.querySelector('.screen-exit'), null, { timeout: 10000 });
  // While "Loading model…" shows, app.results still describes Alpha wall.
  for (const id of ['tool-build', 'tool-export', 'tool-view', 'viewer-info']) await by(page, id).evaluate((el) => el.click());
  check(!(await page.locator('.viewer-loading').evaluate((el) => el.hidden)), 'the tools were tapped while the scan loads');
  check((await page.locator('.sheet-wrap').count()) === 0, 'no tool opens a sheet while the scan loads (it would show the previous scan)');
  await page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 30000 });
  await by(page, 'tool-build').tap();
  await by(page, 'sheet-build').waitFor();
  check((await by(page, 'build-mesh').count()) === 0, "the Build sheet shows the opened scan's mesh as built");
  await context.close();
});

await scenario('camera-error-init', async () => {
  const { page, context } = await newPage();
  await page.goto(`${base}?mock=1&mockSpeed=3`);
  await by(page, 'new-scan').waitFor();
  // A slow engine init (GPU programs compiled the first time).
  await page.evaluate(() => {
    const e = globalThis.__sfmApp.engine;
    const orig = e.init.bind(e);
    e.init = async (o) => {
      await new Promise((r) => setTimeout(r, 3000));
      const res = await orig(o);
      globalThis.__initDone = true;
      return res;
    };
  });
  await by(page, 'new-scan').tap();
  await by(page, 'start-capture').tap();
  await page.locator('.capture-card-title', { hasText: 'Preparing 3D engine' }).waitFor({ timeout: 30000 });
  // Another app takes the camera: the track ends and the automatic restart fails.
  await page.evaluate(() => {
    navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException('Could not start video source', 'NotReadableError'));
    document.querySelector('video').srcObject.getVideoTracks()[0].dispatchEvent(new Event('ended'));
  });
  await by(page, 'capture-error').waitFor({ timeout: 5000 });
  check(/The camera stopped/.test((await by(page, 'capture-error').textContent()) ?? ''), 'the camera error card is shown during the engine init');
  await page.waitForFunction(() => globalThis.__initDone === true, null, { timeout: 15000 });
  await sleep(500);
  check(await by(page, 'capture-error').isVisible(), 'the camera error card stays once the engine is ready');
  check((await phaseOf(page)) === 'error', `the capture does not run with a stopped camera (phase ${await phaseOf(page)})`);
  await page.evaluate(() => {
    delete navigator.mediaDevices.getUserMedia;
  });
  await by(page, 'capture-retry').tap();
  await page.waitForFunction(() => Number(document.querySelector('[data-testid="kf-count"]')?.textContent) >= 2, null, { timeout: 30000 });
  check((await phaseOf(page)) === 'running', '"Reconnect camera" starts the scan');
  await context.close();
});

await scenario('memory-limit-background', async () => {
  const { page, context } = await newPage();
  await startCapture(page, `${base}?mock=1&mockSpeed=3&mockMemoryLimit=6`);
  await page.waitForFunction(() => document.querySelector('.capture')?.getAttribute('data-phase') === 'running', null, { timeout: 45000 });
  const setHidden = (hidden) =>
    page.evaluate((h) => {
      Object.defineProperty(document, 'hidden', { value: h, configurable: true });
      Object.defineProperty(document, 'visibilityState', { value: h ? 'hidden' : 'visible', configurable: true });
      document.dispatchEvent(new Event('visibilitychange'));
    }, hidden);
  // The app goes to the background while the frame that reaches the cap is in flight: its
  // reports (the error event and the marked update) arrive with the capture interrupted.
  await page.evaluate(() => {
    const e = globalThis.__sfmApp.engine;
    for (const em of [e.errorE, e.live]) {
      const orig = em.emit.bind(em);
      em.emit = (x) => {
        if (!globalThis.__hidden && (x?.reason === 'out-of-memory' || x?.timings?.memoryLimited)) {
          globalThis.__hidden = true;
          Object.defineProperty(document, 'hidden', { value: true, configurable: true });
          Object.defineProperty(document, 'visibilityState', { value: 'hidden', configurable: true });
          document.dispatchEvent(new Event('visibilitychange'));
        }
        orig(x);
      };
    }
  });
  await page.waitForFunction(() => globalThis.__hidden === true, null, { timeout: 90000 });
  await sleep(300);
  check((await phaseOf(page)) === 'interrupted', `the cap was reached while interrupted (phase ${await phaseOf(page)})`);
  check((await by(page, 'capture-memory-limit').count()) === 0, 'no prompt while the capture is interrupted');
  await setHidden(false);
  await by(page, 'resume').waitFor({ timeout: 10000 });
  await by(page, 'resume').tap();
  await by(page, 'capture-memory-limit').waitFor({ timeout: 15000 });
  check((await phaseOf(page)) === 'paused', 'the memory prompt comes once the capture runs again');
  await context.close();
});

await scenario('camera-error-init-retry', async () => {
  const { page, context } = await newPage();
  await page.goto(`${base}?mock=1&mockSpeed=3`);
  await by(page, 'new-scan').waitFor();
  // The first engine init fails after a while (GPU programs that do not compile); the retry works.
  await page.evaluate(() => {
    const e = globalThis.__sfmApp.engine;
    const orig = e.init.bind(e);
    let calls = 0;
    e.init = async (o) => {
      if (++calls === 1) {
        await new Promise((r) => setTimeout(r, 3000));
        throw new Error('GPU programs failed to compile');
      }
      return orig(o);
    };
  });
  await by(page, 'new-scan').tap();
  await by(page, 'start-capture').tap();
  await page.locator('.capture-card-title', { hasText: 'Preparing 3D engine' }).waitFor({ timeout: 30000 });
  // The camera stops during the init (its automatic restart fails), then the init fails: the
  // engine card replaces the camera card, and "Try again" succeeds while the camera is still stopped.
  await page.evaluate(() => {
    navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException('Could not start video source', 'NotReadableError'));
    document.querySelector('video').srcObject.getVideoTracks()[0].dispatchEvent(new Event('ended'));
  });
  await page.locator('[data-testid="capture-error"]', { hasText: 'The camera stopped' }).waitFor({ timeout: 5000 });
  await page.locator('[data-testid="capture-error"]', { hasText: 'Processing engine unavailable' }).waitFor({ timeout: 15000 });
  await by(page, 'capture-retry').tap();
  await page.locator('[data-testid="capture-error"]', { hasText: 'The camera stopped' }).waitFor({ timeout: 10000 });
  check((await page.locator('.capture-card-title', { hasText: 'Preparing 3D engine' }).count()) === 0, 'no "Preparing 3D engine" spinner left once the engine is ready');
  check(/Reconnect camera/.test((await by(page, 'capture-retry').textContent()) ?? ''), 'the camera card offers "Reconnect camera" again');
  await page.evaluate(() => {
    delete navigator.mediaDevices.getUserMedia;
  });
  await by(page, 'capture-retry').tap();
  await page.waitForFunction(() => Number(document.querySelector('[data-testid="kf-count"]')?.textContent) >= 2, null, { timeout: 30000 });
  check((await phaseOf(page)) === 'running', '"Reconnect camera" starts the scan');
  await context.close();
});

await scenario('empty-dense', async () => {
  // A densify whose fusion kept no point: the viewer shows the sparse cloud (as exports do).
  const { page, context } = await newPage();
  await page.goto(`${base}?mock=1`);
  await by(page, 'new-scan').waitFor();
  await seedSession(page, { id: 'seed-empty-dense', name: 'Empty dense', mode: 'object', status: 'dense', stats: { quality: 'fast', keyframes: 20 } });
  await page.reload();
  await by(page, 'new-scan').waitFor();
  await page.evaluate(() => {
    const e = globalThis.__sfmApp.engine;
    const orig = e.loadSession.bind(e);
    e.loadSession = async (...a) => ({ ...(await orig(...a)), dense: { count: 0, positions: new Float32Array(0), colors: new Uint8Array(0) } });
  });
  await by(page, 'session-item').first().tap();
  // (the title is set with the content, once the scan is loaded)
  await page.locator('.viewer-title', { hasText: 'Empty dense' }).waitFor({ timeout: 30000 });
  const sub = (await page.locator('.viewer-sub').textContent()) ?? '';
  check(/points/.test(sub) && !/^0 points/.test(sub), `the subtitle counts the sparse points, not the empty dense cloud (got "${sub}")`);
  await context.close();
});

await scenario('splat-build', async () => {
  const hostGpu = process.env.E2E_GPU === 'host';
  const { page, context } = await newPage();
  // slow stages (7.8 s of training at fast): Cancel lands during the training
  await page.goto(`${base}?mock=1&mockSpeed=0.2`);
  await by(page, 'new-scan').waitFor();
  await page.evaluate(async () => {
    const { SessionStore } = await import('/src/io/storage.ts');
    const s = await SessionStore.open();
    await s.createSession({ id: 'seed-splat', name: 'Garden bench', mode: 'object', status: 'dense', stats: { quality: 'fast', keyframes: 24 } });
  });
  /** Pause / resume the mock's training (it pauses while hidden): slow screenshots and audits must not outlast it. */
  const freeze = (on) => page.evaluate((v) => globalThis.__sfmApp.engine.setHidden(v), on);
  /** A hash of the card's preview pixels ('' before the first snapshot). */
  const previewPixels = () =>
    page.evaluate(() => {
      const c = document.querySelector('[data-testid="splat-preview"]');
      if (!c || !c.width || c.closest('[hidden]')) return '';
      const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
      let hash = 0;
      for (let i = 0; i < d.length; i += 3) hash = (Math.imul(hash, 31) + d[i]) >>> 0;
      return `${c.width}×${c.height}:${hash}`;
    });
  await page.reload();
  await by(page, 'session-item').first().tap();
  await page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 30000 });
  check(await by(page, 'layer-switch').isHidden(), 'no layer pill with points only');
  await by(page, 'tool-build').tap();
  await by(page, 'build-splat').waitFor({ timeout: 5000 });
  check(/Gaussian splats trained on your photos/.test((await by(page, 'sheet-build').textContent()) ?? ''), 'Build sheet explains the photoreal model');
  await sleep(300);
  await shot(page, 'state-splat-build-sheet');
  await audit(page, 'splat build sheet');
  await by(page, 'build-splat').tap();
  await by(page, 'splat-progress').waitFor({ state: 'visible', timeout: 10000 });
  await page.waitForFunction(() => /iterations/.test(document.querySelector('[data-testid="splat-progress-text"]')?.textContent ?? ''), null, { timeout: 30000 });
  // "Use it now" waits for the end of the coarse-to-fine schedule (30 %: iteration 900 of 3000)
  check(await by(page, 'splat-use-now').isDisabled(), '"Use it now" disabled early in the training');
  check(/Usable from iteration 900/.test((await by(page, 'splat-use-note').textContent()) ?? ''), 'its note says from when');
  // the preview snapshot replaces the spinner; its pixels change as the model trains
  await by(page, 'splat-preview').waitFor({ state: 'visible', timeout: 10000 });
  const seenPixels = new Set([await previewPixels()]);
  for (const t0 = Date.now(); seenPixels.size < 3 && Date.now() - t0 < 15000; ) {
    await sleep(150);
    seenPixels.add(await previewPixels());
  }
  seenPixels.delete('');
  check(seenPixels.size >= 3, `the preview's pixels change at least twice before Cancel (${seenPixels.size} distinct)`);
  await freeze(true);
  await shot(page, 'state-splat-progress');
  await audit(page, 'splat progress');
  // the other builds wait while it runs
  await by(page, 'tool-build').tap();
  await by(page, 'sheet-build').waitFor();
  check(await by(page, 'build-mesh').isDisabled(), 'mesh build disabled while the photoreal model trains');
  await page.keyboard.press('Escape');
  await freeze(false);
  await by(page, 'sheet-build').waitFor({ state: 'detached' });
  await by(page, 'splat-cancel').tap();
  await by(page, 'dialog-confirm').tap();
  await by(page, 'splat-progress').waitFor({ state: 'hidden', timeout: 15000 });
  await by(page, 'splat-resume').waitFor({ state: 'visible', timeout: 5000 });
  check(true, 'a cancelled build offers "Photoreal model interrupted · Resume"');
  await sleep(300);
  await shot(page, 'state-splat-resume');
  await audit(page, 'splat resume');
  await by(page, 'tool-build').tap();
  check(/Resume/.test((await by(page, 'build-splat').textContent()) ?? ''), 'Build sheet: Resume at the interrupted quality');
  await page.keyboard.press('Escape');
  await by(page, 'sheet-build').waitFor({ state: 'detached' });
  await by(page, 'splat-resume-button').tap();
  await by(page, 'splat-progress').waitFor({ state: 'visible', timeout: 10000 });
  const firstText = await page.waitForFunction(() => {
    const t = document.querySelector('[data-testid="splat-progress-text"]')?.textContent ?? '';
    return /iterations/.test(t) ? t : null;
  }, null, { timeout: 30000 });
  const it0 = Number(((await firstText.jsonValue()) ?? '').replace(/,/g, '').match(/^(\d+)/)?.[1] ?? 0);
  check(it0 > 0, `the resumed run continues (first progress at iteration ${it0})`);
  // the preview, larger: a sheet that follows the snapshots, with its own "Use it now"
  await by(page, 'splat-preview-open').waitFor({ state: 'visible', timeout: 10000 });
  await freeze(true);
  await by(page, 'splat-preview-open').tap();
  await by(page, 'sheet-splat-preview').waitFor({ state: 'visible', timeout: 5000 });
  check(/^(Iteration [\d,]+ of 3,000|Initial Gaussians)/.test((await by(page, 'splat-preview-caption').textContent()) ?? ''), 'the larger preview has the iteration as its caption');
  await sleep(400);
  await shot(page, 'state-splat-preview');
  await audit(page, 'splat preview sheet');
  await page.keyboard.press('Escape');
  await by(page, 'sheet-splat-preview').waitFor({ state: 'detached' });
  await freeze(false);
  // "Use it now" from iteration 900 on: the model is stored where it is, continuable later
  await page.waitForFunction(() => document.querySelector('[data-testid="splat-use-now"]')?.disabled === false, null, { timeout: 30000 });
  await by(page, 'splat-use-now').tap();
  await by(page, 'splat-progress').waitFor({ state: 'hidden', timeout: 30000 });
  check(await by(page, 'splat-resume').isHidden(), 'no "interrupted" card for a model ended early');
  await by(page, 'tool-build').tap();
  await by(page, 'build-splat').waitFor({ timeout: 5000 });
  const early = (await by(page, 'sheet-build').textContent()) ?? '';
  check(/Continue/.test((await by(page, 'build-splat').textContent()) ?? '') && /Stopped early at [\d,]+ of 3,000 iterations/.test(early), 'Build sheet: Continue the training of the model ended early');
  await shot(page, 'state-splat-continue');
  await by(page, 'build-splat').tap();
  await by(page, 'splat-progress').waitFor({ state: 'visible', timeout: 10000 });
  await by(page, 'splat-progress').waitFor({ state: 'hidden', timeout: 90000 });
  check(await by(page, 'splat-resume').isHidden(), 'no resume card once the model exists');
  await by(page, 'tool-build').tap();
  await by(page, 'sheet-build').waitFor();
  check(!/Stopped early/.test((await by(page, 'sheet-build').textContent()) ?? ''), 'the continued model is complete (no Continue)');
  await page.keyboard.press('Escape');
  await by(page, 'sheet-build').waitFor({ state: 'detached' });
  const segments = () => page.locator('[data-testid="layer-switch"] .segment').evaluateAll((els) => els.map((e) => e.dataset.value));
  if (hostGpu) {
    await by(page, 'layer-switch').waitFor({ state: 'visible', timeout: 15000 });
    const segs = await segments();
    check(segs.join(',') === 'splats,points', `layer pill Splats / Points (got ${segs})`);
    await page.waitForFunction(() => document.querySelector('[data-testid="layer-switch"]')?.dataset.layer === 'splats', null, { timeout: 30000 });
    await sleep(500);
    await shot(page, 'state-splat-shown');
    await audit(page, 'splats shown');
  } else {
    // SwiftShader: the splat layer refuses the software adapter (for good, since the model ended
    // early was loaded): the points stay, and the layer pill has nothing to switch to
    await page.waitForFunction(() => document.querySelector('.viewer-screen')?.dataset.splats === 'unavailable', null, { timeout: 30000 });
    const segs = await segments();
    check((await by(page, 'layer-switch').isHidden()) && !segs.includes('splats'), `software adapter: splats refused, points only (layer pill hidden, segments ${segs})`);
  }
  await by(page, 'tool-export').tap();
  await by(page, 'export-ply-splat').waitFor();
  check(!(await by(page, 'export-ply-splat').getAttribute('class'))?.includes('disabled'), 'export sheet offers the Gaussian-splat PLY');
  await page.keyboard.press('Escape');
  // reopened later: the stored splats come back (and are the default view)
  await page.reload();
  await by(page, 'session-item').first().waitFor();
  await sleep(300);
  check((await by(page, 'run-interrupted').count()) === 0, 'no "run interrupted" card after a cancelled and a finished build');
  await by(page, 'session-item').first().tap();
  await page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 30000 });
  if (hostGpu) {
    await page.waitForFunction(() => document.querySelector('[data-testid="layer-switch"]')?.dataset.layer === 'splats' && document.querySelector('.viewer-screen')?.dataset.splats === 'ready', null, { timeout: 60000 });
    check(true, 'the reopened scan shows its splats');
  } else {
    await page.waitForFunction(() => !!document.querySelector('.viewer-screen')?.dataset.splats, null, { timeout: 60000 });
    check(true, 'the reopened scan loads its splat model');
  }
  await context.close();
});

await scenario('splat-no-webgpu', async () => {
  const { page, context } = await newPage({ initScript: () => Object.defineProperty(Navigator.prototype, 'gpu', { get: () => undefined, configurable: true }) });
  await page.goto(`${base}?mock=1`);
  await by(page, 'new-scan').waitFor();
  await seedMeshed(page);
  await by(page, 'session-item').first().tap();
  await page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 30000 });
  await by(page, 'tool-build').tap();
  await by(page, 'build-splat').waitFor({ timeout: 5000 });
  check(await by(page, 'build-splat').isDisabled(), 'photoreal build disabled without WebGPU');
  check(/Needs WebGPU/.test((await by(page, 'sheet-build').textContent()) ?? ''), 'the item says it needs WebGPU');
  await context.close();
});

await scenario('coverage', async () => {
  const { page, context } = await newPage();
  await startCapture(page, `${base}?mock=1&mockSpeed=4`);
  await page.waitForFunction(() => document.querySelector('.capture')?.getAttribute('data-phase') === 'running', null, { timeout: 45000 });
  check(await by(page, 'coverage').isVisible(), 'object mode: the coverage ring is shown');
  const prompt = () => page.evaluate(() => document.querySelector('[data-testid="coverage-prompt"]')?.textContent ?? '');
  const covered = () => page.evaluate(() => document.querySelectorAll('.coverage-cell.c2').length);
  const hint = () => page.evaluate(() => document.querySelector('.finish-hint:not([hidden])')?.textContent ?? '');
  const waitPrompt = (re, timeout) =>
    page.waitForFunction((src) => new RegExp(src).test(document.querySelector('[data-testid="coverage-prompt"]')?.textContent ?? ''), re.source, { timeout });
  // part of the first loop
  await waitPrompt(/other side/, 60000);
  await page.waitForFunction(() => document.querySelectorAll('.coverage-cell.c2').length >= 4, null, { timeout: 30000 });
  const part = await covered();
  check(part >= 4 && part < 12, `partial orbit: some sectors covered (${part}), prompt "${await prompt()}"`);
  check(/^Coverage \d+ % · sides missing$/.test(await hint()), `partial orbit: the Finish hint shows the coverage ("${await hint()}")`);
  check((await page.evaluate(() => document.querySelector('.finish-hint')?.getAttribute('aria-live'))) === 'off', 'the readiness hint is not a live region (it changes per covered cell)');
  check(await page.evaluate(() => document.querySelector('.coverage-marker')?.classList.contains('shown')), 'the phone is marked on the ring');
  await holdShot(page, 'state-coverage-partial');
  // the cloud view keeps the ring above the Overview / Follow buttons
  await by(page, 'cloud-toggle').tap();
  await page.waitForSelector('.capture.cloud-view');
  const [ringBox, actionsBox] = await Promise.all([by(page, 'coverage').boundingBox(), page.locator('.cloud-actions').boundingBox()]);
  check(!!ringBox && !!actionsBox && ringBox.y + ringBox.height <= actionsBox.y + 1, 'cloud view: the ring sits above the Overview button');
  await audit(page, 'coverage-cloud');
  await by(page, 'cloud-toggle').tap();
  await page.waitForFunction(() => !document.querySelector('.capture.cloud-view'));
  // Headless only: SwiftShader's WebGL draw of the 3D thumbnail starves the frame grabs once the
  // mock cloud is fully revealed (measured: 0–2 frames per 5 s with it, 29–46 without), so the
  // two loops would take minutes. Hide it for the rest of the capture.
  await page.addStyleTag({ content: '.mini-view { display: none !important; }' });
  const flat = await page.evaluate(() => document.querySelector('.coverage')?.classList.contains('flat'));
  if (flat) {
    // no motion sensor in this browser: azimuth only, one loop completes it
    await waitPrompt(/All sides covered/, 90000);
    check(true, 'azimuth only (no motion): one loop covers all sides');
    await holdShot(page, 'state-coverage-full');
  } else {
    await waitPrompt(/Raise the phone/, 90000);
    check(/top not captured/.test(await hint()), `after the level loop: "${await hint()}"`);
    await waitPrompt(/^Done/, 120000);
    check(/ready to finish/.test(await hint()), `after the higher loop: "${await hint()}"`);
    await holdShot(page, 'state-coverage-full');
  }
  await audit(page, 'coverage');
  await context.close();

  // auto: no ring until the engine resolved 'object'
  const a = await newPage();
  await a.page.goto(`${base}?mock=1&mockSpeed=4`);
  await by(a.page, 'new-scan').tap();
  await by(a.page, 'mode-auto').tap();
  await by(a.page, 'start-capture').tap();
  await a.page.waitForFunction(() => document.querySelector('.capture')?.getAttribute('data-phase') === 'running', null, { timeout: 45000 });
  check(!(await by(a.page, 'coverage').isVisible()), 'auto: no coverage ring before the mode is resolved');
  await a.page.waitForFunction(() => Number(document.querySelector('[data-testid="kf-count"]')?.textContent) >= 9, null, { timeout: 60000 });
  check(await by(a.page, 'coverage').isVisible(), "auto: the ring appears once the engine resolved 'object'");
  await a.context.close();
});

await scenario('killed-dense', async () => {
  const id = 'seed-killed-dense';
  const { page, context } = await newPage();
  // slow stages: the densify is still running when the page is killed
  await page.goto(`${base}?mock=1&mockSpeed=0.3`);
  await by(page, 'new-scan').waitFor();
  await seedSession(page, { id, name: 'Stone wall', mode: 'scene', status: 'sparse', stats: { quality: 'fast', keyframes: 30 } });
  await page.reload();
  await by(page, 'session-item').first().tap();
  await viewerLoaded(page);
  await by(page, 'tool-build').tap();
  await by(page, 'build-dense').tap();
  await page.waitForSelector('[data-testid="stage-depth_maps"][data-status="active"]', { timeout: 20000 });
  const open = await journal(page);
  check(open?.step === 'densify' && open?.visible === true && open?.sessionId === id, `the run journal holds the running densify (${JSON.stringify(open)})`);
  check(!(await by(page, 'processing-preview').getAttribute('data-light')), 'the first densify is not in light mode');
  // the engine stored its first depth map (MockEngine: the resume marker in the scan record)
  await until(async () => (await storedStats(page, id))?.denseResumablePass === 1, 20000, 'a stored depth map');
  await killPage(page);
  const card = by(page, 'run-interrupted');
  await card.waitFor({ timeout: 15000 });
  const text = (await card.textContent()) ?? '';
  check(/Stone wall/.test(text) && /dense cloud/.test(text), `the card names the scan and the dense cloud ("${text}")`);
  check((await card.getAttribute('data-kind')) === 'killed' && /while the app was open/.test(text), 'stopped while the app was open (killed, not evicted)');
  await sleep(400);
  await shot(page, 'state-killed-dense');
  await audit(page, 'killed-dense card');
  await by(page, 'run-interrupted-resume').tap();
  await viewerLoaded(page);
  await by(page, 'dense-resume').waitFor({ state: 'visible', timeout: 10000 });
  check(true, 'Resume opens the viewer with its dense-resume card');
  const acked = await journal(page);
  check(acked?.kills?.densify === 1 && !acked?.interrupted && acked?.step === null, `the kill is counted and the card answered (${JSON.stringify(acked)})`);
  await by(page, 'dense-resume-button').tap();
  await page.waitForSelector('.processing', { timeout: 20000 });
  check((await page.locator('.proc-title').textContent()) === 'Resuming dense cloud', 'the densify resumes from the saved depth maps');
  check((await by(page, 'processing-preview').getAttribute('data-light')) === '1', 'the retry after the kill runs in light mode');
  // (not the viewer screen still fading out)
  await page.waitForSelector('.viewer-screen:not(.screen-exit)', { timeout: 120000 });
  await viewerLoaded(page);
  await sleep(500);
  check(await by(page, 'dense-resume').isHidden(), 'the resumed densify completed (no resume offer)');
  const done = await journal(page);
  check(done?.step === null && !done?.kills?.densify, `the clean run closed the journal and reset the kill count (${JSON.stringify(done)})`);
  await page.reload();
  await by(page, 'session-item').first().waitFor();
  await sleep(500);
  check((await by(page, 'run-interrupted').count()) === 0, 'no card once the resumed run finished');
  await context.close();
});

await scenario('killed-splat', async () => {
  const id = 'seed-killed-splat';
  const { page, context } = await newPage();
  await page.goto(`${base}?mock=1&mockSpeed=0.25`);
  await by(page, 'new-scan').waitFor();
  await seedSession(page, { id, name: 'Garden bench', mode: 'object', status: 'meshed', stats: { quality: 'fast', keyframes: 24 } });
  await page.reload();
  await by(page, 'session-item').first().tap();
  await viewerLoaded(page);
  // (the 3D view is created after "Loading model…" hides: its module loads first)
  await page.waitForFunction(() => !!globalThis.__sfmApp.debug.viewer, null, { timeout: 30000 });
  const full = await viewerStats(page);
  check(!!full && full.densePoints > 0 && full.triangles > 0, `the viewer shows the dense cloud and the mesh (${JSON.stringify(full)})`);
  await by(page, 'tool-build').tap();
  await by(page, 'build-splat').tap();
  await by(page, 'splat-progress').waitFor({ state: 'visible', timeout: 10000 });
  check((await journal(page))?.step === 'splat', 'the run journal holds the photoreal build');
  check(!(await page.locator('.viewer-screen').getAttribute('data-light')), 'the first build is not in light mode');
  check((await viewerStats(page))?.triangles === full?.triangles, 'the first build keeps the mesh');
  // a checkpoint is stored (MockEngine: the scan record's splatResumableIteration)
  await until(async () => (await storedStats(page, id))?.splatResumableIteration >= 500, 30000, 'a splat checkpoint');
  await killPage(page);
  const card = by(page, 'run-interrupted');
  await card.waitFor({ timeout: 15000 });
  const text = (await card.textContent()) ?? '';
  check(/Garden bench/.test(text) && /photoreal model/.test(text) && (await card.getAttribute('data-kind')) === 'killed', `the card names the scan and the photoreal build ("${text}")`);
  await sleep(400);
  await shot(page, 'state-killed-splat');
  await by(page, 'run-interrupted-resume').tap();
  await viewerLoaded(page);
  await by(page, 'splat-resume').waitFor({ state: 'visible', timeout: 10000 });
  check(true, 'Resume opens the viewer with its splat-resume card');
  await by(page, 'splat-resume-button').tap();
  await by(page, 'splat-progress').waitFor({ state: 'visible', timeout: 10000 });
  const firstText = await page.waitForFunction(() => {
    const t = document.querySelector('[data-testid="splat-progress-text"]')?.textContent ?? '';
    return /iterations/.test(t) ? t : null;
  }, null, { timeout: 30000 });
  const it0 = Number(((await firstText.jsonValue()) ?? '').replace(/,/g, '').match(/^(\d+)/)?.[1] ?? 0);
  check(it0 >= 500, `the resumed build continues from the checkpoint (first progress at iteration ${it0})`);
  check((await page.locator('.viewer-screen').getAttribute('data-light')) === '1', 'the rebuild after the kill runs in light mode');
  const light = await viewerStats(page);
  check(!!light && light.densePoints > 0 && light.densePoints <= 1_500_000 && light.triangles === 0, `light mode: ≤ 1.5 M points shown and no mesh (${JSON.stringify(light)})`);
  const segs = await page.locator('[data-testid="layer-switch"] .segment').evaluateAll((els) => els.map((e) => e.dataset.value));
  check(!segs.includes('mesh'), `no mesh layer offered during the light build (${segs})`);
  await by(page, 'splat-progress').waitFor({ state: 'hidden', timeout: 90000 });
  check(!(await page.locator('.viewer-screen').getAttribute('data-light')), 'light mode ends with the build');
  const after = await viewerStats(page);
  check(after?.densePoints === full?.densePoints && after?.triangles === full?.triangles, `the full cloud and the mesh are back (${JSON.stringify(after)})`);
  const done = await journal(page);
  check(done?.step === null && !done?.kills?.splat, `the finished build closed the journal and reset the kill count (${JSON.stringify(done)})`);
  await page.reload();
  await by(page, 'session-item').first().waitFor();
  await sleep(500);
  check((await by(page, 'run-interrupted').count()) === 0, 'no card once the resumed build finished');
  await context.close();
});

await scenario('evicted-refine', async () => {
  const { page, context } = await newPage();
  await page.goto(`${base}?mock=1&mockSpeed=0.3`);
  await by(page, 'new-scan').waitFor();
  await seedSession(page, { id: 'seed-evicted', name: 'Back yard', mode: 'scene', status: 'captured', stats: { quality: 'fast', keyframes: 20 } });
  await page.reload();
  await by(page, 'session-item').first().tap();
  await page.waitForSelector('[data-testid="stage-matching"][data-status="active"]', { timeout: 20000 });
  check((await journal(page))?.step === 'refine', 'the run journal holds the running refine');
  // the app goes to the background, and iOS evicts it there
  await setPageHidden(page, true);
  check((await journal(page))?.visible === false, 'the journal records that the page is hidden');
  await killPage(page);
  const card = by(page, 'run-interrupted');
  await card.waitFor({ timeout: 15000 });
  const text = (await card.textContent()) ?? '';
  check(/Back yard/.test(text) && /in the background/.test(text) && (await card.getAttribute('data-kind')) === 'evicted', `the card says the run stopped in the background ("${text}")`);
  await sleep(400);
  await shot(page, 'state-evicted-refine');
  await audit(page, 'evicted-refine card');
  await by(page, 'run-interrupted-resume').tap();
  await page.waitForSelector('.processing', { timeout: 20000 });
  check((await page.locator('.proc-title').textContent()) === 'Reconstructing', 'Resume reconstructs the captured scan');
  await page.waitForSelector('.viewer-screen:not(.screen-exit)', { timeout: 120000 });
  const done = await journal(page);
  check(done?.step === null && !done?.interrupted, `the refine finished and closed the journal (${JSON.stringify(done)})`);
  await context.close();
});

await scenario('killed-splat-early', async () => {
  const id = 'seed-killed-splat-early';
  const { page, context } = await newPage();
  // slow stages: the build is still preparing (no checkpoint) when the page is killed
  await page.goto(`${base}?mock=1&mockSpeed=0.25`);
  await by(page, 'new-scan').waitFor();
  await seedSession(page, { id, name: 'Old gate', mode: 'object', status: 'sparse', stats: { quality: 'fast', keyframes: 24 } });
  await page.reload();
  await by(page, 'session-item').first().tap();
  await viewerLoaded(page);
  await by(page, 'tool-build').tap();
  await by(page, 'build-splat').tap();
  await by(page, 'splat-progress').waitFor({ state: 'visible', timeout: 10000 });
  check((await journal(page))?.stage === 'splat_prepare', 'the run journal holds the preparing photoreal build');
  await killPage(page);
  check(typeof (await storedStats(page, id))?.splatResumableIteration !== 'number', 'no checkpoint was stored');
  const card = by(page, 'run-interrupted');
  await card.waitFor({ timeout: 15000 });
  const text = (await card.textContent()) ?? '';
  check(/or starts again if none was saved/.test(text) && !/lighter/.test(text), `the card promises no saved training and, for a sparse scan, no lighter view ("${text}")`);
  await by(page, 'run-interrupted-resume').tap();
  await viewerLoaded(page);
  const hint = page.locator('.toast:not(.hide)', { hasText: 'before any training was saved' });
  await hint.waitFor({ timeout: 10000 });
  check(await by(page, 'splat-resume').isHidden(), 'nothing to resume: no splat-resume card');
  await sleep(400);
  await shot(page, 'state-killed-splat-early');
  await hint.locator('.toast-action').tap();
  await by(page, 'sheet-build').waitFor({ timeout: 5000 });
  check((await by(page, 'build-splat').count()) === 1, 'the hint opens the Build sheet with the photoreal build');
  await context.close();
});

const notRun = SCENARIOS.filter((n) => want(n) && !ran.has(n));
check(!notRun.length, `every requested scenario ran (not run: ${notRun.join(', ') || 'none'})`);
const errs = consoleErrors.filter((e) => !/favicon/i.test(e));
check(errs.length === 0, `no console errors (got ${errs.length}: ${errs.slice(0, 5).join(' | ')})`);
console.log(failures.length ? `[states] FAILED (${failures.length})` : '[states] OK');
await close();
process.exit(failures.length ? 1 : 0);
