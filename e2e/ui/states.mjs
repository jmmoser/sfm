#!/usr/bin/env node
/**
 * UI edge states at iPhone size, each in a fresh context (isolated storage), with screenshots in
 * e2e/out/ui/state-*.png:
 *   engine-missing   real EngineClient + placeholder worker → clear "engine unavailable" card
 *   install-hint     iOS Safari UA, not standalone → "Install to Home Screen" card
 *   cpu-mode         no WebGPU → "CPU mode — slower" banner
 *   lockdown         no WebGPU, camera, IndexedDB → Lockdown Mode explanation, New scan disabled
 *   camera-denied    getUserMedia NotAllowedError → recovery card
 *   landscape        capture UI in landscape (852×393)
 *   interrupted      page hidden during capture → paused, resume needs a tap
 *   processing-error refine fails → error banner with retry
 *   storage-full-start   IndexedDB quota exceeded → the scan is blocked with an explanation
 *   storage-full-capture keyframes cannot be stored → capture pauses, "Finish scan" works
 *   engine-dead      EngineClient gave up (respawn cap) → error card, no toast storm
 *   export-reopen    export finishes while its sheet is closed → the reopened sheet shows Save
 *   open-other-session processing a stored scan after viewing another: no stale name / stages
 *   focus            keyboard navigation moves focus to the new screen's heading
 * Usage: node e2e/ui/states.mjs [name ...] [--verbose]
 */
import { launch, shot, sleep } from './lib.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const want = (n) => !args.length || args.includes(n);
const { newPage, close, consoleErrors, base } = await launch({ verbose: process.argv.includes('--verbose') });
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
  if (!want(name)) return;
  console.log(`[states] — ${name}`);
  try {
    await fn();
  } catch (e) {
    failures.push(`${name}: ${e?.message ?? e}`);
    console.error(e);
  }
}

await scenario('engine-missing', async () => {
  const { page, context } = await newPage();
  await startCapture(page, base);
  await page.waitForSelector('[data-testid="capture-error"]', { timeout: 60000 });
  const text = await page.locator('[data-testid="capture-error"]').textContent();
  check(/engine not implemented yet/i.test(text ?? ''), 'placeholder worker error is shown');
  await shot(page, 'state-engine-missing');
  await by(page, 'capture-back').tap();
  await by(page, 'new-scan').waitFor();
  check(true, 'back to home from the error card');
  await context.close();
  // The placeholder worker answers with an error: expected console noise is none.
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
  await page.waitForFunction(() => document.querySelector('.processing .topbar-title')?.textContent === 'Beta yard', null, { timeout: 15000 });
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

const errs = consoleErrors.filter((e) => !/favicon/i.test(e));
check(errs.length === 0, `no console errors (got ${errs.length}: ${errs.slice(0, 5).join(' | ')})`);
console.log(failures.length ? `[states] FAILED (${failures.length})` : '[states] OK');
await close();
process.exit(failures.length ? 1 : 0);
