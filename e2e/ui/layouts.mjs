#!/usr/bin/env node
/**
 * Layout checks of every screen at several device sizes (mock engine, fake camera), with
 * screenshots in e2e/out/ui/layout-<device>-<screen>.png:
 *   se          iPhone SE 375×667 @2x (smallest supported phone)
 *   landscape   iPhone 852×393 @3x
 *   ipad        iPad 1180×820 @2x landscape
 * On every screen (and open sheet) it asserts:
 *   - no horizontal overflow of the page or the screen;
 *   - every visible control lies horizontally inside the viewport;
 *   - every visible control has a touch target ≥ 44×44 CSS px (switches: their row);
 *   - no clipped single-line text (overflow hidden without an ellipsis).
 * Sessions are seeded directly into IndexedDB (one meshed scan with a very long name, one
 * interrupted capture), so the Home list and the Viewer are populated without a capture.
 * Usage: node e2e/ui/layouts.mjs [se|landscape|ipad ...] [--verbose]
 */
import { launch, shot, sleep } from './lib.mjs';

const DEVICES = {
  se: { viewport: { width: 375, height: 667 }, deviceScaleFactor: 2 },
  landscape: { viewport: { width: 852, height: 393 }, deviceScaleFactor: 3 },
  ipad: { viewport: { width: 1180, height: 820 }, deviceScaleFactor: 2 },
};
const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const devices = args.length ? args : Object.keys(DEVICES);
const { newPage, close, consoleErrors, base } = await launch({ verbose: process.argv.includes('--verbose') });
const failures = [];
const check = (cond, msg) => {
  if (!cond) {
    failures.push(msg);
    console.error(`[layouts] FAIL: ${msg}`);
  }
};
const by = (page, id) => page.locator(`[data-testid="${id}"]`);

/** Runs in the page: layout problems of the current screen + open overlays. */
function audit() {
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

async function auditAndShot(page, device, name) {
  await sleep(450); // sheet / screen transitions
  const problems = await page.evaluate(audit);
  for (const p of problems) check(false, `${device}/${name}: ${p}`);
  await shot(page, `layout-${device}-${name}`);
}

async function seed(page) {
  await page.evaluate(async () => {
    const { SessionStore } = await import('/src/io/storage.ts');
    const s = await SessionStore.open();
    await s.createSession({
      id: 'seed-meshed',
      name: 'Grandmother’s ceramic vase on the living-room table (second try, evening light)',
      mode: 'object',
      status: 'meshed',
      stats: { quality: 'balanced', keyframes: 42 },
    });
    await s.createSession({ id: 'seed-incomplete', name: 'Backyard', mode: 'scene', status: 'capturing', stats: { quality: 'fast', keyframes: 2 } });
  });
}

async function closeSheet(page) {
  await page.locator('.sheet-close').last().tap();
  await page.waitForFunction(() => !document.querySelector('.sheet-wrap'), null, { timeout: 5000 });
}

for (const device of devices) {
  const dev = DEVICES[device];
  if (!dev) throw new Error(`unknown device ${device}`);
  console.log(`[layouts] — ${device} ${dev.viewport.width}×${dev.viewport.height}`);
  const { page, context } = await newPage({ contextOptions: dev });
  try {
    await page.goto(`${base}?mock=1&mockSpeed=4`);
    await by(page, 'new-scan').waitFor();
    await seed(page);
    await page.reload();
    await by(page, 'session-item').first().waitFor({ timeout: 10000 });
    await auditAndShot(page, device, 'home');
    await page.evaluate(() => document.querySelector('.home')?.scrollTo(0, 4000));
    await auditAndShot(page, device, 'home-bottom');

    await by(page, 'session-more').first().tap();
    await auditAndShot(page, device, 'session-actions');
    await closeSheet(page);

    await by(page, 'new-scan').tap();
    await by(page, 'start-capture').waitFor();
    await auditAndShot(page, device, 'setup');
    await page.evaluate(() => document.querySelector('.setup')?.scrollTo(0, 4000));
    await auditAndShot(page, device, 'setup-bottom');
    await by(page, 'back').tap();

    // Viewer of the seeded meshed scan (the newest session is listed first).
    await page.locator('[data-testid="session-item"]', { hasText: 'ceramic vase' }).tap();
    await page.waitForSelector('.viewer-screen');
    await page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 30000 });
    await auditAndShot(page, device, 'viewer');
    await by(page, 'tool-export').tap();
    await by(page, 'sheet-export').waitFor();
    await auditAndShot(page, device, 'export');
    await closeSheet(page);
    await by(page, 'viewer-info').tap();
    await by(page, 'sheet-info').waitFor();
    await auditAndShot(page, device, 'info');
    await closeSheet(page);
    await by(page, 'tool-measure').tap();
    await auditAndShot(page, device, 'measure');
    await by(page, 'tool-measure').tap();
    await by(page, 'viewer-back').tap();

    // Capture → processing.
    await by(page, 'new-scan').tap();
    await by(page, 'start-capture').tap();
    await page.waitForFunction(() => document.querySelector('.capture')?.getAttribute('data-phase') === 'running', null, { timeout: 45000 });
    await page.waitForFunction(() => Number(document.querySelector('[data-testid="kf-count"]')?.textContent) >= 5, null, { timeout: 90000 });
    await page.evaluate(() => globalThis.__sfmApp?.debug.holdFrames?.(true));
    await auditAndShot(page, device, 'capture');
    await by(page, 'mini-view').tap();
    await auditAndShot(page, device, 'capture-3d');
    await page.locator('.mini-collapse').tap();
    await by(page, 'pause').tap();
    await auditAndShot(page, device, 'capture-paused');
    await page.evaluate(() => globalThis.__sfmApp?.debug.holdFrames?.(false));
    await by(page, 'pause').tap();
    await page.waitForFunction(() => document.querySelector('.capture')?.getAttribute('data-phase') === 'running');
    await by(page, 'finish').tap();
    await page.waitForSelector('.processing', { timeout: 20000 });
    await auditAndShot(page, device, 'processing');
    await page.waitForSelector('.viewer-screen', { timeout: 120000 });
    await by(page, 'viewer-back').tap();

    await by(page, 'open-diagnostics').tap();
    await page.waitForSelector('.feature-grid', { timeout: 30000 });
    await auditAndShot(page, device, 'diagnostics');
  } catch (e) {
    failures.push(`${device}: ${e?.message ?? e}`);
    console.error(e);
    await shot(page, `layout-${device}-zz-failure`).catch(() => {});
  }
  await context.close();

  // Recovery card (camera denied) — must stay reachable on short screens.
  const denied = await newPage({
    contextOptions: dev,
    initScript: () => {
      const md = navigator.mediaDevices;
      if (md) md.getUserMedia = () => Promise.reject(new DOMException('Permission denied', 'NotAllowedError'));
    },
  });
  try {
    await denied.page.goto(`${base}?mock=1`);
    await by(denied.page, 'new-scan').tap();
    await by(denied.page, 'start-capture').tap();
    await denied.page.waitForSelector('[data-testid="capture-error"]', { timeout: 30000 });
    const back = by(denied.page, 'capture-back');
    await back.scrollIntoViewIfNeeded();
    const box = await back.boundingBox();
    const vh = dev.viewport.height;
    check(!!box && box.y >= 0 && box.y + box.height <= vh, `${device}: "Back to home" of the recovery card is reachable (y ${box?.y?.toFixed(0)}, h ${box?.height})`);
    await auditAndShot(denied.page, device, 'camera-denied');
    await back.tap();
    await by(denied.page, 'new-scan').waitFor({ timeout: 10000 });
  } catch (e) {
    failures.push(`${device} camera-denied: ${e?.message ?? e}`);
    console.error(e);
  }
  await denied.context.close();
}

const errs = consoleErrors.filter((e) => !/favicon/i.test(e));
check(errs.length === 0, `no console errors (got ${errs.length}: ${errs.slice(0, 5).join(' | ')})`);
console.log(failures.length ? `[layouts] FAILED (${failures.length})` : '[layouts] OK');
await close();
process.exit(failures.length ? 1 : 0);
