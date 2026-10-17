#!/usr/bin/env node
/**
 * UI smoke test of the whole mock flow at iPhone size (393×852 @3x, touch):
 *   Home → Setup → Start (fake camera) → Capture (wait for keyframes) → Finish → Processing →
 *   Viewer (info, display, measure, build mesh) → Export (prepare + save → download) → Home → Diagnostics
 * Asserts no console errors / page exceptions and writes screenshots of every screen to
 * e2e/out/ui/. Usage: node e2e/ui/smoke.mjs [--verbose] [--keep-open]
 */
import { readFile } from 'node:fs/promises';
import { launch, shot, sleep } from './lib.mjs';

const verbose = process.argv.includes('--verbose');
const { page, close, consoleErrors, base } = await launch({ verbose });
const t0 = Date.now();
const step = (m) => console.log(`[ui +${((Date.now() - t0) / 1000).toFixed(1)}s] ${m}`);
const failures = [];
const check = (cond, msg) => {
  if (!cond) {
    failures.push(msg);
    console.error(`[ui] FAIL: ${msg}`);
  }
};

const byTest = (id) => page.locator(`[data-testid="${id}"]`);

/** Screenshot of the running capture screen (frame delivery held: see App.debug). */
async function captureShot(name) {
  await page.evaluate(() => globalThis.__sfmApp?.debug.holdFrames?.(true));
  await sleep(700);
  await shot(page, name);
  await page.evaluate(() => globalThis.__sfmApp?.debug.holdFrames?.(false));
}

async function closeSheet() {
  await page.locator('.sheet-close').last().tap();
  await page.waitForFunction(() => !document.querySelector('.sheet-wrap'), null, { timeout: 5000 });
}

try {
  step('home');
  await page.goto(`${base}?mock=1&mockSpeed=2`);
  await byTest('new-scan').waitFor();
  await page.waitForSelector('.empty-state', { timeout: 10000 });
  await sleep(500);
  await shot(page, '01-home');

  step('setup');
  await byTest('new-scan').tap();
  await byTest('start-capture').waitFor();
  await sleep(400);
  await shot(page, '02-setup');
  await byTest('mode-scene').tap();
  check((await byTest('toggle-gps').isChecked()) === true, 'GPS defaults on for large area');
  await byTest('mode-object').tap();
  check((await byTest('toggle-gps').isChecked()) === false, 'GPS defaults off for objects');
  await sleep(300);
  await shot(page, '02b-setup-object');

  step('start capture');
  await byTest('start-capture').tap();
  await page.waitForSelector('.capture');
  await page.waitForFunction(() => document.querySelector('.capture')?.getAttribute('data-phase') === 'running', null, { timeout: 45000 });
  step('capturing');
  await page.waitForFunction(() => Number(document.querySelector('[data-testid="kf-count"]')?.textContent) >= 3, null, { timeout: 60000 });
  await sleep(400);
  await captureShot('03-capture-early');
  await page.waitForFunction(() => Number(document.querySelector('[data-testid="kf-count"]')?.textContent) >= 8, null, { timeout: 90000 });
  // Overlay keypoints are drawn on the canvas.
  const overlayPixels = await page.evaluate(() => {
    const c = document.querySelector('.capture-overlay');
    if (!c) return 0;
    const ctx = c.getContext('2d');
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 3; i < d.length; i += 4 * 7) if (d[i] > 0) n++;
    return n;
  });
  check(overlayPixels > 0, 'keypoint overlay drawn');
  await captureShot('04-capture');
  check(!(await byTest('finish').isDisabled()), 'finish enabled after ≥ 5 keyframes');

  step('pause / resume');
  await byTest('pause').tap();
  await page.waitForFunction(() => document.querySelector('.capture')?.getAttribute('data-phase') === 'paused');
  await shot(page, '05-capture-paused');
  await byTest('pause').tap();
  await page.waitForFunction(() => document.querySelector('.capture')?.getAttribute('data-phase') === 'running');

  step('expand live 3D');
  await byTest('mini-view').tap();
  await sleep(1200);
  await captureShot('06-capture-3d');
  await page.locator('.mini-collapse').tap();
  await sleep(500);

  step('finish');
  await byTest('finish').tap();
  await page.waitForSelector('.processing', { timeout: 20000 });
  await page.waitForSelector('[data-testid="stage-matching"][data-status="active"], [data-testid="stage-reconstruction"][data-status="active"]', { timeout: 20000 });
  await sleep(1200);
  await shot(page, '07-processing');

  step('viewer');
  await page.waitForSelector('.viewer-screen', { timeout: 120000 });
  await page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 30000 });
  await sleep(2000);
  await shot(page, '09-viewer');

  step('info sheet');
  await byTest('viewer-info').tap();
  await byTest('sheet-info').waitFor();
  await sleep(500);
  await shot(page, '10-info');
  await closeSheet();

  step('display sheet');
  await byTest('tool-view').tap();
  await byTest('sheet-display').waitFor();
  await sleep(400);
  await shot(page, '11-display');
  await byTest('display-edl').locator('xpath=..').tap();
  await closeSheet();

  step('measure');
  await byTest('tool-measure').tap();
  const box = await page.locator('.viewer-canvas').boundingBox();
  await page.touchscreen.tap(box.x + box.width * 0.45, box.y + box.height * 0.5);
  await sleep(400);
  await page.touchscreen.tap(box.x + box.width * 0.58, box.y + box.height * 0.56);
  await sleep(800);
  const measured = (await byTest('measure-value').textContent())?.trim() ?? '';
  step(`measured: "${measured}"`);
  await shot(page, '12-measure');
  await byTest('tool-measure').tap();

  step('build mesh');
  await byTest('tool-build').tap();
  await byTest('sheet-build').waitFor();
  await sleep(400);
  await shot(page, '13-build');
  await byTest('build-mesh').tap();
  await page.waitForSelector('.processing', { timeout: 20000 });
  await page.waitForSelector('[data-testid="stage-meshing"]:is([data-status="active"], [data-status="done"])', { timeout: 30000 });
  await shot(page, '14-processing-mesh');
  await page.waitForSelector('.viewer-screen', { timeout: 120000 });
  await page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 30000 });
  await sleep(2000);
  await shot(page, '15-viewer-mesh');

  step('rebuild at another quality');
  await byTest('tool-build').tap();
  await byTest('sheet-build').waitFor();
  check(!(await page.locator('[data-testid="build-mesh"]').count()), 'the mesh shows Built at its own quality');
  const other = (await page.locator('[data-testid="sheet-build"] .segment[aria-checked="true"]').getAttribute('data-value')) === 'max' ? 'fast' : 'max';
  await page.locator(`[data-testid="sheet-build"] .segment[data-value="${other}"]`).tap();
  await byTest('build-mesh').waitFor({ timeout: 2000 });
  check(/Rebuild/.test((await byTest('build-mesh').textContent()) ?? ''), 'another quality offers Rebuild');
  await sleep(300);
  await shot(page, '15b-rebuild');
  await page.keyboard.press('Escape');
  await byTest('sheet-build').waitFor({ state: 'detached' });

  step('export');
  await byTest('tool-export').tap();
  await byTest('sheet-export').waitFor();
  await sleep(300);
  await shot(page, '16-export');
  await byTest('export-prepare-ply-points').tap();
  await byTest('export-save-ply-points').waitFor({ timeout: 30000 });
  await byTest('export-prepare-glb').tap();
  await byTest('export-save-glb').waitFor({ timeout: 30000 });
  await sleep(300);
  await shot(page, '17-export-ready');
  const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 15000 }), byTest('export-save-ply-points').tap()]);
  const name = dl.suggestedFilename();
  const bytes = await readFile(await dl.path());
  step(`downloaded ${name} (${bytes.length} bytes)`);
  check(name.endsWith('.ply'), 'PLY filename');
  check(bytes.subarray(0, 3).toString() === 'ply', 'PLY magic');
  const [dl2] = await Promise.all([page.waitForEvent('download', { timeout: 15000 }), byTest('export-save-glb').tap()]);
  const glb = await readFile(await dl2.path());
  check(glb.subarray(0, 4).toString() === 'glTF', 'GLB magic');
  await byTest('export-prepare-colmap-zip').tap();
  await byTest('export-save-colmap-zip').waitFor({ timeout: 30000 });
  const [dl3] = await Promise.all([page.waitForEvent('download', { timeout: 15000 }), byTest('export-save-colmap-zip').tap()]);
  const zip = await readFile(await dl3.path());
  check(zip.readUInt32LE(0) === 0x04034b50, 'ZIP magic');
  await closeSheet();

  step('home with session');
  await byTest('viewer-back').tap();
  await byTest('session-item').first().waitFor({ timeout: 10000 });
  await sleep(800);
  await shot(page, '18-home-sessions');
  await byTest('session-more').first().tap();
  await sleep(400);
  await shot(page, '19-session-actions');
  await page.locator('.sheet-close').last().tap();
  await sleep(400);

  step('reopen session');
  await byTest('session-item').first().tap();
  await page.waitForSelector('.viewer-screen', { timeout: 30000 });
  await page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 30000 });
  await sleep(1000);
  await byTest('viewer-back').tap();
  await byTest('session-item').first().waitFor();

  step('diagnostics');
  await byTest('open-diagnostics').tap();
  await page.waitForSelector('.feature-grid', { timeout: 30000 });
  await sleep(500);
  await shot(page, '20-diagnostics');
  await page.evaluate(() => document.querySelector('.diagnostics')?.scrollTo(0, 900));
  await sleep(300);
  await shot(page, '21-diagnostics-2');
  await byTest('back').tap();
  await byTest('new-scan').waitFor();

  step('rename + delete');
  await byTest('session-more').first().tap();
  await byTest('session-rename').tap();
  await page.locator('.text-input').fill('Ceramic vase');
  await sleep(200);
  await shot(page, '22-rename');
  await byTest('dialog-confirm').tap();
  await page.waitForFunction(() => document.querySelector('.session-name')?.textContent === 'Ceramic vase', null, { timeout: 5000 });
  await byTest('session-more').first().tap();
  await byTest('session-delete').tap();
  await sleep(300);
  await shot(page, '23-delete-confirm');
  await byTest('dialog-confirm').tap();
  await page.waitForSelector('.empty-state', { timeout: 5000 });
} catch (e) {
  failures.push(`exception: ${e?.message ?? e}`);
  console.error(e);
  await shot(page, 'zz-failure').catch(() => {});
}

const errs = consoleErrors.filter((e) => !/favicon/i.test(e));
check(errs.length === 0, `no console errors (got ${errs.length}: ${errs.slice(0, 5).join(' | ')})`);
step(failures.length ? `FAILED (${failures.length})` : 'OK');
if (!process.argv.includes('--keep-open')) await close();
process.exit(failures.length ? 1 : 0);
