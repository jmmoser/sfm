#!/usr/bin/env node
/**
 * UI flow of the raw capture recorder with the mock engine at iPhone size (fake camera):
 *   Setup › Options › "Record raw capture" (+ VideoFrame grab) → capture until ≥ 6 keyframes →
 *   Finish → Viewer → Home → Diagnostics › Raw capture recordings (listed with its size) →
 *   Export: Prepare + Save every ZIP part (downloads) → e2e/import-recording.mjs into
 *   <out>/<name> (checks every frame's JPEG) → Delete (the list is empty again).
 * Asserts no console errors / page exceptions; screenshots go to e2e/out/ui/.
 * Replay the imported recording afterwards with `node e2e/run-e2e.mjs <name> --engine mock`
 * (import into e2e/data, the default) — the plumbing check of the verbatim replay + GT-free scoring.
 * Usage: node e2e/ui/recorder.mjs [--out DIR] [--name NAME] [--verbose]
 */
import { execFileSync } from 'node:child_process';
import { mkdir, rm, writeFile, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { launch, OUT, ROOT, shot, sleep } from './lib.mjs';

const argv = process.argv.slice(2);
const arg = (k, d) => (argv.includes(k) ? argv[argv.indexOf(k) + 1] : d);
const verbose = argv.includes('--verbose');
const outDir = arg('--out', join(ROOT, 'e2e', 'data'));
const name = arg('--name', 'ui-recording');
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

try {
  step('setup: record raw capture on');
  await page.goto(`${base}?mock=1&mockSpeed=4`);
  await byTest('new-scan').tap();
  await byTest('start-capture').waitFor();
  check((await byTest('toggle-record').isChecked()) === false, 'recording is off by default');
  await byTest('toggle-record').check();
  if (!(await byTest('toggle-videoframe').isChecked())) await byTest('toggle-videoframe').check();
  await page.evaluate(() => document.querySelector('[data-testid="toggle-record"]')?.scrollIntoView({ block: 'center' }));
  await sleep(300);
  await shot(page, 'rec-01-setup');

  step('capture');
  await byTest('start-capture').tap();
  await page.waitForFunction(() => document.querySelector('.capture')?.getAttribute('data-phase') === 'running', null, { timeout: 45000 });
  await page.waitForFunction(() => /Recording raw capture/.test(document.querySelector('.toast-host')?.textContent ?? ''), null, { timeout: 10000 }).catch(() => check(false, 'a toast says the capture is recorded'));
  await page.waitForFunction(() => Number(document.querySelector('[data-testid="kf-count"]')?.textContent) >= 6, null, { timeout: 90000 });

  step('finish → viewer → home');
  await byTest('finish').tap();
  await page.waitForSelector('.viewer-screen', { timeout: 180000 });
  await byTest('viewer-back').tap();
  await byTest('session-item').first().waitFor({ timeout: 10000 });

  step('diagnostics: the recording is listed');
  await byTest('open-diagnostics').tap();
  await byTest('recording-item').first().waitFor({ timeout: 30000 });
  const items = await byTest('recording-item').count();
  check(items === 1, `one recording listed (got ${items})`);
  const sub = await byTest('recording-item').first().locator('.export-sub').textContent();
  step(`listed: ${sub}`);
  const frames = Number(/(\d+) frames/.exec(sub ?? '')?.[1] ?? 0);
  check(frames >= 6, `the recording has frames (${frames})`);
  await byTest('recording-item').first().scrollIntoViewIfNeeded();
  await sleep(300);
  await shot(page, 'rec-02-diagnostics');

  step('export: prepare + save every part');
  await byTest('recording-export').first().tap();
  await byTest('sheet-recording-export').waitFor();
  await byTest('recording-part-1').waitFor({ timeout: 30000 });
  const parts = await page.locator('[data-testid^="recording-part-"]').count();
  const dlDir = join(OUT, 'recording-parts');
  await rm(dlDir, { recursive: true, force: true });
  await mkdir(dlDir, { recursive: true });
  const files = [];
  for (let p = 1; p <= parts; p++) {
    await byTest(`recording-prepare-${p}`).tap();
    await byTest(`recording-save-${p}`).waitFor({ timeout: 60000 });
    const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 15000 }), byTest(`recording-save-${p}`).tap()]);
    const file = join(dlDir, dl.suggestedFilename());
    await writeFile(file, await readFile(await dl.path()));
    files.push(file);
    step(`saved ${dl.suggestedFilename()}`);
  }
  await sleep(300);
  await shot(page, 'rec-03-export');
  check(files.every((f) => /^sfm-rec-[\w-]+\.zip$/.test(f.split('/').pop())), 'part file names');
  await page.locator('.sheet-close').last().tap();
  await page.waitForFunction(() => !document.querySelector('.sheet-wrap'), null, { timeout: 5000 });

  step(`import → ${join(outDir, name)}`);
  let imported = '';
  try {
    imported = execFileSync(process.execPath, [join(ROOT, 'e2e', 'import-recording.mjs'), ...files, '--out', outDir, '--name', name, '--force'], { encoding: 'utf8' });
  } catch (e) {
    check(false, `import failed: ${e.stdout ?? ''}${e.message}`);
  }
  process.stdout.write(imported);
  check(/PROBLEM/.test(imported) === false && imported.includes(`${frames} frames`), 'the import found every frame');

  step('delete');
  await byTest('recording-delete').first().tap();
  await byTest('dialog-confirm').tap();
  await byTest('recordings-empty').waitFor({ timeout: 10000 });
} catch (e) {
  failures.push(`exception: ${e?.message ?? e}`);
  console.error(e);
  await shot(page, 'rec-zz-failure').catch(() => {});
}

const errs = consoleErrors.filter((e) => !/favicon/i.test(e));
check(errs.length === 0, `no console errors (got ${errs.length}: ${errs.slice(0, 5).join(' | ')})`);
step(failures.length ? `FAILED (${failures.length})` : 'OK');
await close();
process.exit(failures.length ? 1 : 0);
