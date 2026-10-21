#!/usr/bin/env node
/**
 * On-device benchmark & self-test (Diagnostics → "Run benchmark") in headless Chromium with
 * SwiftShader WebGPU, at CI size (`?benchScale=small`; SwiftShader is a CPU emulation of a GPU,
 * its timings say nothing about an iPhone). Scenarios (each in a fresh context):
 *   run        full run: every test passes (GPU SIFT / matcher / PatchMatch parity and accuracy, BA),
 *              estimates per quality preset, "Copy report" puts text + JSON on the clipboard,
 *              the UI stays responsive while it runs, layout audit at 393 px and 375 px
 *   cancel     Cancel while running → "Cancelled" state, Run is available again
 *   no-webgpu  the page has no WebGPU → CPU-only results, nothing crashes
 *   loaded     a scan is open in the engine → Run asks first ("Free memory and run" releases the
 *              engine, then runs; "Run anyway"); the scan opens again afterwards
 * Prints the text report of the run. Screenshots: e2e/out/ui/bench-*.png.
 * Usage: node e2e/ui/bench.mjs [run|cancel|no-webgpu|loaded|closed ...] [--scale small|full] [--verbose]
 */
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { OUT, launch, layoutAudit, shot, sleep } from './lib.mjs';

const argv = process.argv.slice(2);
const scaleIdx = argv.indexOf('--scale');
const scaleEq = argv.find((a) => a.startsWith('--scale='));
const scale = scaleEq ? scaleEq.slice('--scale='.length) : scaleIdx >= 0 ? argv[scaleIdx + 1] : 'small';
const args = argv.filter((a, i) => !a.startsWith('--') && argv[i - 1] !== '--scale');
const SCENARIOS = ['run', 'cancel', 'no-webgpu', 'loaded', 'closed'];
// A misspelled name or option (or `full` without --scale) would run nothing, or the wrong thing, and still pass.
const unknown = args.filter((a) => !SCENARIOS.includes(a));
const badFlags = argv.filter((a) => a.startsWith('--') && a !== '--scale' && a !== '--verbose' && a !== scaleEq);
if (unknown.length || badFlags.length || !['small', 'full'].includes(scale)) {
  if (unknown.length) console.error(`[bench] unknown scenario(s): ${unknown.join(', ')}\n[bench] known: ${SCENARIOS.join(', ')}`);
  else if (badFlags.length) console.error(`[bench] unknown option(s): ${badFlags.join(', ')} (known: --scale small|full, --verbose)`);
  else console.error(`[bench] --scale must be small | full, got ${scale}`);
  process.exit(2);
}
const want = (n) => !args.length || args.includes(n);
const { newPage, close, consoleErrors, base } = await launch({ verbose: argv.includes('--verbose') });
const failures = [];
const check = (cond, msg) => {
  if (!cond) {
    failures.push(msg);
    console.error(`[bench] FAIL: ${msg}`);
  } else console.log(`[bench] ok: ${msg}`);
};
const by = (page, id) => page.locator(`[data-testid="${id}"]`);
const url = (extra = '') => `${base}?mock=1&screen=diagnostics&benchScale=${scale}${extra}`;
const TIMEOUT = scale === 'full' ? 3_600_000 : 900_000;

async function scenario(name, fn) {
  if (!SCENARIOS.includes(name)) throw new Error(`scenario ${name} is missing from SCENARIOS`);
  if (!want(name)) return;
  console.log(`[bench] — ${name}`);
  try {
    await fn();
  } catch (e) {
    failures.push(`${name}: ${e?.message ?? e}`);
    console.error(e);
  }
}

async function waitReport(page) {
  await page.waitForFunction(() => !!globalThis.__sfmApp?.debug.benchReport, null, { timeout: TIMEOUT, polling: 500 });
  return page.evaluate(() => globalThis.__sfmApp.debug.benchReport);
}

async function auditShot(page, name) {
  await sleep(350);
  const problems = await page.evaluate(layoutAudit);
  for (const p of problems) check(false, `${name}: ${p}`);
  await shot(page, name);
}

await scenario('run', async () => {
  const { page, context } = await newPage({ contextOptions: { permissions: ['clipboard-read', 'clipboard-write'] } });
  await page.goto(url());
  await by(page, 'bench-run').waitFor({ timeout: 30000 });
  await page.waitForSelector('.feature-grid', { timeout: 30000 });
  await auditShot(page, 'bench-idle');
  const t0 = Date.now();
  await by(page, 'bench-run').tap();
  await by(page, 'bench-cancel').waitFor({ timeout: 10000 });
  // The UI thread stays responsive while the worker computes: rAF keeps ticking.
  await page.waitForSelector('[data-testid="bench-sift"][data-status="running"]', { timeout: 30000 });
  const frames = await page.evaluate(
    () =>
      new Promise((res) => {
        let n = 0;
        const t = performance.now();
        const tick = () => {
          n++;
          if (performance.now() - t < 1000) requestAnimationFrame(tick);
          else res(n);
        };
        requestAnimationFrame(tick);
      }),
  );
  check(frames >= 10, `UI responsive during the benchmark (${frames} animation frames in 1 s)`);
  await sleep(1500);
  await auditShot(page, 'bench-running');
  const rep = await waitReport(page);
  console.log(`[bench] finished in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  await by(page, 'bench-copy').waitFor({ timeout: 10000 });
  check(rep.device.gpu.available === true, `WebGPU used (${rep.device.gpu.adapter ?? rep.device.gpu.reason})`);
  check(rep.tests.length === 4, '4 tests reported');
  for (const t of rep.tests) {
    check(t.status === 'pass', `${t.id}: ${t.status} — ${t.summary || t.error} ${t.checks.filter((c) => !c.pass).map((c) => `[${c.name}: ${c.detail}]`).join(' ')}`);
    check(t.backend === (t.id === 'ba' ? 'cpu' : 'gpu'), `${t.id} measured the ${t.id === 'ba' ? 'CPU (no GPU path)' : 'GPU'} path`);
  }
  check(rep.estimates.length === 3, 'estimates for fast / balanced / max');
  const chips = await page.$$eval('.bench-row[data-status]', (els) => els.map((e) => e.getAttribute('data-status')));
  check(chips.length === 4 && chips.every((s) => s === 'pass'), `pass chips shown (${chips.join(', ')})`);
  await page.evaluate(() => document.querySelector('.diagnostics')?.scrollTo(0, 0));
  await auditShot(page, 'bench-done');
  // Timing details of the SIFT test, then the estimates.
  await page.locator('[data-testid="bench-sift"] details summary').tap();
  await sleep(300);
  await page.locator('[data-testid="bench-sift"] details').scrollIntoViewIfNeeded();
  await auditShot(page, 'bench-done-timings');
  await by(page, 'bench-estimate-max').scrollIntoViewIfNeeded();
  await auditShot(page, 'bench-done-estimates');
  // Copy report → clipboard holds the text report and the JSON.
  await by(page, 'bench-copy').tap();
  await page.locator('.toast', { hasText: 'report copied' }).waitFor({ timeout: 5000 });
  const clip = await page.evaluate(() => navigator.clipboard.readText());
  check(clip.startsWith('3D Scan — benchmark & self-test'), 'clipboard: text report');
  const json = JSON.parse(clip.slice(clip.indexOf('--- JSON ---') + 12));
  check(json.version === 1 && json.tests.length === 4, 'clipboard: JSON report');
  const text = clip.slice(0, clip.indexOf('\n\n--- JSON ---'));
  console.log(`\n${text}\n`);
  await writeFile(join(OUT, 'bench-report.txt'), clip);
  // Leaving and re-opening Diagnostics keeps the results.
  await by(page, 'back').tap();
  await by(page, 'open-diagnostics').tap();
  // the previous Diagnostics screen fades out for 180 ms (still in the DOM, with its own verdict)
  await page.waitForFunction(() => document.querySelectorAll('[data-screen="diagnostics"]').length === 1, null, { timeout: 5000 });
  await by(page, 'bench-verdict').waitFor({ timeout: 10000 });
  const again = await page.$$eval('.bench-row[data-status]', (els) => els.map((e) => e.getAttribute('data-status')));
  check(again.length === 4 && again.every((s) => s === 'pass'), `results kept after re-opening Diagnostics (${again.join(', ')})`);
  // Smallest phone width.
  await page.setViewportSize({ width: 375, height: 667 });
  await page.evaluate(() => document.querySelector('.diagnostics')?.scrollTo(0, 0));
  await auditShot(page, 'bench-done-se');
  await context.close();
});

await scenario('cancel', async () => {
  const { page, context } = await newPage();
  await page.goto(url());
  await by(page, 'bench-run').waitFor({ timeout: 30000 });
  await by(page, 'bench-run').tap();
  await page.waitForSelector('[data-testid="bench-sift"][data-status="running"]', { timeout: 30000 });
  await sleep(1500);
  const t0 = Date.now();
  await by(page, 'bench-cancel').tap();
  const rep = await waitReport(page);
  const dt = Date.now() - t0;
  check(rep.cancelled === true, 'report marked cancelled');
  check(dt < 15000, `cancel takes effect quickly (${(dt / 1000).toFixed(1)} s)`);
  await by(page, 'bench-run').waitFor({ timeout: 5000 });
  check(!(await by(page, 'bench-run').isDisabled()), 'Run is available again after a cancel');
  const states = await page.$$eval('.bench-row[data-status]', (els) => els.map((e) => e.getAttribute('data-status')));
  check(states.includes('cancelled'), `cancelled tests shown (${states.join(', ')})`);
  await page.evaluate(() => document.querySelector('.diagnostics')?.scrollTo(0, 0));
  await auditShot(page, 'bench-cancelled');
  await context.close();
});

await scenario('no-webgpu', async () => {
  const { page, context } = await newPage({ initScript: () => Object.defineProperty(Navigator.prototype, 'gpu', { get: () => undefined, configurable: true }) });
  await page.goto(url());
  await by(page, 'bench-run').waitFor({ timeout: 30000 });
  await by(page, 'bench-run').tap();
  const rep = await waitReport(page);
  check(rep.device.gpu.available === false, `no GPU used (${rep.device.gpu.reason})`);
  for (const t of rep.tests) check(t.backend === 'cpu' && t.status === 'pass', `${t.id}: CPU ${t.status} — ${t.summary || t.error}`);
  check(rep.estimates.length === 3 && rep.estimates.every((e) => e.features === 'cpu' && e.dense === 'cpu'), 'CPU-only estimates');
  const verdictText = await by(page, 'bench-verdict').textContent();
  check(/No WebGPU — CPU results only/.test(verdictText ?? ''), 'the verdict says the results are CPU-only');
  await page.evaluate(() => document.querySelector('.diagnostics')?.scrollTo(0, 0));
  await auditShot(page, 'bench-no-webgpu');
  await context.close();
});

await scenario('loaded', async () => {
  const { page, context } = await newPage();
  await page.goto(`${base}?mock=1&benchScale=${scale}`);
  await by(page, 'new-scan').waitFor();
  await page.evaluate(async () => {
    const { SessionStore } = await import('/src/io/storage.ts');
    const s = await SessionStore.open();
    await s.createSession({ id: 'seed-bench', name: 'Loaded vase', mode: 'object', status: 'meshed', stats: { quality: 'fast', keyframes: 20 } });
  });
  await page.reload();
  // open the scan: the engine now holds it
  await by(page, 'session-item').first().tap();
  await page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 30000 });
  await by(page, 'viewer-back').tap();
  await by(page, 'open-diagnostics').tap();
  await by(page, 'bench-run').waitFor({ timeout: 30000 });
  // "Run anyway" asks first, then runs with the scan loaded
  await by(page, 'bench-run').tap();
  await by(page, 'bench-run-anyway').waitFor({ timeout: 5000 });
  check(await by(page, 'bench-release-run').isVisible(), 'a loaded scan: Run asks first (free memory / run anyway)');
  await auditShot(page, 'bench-loaded-ask');
  await by(page, 'bench-run-anyway').tap();
  await by(page, 'bench-cancel').waitFor({ timeout: 10000 });
  check((await page.evaluate(() => globalThis.__sfmApp.engine.status.state)) === 'ready', '"Run anyway" keeps the scan loaded');
  await by(page, 'bench-cancel').tap();
  await waitReport(page);
  await by(page, 'bench-run').waitFor({ timeout: 10000 });
  // "Free memory and run": the engine is released before the benchmark starts
  await page.evaluate(() => (globalThis.__sfmApp.debug.benchReport = undefined));
  await by(page, 'bench-run').tap();
  await by(page, 'bench-release-run').tap();
  await by(page, 'bench-cancel').waitFor({ timeout: 10000 });
  const st = await page.evaluate(() => ({ state: globalThis.__sfmApp.engine.status.state, loaded: globalThis.__sfmApp.loadedSessionId }));
  check(st.state === 'idle' && st.loaded === null, `"Free memory and run" released the engine (${JSON.stringify(st)})`);
  await by(page, 'bench-cancel').tap();
  await waitReport(page);
  // released and nothing loaded: Run starts right away (no question)
  await by(page, 'bench-run').waitFor({ timeout: 10000 });
  await page.evaluate(() => (globalThis.__sfmApp.debug.benchReport = undefined));
  await by(page, 'bench-run').tap();
  await by(page, 'bench-cancel').waitFor({ timeout: 10000 });
  check((await page.locator('[data-testid="bench-run-anyway"]').count()) === 0, 'no question once nothing is loaded');
  await by(page, 'bench-cancel').tap();
  await waitReport(page);
  // the scan opens again (loaded from storage)
  await by(page, 'back').tap();
  await by(page, 'session-item').first().tap();
  await page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 30000 });
  check(/Loaded vase/.test((await page.locator('.viewer-title').textContent()) ?? ''), 'the scan opens again after the release');
  await context.close();
});

await scenario('closed', async () => {
  const { page, context } = await newPage();
  await page.goto(`${base}?mock=1&benchScale=${scale}`);
  await by(page, 'new-scan').waitFor();
  await page.evaluate(async () => {
    const { SessionStore } = await import('/src/io/storage.ts');
    const s = await SessionStore.open();
    await s.createSession({ id: 'seed-closed', name: 'Deleted vase', mode: 'object', status: 'meshed', stats: { quality: 'fast', keyframes: 20 } });
  });
  await page.reload();
  // open the scan, then delete it: the engine lets go of it (closed) but its worker stays
  await by(page, 'session-item').first().tap();
  await page.waitForFunction(() => document.querySelector('.viewer-loading')?.hasAttribute('hidden'), null, { timeout: 30000 });
  await by(page, 'viewer-back').tap();
  await page.evaluate(() => globalThis.__sfmApp.deleteSession('seed-closed'));
  const before = await page.evaluate(() => ({ ...globalThis.__sfmApp.engine.status, loaded: globalThis.__sfmApp.loadedSessionId }));
  check(before.state === 'idle' && before.reason === 'closed' && before.loaded === null, `deleting the open scan closes it (${JSON.stringify(before)})`);
  await by(page, 'open-diagnostics').tap();
  await by(page, 'bench-run').waitFor({ timeout: 30000 });
  // nothing loaded: no question, but what the closed engine still parks is freed first
  await by(page, 'bench-run').tap();
  await by(page, 'bench-cancel').waitFor({ timeout: 10000 });
  check((await page.locator('[data-testid="bench-run-anyway"]').count()) === 0, 'no question after a delete');
  const st = await page.evaluate(() => globalThis.__sfmApp.engine.status);
  check(st.state === 'idle' && st.reason === 'released', `the closed engine was released before the benchmark (${JSON.stringify(st)})`);
  await by(page, 'bench-cancel').tap();
  await waitReport(page);
  await context.close();
});

const errs = consoleErrors.filter((e) => !/favicon/i.test(e));
check(errs.length === 0, `no console errors (got ${errs.length}: ${errs.slice(0, 5).join(' | ')})`);
console.log(failures.length ? `[bench] FAILED (${failures.length})` : '[bench] OK');
await close();
process.exit(failures.length ? 1 : 0);
