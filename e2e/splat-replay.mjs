#!/usr/bin/env node
/**
 * Splat-stage experiments on a frozen engine session: re-runs ONLY the splat stage (the engine's
 * `runSplat`, src/engine/splat.ts) on the inputs of one e2e run, once per variant, and scores each
 * with e2e/nvs.mjs on the held-out frames and on off-trajectory test datasets.
 *
 *   # once per scene: an e2e run that dumps its session after mesh (keyframes, sparse model,
 *   # dense cloud, filtered depth maps) to <run>/session/
 *   E2E_GPU=host node e2e/run-e2e.mjs object-real --engine worker --splat --tag gr-base \
 *     --set persist=true --set dumpSession=true
 *   # then any number of variants (trainer overrides = SplatTrainerOptions fields)
 *   E2E_GPU=host node e2e/splat-replay.mjs e2e/out/e2e/object-real-gr-base \
 *     --variant base='{}' --variant geo='{"geometry":{"depth":[0.5,0.05]}}' \
 *     [--tests held,sim3,object-real-up12,…] [--out DIR] [--skip-nvs] [--nvs-only] [--repeat N]
 *     [--splat-previews]
 *   (--nvs-only: no training; scores the variants' existing splats.ply on the tests missing from
 *   summary.json — resumes an interrupted run; --splat-previews: every variant of this invocation
 *   trains with SplatOptions.preview, as the app does, and records the snapshots' arrival times and
 *   PSNR. The preview cost is the stage-time change between two sequential invocations into the
 *   same --out, without and with it, under different variant names, e.g.
 *     … --variant off='{}' --skip-nvs   then   … --variant on='{}' --splat-previews --skip-nvs)
 *
 * Writes <out>/<variant>/{result.json (copied from the run), splats.ply, replay.json,
 * nvs-<test>/nvs.json} and <out>/summary.json; prints a table of full-frame PSNR per test set and
 * the splat stage times. Default --out: <run>-replay; default --tests: held, sim3 and every
 * e2e/data/<dataset>-* directory (the off-trajectory sets of the run's dataset).
 */
import { access, copyFile, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { basename, dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { startHarnessBrowser } from './lib/e2e-browser.mjs';
import { parseNvsArgs } from './lib/nvs-args.mjs';
import { table } from './lib/e2e-checks.mjs';
import { runNvs } from './nvs.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const exists = (p) => access(p).then(() => true, () => false);
const f2 = (v, d = 2) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : '—');

/** Preview timings of a replay result (--splat-previews): count, first arrival, largest gap while training (s), PSNR first → last. */
function previewSummary(r) {
  const pv = r.previews ?? [];
  if (!pv.length) return null;
  // (gaps between the snapshots of the training, not the one before the view uploads)
  const train = pv.filter((p) => p.iteration > 0);
  const gaps = train.slice(1).map((p, i) => p.ms - train[i].ms);
  return {
    count: pv.length,
    firstS: pv[0].ms / 1000,
    firstTrainingS: r.firstTrainingMs != null ? r.firstTrainingMs / 1000 : null,
    maxGapS: gaps.length ? Math.max(...gaps) / 1000 : null,
    psnrFirst: pv[0].psnr,
    psnrLast: pv[pv.length - 1].psnr,
    psnrFinal: r.stats?.splatPreviewFinalPsnr ?? null,
  };
}

function parse(argv) {
  const o = { run: null, variants: [], tests: null, out: null, skipNvs: false, repeat: 1, verbose: false, ssim: true, previews: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === '--variant') {
      const v = next();
      const k = v.indexOf('=');
      if (k < 1) throw new Error(`--variant name=json, got ${v}`);
      o.variants.push({ name: v.slice(0, k), trainer: JSON.parse(v.slice(k + 1)) });
    } else if (a === '--variants') {
      o.variantsFile = next();
    } else if (a === '--tests') o.tests = next().split(',').filter(Boolean);
    else if (a === '--out') o.out = next();
    else if (a === '--skip-nvs') o.skipNvs = true;
    else if (a === '--nvs-only') o.nvsOnly = true;
    else if (a === '--no-ssim') o.ssim = false;
    else if (a === '--repeat') o.repeat = Math.max(1, Number(next()) | 0);
    else if (a === '--splat-previews') o.previews = true;
    else if (a === '--verbose') o.verbose = true;
    else if (a.startsWith('--')) throw new Error(`unknown option ${a}`);
    else o.run = a;
  }
  if (!o.run) throw new Error('usage: node e2e/splat-replay.mjs <runDir> --variant name=json … (see the file header)');
  return o;
}

const o = parse(process.argv.slice(2));
const runDir = resolve(o.run);
if (o.variantsFile) for (const [name, trainer] of Object.entries(JSON.parse(await readFile(o.variantsFile, 'utf8')))) o.variants.push({ name, trainer });
if (!o.variants.length) o.variants.push({ name: 'base', trainer: {} });
const result = JSON.parse(await readFile(join(runDir, 'result.json'), 'utf8'));
const dataset = result.config?.dataset ?? basename(runDir).replace(/-[^-]+$/, '');
const mode = result.config?.mode ?? 'object';
const quality = result.config?.quality ?? 'fast';
if (!(await exists(join(runDir, 'session', 'manifest.json')))) throw new Error(`${runDir}/session/manifest.json missing (run e2e/run-e2e.mjs with --set persist=true --set dumpSession=true)`);
const relRun = relative(ROOT, runDir);
if (relRun.startsWith('..')) throw new Error('the run directory must be inside the repository (served by Vite)');
let tests = o.tests;
if (!tests) {
  const all = await readdir(join(ROOT, 'e2e', 'data'));
  tests = ['held', 'sim3', ...all.filter((d) => d.startsWith(`${dataset}-`)).sort()];
}
const outRoot = resolve(o.out ?? `${runDir}-replay`);
await mkdir(outRoot, { recursive: true });
console.log(`[replay] ${dataset} (${mode}, ${quality}) from ${runDir}; variants ${o.variants.map((v) => v.name).join(', ')}; tests ${tests.join(', ')} → ${outRoot}`);

const summary = (await exists(join(outRoot, 'summary.json'))) ? JSON.parse(await readFile(join(outRoot, 'summary.json'), 'utf8')) : {};
const browser = o.nvsOnly ? null : await startHarnessBrowser({ root: ROOT, verbose: o.verbose });
let page = null;
let closePage = null;
try {
  for (const v of browser ? o.variants : []) {
    for (let rep = 0; rep < o.repeat; rep++) {
      const name = o.repeat > 1 ? `${v.name}-r${rep + 1}` : v.name;
      const vdir = join(outRoot, name);
      await mkdir(vdir, { recursive: true });
      if (!page) {
        const p = await browser.openPage({
          outDir: outRoot,
          onLog: (m) => {
            if (o.verbose || !/uploading views|splat_prepare|splat_training/.test(m)) console.log(`[replay:page] ${m}`);
          },
          onConsole: (type, text) => {
            if (type === 'error' || type === 'warning' || o.verbose) console.log(`[replay:page:${type}] ${text}`);
          },
        });
        page = p.page;
        closePage = p.close;
      }
      console.log(`[replay] === ${name}: trainer ${JSON.stringify(v.trainer)}`);
      const t0 = Date.now();
      // a browser killed mid-run leaves page.evaluate pending: time out, reopen and retry once
      let r = null;
      for (let attempt = 0; attempt < 2 && !r; attempt++) {
        let timer;
        r = await Promise.race([
          page.evaluate((c) => window.replaySplat(c), { dump: `/${relRun}/session/`, quality, mode, splatConfig: { trainer: v.trainer }, ...(o.previews ? { splat: { preview: true } } : {}), outPrefix: `${name}/` }),
          new Promise((res) => (timer = setTimeout(() => res(null), 900_000))),
        ]).finally(() => clearTimeout(timer));
        if (!r) {
          console.log(`[replay] ${name}: splat stage timed out${attempt ? '' : ', retrying in a new page'}`);
          await closePage?.().catch(() => {});
          const p = await browser.openPage({ outDir: outRoot, onLog: (m) => (o.verbose ? console.log(`[replay:page] ${m}`) : undefined) });
          page = p.page;
          closePage = p.close;
        }
      }
      if (!r) throw new Error(`${name}: splat stage failed twice`);
      const st = r.stats;
      console.log(
        `[replay] ${name}: ${r.count} Gaussians, ${r.iterations} it, stage ${(r.ms / 1000).toFixed(1)} s (prepare ${(st.splatPrepareMs / 1000).toFixed(1)} s, train ${(st.splatTrainMs / 1000).toFixed(1)} s at ${st.splatItPerSec} it/s), loss ${st.splatLoss}` +
          `${st.splatDepthViews !== undefined ? `, depth targets ${st.splatDepthViews} views (${st.splatDepthMs} ms)` : ''} (${((Date.now() - t0) / 1000).toFixed(1)} s)`,
      );
      const pv = previewSummary(r);
      if (pv)
        console.log(
          `[replay] ${name}: ${pv.count} previews, first at ${f2(pv.firstS, 1)} s (first iteration reported at ${f2(pv.firstTrainingS, 1)} s), max gap while training ${f2(pv.maxGapS, 1)} s, ` +
            `PSNR vs the training photo ${f2(pv.psnrFirst)} → ${f2(pv.psnrLast)} dB (final model ${f2(pv.psnrFinal)} dB)`,
        );
      if (r.gpuMs) console.log(`[replay] ${name}: GPU ms per iteration ${Object.entries(r.gpuMs).map(([k, ms]) => `${k} ${(ms / r.profiledIterations).toFixed(2)}`).join(', ')} (${r.profiledIterations} it)`);
      await copyFile(join(runDir, 'result.json'), join(vdir, 'result.json'));
      await writeFile(join(vdir, 'replay.json'), JSON.stringify({ variant: v, splatPreviews: o.previews, ...r }, null, 1));
      const entry = { trainer: v.trainer, previews: pv, stageS: r.ms / 1000, prepareS: st.splatPrepareMs / 1000, trainS: st.splatTrainMs / 1000, itPerSec: st.splatItPerSec, loss: st.splatLoss, count: r.count, stats: st, nvs: summary[name]?.nvs ?? {} };
      summary[name] = entry;
      await writeFile(join(outRoot, 'summary.json'), JSON.stringify(summary, null, 1));
    }
  }
} finally {
  if (closePage) await closePage();
  await browser?.close();
}

if (!o.skipNvs) {
  for (const v of o.variants) {
    for (let rep = 0; rep < o.repeat; rep++) {
      const name = o.repeat > 1 ? `${v.name}-r${rep + 1}` : v.name;
      const vdir = join(outRoot, name);
      if (!summary[name]) continue;
      for (const t of tests) {
        if (o.nvsOnly && summary[name].nvs?.[t]) continue;
        const args = [join(ROOT, 'e2e', 'data', dataset), vdir, '--renderers', 'splat'];
        if (!o.ssim) args.push('--no-ssim');
        if (t === 'sim3') args.push('--pose', 'sim3', '--out', join(vdir, 'nvs-sim3'));
        else if (t !== 'held') args.push('--test-dataset', t);
        // a browser that dies mid-run leaves page.evaluate pending forever: time out and retry once
        let res = null;
        for (let attempt = 0; attempt < 3 && !res; attempt++) {
          let timer;
          res = await Promise.race([
            runNvs(parseNvsArgs(args), { log: (m) => (o.verbose ? console.log(`[nvs] ${m}`) : undefined) }),
            new Promise((r) => (timer = setTimeout(() => r(null), 150_000))),
          ]).finally(() => clearTimeout(timer));
          if (!res) console.log(`[replay] ${name} ${t}: nvs timed out${attempt ? '' : ', retrying'}`);
        }
        if (!res) continue;
        const s = res.summary.splat;
        summary[name].nvs[t] = { psnr: s.full.psnr.mean, ssim: s.full.ssim?.mean ?? null, ccPsnr: s.cc.full.psnr.mean, coverage: s.coverage.mean, distM: res.distance.distM?.mean ?? null };
        console.log(`[replay] ${name} ${t}: full PSNR ${f2(s.full.psnr.mean)} SSIM ${f2(s.full.ssim?.mean, 3)} cc ${f2(s.cc.full.psnr.mean)} (${res.timing.totalS} s)`);
        await writeFile(join(outRoot, 'summary.json'), JSON.stringify(summary, null, 1));
      }
    }
  }
}

const offs = tests.filter((t) => t !== 'held' && t !== 'sim3');
const rows = [['variant', 'stage s', 'train s', 'it/s', 'previews', ...tests.map((t) => t.replace(`${dataset}-`, '')), 'off mean']];
for (const [name, e] of Object.entries(summary)) {
  const off = offs.map((t) => e.nvs?.[t]?.psnr).filter((x) => Number.isFinite(x));
  rows.push([name, f2(e.stageS, 1), f2(e.trainS, 1), f2(e.itPerSec, 1), e.previews ? String(e.previews.count) : '—', ...tests.map((t) => f2(e.nvs?.[t]?.psnr)), off.length === offs.length && off.length ? f2(off.reduce((a, b) => a + b, 0) / off.length) : '—']);
}
console.log(`\n[replay] ${dataset}: full-frame PSNR (dB) per test set\n${table(rows, [0])}`);
