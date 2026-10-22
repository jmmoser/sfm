#!/usr/bin/env node
/**
 * End-to-end run of the whole engine on synthetic datasets (headless Chromium + SwiftShader).
 *
 *   node e2e/run-e2e.mjs [object|yard|all|<dataset> ...] [--engine worker|inline|mock]
 *        (<dataset> = any directory of e2e/data, e.g. the presets object-real / yard-real;
 *        the capture mode and thresholds follow its meta.scene)
 *        [--quality fast|balanced|max] [--stride N] [--max-frames N] [--no-dense] [--no-mesh]
 *        [--splat [--splat-iterations N] [--splat-max N]] [--nvs] [--timeout-min M] … (--help for every option)
 *
 * Starts Vite (root = repo, free port) and Chromium with the WebGPU flags, opens harness.html
 * per dataset and calls window.runPipeline(config) (src/harness/engine): frames are decoded
 * from e2e/data/<dataset>/frames/*.jpg and fed to the engine with IMU / GPS / compass like the
 * FramePump would, then finishCapture → refine → densify → mesh → [splat] → exports. Progress is
 * streamed to stdout. Writes to e2e/out/e2e/<dataset>[-tag]/:
 *   result.json   harness result + the e2e/metrics.mjs input fields (intrinsics, images, metric,
 *                 numInputImages, pointsFile) — re-score with `node e2e/metrics.mjs e2e/data/<d> result.json`
 *   sparse.ply / dense.ply / mesh.ply / splats.ply   (streamed from the page; splats with --splat)
 *   metrics.json  metrics (e2e/lib/metrics-core.mjs) + sparse-cloud accuracy + live-trajectory
 *                 metrics + threshold checks (e2e/thresholds.json) [+ novel-view summary with --nvs]
 *   nvs/nvs.json  with --nvs: novel-view evaluation of the output (e2e/nvs.mjs; informational)
 * Exit code: 0 when every pipeline finished without errors and every enforced check passed,
 * 1 otherwise, 2 for usage errors. Thresholds are not enforced for --engine mock unless --enforce.
 */
import { access, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadDataset } from './lib/dataset.mjs';
import { computeMetrics, formatTable } from './lib/metrics-core.mjs';
import { parsePoints } from './lib/points.mjs';
import { ArgError, USAGE, parseArgs } from './lib/e2e-args.mjs';
import { startHarnessBrowser } from './lib/e2e-browser.mjs';
import { plyFromBase64 } from './lib/e2e-ply.mjs';
import { SUMMARY_HEADER, fmtNum, formatChecks, metricsInput, resolveThresholds, runChecks, summaryRow, table } from './lib/e2e-checks.mjs';
import { parseNvsArgs } from './lib/nvs-args.mjs';
import { runNvs } from './nvs.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function exists(p) {
  try {
    await access(p);
    return true;
  } catch {
    return false;
  }
}

function withTimeout(promise, ms, what) {
  let timer;
  return Promise.race([
    promise.finally(() => clearTimeout(timer)),
    new Promise((_, rej) => {
      timer = setTimeout(() => rej(new Error(`${what} timed out after ${ms} ms`)), ms);
    }),
  ]);
}

/** Run window.runPipeline with a hard timeout; on timeout abort and collect the partial result. */
async function runOnPage(page, cfg, timeoutMs, log) {
  const run = page.evaluate((c) => window.runPipeline(c), cfg).then(
    (r) => ({ r }),
    (e) => ({ e }),
  );
  let timer;
  const timeout = new Promise((res) => {
    timer = setTimeout(() => res({ timeout: true }), timeoutMs);
  });
  const crash = new Promise((res) => page.once('crash', () => res({ crash: true })));
  const first = await Promise.race([run, timeout, crash]);
  clearTimeout(timer);
  if (first.r) return { result: first.r, error: null };
  if (first.e) return { result: null, error: `page.evaluate failed: ${first.e.message}` };
  if (first.crash) return { result: null, error: 'page crashed (renderer died; out of memory?)' };
  log(`TIMEOUT after ${timeoutMs >= 120_000 ? `${Math.round(timeoutMs / 60_000)} min` : `${Math.round(timeoutMs / 1000)} s`}: aborting the pipeline`);
  await withTimeout(page.evaluate(() => window.harnessAbort('runner timeout')), 15_000, 'harnessAbort').catch((e) => log(`abort failed: ${e.message}`));
  const late = await Promise.race([run, sleep(120_000).then(() => null)]);
  if (late?.r) return { result: late.r, error: 'timeout' };
  const snap = await withTimeout(page.evaluate(() => window.harnessSnapshot()), 15_000, 'harnessSnapshot').catch(() => null);
  return { result: snap, error: 'timeout (snapshot of the unfinished run)' };
}

function printTimings(r, log) {
  const t = r.timings ?? {};
  log(`timings (s): ${Object.entries(t).map(([k, v]) => `${k} ${fmtNum(v / 1000, 1)}`).join(', ')}`);
  if (r.stageTimings?.length) {
    log(`engine stages (s): ${r.stageTimings.map((s) => `${s.step}/${s.stage} ${fmtNum(s.durationMs / 1000, 1)}`).join(', ')}`);
  }
  const lv = r.live ?? {};
  log(
    `live: ${lv.delivered} frames → ${lv.keyframes} keyframes, ${lv.registered} registered, ${lv.points} points; init ${lv.timeToInit ? `after ${lv.timeToInit.delivered} frames (${fmtNum(lv.timeToInit.captureMs / 1000, 1)} s capture)` : 'never'}; frame ms p50 ${fmtNum(lv.frameMs?.median, 0)} p90 ${fmtNum(lv.frameMs?.p90, 0)} max ${fmtNum(lv.frameMs?.max, 0)}; decisions ${JSON.stringify(r.decisions)}`,
  );
  if (r.sparse) {
    log(`sparse: ${r.sparse.numCameras} cameras (${r.sparse.images.length} mapped via ${r.sparse.mapping}), ${r.sparse.pointsCount} points, metric ${JSON.stringify(r.sparse.metric)}${r.sparse.intrinsicsNote ? `; NOTE ${r.sparse.intrinsicsNote}` : ''}`);
  }
  if (r.extend) {
    const x = r.extend;
    log(
      `extend: reload after ${x.firstFrames} frames (first refine ${x.first.registeredImages ?? '–'} of ${x.first.keyframes} keyframes${x.first.error ? `, ${x.first.error}` : ''}), ${x.skippedFrames} skipped, resumed at #${x.resumeIndex} with ${x.resumedKeyframes ?? '–'} keyframes; ` +
        `second segment ${x.secondKeyframes ?? '–'} keyframes, ${x.secondRegisteredLive ?? '–'} registered live, ${x.secondRegisteredFinal ?? '–'} in the final model`,
    );
  }
  if (r.dense) log(`dense: ${r.dense.count} points${r.dense.file ? ` → ${r.dense.file}` : ''}`);
  if (r.mesh) log(`mesh: ${r.mesh.vertexCount} vertices, ${r.mesh.triangleCount} triangles`);
  if (r.splat) {
    const s = r.splat.stats ?? {};
    log(
      `splat: ${r.splat.count} Gaussians, ${r.splat.iterations} iterations in ${fmtNum(r.splat.ms / 1000, 1)} s (prepare ${fmtNum((s.splatPrepareMs ?? 0) / 1000, 1)} s, train ${fmtNum((s.splatTrainMs ?? 0) / 1000, 1)} s at ${s.splatItPerSec} it/s, finish ${fmtNum((s.splatFinishMs ?? 0) / 1000, 1)} s); ` +
        `${s.splatViews} views ${s.splatWidth}×${s.splatHeight}, init ${s.splatInitGaussians} (dense ${s.splatInitDense ?? 0}, sparse ${s.splatInitSparse ?? 0}, shell ${s.splatShell ?? 0}), memory plan ≈ ${s.splatMemPeakMB} MB (GPU ${s.splatGpuMB} MB)` +
        `${s.splatDegraded ? `, degraded: ${s.splatDegraded}` : ''}; checkpoints ${s.splatCheckpoints} × ${s.splatCheckpoints ? Math.round(s.splatCheckpointMs / s.splatCheckpoints) : 0} ms${r.splat.file ? ` → ${r.splat.file}` : ''}`,
    );
  }
  if (r.exports && Object.keys(r.exports).length) {
    log(`exports: ${Object.entries(r.exports).map(([k, v]) => `${k} ${'bytes' in v ? `${fmtNum(v.bytes / 1048576, 2)} MB` : `ERROR ${v.error}`}`).join(', ')}`);
  }
  if (r.engine?.respawns) log(`WARNING: the engine worker was respawned ${r.engine.respawns} time(s) (state lost)`);
  for (const e of r.errors ?? []) log(`error [${e.step}] ${e.name}: ${e.message}`);
}

async function runDataset(browser, name, opts, outRoot, thresholdsSpec) {
  const runName = opts.tag ? `${name}-${opts.tag}` : name;
  const log = (m) => console.log(`[${runName}] ${m}`);
  const dsDir = join(ROOT, 'e2e', 'data', name);
  if (!(await exists(join(dsDir, 'meta.json')))) {
    const known = ['object', 'yard', 'object-real', 'yard-real'].includes(name); // scenes and generator presets
    log(`dataset missing: ${dsDir} (generate it: node e2e/generate-synth.mjs ${known ? name : '<scene|preset> --name ' + name})`);
    return { name: runName, result: null, metrics: null, checks: [], failed: true };
  }
  const dir = join(outRoot, runName);
  await mkdir(dir, { recursive: true });
  for (const f of ['result.json', 'metrics.json', 'sparse.ply', 'dense.ply', 'mesh.ply', 'splats.ply']) await rm(join(dir, f), { force: true });

  const cfg = { ...structuredClone(opts.config), dataset: name, output: { prefix: `${runName}/` } };
  log(`=== run ${JSON.stringify(cfg)}`);
  const t0 = Date.now();
  const { page, state, close } = await browser.openPage({
    outDir: outRoot,
    onLog: (m) => log(m),
    onConsole: (type, text) => {
      if (type === 'error' || type === 'warning' || opts.verbose) log(`[page:${type}] ${text}`);
    },
  });
  let res;
  try {
    res = await runOnPage(page, cfg, opts.timeoutMin * 60_000, log);
  } finally {
    if (state.navigations > 0) log(`WARNING: the page navigated ${state.navigations} time(s) during the run (Vite optimiser reload?)`);
    await close();
  }
  const wall = (Date.now() - t0) / 1000;
  const r = res.result;
  if (res.error) log(`run error: ${res.error}`);
  if (!r) {
    await writeFile(join(dir, 'result.json'), JSON.stringify({ ok: false, error: res.error, config: cfg }, null, 2));
    return { name: runName, result: null, metrics: null, checks: [], failed: true };
  }
  if (res.error) r.errors = [...(r.errors ?? []), { step: 'runner', name: 'Error', message: res.error }];
  if (res.error) r.ok = false;

  // ---- files ----
  let pointsFile = null;
  if (r.densePoints) {
    await writeFile(join(dir, 'dense.ply'), plyFromBase64(r.densePoints.count, r.densePoints.xyz, r.densePoints.rgb));
    if (r.dense) r.dense.file = `${runName}/dense.ply`;
    delete r.densePoints;
    pointsFile = 'dense.ply';
  } else if (r.dense?.file) pointsFile = 'dense.ply';
  const input = metricsInput(r, { pointsFile });
  await writeFile(join(dir, 'result.json'), JSON.stringify({ ...input, ...r, runner: { wallSeconds: wall, error: res.error } }, null, 1));
  printTimings(r, log);

  // ---- metrics ----
  const tm = Date.now();
  let metrics = null;
  let sparseCloud = null;
  let liveMetrics = null;
  const sparsePly = join(dir, 'sparse.ply');
  const hasSparsePly = await exists(sparsePly);
  try {
    const dataset = await loadDataset(dsDir, { points: !!pointsFile || hasSparsePly });
    const points = pointsFile ? parsePoints(new Uint8Array(await readFile(join(dir, pointsFile)))) : undefined;
    metrics = computeMetrics(dataset, input, { points });
    if (hasSparsePly && input.images.length >= 3) {
      try {
        const sp = parsePoints(new Uint8Array(await readFile(sparsePly)));
        const d = computeMetrics(dataset, input, { points: sp }).dense;
        if (d) sparseCloud = { numPoints: d.numPointsRaw, evaluated: d.numPointsEvaluated, accuracyMedian: d.accuracyMedian, accuracyP90: d.accuracyP90, precision: d.precision, tau: d.tau };
      } catch (e) {
        log(`sparse-cloud metrics failed: ${e.message}`);
      }
    }
    if (r.live?.images && r.live.images.length >= 3) {
      const lm = computeMetrics(dataset, { images: r.live.images, metric: { metric: false, gravityAligned: false, northAligned: false } });
      liveMetrics = { registered: lm.registration.numRegistered, ateRelative: lm.trajectory?.ateRelative ?? null, ateRmse: lm.trajectory?.ateRmse ?? null, rotErrDegMedian: lm.trajectory?.rotErrDegMedian ?? null, mapping: 'keyframe order (unverified)' };
    }
  } catch (e) {
    log(`metrics failed: ${e.stack ?? e.message}`);
  }
  const meta = JSON.parse(await readFile(join(dsDir, 'meta.json'), 'utf8'));
  const th = resolveThresholds(thresholdsSpec, meta.scene, name);
  const ctx = {
    dense: cfg.densify !== false,
    claims: input.metric,
    sensors: {
      imu: cfg.useImu !== false,
      gps: cfg.useGps !== false && (await exists(join(dsDir, 'geo.json'))),
      heading: cfg.useHeading !== false && (await exists(join(dsDir, 'heading.json'))),
    },
  };
  const checks = metrics ? runChecks(metrics, th, ctx) : Object.keys(th.limits).map((k) => ({ name: k, label: k, value: null, limit: th.limits[k], status: 'fail', reason: 'no metrics' }));
  const enforced = !opts.noThresholds && (cfg.engine !== 'mock' || opts.enforce);
  if (metrics) log(`metrics (${((Date.now() - tm) / 1000).toFixed(1)} s):\n${formatTable(metrics)}`);
  if (sparseCloud) log(`sparse cloud: ${sparseCloud.evaluated} points in ROI, accuracy p50 ${fmtNum(sparseCloud.accuracyMedian)} p90 ${fmtNum(sparseCloud.accuracyP90)} m, precision ${fmtNum(sparseCloud.precision, 3)} (tau ${fmtNum(sparseCloud.tau)})`);
  if (liveMetrics) log(`live trajectory (${liveMetrics.registered} keyframe poses, ${liveMetrics.mapping}): ATE/extent ${fmtNum(liveMetrics.ateRelative)}, rotation p50 ${fmtNum(liveMetrics.rotErrDegMedian, 3)}°`);
  log(`checks (${enforced ? 'enforced' : 'NOT enforced'}${cfg.engine === 'mock' ? ' — mock engine: metrics are meaningless' : ''}):\n${formatChecks(checks)}`);
  // ---- novel-view evaluation (informational: never fails the run) ----
  let nvs = null;
  if (opts.nvs) {
    if (!pointsFile && !(await exists(join(dir, 'mesh.ply'))) && !(await exists(join(dir, 'splats.ply')))) log('nvs skipped: no dense.ply, mesh.ply or splats.ply');
    else {
      try {
        const res = await runNvs({ ...parseNvsArgs([dsDir, dir]), views: opts.nvsViews }, { log: (m) => log(`[nvs] ${m}`) });
        nvs = { file: 'nvs/nvs.json', views: res.views.length, summary: res.summary };
      } catch (e) {
        log(`nvs failed: ${e.message}`);
        nvs = { error: e.message };
      }
    }
  }
  await writeFile(
    join(dir, 'metrics.json'),
    JSON.stringify(
      { dataset: name, run: runName, engine: cfg.engine, pipelineOk: !!r.ok, errors: r.errors ?? [], metrics, sparseCloud, live: liveMetrics, checks, thresholdsEnforced: enforced, ...(r.splat ? { splat: r.splat } : {}), ...(nvs ? { nvs } : {}) },
      null,
      2,
    ),
  );
  log(`wrote ${dir}/{result.json, metrics.json${r.files?.length ? ', ' + r.files.map((f) => f.split('/').pop()).join(', ') : ''}} (${wall.toFixed(1)} s)`);
  const checkFail = enforced && checks.some((c) => c.status === 'fail');
  return { name: runName, result: r, metrics, checks, enforced, failed: !r.ok || checkFail };
}

const isMain = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  let opts;
  try {
    opts = parseArgs(process.argv.slice(2));
  } catch (e) {
    if (e instanceof ArgError) {
      console.error(`error: ${e.message}\n${USAGE}`);
      process.exit(2);
    }
    throw e;
  }
  if (opts.help) {
    console.log(USAGE);
    process.exit(0);
  }
  const outRoot = resolve(opts.out ?? join(ROOT, 'e2e', 'out', 'e2e'));
  await mkdir(outRoot, { recursive: true });
  const thresholdsPath = resolve(opts.thresholds ?? join(ROOT, 'e2e', 'thresholds.json'));
  let thresholdsSpec;
  try {
    thresholdsSpec = JSON.parse(await readFile(thresholdsPath, 'utf8'));
  } catch (e) {
    console.error(`error: cannot read thresholds ${thresholdsPath}: ${e.message}`);
    process.exit(2);
  }
  const browser = await startHarnessBrowser({ root: ROOT, verbose: opts.verbose, headed: opts.headed });
  console.log(`[e2e] harness at ${browser.base}/harness.html; output ${outRoot}`);
  const runs = [];
  let crashed = false;
  try {
    for (const d of opts.datasets) runs.push(await runDataset(browser, d, opts, outRoot, thresholdsSpec));
  } catch (e) {
    console.error(e);
    crashed = true;
  } finally {
    await browser.close();
  }
  if (runs.length) {
    console.log('\n=== e2e summary ===');
    console.log(table([SUMMARY_HEADER, ...runs.map((x) => summaryRow(x.name, x.result, x.metrics, x.checks))]));
    for (const x of runs) {
      for (const e of (x.result?.errors ?? []).slice(0, 5)) console.log(`  ${x.name}: [${e.step}] ${e.name}: ${e.message}`);
      if (!x.result) console.log(`  ${x.name}: no result`);
      const fails = x.checks.filter((c) => c.status === 'fail').map((c) => c.name);
      if (fails.length) console.log(`  ${x.name}: failed checks ${fails.join(', ')}${x.enforced ? '' : ' (not enforced)'}`);
    }
  }
  const failed = crashed || runs.some((x) => x.failed);
  console.log(failed ? '\n[e2e] FAILED' : '\n[e2e] OK');
  process.exit(failed ? 1 : 0);
}
