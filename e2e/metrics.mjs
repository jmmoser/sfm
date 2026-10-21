#!/usr/bin/env node
/**
 * Score a reconstruction against a synthetic dataset.
 *
 *   node e2e/metrics.mjs <datasetDir> <result.json> [--tau T] [--out metrics.json]
 *        [--no-roi] [--min-visibility N] [--dense-align sim3|sim3+icp|se3|none] [--max-points N]
 *        [--focus-points mesh.ply]
 *
 * result.json: { intrinsics: {f,cx,cy,k1,k2}, images: [{index, pose: {R: number[9], t: number[3]}}],
 *   metric: {metric, gravityAligned, northAligned}, pointsFile?: 'binary/ascii PLY or Float32 xyz',
 *   numInputImages?, objectRoi? }  (pointsFile is resolved relative to the result file).
 * --focus-points: points (e.g. a run's mesh.ply) scored against meta.focus (metrics `focus`).
 * Prints a table and writes metrics.json next to the result (or --out).
 * Library use: import { evaluate, computeMetrics, formatTable } from './metrics.mjs'.
 */
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadDataset } from './lib/dataset.mjs';
import { computeMetrics, formatTable } from './lib/metrics-core.mjs';
import { parsePoints } from './lib/points.mjs';

export { computeMetrics, formatTable, loadDataset, parsePoints };

/** Load dataset + result (+ points) from disk and compute metrics. */
export async function evaluate(datasetDir, resultPath, opts = {}) {
  const result = JSON.parse(await readFile(resultPath, 'utf8'));
  let points = opts.points;
  if (!points && result.pointsFile) {
    const pf = isAbsolute(result.pointsFile) ? result.pointsFile : join(dirname(resultPath), result.pointsFile);
    points = parsePoints(new Uint8Array(await readFile(pf)));
  }
  let focusPoints = opts.focusPoints;
  if (!focusPoints && opts.focusPointsFile) focusPoints = parsePoints(new Uint8Array(await readFile(opts.focusPointsFile)));
  const dataset = await loadDataset(datasetDir, { points: !!points || !!focusPoints });
  return computeMetrics(dataset, result, { ...opts, points, focusPoints });
}

export function parseArgs(argv) {
  const o = { positional: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${a} needs a value`);
      return argv[++i];
    };
    const num = (min) => {
      const v = Number(next());
      if (!Number.isFinite(v) || v < min) throw new Error(`${a} needs a number ≥ ${min}`);
      return v;
    };
    if (a === '--tau') {
      o.tau = num(0);
      if (!(o.tau > 0)) throw new Error('--tau must be > 0');
    } else if (a === '--out') o.out = next();
    else if (a === '--no-roi') o.roi = false;
    else if (a === '--min-visibility') o.minVisibility = num(0);
    else if (a === '--dense-align') {
      o.denseAlign = next();
      if (!['sim3', 'sim3+icp', 'se3', 'none'].includes(o.denseAlign)) throw new Error(`--dense-align must be sim3 | sim3+icp | se3 | none, got ${o.denseAlign}`);
    } else if (a === '--max-points') o.maxPoints = Math.round(num(1));
    else if (a === '--focus-points') o.focusPointsFile = next();
    else if (a.startsWith('--')) throw new Error(`unknown option ${a}`);
    else o.positional.push(a);
  }
  return o;
}

const isMain = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
const USAGE = 'usage: node e2e/metrics.mjs <datasetDir> <result.json> [--tau T] [--out metrics.json] [--no-roi] [--min-visibility N] [--dense-align sim3|sim3+icp|se3|none] [--max-points N] [--focus-points mesh.ply]';
if (isMain) {
  let o;
  try {
    o = parseArgs(process.argv.slice(2));
  } catch (e) {
    console.error(`error: ${e.message}\n${USAGE}`);
    process.exit(2);
  }
  if (o.positional.length !== 2) {
    console.error(USAGE);
    process.exit(2);
  }
  const [datasetDir, resultPath] = o.positional;
  const t0 = Date.now();
  evaluate(datasetDir, resultPath, o)
    .then(async (m) => {
      console.log(formatTable(m));
      const out = o.out ?? join(dirname(resultPath), 'metrics.json');
      await writeFile(out, JSON.stringify(m, null, 2));
      console.log(`\nwrote ${out} (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
    })
    .catch((e) => {
      console.error(e);
      process.exit(1);
    });
}
