#!/usr/bin/env node
/**
 * Dense MVS validation with ground-truth poses (headless).
 *
 *   node e2e/run-dense-gt.mjs [object|yard|all|<dataset> ...] [--max-size N] [--views N] [--stride N]
 *        [--first N] [--frames i,j,...] [--sources N] [--iterations N] [--geom-iterations N] [--no-geometric]
 *        [--multiscale] [--tsdf] [--debug-views N] [--sparse N] [--tag NAME] [--out DIR]
 *        [--set key=json ...] [--no-render] [--no-metrics] [--verbose]
 *        [--save-maps] [--load-maps RUN] [--fusion-input geo|photo] [--no-consistency] [--oracle]
 *
 * Starts a Vite dev server, opens dense-gt.html in headless Chromium (SwiftShader WebGPU)
 * and calls window.runDenseGT(config) per dataset (src/harness/dense-gt/main.ts): GT poses,
 * undistortion, GT-derived sparse points + source-view selection, PatchMatchGPU (photometric
 * + geometric), fusion, optional TSDF. Writes to <out> (default e2e/out/dense-gt/):
 *   <name>.ply              fused cloud (binary PLY, GT world frame)
 *   <name>.mesh.ply         TSDF mesh (--tsdf)
 *   <name>.result.json      result in e2e/metrics.mjs format (GT poses of the used frames)
 *   <name>.stats.json       per-view stats/timings + per-stage timings + metrics
 *   <name>/debug/*.png      per-view panels (image, GT depth, PatchMatch depth/error/cost, fused pixels)
 *   <name>/render-*.png     viewer screenshots of the fused cloud (and mesh)
 * and scores the cloud with e2e/lib/metrics-core.mjs (identity alignment: the cloud is
 * already in the GT frame), plus completeness restricted to GT samples visible in the used
 * views, and accuracy / completeness breakdowns by region and camera distance.
 * `<name>` is the dataset name plus `-<tag>` when --tag is given.
 * --save-maps also writes every depth map (<name>/maps/{half,photo,geo}-NNNNNN.f32: Float32
 * [W, H, depth, normal×3, cost]); --load-maps RUN (e.g. `object`, or `{scene}` for several
 * scenes) skips PatchMatch and reuses those maps, so fusion / TSDF settings can be compared
 * in seconds (e.g. `--load-maps {scene} --tag mv2 --set fusion.minViews=2`). --oracle fuses the
 * GT surfel depth maps instead of PatchMatch output: the upper bound for the given resolution,
 * views and fusion settings (separates MVS errors from coverage / fusion limits).
 */
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir, writeFile, appendFile, readFile } from 'node:fs/promises';
import { createServer as createNetServer } from 'node:net';
import { dirname, join, resolve, isAbsolute, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadDataset } from './lib/dataset.mjs';
import { computeMetrics, formatTable } from './lib/metrics-core.mjs';
import { parsePoints } from './lib/points.mjs';
import { KdTree } from './lib/kdtree.mjs';
import { HashGrid } from './lib/grid.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
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

const USAGE = `usage: node e2e/run-dense-gt.mjs [object|yard|all|<dataset> ...] [--max-size N] [--views N] [--stride N] [--first N]
       [--frames i,j,...] [--sources N] [--iterations N] [--geom-iterations N] [--no-geometric] [--multiscale] [--tsdf]
       [--debug-views N] [--sparse N] [--tag NAME] [--out DIR] [--set key=json ...] [--no-render] [--no-metrics] [--verbose]
       [--save-maps] [--load-maps RUN] [--fusion-input geo|photo] [--no-consistency] [--oracle]`;

export function parseArgs(argv) {
  const o = { scenes: [], cfg: {}, render: true, metrics: true, verbose: false, out: null, tag: null };
  const setPath = (obj, path, val) => {
    const ks = path.split('.');
    let cur = obj;
    for (const k of ks.slice(0, -1)) cur = cur[k] = cur[k] && typeof cur[k] === 'object' ? cur[k] : {};
    cur[ks[ks.length - 1]] = val;
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${a} needs a value`);
      return argv[++i];
    };
    const int = (min = 0) => {
      const v = Number(next());
      if (!Number.isInteger(v) || v < min) throw new Error(`${a} needs an integer ≥ ${min}`);
      return v;
    };
    if (a === '--max-size') o.cfg.maxSize = int(16);
    else if (a === '--views') o.cfg.maxViews = int(2);
    else if (a === '--stride') o.cfg.stride = int(1);
    else if (a === '--first') o.cfg.first = int(0);
    else if (a === '--frames') o.cfg.frames = next().split(',').map((s) => Number(s));
    else if (a === '--sources') o.cfg.numSources = int(1);
    else if (a === '--iterations') setPath(o.cfg, 'pm.iterations', int(1));
    else if (a === '--geom-iterations') setPath(o.cfg, 'geo.geometricIterations', int(1));
    else if (a === '--no-geometric') o.cfg.geometric = false;
    else if (a === '--multiscale') o.cfg.multiScale = true;
    else if (a === '--tsdf') o.cfg.tsdf = true;
    else if (a === '--debug-views') o.cfg.debugViews = int(0);
    else if (a === '--sparse') o.cfg.sparseCount = int(10);
    else if (a === '--tag') o.tag = next();
    else if (a === '--save-maps') o.cfg.saveMaps = true;
    else if (a === '--load-maps') o.loadMaps = next();
    else if (a === '--fusion-input') {
      o.cfg.fusionInput = next();
      if (!['geo', 'photo'].includes(o.cfg.fusionInput)) throw new Error('--fusion-input must be geo | photo');
    } else if (a === '--no-consistency') o.cfg.consistencyCoverage = false;
    else if (a === '--oracle') o.cfg.oracle = true;
    else if (a === '--out') o.out = next();
    else if (a === '--no-render') o.render = false;
    else if (a === '--no-metrics') o.metrics = false;
    else if (a === '--verbose') o.verbose = true;
    else if (a === '--set') {
      const kv = next();
      const eq = kv.indexOf('=');
      if (eq <= 0) throw new Error(`--set needs key=json, got ${kv}`);
      let val;
      try {
        val = JSON.parse(kv.slice(eq + 1));
      } catch {
        throw new Error(`--set ${kv}: value is not JSON`);
      }
      setPath(o.cfg, kv.slice(0, eq), val);
    } else if (a === '--help' || a === '-h') o.help = true;
    else if (a.startsWith('--')) throw new Error(`unknown option ${a}`);
    else o.scenes.push(a);
  }
  if (o.scenes.length === 0 || o.scenes.includes('all')) o.scenes = ['object', 'yard'];
  if (o.tag !== null && !/^[\w.-]+$/.test(o.tag)) throw new Error('--tag must be a plain name');
  if (o.loadMaps !== undefined && !/^[\w.{}-]+$/.test(o.loadMaps)) throw new Error('--load-maps must be a run name (e.g. object or {scene}-base)');
  for (const s of o.scenes) if (!/^[\w.-]+$/.test(s)) throw new Error(`bad dataset name ${s}`);
  return o;
}

function freePort() {
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

/** Start Vite + headless Chromium on dense-gt.html; returns { page, close }. */
export async function openDensePage({ verbose = false, outDir }) {
  const port = await freePort();
  const server = await createServer({
    root: ROOT,
    configFile: join(ROOT, 'vite.config.ts'),
    logLevel: verbose ? 'info' : 'warn',
    clearScreen: false,
    server: { port, strictPort: true, host: '127.0.0.1', hmr: false, open: false },
    optimizeDeps: { entries: ['dense-gt.html'] },
  });
  await server.listen();
  const browser = await chromium.launch({ headless: true, args: CHROMIUM_ARGS });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.setDefaultTimeout(0);
  page.on('console', (m) => {
    if (m.type() === 'error' || m.type() === 'warning' || verbose) console.log(`[page:${m.type()}] ${m.text()}`);
  });
  page.on('pageerror', (e) => console.error(`[page:exception] ${e.message}`));
  await page.exposeFunction('__denseWrite', async (path, data, encoding, append) => {
    if (typeof path !== 'string' || isAbsolute(path) || path.split(/[\\/]/).includes('..')) throw new Error(`bad path ${path}`);
    const file = join(outDir, path);
    if (!file.startsWith(outDir + sep)) throw new Error(`bad path ${path}`);
    await mkdir(dirname(file), { recursive: true });
    const buf = Buffer.from(data, encoding === 'base64' ? 'base64' : 'utf8');
    if (append) await appendFile(file, buf);
    else await writeFile(file, buf);
  });
  await page.exposeFunction('__denseLog', (msg) => console.log(msg));
  await page.goto(`http://127.0.0.1:${port}/dense-gt.html`);
  await page.waitForFunction(() => window.denseGTReady === true, null, { timeout: 180_000 });
  const close = async () => {
    await browser.close().catch(() => {});
    await server.close().catch(() => {});
  };
  return { page, close };
}

// ---------------------------------------------------------------------------- scoring helpers

function inBox(b, x, y, z) {
  return x >= b.min[0] && x <= b.max[0] && y >= b.min[1] && y <= b.max[1] && z >= b.min[2] && z <= b.max[2];
}

function centreOf(pose) {
  const R = pose.R, t = pose.t;
  return [-(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]), -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]), -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2])];
}

/** Accuracy per point (same definition as metrics-core: nearest GT sample, point-to-plane refined with lateral slack h). */
function accuracyPerPoint(P, dataset, maxD) {
  const sets = [];
  const add = (set, h) => {
    if (!set) return;
    const n = set.points.length / 3;
    const N = set.normals ? Float32Array.from(set.normals, (v) => v / 127) : null;
    sets.push({ G: set.points, N, h, tree: new KdTree(set.points), n });
  };
  const sb = dataset.meta.gt?.spacing ?? 0.01;
  add(dataset.gt, sb);
  if (dataset.gtFocus) add(dataset.gtFocus, dataset.meta.gt?.focus?.spacing ?? sb);
  const nP = P.length / 3;
  const acc = new Float64Array(nP);
  for (let i = 0; i < nP; i++) {
    const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
    let dBest = Infinity;
    for (const S of sets) {
      const nn = S.tree.nearest(x, y, z, Math.min(maxD, dBest + S.h));
      if (nn.index < 0) continue;
      let d = nn.dist;
      const j = nn.index;
      if (S.N) {
        const nx = S.N[3 * j], ny = S.N[3 * j + 1], nz = S.N[3 * j + 2];
        if (nx || ny || nz) {
          const dn = Math.abs((x - S.G[3 * j]) * nx + (y - S.G[3 * j + 1]) * ny + (z - S.G[3 * j + 2]) * nz);
          const dl = Math.sqrt(Math.max(0, d * d - dn * dn));
          const lat = Math.max(0, dl - S.h);
          d = Math.sqrt(dn * dn + lat * lat);
        }
      }
      if (d < dBest) dBest = d;
    }
    acc[i] = dBest;
  }
  return acc;
}

function pct(a, p) {
  if (!a.length) return NaN;
  const s = Float64Array.from(a).sort();
  return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))];
}

/** Breakdowns: accuracy/precision by region (object: focus box vs rest) or by camera distance; completeness by camera distance. */
function breakdown(dataset, points, centres, tau, compVis) {
  const meta = dataset.meta;
  const roi = meta.roi;
  const pts = [];
  for (let i = 0; i < points.length; i += 3) {
    const x = points[i], y = points[i + 1], z = points[i + 2];
    if (roi && !inBox(roi, x, y, z)) continue;
    pts.push(x, y, z);
  }
  const P = Float64Array.from(pts);
  const nP = P.length / 3;
  const acc = accuracyPerPoint(P, dataset, 10 * tau);
  const camDist = (x, y, z) => {
    let d = Infinity;
    for (const c of centres) d = Math.min(d, Math.hypot(x - c[0], y - c[1], z - c[2]));
    return d;
  };
  const bins = meta.scene === 'object' ? [0, 0.3, 0.6, 1, 2, Infinity] : [0, 5, 10, 20, 40, Infinity];
  const binOf = (d) => {
    for (let b = 0; b < bins.length - 1; b++) if (d < bins[b + 1]) return b;
    return bins.length - 2;
  };
  const byDist = bins.slice(0, -1).map((lo, b) => ({ range: `${lo}-${bins[b + 1]}`, n: 0, within: 0, acc: [], rel: [] }));
  const region = { focus: { n: 0, within: 0, acc: [] }, rest: { n: 0, within: 0, acc: [] } };
  for (let i = 0; i < nP; i++) {
    const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
    const d = camDist(x, y, z);
    const B = byDist[binOf(d)];
    B.n++;
    B.acc.push(acc[i]);
    B.rel.push(acc[i] / d);
    if (acc[i] <= tau) B.within++;
    if (meta.focus) {
      const R = inBox(meta.focus, x, y, z) ? region.focus : region.rest;
      R.n++;
      R.acc.push(acc[i]);
      if (acc[i] <= tau) R.within++;
    }
  }
  const fmtBin = (B) => ({ range: B.range, points: B.n, precision: B.n ? B.within / B.n : null, accMedian: pct(B.acc, 50), accP90: pct(B.acc, 90), relMedian: B.rel ? pct(B.rel, 50) : undefined });
  // completeness by camera distance (completeness set, all-frame visibility ≥ 1 and subset visibility ≥ 1)
  const compSet = dataset.gtFocus ?? dataset.gt;
  const compRoi = dataset.gtFocus ? null : roi;
  const grid = nP ? new HashGrid(P, tau) : null;
  const cb = bins.slice(0, -1).map((lo, b) => ({ range: `${lo}-${bins[b + 1]}`, n: 0, covered: 0, nSub: 0, coveredSub: 0 }));
  const nC = compSet.points.length / 3;
  for (let i = 0; i < nC; i++) {
    if (compSet.visibility && compSet.visibility[i] < 1) continue;
    const x = compSet.points[3 * i], y = compSet.points[3 * i + 1], z = compSet.points[3 * i + 2];
    if (compRoi && !inBox(compRoi, x, y, z)) continue;
    const B = cb[binOf(camDist(x, y, z))];
    const hit = grid ? grid.hasWithin(x, y, z, tau) : false;
    B.n++;
    if (hit) B.covered++;
    if (compVis && compVis[i] >= 1) {
      B.nSub++;
      if (hit) B.coveredSub++;
    }
  }
  return {
    accuracyByCameraDistance: byDist.map(fmtBin),
    accuracyByRegion: meta.focus && meta.scene === 'object' ? { focus: fmtBin({ ...region.focus, range: 'focus box' }), rest: fmtBin({ ...region.rest, range: 'ROI minus focus box' }) } : null,
    completenessByCameraDistance: cb.map((B) => ({ range: B.range, samples: B.n, completeness: B.n ? B.covered / B.n : null, samplesSubsetVisible: B.nSub, completenessSubsetVisible: B.nSub ? B.coveredSub / B.nSub : null })),
  };
}

const f = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? '—' : typeof v === 'number' ? (Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : v.toFixed(d)) : String(v));
const pc = (v) => (v === null || v === undefined || Number.isNaN(v) ? '—' : `${(100 * v).toFixed(1)}%`);

function viewTable(stats) {
  const rows = [['frame', 'src', 'z range', 'photo ms', 'geo ms', 'photo valid/≤1%/>5%', 'geo valid/≤1%/>5%', 'geo surf ≤1%', 'fused px', 'fused ≤1%', 'exact geo ≤1%']];
  for (const v of stats.views) {
    const e = v.eval;
    rows.push([
      String(v.frame),
      String(v.sources.length),
      `${f(v.depthMin, 2)}–${f(v.depthMax, 2)}`,
      f(v.photo.ms, 0),
      v.geo ? f(v.geo.ms, 0) : '—',
      e ? `${pc(e.photo.all.valid)}/${pc(e.photo.all.within1)}/${pc(e.photo.all.outlier5)}` : '—',
      e?.geo ? `${pc(e.geo.all.valid)}/${pc(e.geo.all.within1)}/${pc(e.geo.all.outlier5)}` : '—',
      e?.geo ? pc(e.geo.surface.within1) : '—',
      pc(v.fusedPx),
      e ? pc(e.fused.all.within1) : '—',
      v.exact?.geo ? pc(v.exact.geo.all.within1) : '—',
    ]);
  }
  const w = rows[0].map((_, c) => Math.max(...rows.map((r) => r[c].length)));
  return rows.map((r) => r.map((s, c) => s.padStart(w[c])).join('  ')).join('\n');
}

function aggTable(agg) {
  const keys = Object.keys(agg).filter((k) => agg[k] && agg[k].gtPixels > 0);
  const rows = [['set', 'GT px', 'valid', '≤0.5%', '≤1%', '>5%', '≤tau', 'compl@1%', 'med rel', 'med normal°']];
  for (const k of keys) {
    const a = agg[k];
    rows.push([k, String(a.gtPixels), pc(a.valid), pc(a.within05), pc(a.within1), pc(a.outlier5), pc(a.withinTau), pc(a.completeness1), f(a.medianOfMedianRel, 4), f(a.medianNormalDeg, 1)]);
  }
  const w = rows[0].map((_, c) => Math.max(...rows.map((r) => r[c].length)));
  return rows.map((r) => r.map((s, c) => (c === 0 ? s.padEnd(w[c]) : s.padStart(w[c]))).join('  ')).join('\n');
}

// ---------------------------------------------------------------------------- main

async function runScene(page, scene, opts, outDir) {
  const name = opts.tag ? `${scene}-${opts.tag}` : scene;
  const cfg = { dataset: scene, ...structuredClone(opts.cfg) };
  if (opts.loadMaps) {
    const rel = relative(ROOT, outDir).split(sep).join('/');
    if (rel.startsWith('..')) throw new Error('--load-maps needs --out inside the repository (served by Vite)');
    cfg.loadMaps = `/${rel}/${opts.loadMaps.replace('{scene}', scene)}/maps/`;
  }
  console.log(`\n[dense-gt] === ${name}: config ${JSON.stringify(cfg)}`);
  const t0 = Date.now();
  const summary = await page.evaluate((c) => window.runDenseGT(c).then((r) => window.denseGTSummary(r)), cfg);
  const wall = (Date.now() - t0) / 1000;
  const written = await page.evaluate((p) => window.denseGTWrite(p), name);
  if (opts.render) {
    const isObj = summary.dataset.scene === 'object';
    const ds = await loadDataset(join(ROOT, 'e2e', 'data', scene), { points: false });
    const fz = ds.meta.focus;
    const fc = fz ? [0, 1, 2].map((a) => 0.5 * (fz.min[a] + fz.max[a])) : [0, 0, 0];
    const shots = [
      { name: 'fit', kind: 'fit' },
      { name: `cam${summary.frames[0]}`, kind: 'camera', frame: summary.frames[0], width: summary.cam.width * 2, height: summary.cam.height * 2 },
      { name: `cam${summary.frames[Math.floor(summary.frames.length / 2)]}`, kind: 'camera', frame: summary.frames[Math.floor(summary.frames.length / 2)], width: summary.cam.width * 2, height: summary.cam.height * 2 },
      isObj
        ? { name: 'focus', kind: 'orbit', target: fc, azimuthDeg: -60, elevationDeg: 35, distance: 0.7 }
        : { name: 'top', kind: 'orbit', target: fc, azimuthDeg: -90, elevationDeg: 60, distance: 45 },
    ];
    if (summary.mesh) shots.push({ name: 'mesh', kind: isObj ? 'orbit' : 'fit', target: fc, azimuthDeg: -60, elevationDeg: 35, distance: 0.7, mesh: true });
    try {
      written.push(...(await page.evaluate((o) => window.denseGTRender(o), { prefix: name, shots })));
    } catch (e) {
      console.warn(`[dense-gt] render failed: ${e.message}`);
    }
  }
  console.log(`[dense-gt] ${name}: ${summary.cloud.count} points, ${wall.toFixed(1)} s wall; wrote ${written.join(', ')}`);
  console.log('\nper-stage timings (ms): ' + JSON.stringify(Object.fromEntries(Object.entries(summary.timings).map(([k, v]) => [k, Math.round(v)]))));
  console.log('\nper-view (eval vs surfel GT; exact = dataset depth map):\n' + viewTable(summary));
  if (summary.aggregate) console.log('\nper-view aggregates (pixel-weighted):\n' + aggTable(summary.aggregate));

  const datasetDir = join(ROOT, 'e2e', 'data', scene);
  const ds0 = await loadDataset(datasetDir, { points: false });
  const result = {
    intrinsics: { ...ds0.meta.intrinsics },
    images: summary.frames.map((i) => ({ index: i, pose: ds0.frames.find((x) => x.index === i).pose })),
    metric: { metric: true, gravityAligned: true, northAligned: true },
    pointsFile: `${name}.ply`,
    numInputImages: summary.frames.length,
  };
  await writeFile(join(outDir, `${name}.result.json`), JSON.stringify(result, null, 2));
  const out = { name, wallSeconds: wall, summary };
  if (opts.metrics) {
    const tm = Date.now();
    const dataset = await loadDataset(datasetDir);
    const points = parsePoints(new Uint8Array(await readFile(join(outDir, `${name}.ply`))));
    const m = computeMetrics(dataset, result, { points, denseAlign: 'none' });
    out.metrics = m;
    console.log(`\nmetrics (identity alignment):\n${formatTable(m)}`);
    // completeness restricted to GT samples visible in the used views
    let compVis = null;
    try {
      const b = await readFile(join(outDir, `${name}.compvis.u8`));
      compVis = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
    } catch {}
    if (compVis) {
      // visible in the dataset (any frame) AND in at least one of the used views
      const set0 = dataset.gtFocus ?? dataset.gt;
      const both = set0.visibility ? Uint8Array.from(compVis, (v, i) => (set0.visibility[i] >= 1 ? v : 0)) : compVis;
      const sub = { ...dataset };
      if (dataset.gtFocus) sub.gtFocus = { ...dataset.gtFocus, visibility: both };
      else sub.gt = { ...dataset.gt, visibility: both };
      const ms = computeMetrics(sub, result, { points, denseAlign: 'none' });
      out.metricsSubsetVisible = ms.dense;
      console.log(`completeness (GT visible in the ${summary.frames.length} used views): ${f(ms.dense.completeness, 4)}  (${ms.dense.numGtEvaluated} samples), F-score ${f(ms.dense.fscore, 4)}`);
    }
    const centres = result.images.map((im) => centreOf(im.pose));
    out.breakdown = breakdown(dataset, points, centres, m.dense.tau, compVis);
    console.log('accuracy by camera distance: ' + out.breakdown.accuracyByCameraDistance.map((b) => `[${b.range} m: n=${b.points} prec ${pc(b.precision)} med ${f(b.accMedian, 4)} p90 ${f(b.accP90, 4)} rel ${f(b.relMedian, 4)}]`).join(' '));
    if (out.breakdown.accuracyByRegion) {
      const r = out.breakdown.accuracyByRegion;
      console.log(`accuracy by region: focus box n=${r.focus.points} prec ${pc(r.focus.precision)} med ${f(r.focus.accMedian, 4)} p90 ${f(r.focus.accP90, 4)}; rest n=${r.rest.points} prec ${pc(r.rest.precision)} med ${f(r.rest.accMedian, 4)} p90 ${f(r.rest.accP90, 4)}`);
    }
    console.log('completeness by camera distance: ' + out.breakdown.completenessByCameraDistance.map((b) => `[${b.range} m: ${pc(b.completeness)} of ${b.samples}; subset-visible ${pc(b.completenessSubsetVisible)} of ${b.samplesSubsetVisible}]`).join(' '));
    if (summary.mesh) {
      try {
        const mp = parsePoints(new Uint8Array(await readFile(join(outDir, `${name}.mesh.ply`))));
        const mm = computeMetrics(dataset, result, { points: mp, denseAlign: 'none' });
        out.meshMetrics = mm.dense;
        console.log(`mesh vertices: acc median ${f(mm.dense.accuracyMedian, 4)} p90 ${f(mm.dense.accuracyP90, 4)}, precision ${f(mm.dense.precision, 4)}, completeness ${f(mm.dense.completeness, 4)}, F ${f(mm.dense.fscore, 4)}`);
      } catch (e) {
        console.warn(`mesh metrics failed: ${e.message}`);
      }
    }
    console.log(`(scoring ${((Date.now() - tm) / 1000).toFixed(1)} s)`);
  }
  await writeFile(join(outDir, `${name}.stats.json`), JSON.stringify(out, null, 1));
  return out;
}

const isMain = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  let opts;
  try {
    opts = parseArgs(process.argv.slice(2));
  } catch (e) {
    console.error(`error: ${e.message}\n${USAGE}`);
    process.exit(2);
  }
  if (opts.help) {
    console.log(USAGE);
    process.exit(0);
  }
  const outDir = resolve(opts.out ?? join(ROOT, 'e2e', 'out', 'dense-gt'));
  await mkdir(outDir, { recursive: true });
  const { page, close } = await openDensePage({ verbose: opts.verbose, outDir });
  const results = [];
  try {
    for (const s of opts.scenes) results.push(await runScene(page, s, opts, outDir));
  } catch (e) {
    console.error(e);
    process.exitCode = 1;
  } finally {
    await close();
  }
  if (results.length > 1 && opts.metrics) {
    console.log('\n=== summary (identity alignment, tau = dataset denseTau) ===');
    const rows = [['scene', 'views', 'size', 'points', 'acc med', 'acc p90', 'precision', 'completeness', 'compl (subset-vis)', 'F-score', 'wall s']];
    for (const r of results) {
      const d = r.metrics?.dense;
      rows.push([r.name, String(r.summary.frames.length), `${r.summary.cam.width}x${r.summary.cam.height}`, String(r.summary.cloud.count), f(d?.accuracyMedian, 4), f(d?.accuracyP90, 4), f(d?.precision, 4), f(d?.completeness, 4), f(r.metricsSubsetVisible?.completeness, 4), f(d?.fscore, 4), f(r.wallSeconds, 0)]);
    }
    const w = rows[0].map((_, c) => Math.max(...rows.map((r) => r[c].length)));
    console.log(rows.map((r) => r.map((s, c) => s.padStart(w[c])).join('  ')).join('\n'));
  }
  process.exit(process.exitCode ?? 0);
}
