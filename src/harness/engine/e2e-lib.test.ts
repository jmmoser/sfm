/**
 * Tests of the e2e runner helpers (e2e/lib/e2e-args.mjs, e2e/lib/e2e-checks.mjs; plain JS typed
 * via .d.mts) and of the shipped thresholds file.
 */
import { describe, expect, it } from 'vitest';
import { ArgError, parseArgs } from '../../../e2e/lib/e2e-args.mjs';
import { metricsInput, resolveThresholds, runChecks, skipChecks, summaryRow, CHECKS, formatChecks } from '../../../e2e/lib/e2e-checks.mjs';
import { computeSelfMetrics, formatSelfTable, isGtFree, parseGyroBreaks } from '../../../e2e/lib/self-metrics.mjs';
import { readStoreZip, safeEntryName } from '../../../e2e/lib/zip-read.mjs';
import { isDatasetName, isStrictChild, partOf } from '../../../e2e/import-recording.mjs';
import { createZip } from '../../io/zip';
import type { Metrics } from '../../../e2e/lib/metrics-core.mjs';
import { plyFromBase64 } from '../../../e2e/lib/e2e-ply.mjs';
import { parsePoints } from '../../../e2e/lib/points.mjs';
import { decodePLY } from '../../io/ply';
import thresholdsJson from '../../../e2e/thresholds.json';

describe('parseArgs', () => {
  it('defaults: object, worker, dense + mesh + exports; the quality is left to the harness (fast, or a recording\'s own)', () => {
    const o = parseArgs([]);
    expect(o.datasets).toEqual(['object']);
    expect(o.config).toMatchObject({ engine: 'worker', densify: true, mesh: true, exports: true, logFrames: true });
    expect(o.config.quality).toBeUndefined();
    expect(o.timeoutMin).toBe(120);
  });

  it('parses the documented options', () => {
    const o = parseArgs(['all', '--engine', 'mock', '--quality', 'balanced', '--stride', '4', '--max-frames', '40', '--no-dense', '--timeout-min', '5', '--hfov', 'gt', '--no-gps', '--mapping', 'inline', '--set', 'timeouts.frameMs=600000']);
    expect(o.datasets).toEqual(['object', 'yard']);
    expect(o.config).toMatchObject({
      engine: 'mock',
      quality: 'balanced',
      frameStride: 4,
      maxFrames: 40,
      densify: false,
      mesh: false, // mesh needs dense
      hfovPriorDeg: 'gt',
      useGps: false,
      engineConfig: { mapping: 'inline' },
      timeouts: { frameMs: 600000 },
    });
    expect(o.timeoutMin).toBe(5);
    const d = parseArgs(['yard', '--dense-size', '480', '--dense-views', '12', '--mesh-triangles', '50000', '--export', 'glb,ply-points', '--quiet']);
    expect(d.config.densify).toEqual({ maxImageSize: 480, maxViews: 12 });
    expect(d.config.mesh).toEqual({ targetTriangles: 50000 });
    expect(d.config.exports).toEqual(['glb', 'ply-points']);
    expect(d.config.logFrames).toBe(false);
    expect(d.nvs).toBe(false);
    const sp = parseArgs(['object-real', '--splat', '--splat-iterations', '500']);
    expect(sp.config.splat).toEqual({ iterations: 500 });
    expect(parseArgs(['yard', '--no-dense', '--splat']).config).toMatchObject({ densify: false, splat: true });
    expect(parseArgs(['yard', '--splat-max', '100000']).config.splat).toEqual({ maxGaussians: 100000 });
    expect(parseArgs(['yard']).config.splat).toBe(false);
    expect(parseArgs(['yard', '--export', 'ply-splat']).config.exports).toEqual(['ply-splat']);
    const nv = parseArgs(['object-real', '--nvs', '--nvs-views', '8']);
    expect(nv.nvs).toBe(true);
    expect(nv.nvsViews).toBe(8);
    expect(parseArgs(['object-real', '--extend-at', '0.6', '--extend-skip-s', '2']).config).toMatchObject({ extendAt: 0.6, extendSkipS: 2 });
    // baseline of the object-box scores: mesh the whole scene in object mode
    expect(parseArgs(['object-real', '--no-object-roi']).config.mesh).toEqual({ objectRoi: false });
    expect(parseArgs(['object-real', '--no-object-roi', '--mesh-triangles', '9000']).config.mesh).toEqual({ objectRoi: false, targetTriangles: 9000 });
  });

  it('rejects bad input', () => {
    expect(() => parseArgs(['--engine', 'gpu'])).toThrow(ArgError);
    expect(() => parseArgs(['--stride', '0'])).toThrow(/integer/);
    expect(() => parseArgs(['--bogus'])).toThrow(/unknown option/);
    expect(() => parseArgs(['../etc'])).toThrow(/bad dataset/);
    expect(() => parseArgs(['--export', 'obj'])).toThrow(/unknown format/);
    expect(() => parseArgs(['--set', '__proto__.x=1'])).toThrow(/bad --set key/);
    expect(() => parseArgs(['--set', 'a=notjson'])).toThrow(/not JSON/);
    expect(() => parseArgs(['--extend-at', '1'])).toThrow(/fraction/);
    expect(() => parseArgs(['--extend-skip-s', '-1'])).toThrow(/seconds/);
  });

  it('--device phone | local-bridge, --public-url, --port', () => {
    expect(parseArgs([])).toMatchObject({ device: 'desktop', publicUrl: null, port: null });
    const p = parseArgs(['object-real', '--device', 'phone', '--splat', '--public-url', 'https://abc.trycloudflare.com/', '--port', '5190']);
    expect(p).toMatchObject({ datasets: ['object-real'], device: 'phone', publicUrl: 'https://abc.trycloudflare.com', port: 5190 });
    expect(p.config.splat).toBe(true);
    expect(parseArgs(['all', '--engine', 'mock', '--device', 'local-bridge'])).toMatchObject({ device: 'local-bridge', datasets: ['object', 'yard'] });
    expect(() => parseArgs(['--device', 'ipad'])).toThrow(/desktop \| phone \| local-bridge/);
    expect(() => parseArgs(['--public-url', 'https://x.example'])).toThrow(/need --device phone/);
    expect(() => parseArgs(['--port', '5180'])).toThrow(/need --device phone/);
    expect(() => parseArgs(['--device', 'local-bridge', '--public-url', 'https://x.example'])).toThrow(/for --device phone/);
    expect(() => parseArgs(['--device', 'phone', '--public-url', 'nope'])).toThrow(/needs a URL/);
    expect(() => parseArgs(['--device', 'phone', '--public-url', 'ftp://x.example'])).toThrow(/http\(s\)/);
    expect(() => parseArgs(['--device', 'phone', '--public-url', 'https://x.example/sub'])).toThrow(/must be an origin/);
    expect(() => parseArgs(['--device', 'phone', '--port', '70000'])).toThrow(/1–65535/);
    expect(() => parseArgs(['--device', 'phone', '--headed'])).toThrow(/no effect/);
  });
});

function fakeMetrics(over: Partial<Metrics> = {}): Metrics {
  return {
    dataset: 'object',
    scene: 'object',
    numFrames: 200,
    registration: { numFrames: 200, numRegistered: 19, rate: 0.095, numInput: 20, rateOfInput: 0.95, unknownIndices: 0 },
    trajectory: {
      sim3: { s: 1, R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0] },
      ateRmse: 0.005,
      ateMedian: 0.004,
      ateMax: 0.01,
      sceneExtent: 1.36,
      ateRelative: 0.005 / 1.36,
      rotErrDegMean: 0.2,
      rotErrDegMedian: 0.2,
      rotErrDegMax: 0.5,
      rpeRotDegMean: 0.1,
      rpeRotDegMax: 0.3,
    },
    metric: { scale: 1.02, scaleError: 1 / 1.02 - 1, ateSe3Rmse: 0.01, ateSe3Relative: 0.01 },
    frame: { gravityAligned: true, northAligned: false, gravityErrorDeg: 0.5, gravityErrorDegSim3: 0.5, headingErrorDeg: -40, headingErrorDegSim3: -40 },
    intrinsics: { fGt: 753, fRes: 760, focalErrorPct: 0.93, ppErrorPx: 1, k1Error: 0, k2Error: 0, cornerDistortionErrorPx: 0.1 },
    dense: null,
    focus: null,
    objectRoi: null,
    ...over,
  } as Metrics;
}

describe('thresholds + checks', () => {
  const spec = thresholdsJson as unknown as Record<string, Record<string, unknown>>;

  it('the shipped thresholds only use known check names and the requested initial values', () => {
    const datasets = spec.datasets as Record<string, Record<string, unknown>>;
    for (const limits of [spec.default, spec.object, spec.yard, ...Object.values(datasets)]) {
      for (const k of Object.keys(limits)) if (k !== 'claims') expect(Object.keys(CHECKS)).toContain(k);
    }
    // object-real: the object box acceptance (IoU ≥ 0.6, ⊇ focus − 1 cm, ≤ 5 % mesh outside focus ± 2 cm)
    const or = resolveThresholds(spec, 'object', 'object-real');
    expect(or.limits.objectRoiIoU).toEqual({ min: 0.6 });
    expect(or.limits.objectRoiContainsFocus).toEqual({ min: 1 });
    expect(or.limits.meshFocusOutside).toEqual({ max: 0.05 });
    expect(or.limits.meshFocusCompleteness).toBeUndefined();
    expect(resolveThresholds(spec, 'yard', 'yard-real').limits.objectRoiIoU).toBeUndefined();
    const o = resolveThresholds(spec, 'object', 'synth40-object');
    expect(o.limits.registrationRate).toEqual({ min: 0.9 });
    expect(o.limits.ateRelative).toEqual({ max: 0.01 });
    expect(o.limits.focalErrorPct).toEqual({ max: 2 });
    expect(o.limits.scaleError).toEqual({ max: 0.1 });
    expect(o.limits.denseAccuracyMedianRel).toEqual({ max: 0.01 });
    expect(o.limits.denseCompleteness).toEqual({ min: 0.5 });
    const y = resolveThresholds(spec, 'yard', 'yard');
    expect(y.limits.ateRelative).toEqual({ max: 0.02 });
    expect(y.claims.northAligned).toBe(true);
    const custom = resolveThresholds({ ...spec, datasets: { mine: { ateRelative: { max: 0.5 } } } }, 'object', 'mine');
    expect(custom.limits.ateRelative).toEqual({ max: 0.5 });
  });

  it('pass / fail / skip semantics', () => {
    const th = resolveThresholds(spec, 'object', 'object');
    const ctx = { dense: false, claims: { metric: true, gravityAligned: true, northAligned: false }, sensors: { imu: true, gps: false, heading: true } };
    const res = runChecks(fakeMetrics(), th, ctx);
    const by = Object.fromEntries(res.map((c) => [c.name, c]));
    expect(by.registrationRate.status).toBe('pass');
    expect(by.ateRelative.status).toBe('pass');
    expect(by.focalErrorPct.status).toBe('pass');
    expect(by.scaleError.status).toBe('pass');
    expect(by.gravityErrorDeg.status).toBe('pass');
    expect(by.headingErrorDeg.status).toBe('skip'); // north not claimed: −40° is not judged
    expect(by.denseCompleteness.status).toBe('skip');
    expect(by['claims.metric'].status).toBe('pass');
    // ordering follows CHECKS, claims last
    expect(res.map((c) => c.name).slice(0, 3)).toEqual(['registrationRate', 'ateRelative', 'focalErrorPct']);
    expect(formatChecks(res)).toMatch(/registration rate/);

    const bad = runChecks(fakeMetrics({ trajectory: null, intrinsics: { ...fakeMetrics().intrinsics!, focalErrorPct: -3 } }), th, { ...ctx, dense: true, claims: { metric: false, gravityAligned: true, northAligned: false } });
    const b = Object.fromEntries(bad.map((c) => [c.name, c]));
    expect(b.ateRelative).toMatchObject({ status: 'fail', reason: 'missing' });
    expect(b.focalErrorPct.status).toBe('fail'); // |−3| > 2
    expect(b.scaleError.status).toBe('skip');
    expect(b.denseCompleteness).toMatchObject({ status: 'fail', reason: 'missing' });
    expect(b['claims.metric'].status).toBe('fail');
    // a withheld sensor does not require the claim
    const noImu = runChecks(fakeMetrics(), th, { ...ctx, claims: { metric: false, gravityAligned: false, northAligned: false }, sensors: { imu: false, gps: false, heading: false } });
    expect(noImu.find((c) => c.name === 'claims.metric')!.status).toBe('skip');
  });

  it('mesh / object-box checks: skip without a mesh run (box checks also with the box off), judge the focus scores', () => {
    const th = { limits: { meshFocusOutside: { max: 0.05 }, meshFocusCompleteness: { min: 0.9 }, objectRoiIoU: { min: 0.6 }, objectRoiContainsFocus: { min: 1 } }, claims: {} };
    const ctx = { dense: true, claims: null, sensors: { imu: true, gps: false, heading: false } };
    expect(runChecks(fakeMetrics(), th, ctx).every((c) => c.status === 'skip' && c.reason === 'no mesh run')).toBe(true);
    expect(runChecks(fakeMetrics(), th, { ...ctx, objectRoi: true }).every((c) => c.status === 'skip' && c.reason === 'no mesh run')).toBe(true);
    const good = fakeMetrics({
      focus: { margin: 0.02, tau: 0.0136, numPoints: 1000, numOutside: 20, outsideFraction: 0.02, numGtEvaluated: 500, completeness: 0.95 },
      objectRoi: { present: true, iou: 0.72, focusCoverage: 1, containsFocusShrunk: true, shrink: 0.01, confidence: 0.9, planeZGt: 0.75 },
    });
    const on = { ...ctx, mesh: true, objectRoi: true };
    expect(runChecks(good, th, on).map((c) => c.status)).toEqual(['pass', 'pass', 'pass', 'pass']);
    // the box requested but not proposed: the box checks fail
    const none = fakeMetrics({ focus: { ...good.focus!, outsideFraction: 0.6 }, objectRoi: { ...good.objectRoi!, present: false, iou: 0, containsFocusShrunk: false } });
    const by = Object.fromEntries(runChecks(none, th, on).map((c) => [c.name, c]));
    expect(by.meshFocusOutside).toMatchObject({ status: 'fail', value: 0.6 });
    expect(by.objectRoiIoU).toMatchObject({ status: 'fail', value: 0 });
    expect(by.objectRoiContainsFocus).toMatchObject({ status: 'fail', value: 0 });
    // --no-object-roi (the unbounded baseline): only the completeness is judged
    const off = Object.fromEntries(runChecks(none, th, { ...ctx, mesh: true, objectRoi: false }).map((c) => [c.name, c]));
    expect(off.meshFocusCompleteness.status).toBe('pass');
    for (const k of ['meshFocusOutside', 'objectRoiIoU', 'objectRoiContainsFocus']) expect(off[k]).toMatchObject({ status: 'skip', reason: 'object box off' });
    // no metrics (e.g. no meta.focus) → missing
    expect(runChecks(fakeMetrics(), th, on).every((c) => c.status === 'fail' && c.reason === 'missing')).toBe(true);
  });

  it('metricsInput builds the e2e/metrics.mjs result format', () => {
    const r = {
      live: { keyframes: 25 },
      sparse: {
        intrinsics: { f: 750, cx: 360, cy: 480, k1: 0, k2: 0, width: 720, height: 960 },
        images: [{ index: 4, id: 0, pose: { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0] }, f: 750, via: 'frameIndex' }],
        numCameras: 1,
        metric: { metric: true, source: 'imu', relativeUncertainty: 0.02, gravityAligned: true, northAligned: false, originLatLonAlt: null },
        stats: { images: 22 },
      },
    };
    const m = metricsInput(r, { pointsFile: 'dense.ply' });
    expect(m.images).toEqual([{ index: 4, pose: { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0] } }]);
    expect(m.numInputImages).toBe(22);
    expect(m.metric).toMatchObject({ metric: true, gravityAligned: true, northAligned: false });
    expect(m.pointsFile).toBe('dense.ply');
    expect('objectRoi' in m).toBe(false); // no mesh run
    expect(metricsInput({ live: { keyframes: 25 }, sparse: { ...r.sparse, stats: {} } }).numInputImages).toBe(25);
    expect(metricsInput({}).images).toEqual([]);
    const roi = { min: [-0.2, -0.1, 0.754], max: [0.2, 0.13, 1.0], planeZ: 0.75, centre: [0, 0, 0.86], confidence: 0.9 };
    expect(metricsInput({ ...r, mesh: { vertexCount: 10, objectRoi: roi } }).objectRoi).toEqual(roi);
    expect(metricsInput({ ...r, mesh: { vertexCount: 10, objectRoi: null } }).objectRoi).toBeNull();
  });
});

describe('plyFromBase64', () => {
  it('round-trips the embedded dense cloud through a binary PLY', () => {
    const xyz = Float32Array.of(0.5, -1, 2, 3, 4.25, -5);
    const rgb = Uint8Array.of(10, 20, 30, 200, 210, 220);
    const b64 = (u: Uint8Array): string => btoa(String.fromCharCode(...u));
    const ply = plyFromBase64(2, b64(new Uint8Array(xyz.buffer)), b64(rgb));
    expect(Array.from(parsePoints(ply))).toEqual(Array.from(xyz));
    const dec = decodePLY(ply);
    expect(dec.points?.count).toBe(2);
    expect(Array.from(dec.points!.colors)).toEqual(Array.from(rgb));
    expect(() => plyFromBase64(3, b64(new Uint8Array(xyz.buffer)), b64(rgb))).toThrow(/need/);
  });
});

describe('GT-free scoring (raw capture recordings)', () => {
  it('isGtFree: a recording, or frames without poses', () => {
    expect(isGtFree({ source: 'sfm-raw-capture' }, [{ index: 0, pose: {} }])).toBe(true);
    expect(isGtFree({ scene: 'object' }, [{ index: 0 }])).toBe(true);
    expect(isGtFree({ scene: 'object' }, [])).toBe(true);
    expect(isGtFree({ scene: 'object', gt: {} }, [{ index: 0, pose: { R: [], t: [] } }])).toBe(false);
  });

  it('parseGyroBreaks reads refine\'s gyro consistency summary', () => {
    expect(parseGyroBreaks('3 (+1 outliers, median 0.42°)')).toEqual({ breaks: 3, outliers: 1, medianDeg: 0.42 });
    expect(parseGyroBreaks('0 (+0 outliers, median 0.05°)')).toEqual({ breaks: 0, outliers: 0, medianDeg: 0.05 });
    expect(parseGyroBreaks(undefined)).toBeNull();
    expect(parseGyroBreaks('garbage')).toBeNull();
  });

  it('computeSelfMetrics: registered fraction, reprojection, tracks, gyro breaks of the chosen model, scale source, focal vs prior, device agreement', () => {
    const r = {
      config: { hfovPriorDegUsed: 67.3 },
      dataset: { name: 'rec', scene: 'recording', numFrames: 120 },
      decisions: { keyframe: 30, skip: 90 },
      live: { delivered: 120, keyframes: 30, registered: 29, trackingLostFrames: 2, timeToInit: { delivered: 9 }, frameMs: { median: 140, p90: 260 } },
      sparse: {
        intrinsics: { f: 1600, cx: 960, cy: 720, k1: 0.01, k2: -0.002, width: 1920, height: 1440 },
        images: Array.from({ length: 28 }, (_, i) => ({ index: i })),
        numCameras: 28,
        pointsCount: 5000,
        metric: { metric: true, source: 'imu+gps', relativeUncertainty: 0.03, gravityAligned: true, northAligned: true },
        stats: { images: 30, registeredImages: 28, meanReprojErrorPx: 0.61, meanTrackLength: 4.2, medianTrackLength: 3, model: 'offline', gyroBreaksLive: '4 (+2 outliers, median 0.90°)', gyroBreaksOffline: '1 (+0 outliers, median 0.30°)', gyroOutliers: '12,17', liveExplained: 0.91, offlineExplained: 0.95, yawSource: 'gps', timeOffsetMs: 21.5 },
      },
      dense: { count: 123456 },
      recording: { device: { frames: 125, keyframes: 31 }, agreement: { compared: 120, same: 114, rate: 0.95, keyframesDevice: 31, keyframesReplay: 30, confusion: {} } },
    };
    const m = computeSelfMetrics(r, { name: 'rec', scene: 'recording' });
    expect(m.groundTruth).toBe(false);
    expect(m.registration).toMatchObject({ delivered: 120, keyframes: 30, numInput: 30, numRegistered: 28 });
    expect(m.registration.rateOfInput).toBeCloseTo(28 / 30, 12);
    expect(m.reprojection.meanPx).toBe(0.61);
    expect(m.tracks).toEqual({ mean: 4.2, median: 3, points: 5000 });
    expect(m.gyro).toMatchObject({ model: 'offline', breaks: 1, breakOutliers: 0, medianErrDeg: 0.3, deregisteredOutliers: 2 });
    expect(m.scale).toMatchObject({ metric: true, source: 'imu+gps', gravityAligned: true, northAligned: true, yawSource: 'gps', timeOffsetMs: 21.5 });
    const fPrior = 960 / Math.tan((67.3 * Math.PI) / 360);
    expect(m.intrinsics!.fPrior).toBeCloseTo(fPrior, 9);
    expect(m.intrinsics!.focalVsPriorPct).toBeCloseTo((100 * (1600 - fPrior)) / fPrior, 9);
    expect(m.agreement?.rate).toBe(0.95);
    expect(formatSelfTable(m)).toMatch(/no ground truth[\s\S]*registered fraction[\s\S]*gyro breaks \(offline model\)\s+1 \(\+0 outliers/);
    // the summary row of run-e2e takes it like computeMetrics output
    expect(summaryRow('rec', { ok: true, timings: { total: 1000 } }, m, [])).toEqual(['rec', 'ok', '28/30', '—', '—', '—', '—', '—', '—', '—', '—', '—', '1', '—']);
  });

  it('computeSelfMetrics never throws on a run that failed before refine', () => {
    const m = computeSelfMetrics({ ok: false, live: { delivered: 3 } }, {});
    expect(m.registration).toMatchObject({ delivered: 3, numRegistered: 0, numInput: null, rateOfInput: null });
    expect(m.scale).toBeNull();
    expect(() => formatSelfTable(m)).not.toThrow();
    expect(computeSelfMetrics(null).registration.delivered).toBe(0);
  });

  it('skipChecks skips every limit and claim', () => {
    const c = skipChecks(resolveThresholds(thresholdsJson, 'recording', 'rec'), 'no ground truth');
    expect(c.length).toBeGreaterThan(3);
    expect(c.every((x) => x.status === 'skip' && x.reason === 'no ground truth')).toBe(true);
    expect(c.map((x) => x.name)).toContain('claims.metric');
    expect(summaryRow('rec', { ok: true }, null, c).at(-1)).toBe('SKIP');
  });
});

describe('recording import (e2e/import-recording.mjs, e2e/lib/zip-read.mjs)', () => {
  it('reads the STORE archives of src/io/zip.ts and checks their CRCs', () => {
    const zip = createZip([
      { name: 'meta.json', data: '{"a":1}' },
      { name: 'frames/000001.jpg', data: Uint8Array.of(0xff, 0xd8, 1, 2) },
    ]);
    const e = readStoreZip(zip);
    expect(e.map((x) => x.name)).toEqual(['meta.json', 'frames/000001.jpg']);
    expect(Array.from(e[1].data)).toEqual([0xff, 0xd8, 1, 2]);
    const bad = zip.slice();
    bad[30 + 'meta.json'.length + 2] ^= 0xff; // a payload byte of meta.json
    expect(() => readStoreZip(bad)).toThrow(/CRC/);
  });

  it('rejects entry names that would leave the dataset directory', () => {
    for (const n of ['../x', 'a/../../x', '/etc/passwd', 'C:/x', 'a\\b', 'a//b', './a']) expect(() => safeEntryName(n), n).toThrow(/unsafe/);
    expect(safeEntryName('frames/000001.jpg')).toBe('frames/000001.jpg');
  });

  it('part names', () => {
    expect(partOf('/x/sfm-rec-20261021-140210-ab12cd34-part2of3.zip')).toEqual({ base: 'sfm-rec-20261021-140210-ab12cd34', part: 2, count: 3 });
    expect(partOf('sfm-rec-1.zip')).toEqual({ base: 'sfm-rec-1', part: 1, count: 1 });
    // an archive named '..zip' derives the name '.'
    expect(partOf('/x/..zip').base).toBe('.');
  });

  it('dataset names: never ".", "..", hidden or a path, so --force only replaces a directory inside the output', () => {
    for (const n of ['.', '..', '.git', '...', 'a/b', '../x', 'a b', '']) expect(isDatasetName(n), n).toBe(false);
    for (const n of ['sfm-rec-20261021-140210-ab12cd34', 'kitchen_1', 'a.b', 'x..']) expect(isDatasetName(n), n).toBe(true);
    expect(isStrictChild('/r/e2e/data', '/r/e2e/data/kitchen')).toBe(true);
    for (const d of ['/r/e2e/data', '/r/e2e/data/..', '/r/e2e', '/r/e2e/data2', '/elsewhere']) expect(isStrictChild('/r/e2e/data', d), d).toBe(false);
  });
});
