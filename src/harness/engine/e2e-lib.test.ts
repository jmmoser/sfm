/**
 * Tests of the e2e runner helpers (e2e/lib/e2e-args.mjs, e2e/lib/e2e-checks.mjs; plain JS typed
 * via .d.mts) and of the shipped thresholds file.
 */
import { describe, expect, it } from 'vitest';
import { ArgError, parseArgs } from '../../../e2e/lib/e2e-args.mjs';
import { metricsInput, resolveThresholds, runChecks, CHECKS, formatChecks } from '../../../e2e/lib/e2e-checks.mjs';
import type { Metrics } from '../../../e2e/lib/metrics-core.mjs';
import { plyFromBase64 } from '../../../e2e/lib/e2e-ply.mjs';
import { parsePoints } from '../../../e2e/lib/points.mjs';
import { decodePLY } from '../../io/ply';
import thresholdsJson from '../../../e2e/thresholds.json';

describe('parseArgs', () => {
  it('defaults: object, worker, fast, dense + mesh + exports', () => {
    const o = parseArgs([]);
    expect(o.datasets).toEqual(['object']);
    expect(o.config).toMatchObject({ engine: 'worker', quality: 'fast', densify: true, mesh: true, exports: true, logFrames: true });
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
    ...over,
  } as Metrics;
}

describe('thresholds + checks', () => {
  const spec = thresholdsJson as unknown as Record<string, Record<string, unknown>>;

  it('the shipped thresholds only use known check names and the requested initial values', () => {
    for (const scope of ['default', 'object', 'yard']) {
      for (const k of Object.keys(spec[scope])) if (k !== 'claims') expect(Object.keys(CHECKS)).toContain(k);
    }
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
    expect(metricsInput({ live: { keyframes: 25 }, sparse: { ...r.sparse, stats: {} } }).numInputImages).toBe(25);
    expect(metricsInput({}).images).toEqual([]);
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
