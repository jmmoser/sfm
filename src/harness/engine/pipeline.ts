/**
 * runPipeline(config): replay a synthetic dataset through the engine like the app would.
 *
 *   fetch meta/frames/imu/geo/heading → for each planned frame (time order): fetch + decode the
 *   JPEG (createImageBitmap, one frame decoded ahead) → FrameInput { bitmap, t, screenAngle, imu
 *   since the previous delivered frame, latest GPS fix / heading ≤ t, gravityDevice, index } →
 *   engine.processFrame (one frame in flight = FramePump back-pressure) → record the LiveUpdate
 *   → finishCapture → refine (progress) → [densify] → [mesh] → [splat] → [exports, sizes only].
 *
 * The live summary carries the object coverage of the last live cameras (`live.coverage`, the
 * capture screen's ring: src/app/coverage.ts) and, with `liveTrace`, of every model update.
 *
 * Every step is guarded: a failure is recorded in `errors` and the steps that depend on it are
 * skipped, so a partial result is always returned. Big outputs (PLY files) are streamed to
 * Node through `window.__e2eWrite` when the runner installed it (e2e/run-e2e.mjs).
 */
import type { GeoFix, ImuSample, PointCloud, ProgressEvent, TriangleMesh } from '../../core/types';
import { encodeMeshPLY, encodePLY } from '../../io/ply';
import { computeCoverage } from '../../app/coverage';
import { stageIdFor } from '../../app/stages';
import {
  ENGINE_ERRORS,
  EXPORT_FORMATS,
  type DenseOptions,
  type EngineOptions,
  type ExportFormat,
  type FrameInput,
  type LiveUpdate,
  type MeshOptions,
  type ScreenAngle,
  type SparseResult,
  type SplatOptions,
  type SplatResult,
} from '../../workers/protocol';
import { debugKeyframes, debugPairs } from './debug-dump';
import { dumpSession } from './replay';
import { createHarnessEngine, type HarnessEngine } from './engines';
import {
  HeadingNorthEmulator,
  cleanGeoFix,
  compassHeading,
  gravityDownAt,
  hfovLongSideDeg,
  latestAtOrBefore,
  numStats,
  planFrames,
  samplesBetween,
  type HeadingRow,
} from './feed';
import { bbox, keyframeSourceIndex, mapCamerasToFrames, medianIntrinsics, normalizeIntrinsics, poseToJSON, stageTimings, type CameraLike, type DeliveredRef } from './result';
import type { CloudSummary, FrameRecord, HarnessConfig, HarnessError, HarnessResult, LiveCoverage, LiveSummary, ProgressRecord, ResolvedHarnessConfig } from './types';

// ------------------------------------------------------------------ dataset

interface DatasetMeta {
  name: string;
  scene: string;
  width: number;
  height: number;
  fps: number;
  numFrames: number;
  screenAngle: ScreenAngle;
  sceneExtent: number;
  intrinsics: { f: number; cx: number; cy: number; k1: number; k2: number; width: number; height: number };
  /** WGS84 [lat°, lon°, alt m] of the dataset's ENU origin (null: no location). */
  originLatLonAlt?: [number, number, number] | null;
}

interface DatasetFrame {
  index: number;
  file: string;
  t: number;
}

interface Dataset {
  baseUrl: string;
  meta: DatasetMeta;
  frames: DatasetFrame[];
  imu: ImuSample[];
  geo: GeoFix[] | null;
  headings: HeadingRow[] | null;
}

function joinUrl(base: string, rel: string): string {
  return base.endsWith('/') ? base + rel : `${base}/${rel}`;
}

/**
 * Fetch + parse a JSON file. Optional files may be missing: a 404, or an HTML page (the Vite dev
 * server answers unknown paths with its index.html fallback) gives null.
 */
async function fetchJSON<T>(url: string, optional = false): Promise<T | null> {
  const r = await fetch(url);
  if (!r.ok) {
    if (optional) return null;
    throw new Error(`fetch ${url}: HTTP ${r.status}`);
  }
  const text = await r.text();
  try {
    return JSON.parse(text) as T;
  } catch (e) {
    if (optional) return null;
    const hint = /^\s*</.test(text) ? ' (got HTML: file missing? Vite serves its index.html for unknown paths)' : '';
    throw new Error(`fetch ${url}: not JSON${hint}: ${(e as Error).message}`);
  }
}

async function loadDataset(baseUrl: string): Promise<Dataset> {
  const [meta, frames, imu, geo, headings] = await Promise.all([
    fetchJSON<DatasetMeta>(joinUrl(baseUrl, 'meta.json')),
    fetchJSON<DatasetFrame[]>(joinUrl(baseUrl, 'frames.json')),
    fetchJSON<ImuSample[]>(joinUrl(baseUrl, 'imu.json'), true),
    fetchJSON<GeoFix[]>(joinUrl(baseUrl, 'geo.json'), true),
    fetchJSON<HeadingRow[]>(joinUrl(baseUrl, 'heading.json'), true),
  ]);
  const byT = <T extends { t: number }>(a: T[] | null): T[] | null => (a ? [...a].sort((x, y) => x.t - y.t) : null);
  return { baseUrl, meta: meta!, frames: frames!, imu: byT(imu) ?? [], geo: byT(geo), headings: byT(headings) };
}

// ------------------------------------------------------------------ helpers

type WriteFn = (path: string, data: string, encoding: 'base64' | 'utf8', append: boolean) => Promise<void>;

declare global {
  interface Window {
    __e2eWrite?: WriteFn;
    __e2eLog?: (msg: string) => void;
  }
}

function blobToBase64(b: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => {
      const s = r.result as string;
      resolve(s.slice(s.indexOf(',') + 1));
    };
    r.onerror = () => reject(r.error);
    r.readAsDataURL(b);
  });
}

/** Stream a binary blob to Node in ≤ 6 MB base64 chunks. */
export async function writeBinary(path: string, data: ArrayBuffer | Blob): Promise<void> {
  const w = window.__e2eWrite;
  if (!w) throw new Error('window.__e2eWrite is not installed');
  const blob = data instanceof Blob ? data : new Blob([data]);
  const CH = 6 << 20;
  if (blob.size === 0) return w(path, '', 'base64', false);
  for (let off = 0; off < blob.size; off += CH) {
    await w(path, await blobToBase64(blob.slice(off, Math.min(blob.size, off + CH))), 'base64', off > 0);
  }
}

function heapBytes(): number | null {
  const m = (performance as Performance & { memory?: { usedJSHeapSize?: number } }).memory;
  return m && typeof m.usedJSHeapSize === 'number' ? m.usedJSHeapSize : null;
}

function errorOf(step: string, e: unknown): HarnessError {
  const o = (e && typeof e === 'object' ? e : {}) as { name?: unknown; message?: unknown; stack?: unknown };
  const err: HarnessError = {
    step,
    name: typeof o.name === 'string' ? o.name : 'Error',
    message: typeof o.message === 'string' ? o.message : String(e),
  };
  if (typeof o.stack === 'string') err.stack = o.stack.split('\n').slice(0, 8).join('\n');
  return err;
}

const round = (v: number, d = 1): number => {
  const k = 10 ** d;
  return Math.round(v * k) / k;
};

function resolveConfig(c: HarnessConfig, meta: DatasetMeta | null): ResolvedHarnessConfig {
  const useImu = c.useImu ?? true;
  const prior = c.hfovPriorDeg ?? 67.3;
  let used: number | null = null;
  if (typeof prior === 'number') used = prior;
  else if (prior === 'gt' && meta) used = round(hfovLongSideDeg(meta.intrinsics.f, meta.width, meta.height), 4);
  return {
    dataset: c.dataset,
    baseUrl: c.baseUrl ?? `/e2e/data/${c.dataset}/`,
    engine: c.engine ?? 'worker',
    mode: c.mode ?? (meta ? (meta.scene === 'object' ? 'object' : 'scene') : 'auto'),
    quality: c.quality ?? 'fast',
    startFrame: c.startFrame ?? 0,
    frameStride: c.frameStride ?? 1,
    maxFrames: c.maxFrames ?? null,
    fps: c.fps ?? null,
    useImu,
    useGps: c.useGps ?? true,
    useHeading: c.useHeading ?? true,
    headingNorth: c.headingNorth ?? (meta?.originLatLonAlt && (c.useGps ?? true) ? 'magnetic' : 'true'),
    captureDateMs: c.captureDateMs ?? Date.now(),
    useGravity: c.useGravity ?? useImu,
    hfovPriorDeg: prior,
    hfovPriorDegUsed: used,
    densify: c.densify ?? false,
    mesh: c.mesh ?? false,
    splat: c.splat ?? false,
    exports: c.exports ?? false,
    persist: c.persist ?? false,
    preferGPU: c.preferGPU ?? true,
    engineConfig: c.engineConfig ?? null,
    mockSpeed: c.mockSpeed ?? 4,
    timeouts: {
      initMs: c.timeouts?.initMs ?? 600_000,
      frameMs: c.timeouts?.frameMs ?? 300_000,
      callMs: c.timeouts?.callMs ?? 120_000,
      inactivityMs: c.timeouts?.inactivityMs ?? 1_800_000,
    },
    output: c.output ?? null,
    embedDensePoints: c.embedDensePoints ?? true,
    maxFrameErrors: c.maxFrameErrors ?? 3,
    logFrames: c.logFrames ?? true,
    liveTrace: c.liveTrace ?? false,
    debugDump: c.debugDump ?? false,
    dumpSession: c.dumpSession ?? false,
  };
}

/** Errors after which further frames are pointless. */
function fatalFrameError(e: unknown): boolean {
  const n = (e as { name?: string } | null)?.name;
  return n === ENGINE_ERRORS.notImplemented || n === ENGINE_ERRORS.notInitialized || (n === ENGINE_ERRORS.crashed && /disposed|stopped/i.test(String((e as Error).message)));
}

function emptyLive(): LiveSummary {
  const s = numStats([]);
  return {
    delivered: 0,
    keyframes: 0,
    registered: 0,
    points: 0,
    timeToInit: null,
    trackingLostFrames: 0,
    unsolicitedUpdates: 0,
    keyframeIndices: [],
    frameMs: s,
    keyframeMs: s,
    cameras: [],
    coverage: null,
    images: null,
  };
}

/** The capture screen's coverage of live cameras (elevation bands only when gravity reaches the engine). */
function liveCoverage(cams: NonNullable<LiveUpdate['cameras']>, elevation: boolean): LiveCoverage {
  const c = computeCoverage(cams, { elevation });
  return {
    stable: c.stable,
    elevation: c.elevation,
    completeness: round(c.completeness, 4),
    covered: c.covered,
    coveredSectors: c.coveredSectors,
    prompt: c.prompt,
    counts: c.counts,
    cameras: cams.length,
  };
}

// ------------------------------------------------------------------ run

export interface RunControl {
  /** Abort the run (the current frame / long operation finishes or is cancelled). */
  abort(reason?: string): void;
  /** Snapshot of the result so far. */
  snapshot(): HarnessResult;
}

export interface RunHandle extends RunControl {
  done: Promise<HarnessResult>;
}

export function startPipeline(config: HarnessConfig, log: (msg: string) => void = (m) => console.log(m)): RunHandle {
  const T0 = performance.now();
  const now = (): number => round(performance.now() - T0, 1);
  let abortReason: string | null = null;
  let engine: HarnessEngine | null = null;

  const result: HarnessResult = {
    ok: false,
    aborted: false,
    config: resolveConfig(config, null),
    dataset: { name: config.dataset, scene: '', numFrames: 0, width: 0, height: 0, fps: 0, screenAngle: 0, sceneExtent: 0, hasGeo: false, hasHeading: false, imuSamples: 0 },
    engine: { kind: config.engine ?? 'worker', init: null, respawns: 0, statusChanges: [] },
    timings: {},
    stageTimings: [],
    frames: [],
    decisions: {},
    guidance: {},
    tracking: {},
    live: emptyLive(),
    progress: [],
    sparse: null,
    dense: null,
    mesh: null,
    exports: {},
    files: [],
    logs: [],
    errors: [],
    memory: {},
  };

  const stepEnds: Record<string, number> = {};
  const lastProgress = new Map<string, ProgressRecord>();
  let curStep = 'capture';
  let sessionId: string | null = null;
  const onProgress = (step: string) => (p: ProgressEvent): void => {
    const rec: ProgressRecord = { step, stage: p.stage, uiStage: stageIdFor(p.stage), progress: p.progress, ms: now() };
    if (p.message !== undefined) rec.message = p.message;
    const key = `${step}\u0000${p.stage}`;
    const prev = lastProgress.get(key);
    const keep =
      !prev || p.progress <= 0 || p.progress >= 1 || Math.abs(p.progress - prev.progress) >= 0.02 || p.message !== prev.message;
    if (keep && result.progress.length < 5000) result.progress.push(rec);
    if (!prev || Math.floor(p.progress * 4) !== Math.floor(prev.progress * 4) || p.progress >= 1) {
      log(`[${step}] ${p.stage}${rec.uiStage && rec.uiStage !== p.stage ? ` (${rec.uiStage})` : ''} ${p.progress < 0 ? '…' : `${Math.round(p.progress * 100)}%`}${p.message ? ` — ${p.message}` : ''}`);
    }
    lastProgress.set(key, rec);
  };

  const mark = (step: string, t0: number): void => {
    result.timings[step] = round(performance.now() - t0, 1);
    stepEnds[step] = now();
    const h = heapBytes();
    if (h !== null) result.memory[step] = h;
  };

  const fail = (step: string, e: unknown): void => {
    const err = errorOf(step, e);
    result.errors.push(err);
    log(`[error] ${step}: ${err.name}: ${err.message}`);
  };

  const finalize = (): HarnessResult => {
    result.stageTimings = stageTimings(result.progress, stepEnds);
    result.timings.total = round(performance.now() - T0, 1);
    result.aborted = abortReason !== null;
    result.ok = result.errors.length === 0 && !result.aborted;
    return result;
  };

  const run = async (): Promise<HarnessResult> => {
    // ---- load ----
    let ds: Dataset;
    let t0 = performance.now();
    try {
      const baseUrl = config.baseUrl ?? `/e2e/data/${config.dataset}/`;
      ds = await loadDataset(baseUrl);
      result.config = resolveConfig(config, ds.meta);
      const m = ds.meta;
      result.dataset = {
        name: m.name,
        scene: m.scene,
        numFrames: ds.frames.length,
        width: m.width,
        height: m.height,
        fps: m.fps,
        screenAngle: m.screenAngle,
        sceneExtent: m.sceneExtent,
        hasGeo: !!ds.geo?.length,
        hasHeading: !!ds.headings?.length,
        imuSamples: ds.imu.length,
      };
      mark('load', t0);
    } catch (e) {
      fail('load', e);
      return finalize();
    }
    const cfg = result.config;
    log(`[harness] dataset ${ds.meta.name} (${ds.meta.scene}, ${ds.frames.length} frames ${ds.meta.width}x${ds.meta.height} θ=${ds.meta.screenAngle}), engine ${cfg.engine}, mode ${cfg.mode}, quality ${cfg.quality}`);

    // ---- engine ----
    t0 = performance.now();
    try {
      engine = await createHarnessEngine(cfg.engine, { timeouts: cfg.timeouts, mockSpeed: cfg.mockSpeed });
    } catch (e) {
      fail('init', e);
      return finalize();
    }
    const eng = engine;
    const coverageElevation = cfg.useImu && cfg.useGravity;
    let liveEvents = 0;
    let lastLive: LiveUpdate | null = null;
    // with background mapping only the model updates carry cameras, frame responses do not
    let lastCams: LiveUpdate | null = null;
    eng.onLive((u) => {
      liveEvents++;
      lastLive = u;
      if (u.cameras) lastCams = u;
      if (cfg.liveTrace && u.cameras) {
        const { counts: _counts, ...coverage } = liveCoverage(u.cameras, coverageElevation);
        (result.liveTrace ??= []).push({
          ms: now(),
          frameIndex: u.frameIndex,
          keyframes: u.numKeyframes,
          registered: u.numRegistered,
          points: u.numPoints,
          poses: u.cameras.map((c) => poseToJSON(c.pose)),
          coverage,
        });
      }
    });
    eng.onLog((l) => {
      if (result.logs.length < 1000 || l.level === 'error' || l.level === 'warn') result.logs.push({ ms: now(), level: l.level, message: l.message });
      if (l.level !== 'debug') log(`[engine:${l.level}] ${l.message}`);
    });
    eng.onError((e) => {
      result.errors.push({ step: `engine-event${e.fatal ? ' (fatal)' : ''}`, name: e.error.name, message: `${e.error.message}${e.reason ? ` [${e.reason}]` : ''}` });
      log(`[engine:error] ${e.fatal ? 'FATAL ' : ''}${e.error.name}: ${e.error.message}`);
    });
    eng.onStatus((s) => {
      result.engine.statusChanges.push({ ms: now(), state: s.state, ...(s.reason ? { reason: s.reason } : {}) });
      result.engine.respawns = s.respawns;
      if (s.state === 'respawning' || s.state === 'dead') log(`[engine] state ${s.state}: ${s.reason ?? ''}`);
    });
    eng.onProgress((p) => {
      if (curStep === 'finishCapture' || curStep === 'capture') onProgress(curStep)(p);
    });

    try {
      const opts: EngineOptions = { mode: cfg.mode, quality: cfg.quality, persist: cfg.persist, preferGPU: cfg.preferGPU };
      if (cfg.hfovPriorDegUsed !== null) opts.hfovPriorDeg = cfg.hfovPriorDegUsed;
      if (cfg.persist) opts.sessionId = sessionId = `e2e-${cfg.dataset}-${Date.now().toString(36)}`;
      // Extension field understood by src/engine (EngineConfig); structured-cloned to the worker.
      if (cfg.engineConfig) (opts as EngineOptions & { engineConfig?: Record<string, unknown> }).engineConfig = cfg.engineConfig;
      result.engine.init = await eng.init(opts);
      mark('init', t0);
      log(`[harness] engine ready: ${JSON.stringify(result.engine.init)} (${result.timings.init} ms)`);
    } catch (e) {
      mark('init', t0);
      fail('init', e);
      return finalize();
    }

    // ---- capture ----
    const plan = planFrames(ds.frames, { startFrame: cfg.startFrame, frameStride: cfg.frameStride, fps: cfg.fps, datasetFps: ds.meta.fps, maxFrames: cfg.maxFrames });
    log(`[harness] replaying ${plan.length} of ${ds.frames.length} frames (start ${cfg.startFrame}, stride ${cfg.frameStride}${cfg.fps ? `, ${cfg.fps} Hz pump` : ''})`);
    const refs: DeliveredRef[] = [];
    const frameMs: number[] = [];
    const keyframeMs: number[] = [];
    let consecutiveErrors = 0;
    let captureOk = true;
    const tCapture = performance.now();
    curStep = 'capture';

    type Decoded = { bitmap: ImageBitmap; ms: number };
    const decode = async (f: DatasetFrame): Promise<Decoded> => {
      const td = performance.now();
      const r = await fetch(joinUrl(ds.baseUrl, f.file));
      if (!r.ok) throw new Error(`fetch ${f.file}: HTTP ${r.status}`);
      const bitmap = await createImageBitmap(await r.blob());
      return { bitmap, ms: performance.now() - td };
    };
    let pending: Promise<Decoded> | null = plan.length ? decode(plan[0]) : null;
    pending?.catch(() => undefined);
    let prevT = -Infinity;
    // compass north reference: the datasets store TRUE-north headings (feed.ts HeadingNorthMode)
    const northEmu = new HeadingNorthEmulator(cfg.headingNorth, ds.meta.originLatLonAlt ?? null, cfg.captureDateMs);
    cfg.headingNorth = northEmu.mode; // 'true' when the dataset has no location
    if (cfg.useHeading && ds.headings?.length)
      log(`[harness] compass north: ${northEmu.mode}${northEmu.mode !== 'true' ? ` (WMM2025 declination at the dataset location ${northEmu.datasetDeclinationDeg.toFixed(3)}°)` : ''}`);

    for (let k = 0; k < plan.length; k++) {
      const f = plan[k];
      let dec: Decoded;
      try {
        dec = await pending!;
      } catch (e) {
        fail(`decode ${f.index}`, e);
        captureOk = false;
        break;
      }
      pending = k + 1 < plan.length ? decode(plan[k + 1]) : null;
      pending?.catch(() => undefined);
      if (abortReason !== null) {
        dec.bitmap.close();
        break;
      }
      const t = f.t;
      const imu = cfg.useImu ? samplesBetween(ds.imu, k === 0 ? t - 1000 : prevT, t) : [];
      const input: FrameInput = {
        bitmap: dec.bitmap,
        t,
        screenAngle: ds.meta.screenAngle,
        imu,
        geo: cfg.useGps ? cleanGeoFix(latestAtOrBefore(ds.geo, t)) : null,
        heading: null,
        index: k,
        segment: 0,
        gravityDevice: cfg.useGravity && cfg.useImu ? gravityDownAt(ds.imu, t) : null,
        imuCalibrationVersion: cfg.useImu ? 0 : -1,
        timeSource: 'captureTime',
      };
      if (cfg.useHeading) {
        // what CompassSensor.headingAt reports: device-top azimuth while the top is within 45° of
        // horizontal (landscape yard), else the optical axis (dataset `heading`) — gravity from
        // the IMU like the sensor's DeviceOrientation (independent of useImu)
        input.heading = northEmu.convert(compassHeading(latestAtOrBefore(ds.headings, t), gravityDownAt(ds.imu, t)), input.geo ?? null);
      }
      prevT = t;
      const rec: FrameRecord = {
        i: k,
        index: f.index,
        t,
        frameIndex: null,
        decision: null,
        guidance: null,
        tracking: null,
        numKeyframes: 0,
        numRegistered: 0,
        numPoints: 0,
        overlay: 0,
        sparse: null,
        cameras: null,
        imu: imu.length,
        timings: {},
        wallMs: 0,
        decodeMs: round(dec.ms, 1),
      };
      const tf = performance.now();
      let upd: LiveUpdate | null = null;
      try {
        upd = await eng.processFrame(input);
        consecutiveErrors = 0;
        if (cfg.debugDump && upd.decision === 'keyframe') {
          const kfs = debugKeyframes(eng);
          if (kfs) (result.debug ??= { keyframeTrace: [] }).keyframeTrace.push({ frameIndex: f.index, keyframes: kfs });
        }
      } catch (e) {
        consecutiveErrors++;
        const err = errorOf(`frame ${f.index}`, e);
        rec.error = `${err.name}: ${err.message}`;
        result.errors.push(err);
        log(`[frame ${k}/${plan.length} #${f.index}] ERROR ${rec.error}`);
        if (fatalFrameError(e) || consecutiveErrors >= cfg.maxFrameErrors) {
          rec.wallMs = round(performance.now() - tf, 1);
          result.frames.push(rec);
          refs.push({ delivered: k, index: f.index, t, engineFrameIndex: null, keyframe: false });
          result.live.delivered = k + 1;
          log(`[harness] aborting capture after ${consecutiveErrors} consecutive frame error(s)`);
          captureOk = false;
          break;
        }
      }
      rec.wallMs = round(performance.now() - tf, 1);
      if (upd) {
        rec.frameIndex = upd.frameIndex;
        rec.decision = upd.decision;
        rec.guidance = upd.guidance;
        rec.tracking = upd.tracking;
        rec.numKeyframes = upd.numKeyframes;
        rec.numRegistered = upd.numRegistered;
        rec.numPoints = upd.numPoints;
        rec.overlay = upd.overlay ? upd.overlay.length >> 1 : 0;
        rec.sparse = upd.sparse ? upd.sparse.positions.length / 3 : null;
        rec.cameras = upd.cameras ? upd.cameras.length : null;
        for (const [key, v] of Object.entries(upd.timings ?? {})) if (typeof v === 'number') rec.timings[key] = round(v, 2);
        result.decisions[upd.decision] = (result.decisions[upd.decision] ?? 0) + 1;
        result.guidance[upd.guidance] = (result.guidance[upd.guidance] ?? 0) + 1;
        result.tracking[upd.tracking] = (result.tracking[upd.tracking] ?? 0) + 1;
        frameMs.push(rec.wallMs);
        if (upd.decision === 'keyframe') {
          keyframeMs.push(rec.wallMs);
          // a held (best-of-window) keyframe comes from an earlier frame than this update's
          result.live.keyframeIndices.push(keyframeSourceIndex(upd.timings, (d) => plan[d]?.index, f.index));
        }
        if (upd.tracking === 'lost') result.live.trackingLostFrames++;
        if (!result.live.timeToInit && upd.tracking === 'tracking') {
          result.live.timeToInit = {
            delivered: k + 1,
            index: f.index,
            captureMs: round(t - plan[0].t, 1),
            wallMs: round(performance.now() - tCapture, 1),
            keyframes: upd.numKeyframes,
          };
        }
      }
      refs.push({ delivered: k, index: f.index, t, engineFrameIndex: upd ? upd.frameIndex : null, keyframe: upd?.decision === 'keyframe' });
      result.frames.push(rec);
      result.live.delivered = k + 1;
      if (upd && (cfg.logFrames || k % 10 === 0 || k === plan.length - 1 || upd.decision === 'keyframe')) {
        log(
          `[frame ${k + 1}/${plan.length} #${f.index}] ${upd.decision.padEnd(13)} ${upd.tracking.padEnd(12)} ${upd.guidance.padEnd(14)} kf=${upd.numKeyframes} reg=${upd.numRegistered} pts=${upd.numPoints} ${Math.round(rec.wallMs)} ms`,
        );
      }
    }
    if (pending) {
      pending.then((d) => d.bitmap.close()).catch(() => undefined);
      pending = null;
    }
    mark('capture', tCapture);
    /**
     * Live summary: the counts from the newest LiveUpdate of any kind, the cameras from the newest
     * one carrying them. Called after the capture AND again after finishCapture: with background
     * mapping the model updates of the keyframes still in the mapping queue arrive as `live` events
     * while finishCapture drains it.
     */
    const applyLive = (): void => {
      const ll = lastLive as LiveUpdate | null;
      const lc = lastCams as LiveUpdate | null;
      result.live.unsolicitedUpdates = Math.max(0, liveEvents - frameMs.length);
      if (ll) {
        result.live.keyframes = ll.numKeyframes;
        result.live.registered = ll.numRegistered;
        result.live.points = ll.numPoints;
      }
      if (lc?.cameras) {
        result.live.cameras = lc.cameras.map((c) => ({ pose: poseToJSON(c.pose), f: c.intrinsics.f }));
        result.live.coverage = liveCoverage(lc.cameras, coverageElevation);
      }
      // cameras lists the registered keyframes only, and keyframeIndices misses a held frame
      // committed by finishCapture: map by order only when both cover every keyframe of `lc`
      result.live.images = null;
      if (lc?.cameras && lc.cameras.length > 0 && lc.cameras.length === lc.numKeyframes && lc.numKeyframes === result.live.keyframeIndices.length) {
        result.live.images = result.live.cameras.map((c, i) => ({ index: result.live.keyframeIndices[i], pose: c.pose }));
      }
    };
    applyLive();
    result.live.frameMs = numStats(frameMs);
    result.live.keyframeMs = numStats(keyframeMs);
    log(
      `[harness] capture done: ${result.live.delivered} frames, ${result.live.keyframes} keyframes, ${result.live.registered} registered, ${result.live.points} points; decisions ${JSON.stringify(result.decisions)} (${Math.round(result.timings.capture)} ms)`,
    );
    if (!captureOk || abortReason !== null || frameMs.length === 0) {
      if (frameMs.length === 0 && captureOk && abortReason === null) fail('capture', new Error('no frame was processed'));
      return finalize();
    }

    // ---- finishCapture ----
    curStep = 'finishCapture';
    t0 = performance.now();
    try {
      await eng.finishCapture();
      mark('finishCapture', t0);
      applyLive();
      if (cfg.debugDump) {
        const d = (result.debug ??= { keyframeTrace: [] });
        d.afterCapture = { keyframes: debugKeyframes(eng), pairs: debugPairs(eng) };
      }
      log(`[harness] mapping queue drained: ${result.live.keyframes} keyframes, ${result.live.registered} registered live, ${result.live.points} points (${result.live.unsolicitedUpdates} background updates)`);
    } catch (e) {
      mark('finishCapture', t0);
      fail('finishCapture', e);
      return finalize();
    }

    // ---- refine ----
    curStep = 'refine';
    t0 = performance.now();
    let sparse: SparseResult;
    try {
      sparse = await eng.refine(onProgress('refine'));
      mark('refine', t0);
      if (cfg.debugDump) {
        const d = (result.debug ??= { keyframeTrace: [] });
        d.afterRefine = { keyframes: debugKeyframes(eng), pairs: debugPairs(eng) };
      }
    } catch (e) {
      mark('refine', t0);
      fail('refine', e);
      return finalize();
    }
    try {
      summarizeSparse(sparse, ds, refs);
    } catch (e) {
      fail('sparse-summary', e);
    }
    const sp = result.sparse;
    if (sp) {
      log(
        `[harness] refine done: ${sp.numCameras} cameras (${sp.images.length} mapped via ${sp.mapping}${sp.unmapped.length ? `, ${sp.unmapped.length} unmapped` : ''}), ${sp.pointsCount} points, f=${sp.intrinsics ? sp.intrinsics.f.toFixed(1) : '?'}, metric ${JSON.stringify(sp.metric)} (${Math.round(result.timings.refine)} ms)`,
      );
    }
    if (cfg.output && cfg.output.sparse !== false && window.__e2eWrite && sparse.points.count > 0) {
      await streamFile('sparse.ply', () => encodePLY(sparse.points, { comments: ['sfm e2e harness: sparse'] }));
    }
    if (abortReason !== null) return finalize();

    // ---- densify ----
    let cloud: PointCloud | null = null;
    if (cfg.densify) {
      curStep = 'densify';
      t0 = performance.now();
      try {
        const dopts: DenseOptions | undefined = typeof cfg.densify === 'object' ? cfg.densify : undefined;
        cloud = await eng.densify(dopts, onProgress('densify'));
        mark('densify', t0);
        result.dense = cloudSummary(cloud);
        log(`[harness] densify done: ${cloud.count} points (${Math.round(result.timings.densify)} ms)`);
      } catch (e) {
        mark('densify', t0);
        fail('densify', e);
      }
      if (cloud && cloud.count > 0) {
        const c = cloud;
        if (cfg.output && cfg.output.dense !== false && window.__e2eWrite) {
          const file = await streamFile('dense.ply', () => encodePLY(c, { comments: ['sfm e2e harness: dense'] }));
          if (file && result.dense) result.dense.file = file;
        } else if (cfg.embedDensePoints) {
          result.densePoints = {
            count: c.count,
            xyz: await blobToBase64(new Blob([c.positions.slice(0, 3 * c.count)])),
            rgb: await blobToBase64(new Blob([c.colors.slice(0, 3 * c.count)])),
          };
        }
      }
    }
    if (abortReason !== null) return finalize();

    // ---- mesh ----
    let mesh: TriangleMesh | null = null;
    if (cfg.mesh && cloud) {
      curStep = 'mesh';
      t0 = performance.now();
      try {
        const mopts: MeshOptions | undefined = typeof cfg.mesh === 'object' ? cfg.mesh : undefined;
        mesh = await eng.mesh(mopts, onProgress('mesh'));
        mark('mesh', t0);
        const b = bbox(mesh.positions, mesh.vertexCount);
        result.mesh = { vertexCount: mesh.vertexCount, triangleCount: mesh.triangleCount, file: null, bboxMin: b?.min ?? null, bboxMax: b?.max ?? null };
        log(`[harness] mesh done: ${mesh.vertexCount} vertices, ${mesh.triangleCount} triangles (${Math.round(result.timings.mesh)} ms)`);
      } catch (e) {
        mark('mesh', t0);
        fail('mesh', e);
      }
      if (mesh && mesh.triangleCount > 0 && cfg.output && cfg.output.mesh !== false && window.__e2eWrite) {
        const m = mesh;
        const file = await streamFile('mesh.ply', () => encodeMeshPLY(m, { comments: ['sfm e2e harness: mesh'] }));
        if (file && result.mesh) result.mesh.file = file;
      }
    } else if (cfg.mesh && !cloud) {
      log('[harness] mesh skipped (no dense cloud)');
    }
    if (abortReason !== null) return finalize();

    // ---- session dump (experiments: re-run the splat stage on the same inputs) ----
    if (cfg.dumpSession && cfg.persist && sessionId && window.__e2eWrite) {
      const td = performance.now();
      try {
        const n = await dumpSession(sessionId, `${result.config.output?.prefix ?? ''}session/`, writeBinary);
        log(`[harness] session dump: ${n.frames} frames, ${n.values} values, ${n.binaries} binaries, ${(n.bytes / 1048576).toFixed(1)} MB (${Math.round(performance.now() - td)} ms)`);
      } catch (e) {
        fail('session dump', e);
      }
    }

    // ---- splat ----
    let splat: SplatResult | null = null;
    if (cfg.splat) {
      curStep = 'splat';
      t0 = performance.now();
      try {
        const sopts: SplatOptions | undefined = typeof cfg.splat === 'object' ? cfg.splat : undefined;
        const rec = onProgress('splat');
        let lastDetail: Record<string, number> | null = null;
        splat = await eng.splat(sopts, (p) => {
          if (p.detail) lastDetail = { ...p.detail };
          rec(p);
        });
        mark('splat', t0);
        result.splat = { ...splat, stats: { ...splat.stats }, lastDetail, file: null };
        const st = splat.stats;
        log(
          `[harness] splat done: ${splat.count} Gaussians, ${splat.iterations} iterations, ${st.splatViews} views ${st.splatWidth}×${st.splatHeight}, ${st.splatItPerSec} it/s, memory plan ≈ ${st.splatMemPeakMB} MB (GPU ${st.splatGpuMB} MB) (${Math.round(result.timings.splat)} ms)`,
        );
      } catch (e) {
        mark('splat', t0);
        fail('splat', e);
      }
      if (splat && splat.count > 0 && cfg.output && cfg.output.splat !== false && window.__e2eWrite) {
        try {
          const tw = performance.now();
          const blob = await eng.exportModel('ply-splat');
          const path = `${result.config.output?.prefix ?? ''}splats.ply`;
          await writeBinary(path, blob);
          result.files.push(path);
          if (result.splat) result.splat.file = path;
          log(`[harness] wrote ${path} (${(blob.size / 1048576).toFixed(1)} MB, ${Math.round(performance.now() - tw)} ms)`);
        } catch (e) {
          fail('write splats.ply', e);
        }
      }
    }

    // ---- exports (sizes only) ----
    if (cfg.exports && abortReason === null) {
      const formats: ExportFormat[] = Array.isArray(cfg.exports)
        ? cfg.exports
        : EXPORT_FORMATS.filter((fm) => (fm !== 'ply-mesh' || !!mesh) && (fm !== 'ply-splat' || !!splat));
      curStep = 'export';
      const te = performance.now();
      for (const fm of formats) {
        const tx = performance.now();
        try {
          const blob = await eng.exportModel(fm);
          result.exports[fm] = { bytes: blob.size, ms: round(performance.now() - tx, 1), type: blob.type };
          log(`[export] ${fm}: ${blob.size} bytes (${Math.round(performance.now() - tx)} ms)`);
        } catch (e) {
          const err = errorOf(`export ${fm}`, e);
          result.exports[fm] = { error: `${err.name}: ${err.message}` };
          result.errors.push(err);
          log(`[export] ${fm}: ERROR ${err.name}: ${err.message}`);
        }
      }
      mark('export', te);
    }
    return finalize();
  };

  async function streamFile(name: string, encode: () => ArrayBuffer): Promise<string | null> {
    const prefix = result.config.output?.prefix ?? '';
    const path = `${prefix}${name}`;
    try {
      const tw = performance.now();
      const buf = encode();
      await writeBinary(path, buf);
      result.files.push(path);
      log(`[harness] wrote ${path} (${(buf.byteLength / 1048576).toFixed(1)} MB, ${Math.round(performance.now() - tw)} ms)`);
      return path;
    } catch (e) {
      fail(`write ${name}`, e);
      return null;
    }
  }

  function cloudSummary(c: PointCloud): CloudSummary {
    const b = bbox(c.positions, c.count);
    return { count: c.count, hasNormals: !!c.normals, hasConfidence: !!c.confidence, bboxMin: b?.min ?? null, bboxMax: b?.max ?? null, file: null };
  }

  function summarizeSparse(sparse: SparseResult, ds: Dataset, refs: DeliveredRef[]): void {
    const cams = sparse.cameras as unknown as CameraLike[];
    const mapped = mapCamerasToFrames(cams, refs);
    const med = medianIntrinsics(cams.map((c) => c.intrinsics));
    const norm = med ? normalizeIntrinsics(med, ds.meta.width, ds.meta.height) : null;
    const stats: Record<string, number | string> = {};
    for (const [k, v] of Object.entries(sparse.stats ?? {})) if (typeof v === 'number' || typeof v === 'string') stats[k] = v;
    result.sparse = {
      intrinsics: norm?.intrinsics ?? null,
      intrinsicsNote: norm?.note ?? null,
      images: mapped.images,
      numCameras: cams.length,
      unmapped: mapped.unmapped,
      duplicates: mapped.duplicates,
      mapping: mapped.mapping,
      metric: { ...sparse.metric, originLatLonAlt: sparse.metric.originLatLonAlt ? [...sparse.metric.originLatLonAlt] : null },
      stats,
      pointsCount: sparse.points.count,
    };
    if (norm?.note) log(`[harness] note: ${norm.note}`);
    if (mapped.unmapped.length) log(`[harness] warning: ${mapped.unmapped.length} camera(s) could not be mapped to dataset frames (ids ${mapped.unmapped.slice(0, 10).join(', ')})`);
  }

  const done = run()
    .catch((e) => {
      fail('harness', e);
      return finalize();
    })
    .finally(() => {
      try {
        engine?.dispose();
      } catch {
        /* ignore */
      }
    });

  return {
    done,
    abort(reason = 'aborted') {
      if (abortReason !== null) return;
      abortReason = reason;
      result.errors.push({ step: 'abort', name: 'AbortError', message: reason });
      log(`[harness] abort requested: ${reason}`);
      try {
        engine?.cancel();
      } catch {
        /* ignore */
      }
    },
    snapshot() {
      return JSON.parse(JSON.stringify({ ...result, stageTimings: stageTimings(result.progress, stepEnds), timings: { ...result.timings, total: round(performance.now() - T0, 1) } })) as HarnessResult;
    },
  };
}

/** Run to completion. */
export function runPipeline(config: HarnessConfig, log?: (msg: string) => void): Promise<HarnessResult> {
  return startPipeline(config, log).done;
}
