/**
 * Splat-stage experiments on a frozen session (e2e/splat-replay.mjs).
 *
 *   dumpSession(sessionId, prefix, write)  — called by the pipeline (`dumpSession: true` with
 *     `persist: true`) after mesh: writes the engine's IndexedDB session store to
 *     `<prefix>manifest.json` + `<prefix>frames/<id>.jpg` + `<prefix>bin/<n>.bin` (keyframe JPEGs
 *     and records, values, binaries incl. the sparse model, the dense cloud and the fusion-filtered
 *     `depth/<id>` maps).
 *   window.replaySplat(cfg)  — loads such a dump into memory stores and runs the engine's splat
 *     stage (`runSplat`, src/engine/splat.ts: the same function `PhotogrammetryEngine.splat` runs)
 *     with the given trainer overrides; writes `<outPrefix>splats.ply` exactly like the engine's
 *     'ply-splat' export (stored f16 model → INRIA PLY) and returns the stage stats. With
 *     `splat: { preview: true }` (splat-replay.mjs --splat-previews) the run gets a progress callback
 *     — runSplat renders previews only for one — and returns the snapshots' arrival times and PSNR
 *     (`previews`, `firstTrainingMs`): the preview cost is the stage time with and without.
 */
import type { ProgressEvent } from '../../core/types';
import type { DepthMap } from '../../dense/types';
import { MemoryDepthMapStore } from '../../dense/types';
import { getGPU } from '../../gpu/device';
import { deserializeDepthMap, deserializePointCloud } from '../../io/serialize';
import { MemoryEngineStore, SessionEngineStore, packSplatModel, unpackSparseModel, unpackSplatModel } from '../../engine/persistence';
import { runSplat } from '../../engine/splat';
import type { EngineConfig, KeyframeRecord } from '../../engine/types';
import { encodeSplatPLY } from '../../splat/ply';
import type { SplatTrainer } from '../../splat/trainer';
import { SPLAT_STAGES, type QualityPreset, type SplatOptions } from '../../workers/protocol';

type WriteBinary = (path: string, data: ArrayBuffer | Blob) => Promise<void>;

const TYPED = { Float32Array, Float64Array, Int32Array, Uint32Array, Uint16Array, Int16Array, Uint8Array, Int8Array, Uint8ClampedArray } as const;
type TypedName = keyof typeof TYPED;

/** JSON with typed arrays tagged (`{ __ta: 'Float64Array', d: [...] }`). */
function encodeJSON(v: unknown): string {
  return JSON.stringify(v, (_k, x: unknown) => {
    if (ArrayBuffer.isView(x) && !(x instanceof DataView)) return { __ta: x.constructor.name, d: Array.from(x as unknown as ArrayLike<number>) };
    if (x instanceof Map) return { __map: [...x.entries()] };
    return x;
  });
}

function decodeJSON<T>(s: string): T {
  return JSON.parse(s, (_k, x: unknown) => {
    if (x && typeof x === 'object' && '__ta' in x) {
      const o = x as { __ta: TypedName; d: number[] };
      const C = TYPED[o.__ta];
      return C ? new C(o.d) : o.d;
    }
    if (x && typeof x === 'object' && '__map' in x) return new Map((x as { __map: [unknown, unknown][] }).__map);
    return x;
  }) as T;
}

interface DumpManifest {
  sessionId: string;
  frames: { id: number; file: string; record: KeyframeRecord }[];
  values: Record<string, unknown>;
  binaries: { key: string; file: string; bytes: number }[];
}

/** Writes the session store `sessionId` (opened from this page: same origin as the worker). */
export async function dumpSession(sessionId: string, prefix: string, write: WriteBinary): Promise<{ frames: number; values: number; binaries: number; bytes: number }> {
  const store = await SessionEngineStore.open(sessionId);
  let bytes = 0;
  try {
    const frames: DumpManifest['frames'] = [];
    for (const r of await store.getFrameRecords()) {
      const img = await store.getFrameImage(r.id);
      if (!img) continue;
      const blob = img instanceof Blob ? img : new Blob([new Uint8Array(img.data).slice()]);
      const file = `frames/${r.id}.${img instanceof Blob ? 'jpg' : 'rgba'}`;
      await write(prefix + file, blob);
      bytes += blob.size;
      frames.push({ id: r.id, file, record: r });
    }
    const values: Record<string, unknown> = {};
    for (const k of await store.listValueKeys()) values[k] = await store.getValue(k);
    const binaries: DumpManifest['binaries'] = [];
    let n = 0;
    for (const key of await store.listBinaryKeys()) {
      if (key.startsWith('splat/')) continue;
      const b = await store.getBinary(key);
      if (!b) continue;
      const file = `bin/${n++}.bin`;
      await write(prefix + file, b);
      bytes += b.byteLength;
      binaries.push({ key, file, bytes: b.byteLength });
    }
    const manifest: DumpManifest = { sessionId, frames, values, binaries };
    await write(prefix + 'manifest.json', new Blob([encodeJSON(manifest)]));
    return { frames: frames.length, values: Object.keys(values).length, binaries: binaries.length, bytes };
  } finally {
    store.close();
  }
}

export interface ReplaySplatConfig {
  /** URL of the dump directory (ends with '/'). */
  dump: string;
  quality?: QualityPreset;
  mode?: 'object' | 'scene';
  /** EngineConfig.splat (preset overrides + `trainer` overrides). */
  splatConfig?: EngineConfig['splat'];
  splat?: SplatOptions;
  /** Path prefix of the written splats.ply (relative to the runner's output directory). */
  outPrefix: string;
  memoryBudgetMB?: number;
  /** Feed the dense stage's filtered depth maps to the stage (default true). */
  depthMaps?: boolean;
}

export interface ReplaySplatResult {
  ms: number;
  count: number;
  iterations: number;
  stats: Record<string, number | string>;
  file: string;
  /** With trainer `profile: true`: accumulated GPU ms per pass and the iterations they cover. */
  gpuMs?: Record<string, number>;
  profiledIterations?: number;
  /**
   * With `splat.preview`: each 'splat_preview' snapshot (iteration, arrival in ms after the stage
   * started, PSNR against that training photo) and the arrival of the first training report.
   */
  previews?: { iteration: number; ms: number; psnr: number }[];
  firstTrainingMs?: number | null;
}

const loaded = new Map<string, Promise<{ store: MemoryEngineStore; depth: MemoryDepthMapStore; manifest: DumpManifest }>>();

async function loadDump(url: string): Promise<{ store: MemoryEngineStore; depth: MemoryDepthMapStore; manifest: DumpManifest }> {
  const get = async (rel: string): Promise<Response> => {
    const r = await fetch(url + rel);
    if (!r.ok) throw new Error(`fetch ${url + rel}: HTTP ${r.status}`);
    return r;
  };
  const manifest = decodeJSON<DumpManifest>(await (await get('manifest.json')).text());
  const store = new MemoryEngineStore('replay');
  await store.ensureSession({ mode: 'object', quality: 'fast', name: 'replay' });
  for (const f of manifest.frames) {
    const b = await (await get(f.file)).blob();
    await store.putFrame(f.id, new Blob([b], { type: 'image/jpeg' }), f.record);
  }
  for (const [k, v] of Object.entries(manifest.values)) await store.putValue(k, v);
  const depth = new MemoryDepthMapStore();
  for (const b of manifest.binaries) {
    // what the splat stage reads: the sparse model, the dense cloud, the filtered depth maps
    if (!/^(sparse|dense|depth)\//.test(b.key)) continue;
    const buf = await (await get(b.file)).arrayBuffer();
    await store.putBinary(b.key, buf);
    const m = /^depth\/(\d+)$/.exec(b.key);
    if (m) await depth.put(Number(m[1]), deserializeDepthMap(buf) as DepthMap);
  }
  return { store, depth, manifest };
}

/** Runs the splat stage on a dumped session (see the module comment). */
export async function replaySplat(cfg: ReplaySplatConfig, log: (m: string) => void, write: WriteBinary): Promise<ReplaySplatResult> {
  let p = loaded.get(cfg.dump);
  if (!p) {
    p = loadDump(cfg.dump);
    loaded.set(cfg.dump, p);
    p.catch(() => loaded.delete(cfg.dump));
  }
  const { store, depth } = await p;
  const sm = await store.getBinary('sparse/model');
  if (!sm) throw new Error('dump has no sparse/model');
  const model = unpackSparseModel(sm);
  const meta = await store.getValue<{ rollingShutter?: unknown }>('sparse/meta');
  model.rollingShutter = (meta?.rollingShutter ?? null) as typeof model.rollingShutter;
  const dp = await store.getBinary('dense/points');
  const cloud = dp ? deserializePointCloud(dp) : null;
  const ctx = await getGPU();
  if (!ctx) throw new Error('no WebGPU');
  const t0 = performance.now();
  let trainer: SplatTrainer | null = null;
  // previews need a progress callback (runSplat renders none without one): record their arrival
  const previews: NonNullable<ReplaySplatResult['previews']> = [];
  let firstTrainingMs: number | null = null;
  const progress = cfg.splat?.preview
    ? (p: ProgressEvent): void => {
        const ms = Math.round(10 * (performance.now() - t0)) / 10;
        if (p.stage === SPLAT_STAGES.preview && p.detail) previews.push({ iteration: p.detail.iteration, ms, psnr: p.detail.psnr });
        else if (p.stage === SPLAT_STAGES.training) firstTrainingMs ??= ms;
      }
    : undefined;
  const out = await runSplat(
    {
      store,
      checkpoints: new MemoryEngineStore('replay-ckpt'),
      model,
      denseCloud: async () => cloud,
      densePoints: cloud?.count ?? 0,
      denseResident: false,
      depthMaps: cfg.depthMaps === false ? null : async () => depth,
      quality: cfg.quality ?? 'fast',
      mode: cfg.mode ?? 'object',
      config: { splat: cfg.splatConfig ?? {}, splatDebug: { onTrainer: (t: SplatTrainer) => (trainer = t) } } as EngineConfig,
      waitVisible: async () => undefined,
      gpuLock: (fn) => fn(),
      log: (level, message) => log(`[splat:${level}] ${message}`),
      memoryBudgetMB: cfg.memoryBudgetMB ?? 1200,
      residentBytes: 100 << 20,
      gpu: async () => ctx,
    },
    cfg.splat ?? {},
    progress,
  );
  const ms = performance.now() - t0;
  // exactly what the engine stores and exports: f16 model → INRIA PLY
  const stored = unpackSplatModel(packSplatModel(out.model)).model;
  const ply = encodeSplatPLY(stored, { antialiased: out.antialiased, comments: ['sfm splat replay'] });
  const file = `${cfg.outPrefix}splats.ply`;
  await write(file, new Blob([ply as Uint8Array<ArrayBuffer>]));
  const st = (trainer as SplatTrainer | null)?.stats();
  return {
    ms,
    count: out.model.count,
    iterations: out.iterations,
    stats: out.stats,
    file,
    ...(st?.profiledIterations ? { gpuMs: st.gpuMs, profiledIterations: st.profiledIterations } : {}),
    ...(progress ? { previews, firstTrainingMs } : {}),
  };
}
