// Command line of e2e/run-e2e.mjs (pure; unit-tested in src/harness/engine/e2e-lib.test.ts).

export const USAGE = `usage: node e2e/run-e2e.mjs [object|yard|all|<dataset> ...] [options]

  --engine worker|inline|mock   engine back-end (default worker = EngineClient + engine worker)
  --quality fast|balanced|max   quality preset (default fast; a recording: its recorded preset)
  --mode object|scene|auto      capture mode (default: object for object datasets, scene otherwise;
                                a recording: its recorded mode)
  --stride N                    replay every Nth frame (default 1)
  --start N                     first frame (default 0)
  --max-frames N                deliver at most N frames
  --fps F                       simulated FramePump rate limit in Hz (drops frames closer than 1/F)
  --no-imu | --no-gps | --no-heading   withhold a sensor stream
  --hfov DEG|gt|none            hfovPriorDeg passed to the engine (default 67.3, iPhone main camera;
                                a recording: its recorded prior)
  --no-dense | --no-mesh        skip densify / mesh (mesh needs dense)
  --dense-size N                DenseOptions.maxImageSize       --dense-views N   DenseOptions.maxViews
  --mesh-voxel V                MeshOptions.voxelSize (world units) --mesh-triangles N MeshOptions.targetTriangles
  --no-object-roi               MeshOptions.objectRoi = false: object mode meshes the whole scene (baseline for
                                the object-box scores in metrics.json; the object-box checks are skipped)
  --splat                       Gaussian splatting after densify / mesh (after refine with --no-dense):
                                engine.splat() → <dir>/splats.ply (scored by --nvs as the 'splat' renderer)
  --splat-iterations N          SplatOptions.iterations         --splat-max N   SplatOptions.maxGaussians
  --splat-size N                SplatOptions.maxImageSize       --splat-views N SplatOptions.maxViews
  --splat-previews              SplatOptions.preview: the training snapshots → <dir>/previews/NNN-it<i>.png,
                                arrival times and PSNR in result.json (splat.previews) and the summary
  --no-export | --export a,b    exports (sizes only); default: every applicable format
  --nvs                         novel-view evaluation of the output (e2e/nvs.mjs: held-out frames,
                                PSNR / SSIM of the points / mesh renders) → <dir>/nvs/nvs.json
  --nvs-views N                 held-out views for --nvs (default 24)
  --timeout-min M               per-dataset timeout in minutes (default 120)
  --out DIR                     output root (default e2e/out/e2e); files go to <out>/<dataset>[-tag]/
  --tag NAME                    suffix of the output directory
  --thresholds FILE             thresholds (default e2e/thresholds.json)
  --no-thresholds               report the checks but never fail on them
  --enforce                     enforce thresholds also for --engine mock (off by default: mock metrics are meaningless)
  --mapping inline|background   src/engine EngineConfig.mapping (sent as engineConfig; default: engine default)
  --extend-at F                 continue the capture after a simulated page reload (EngineOptions.extend): at
                                fraction F (0 < F < 1) of the frames finishCapture + refine, re-init with
                                resume + extend (persist on), the rest with every time rebased to start near 0
  --extend-skip-s S             seconds of frames skipped after the --extend-at split (default 0)
  --mock-speed X                MockEngine time multiplier (default 4)
  --set key=json                any other harness config field by dotted path (e.g. --set timeouts.frameMs=600000)
  --quiet                       log every 10th frame only     --verbose   page console + Vite logs
  --headed                      show the browser
  --device desktop|phone|local-bridge
                                where the harness runs (default desktop: headless Chromium here).
                                phone: serve Vite on 0.0.0.0 with the HTTP bridge, print the URL to
                                open in Safari on the phone, wait for its result, score it like a
                                desktop run (HTTPS from E2E_HTTPS_CERT / E2E_HTTPS_KEY, or --public-url).
                                local-bridge: the same, with headless Chromium here opening the URL.
                                Output tag defaults to the device name (e.g. <dataset>-phone)
  --public-url URL              phone: HTTPS tunnel URL forwarding to --port (Vite then listens on 127.0.0.1)
  --port N                      phone / local-bridge: dev server port (default 5180 for phone, free for local-bridge)`;

export class ArgError extends Error {}

const ENGINES = ['worker', 'inline', 'mock'];
const QUALITIES = ['fast', 'balanced', 'max'];
const MODES = ['object', 'scene', 'auto'];
const FORMATS = ['ply-points', 'ply-mesh', 'glb', 'colmap-zip', 'ply-splat', 'usdz', 'spz'];
const DEVICES = ['desktop', 'phone', 'local-bridge'];

function setPath(obj, path, val) {
  const ks = path.split('.');
  if (ks.some((k) => !k || k === '__proto__' || k === 'constructor' || k === 'prototype')) throw new ArgError(`bad --set key ${path}`);
  let cur = obj;
  for (const k of ks.slice(0, -1)) cur = cur[k] = cur[k] && typeof cur[k] === 'object' ? cur[k] : {};
  cur[ks[ks.length - 1]] = val;
}

/**
 * Parse argv → { datasets, config (harness config without `dataset`), nvs, nvsViews, timeoutMin,
 * out, tag, thresholds, noThresholds, enforce, quiet, verbose, headed, help, device, publicUrl, port }.
 */
export function parseArgs(argv) {
  const o = {
    datasets: [],
    // quality: the harness default ('fast'), or a recording's own preset
    config: { engine: 'worker' },
    dense: true,
    mesh: true,
    denseOpts: {},
    meshOpts: {},
    splat: false,
    splatOpts: {},
    exports: true,
    nvs: false,
    nvsViews: 24,
    timeoutMin: 120,
    out: null,
    tag: null,
    thresholds: null,
    noThresholds: false,
    enforce: false,
    quiet: false,
    verbose: false,
    headed: false,
    help: false,
    sets: [],
    device: 'desktop',
    publicUrl: null,
    port: null,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new ArgError(`${a} needs a value`);
      return argv[++i];
    };
    const int = (min) => {
      const s = next();
      const v = Number(s);
      if (!Number.isInteger(v) || v < min) throw new ArgError(`${a} needs an integer ≥ ${min}, got ${s}`);
      return v;
    };
    const num = (minExclusive) => {
      const s = next();
      const v = Number(s);
      if (!Number.isFinite(v) || !(v > minExclusive)) throw new ArgError(`${a} needs a number > ${minExclusive}, got ${s}`);
      return v;
    };
    const oneOf = (list) => {
      const v = next();
      if (!list.includes(v)) throw new ArgError(`${a} must be ${list.join(' | ')}, got ${v}`);
      return v;
    };
    switch (a) {
      case '--engine': o.config.engine = oneOf(ENGINES); break;
      case '--quality': o.config.quality = oneOf(QUALITIES); break;
      case '--mode': o.config.mode = oneOf(MODES); break;
      case '--stride': o.config.frameStride = int(1); break;
      case '--start': o.config.startFrame = int(0); break;
      case '--max-frames': o.config.maxFrames = int(1); break;
      case '--fps': o.config.fps = num(0); break;
      case '--no-imu': o.config.useImu = false; break;
      case '--no-gps': o.config.useGps = false; break;
      case '--no-heading': o.config.useHeading = false; break;
      case '--hfov': {
        const v = next();
        if (v === 'gt' || v === 'none') o.config.hfovPriorDeg = v;
        else {
          const d = Number(v);
          if (!(d > 1 && d < 179)) throw new ArgError(`--hfov needs degrees in (1, 179), gt or none, got ${v}`);
          o.config.hfovPriorDeg = d;
        }
        break;
      }
      case '--no-dense': o.dense = false; break;
      case '--no-mesh': o.mesh = false; break;
      case '--dense-size': o.denseOpts.maxImageSize = int(16); break;
      case '--dense-views': o.denseOpts.maxViews = int(2); break;
      case '--mesh-voxel': o.meshOpts.voxelSize = num(0); break;
      case '--mesh-triangles': o.meshOpts.targetTriangles = int(4); break;
      case '--no-object-roi': o.meshOpts.objectRoi = false; break;
      case '--splat': o.splat = true; break;
      case '--splat-iterations': o.splatOpts.iterations = int(1); break;
      case '--splat-max': o.splatOpts.maxGaussians = int(1000); break;
      case '--splat-size': o.splatOpts.maxImageSize = int(64); break;
      case '--splat-views': o.splatOpts.maxViews = int(2); break;
      case '--splat-previews': o.splatOpts.preview = true; break;
      case '--no-export': o.exports = false; break;
      case '--export': {
        const v = next();
        if (v === 'none') o.exports = false;
        else if (v === 'all') o.exports = true;
        else {
          const list = v.split(',').map((s) => s.trim()).filter(Boolean);
          for (const f of list) if (!FORMATS.includes(f)) throw new ArgError(`--export: unknown format ${f} (${FORMATS.join(', ')}, all, none)`);
          o.exports = list;
        }
        break;
      }
      case '--nvs': o.nvs = true; break;
      case '--nvs-views': o.nvsViews = int(1); break;
      case '--timeout-min': o.timeoutMin = num(0); break;
      case '--out': o.out = next(); break;
      case '--tag': o.tag = next(); break;
      case '--thresholds': o.thresholds = next(); break;
      case '--no-thresholds': o.noThresholds = true; break;
      case '--enforce': o.enforce = true; break;
      case '--mock-speed': o.config.mockSpeed = num(0); break;
      case '--mapping': o.config.engineConfig = { ...(o.config.engineConfig ?? {}), mapping: oneOf(['inline', 'background']) }; break;
      case '--extend-at': {
        const s = next();
        const v = Number(s);
        if (!(v > 0 && v < 1)) throw new ArgError(`--extend-at needs a fraction in (0, 1), got ${s}`);
        o.config.extendAt = v;
        break;
      }
      case '--extend-skip-s': {
        const s = next();
        const v = Number(s);
        if (!Number.isFinite(v) || v < 0) throw new ArgError(`--extend-skip-s needs seconds ≥ 0, got ${s}`);
        o.config.extendSkipS = v;
        break;
      }
      case '--set': {
        const kv = next();
        const eq = kv.indexOf('=');
        if (eq <= 0) throw new ArgError(`--set needs key=json, got ${kv}`);
        let val;
        try {
          val = JSON.parse(kv.slice(eq + 1));
        } catch {
          throw new ArgError(`--set ${kv}: value is not JSON`);
        }
        o.sets.push([kv.slice(0, eq), val]);
        break;
      }
      case '--quiet': o.quiet = true; break;
      case '--verbose': o.verbose = true; break;
      case '--headed': o.headed = true; break;
      case '--device': o.device = oneOf(DEVICES); break;
      case '--public-url': {
        const v = next();
        let u;
        try {
          u = new URL(v);
        } catch {
          throw new ArgError(`--public-url needs a URL, got ${v}`);
        }
        if (u.protocol !== 'https:' && u.protocol !== 'http:') throw new ArgError(`--public-url must be http(s), got ${v}`);
        // the harness and the bridge are served at the origin's root
        if (u.pathname !== '/' || u.search || u.hash) throw new ArgError(`--public-url must be an origin (https://host[:port]), got ${v}`);
        o.publicUrl = u.origin;
        break;
      }
      case '--port': {
        o.port = int(1);
        if (o.port > 65535) throw new ArgError(`--port needs 1–65535, got ${o.port}`);
        break;
      }
      case '--help':
      case '-h': o.help = true; break;
      default:
        if (a.startsWith('-')) throw new ArgError(`unknown option ${a}`);
        o.datasets.push(a);
    }
  }
  if (o.datasets.length === 0) o.datasets = ['object'];
  if (o.datasets.includes('all')) o.datasets = [...new Set(o.datasets.flatMap((d) => (d === 'all' ? ['object', 'yard'] : [d])))];
  for (const d of o.datasets) if (!/^[\w.-]+$/.test(d)) throw new ArgError(`bad dataset name ${d}`);
  if (o.tag !== null && !/^[\w.-]+$/.test(o.tag)) throw new ArgError('--tag must be a plain name');
  if (o.device === 'desktop' && (o.publicUrl !== null || o.port !== null)) throw new ArgError('--public-url / --port need --device phone or local-bridge');
  if (o.device === 'local-bridge' && o.publicUrl !== null) throw new ArgError('--public-url is for --device phone');
  if (o.device === 'phone' && o.headed) throw new ArgError('--headed has no effect with --device phone');
  if (!o.dense && o.mesh) o.mesh = false; // mesh needs the dense cloud
  o.config.densify = o.dense ? (Object.keys(o.denseOpts).length ? { ...o.denseOpts } : true) : false;
  o.config.mesh = o.mesh ? (Object.keys(o.meshOpts).length ? { ...o.meshOpts } : true) : false;
  if (Object.keys(o.splatOpts).length) o.splat = true;
  o.config.splat = o.splat ? (Object.keys(o.splatOpts).length ? { ...o.splatOpts } : true) : false;
  o.config.exports = o.exports;
  o.config.logFrames = !o.quiet;
  for (const [k, v] of o.sets) setPath(o.config, k, v);
  return o;
}
