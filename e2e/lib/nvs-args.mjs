// Command line of e2e/nvs.mjs (pure; unit-tested in src/harness/nvs/nvs-lib.test.ts).

export const NVS_RENDERERS = ['points', 'points-edl', 'mesh-lit', 'mesh-unlit', 'splat'];

export const NVS_USAGE = `usage: node e2e/nvs.mjs <dataset> <runDir> [options]

  <dataset>                 dataset name under e2e/data, or a dataset directory
  <runDir>                  run output directory (result.json + dense.ply / mesh.ply / splats.ply)
  --renderers a,b,…         ${NVS_RENDERERS.join(', ')} (default: all whose input file exists)
  --views N                 held-out views, evenly spread over the non-keyframes (default 24)
  --frames i,j,…            explicit frame indices instead of --views
  --max-size N|dense|full   long side of the test images (default dense: the run's dense view size)
  --pose relative|sim3      held-out pose: nearest keyframe ∘ GT relative motion (default) or global Sim3
  --focal relative|keyframe|shared   test focal (default relative: nearest keyframe's × GT focal ratio)
  --rs auto|run|gt|none     rolling-shutter correction of the GT images (default auto = the run's model)
  --keep-blurred            also score frames the capture rejected as blurred
  --background R,G,B        full-frame background (default 0,0,0)
  --no-ssim                 skip SSIM (faster)
  --save-images             write GT | render | |diff|·3 PNGs per view and renderer
  --out DIR                 output directory (default <runDir>/nvs)
  --verbose | --headed`;

export class NvsArgError extends Error {}

/** Parse argv → { dataset, runDir, renderers|null, views, frames|null, maxSize, pose, focal, rs, keepBlurred, background, ssim, saveImages, out, verbose, headed, help }. */
export function parseNvsArgs(argv) {
  const o = {
    dataset: null,
    runDir: null,
    renderers: null,
    views: 24,
    frames: null,
    maxSize: 'dense',
    pose: 'relative',
    focal: 'relative',
    rs: 'auto',
    keepBlurred: false,
    background: [0, 0, 0],
    ssim: true,
    saveImages: false,
    out: null,
    verbose: false,
    headed: false,
    help: false,
  };
  const pos = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new NvsArgError(`${a} needs a value`);
      return argv[++i];
    };
    const oneOf = (list) => {
      const v = next();
      if (!list.includes(v)) throw new NvsArgError(`${a} must be ${list.join(' | ')}, got ${v}`);
      return v;
    };
    const intList = () => {
      const v = next();
      const list = v.split(',').map((s) => Number(s.trim()));
      if (!list.length || list.some((x) => !Number.isInteger(x) || x < 0)) throw new NvsArgError(`${a} needs comma-separated integers ≥ 0, got ${v}`);
      return list;
    };
    switch (a) {
      case '--renderers': {
        const list = next().split(',').map((s) => s.trim()).filter(Boolean);
        for (const r of list) if (!NVS_RENDERERS.includes(r)) throw new NvsArgError(`--renderers: unknown renderer ${r} (${NVS_RENDERERS.join(', ')})`);
        if (!list.length) throw new NvsArgError('--renderers needs at least one renderer');
        o.renderers = [...new Set(list)];
        break;
      }
      case '--views': {
        const v = Number(next());
        if (!Number.isInteger(v) || v < 1) throw new NvsArgError(`--views needs an integer ≥ 1`);
        o.views = v;
        break;
      }
      case '--frames': o.frames = intList(); break;
      case '--max-size': {
        const v = next();
        if (v === 'dense' || v === 'full') o.maxSize = v;
        else {
          const n = Number(v);
          if (!Number.isInteger(n) || n < 16) throw new NvsArgError(`--max-size needs an integer ≥ 16, dense or full, got ${v}`);
          o.maxSize = n;
        }
        break;
      }
      case '--pose': o.pose = oneOf(['relative', 'sim3']); break;
      case '--focal': o.focal = oneOf(['relative', 'keyframe', 'shared']); break;
      case '--rs': o.rs = oneOf(['auto', 'run', 'gt', 'none']); break;
      case '--keep-blurred': o.keepBlurred = true; break;
      case '--background': {
        const v = next();
        const c = v.split(',').map(Number);
        if (c.length !== 3 || c.some((x) => !(x >= 0 && x <= 255))) throw new NvsArgError(`--background needs R,G,B in 0…255, got ${v}`);
        o.background = c;
        break;
      }
      case '--no-ssim': o.ssim = false; break;
      case '--save-images': o.saveImages = true; break;
      case '--out': o.out = next(); break;
      case '--verbose': o.verbose = true; break;
      case '--headed': o.headed = true; break;
      case '--help':
      case '-h': o.help = true; break;
      default:
        if (a.startsWith('-')) throw new NvsArgError(`unknown option ${a}`);
        pos.push(a);
    }
  }
  if (o.help) return o;
  if (pos.length !== 2) throw new NvsArgError(`expected <dataset> <runDir>, got ${pos.length} positional argument(s)`);
  [o.dataset, o.runDir] = pos;
  return o;
}
