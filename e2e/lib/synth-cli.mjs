// Command-line parsing for e2e/generate-synth.mjs (kept separate so it can be unit-tested
// without starting Vite / Chromium).

export const SCENE_DEFAULTS = { object: { duration: 25, fps: 8 }, yard: { duration: 40, fps: 6 } };

export const USAGE =
  'usage: node e2e/generate-synth.mjs [object|yard|all] [--frames N] [--fps F] [--size WxH] [--duration S] [--out DIR] [--name NAME] [--seed N] [--blur N] [--no-noise] [--depth-every N] [--gt-points N] [--gt-focus-points N] [--time-offset MS] [--texture-scale X] [--visibility-every N] [--supersample N] [--set key=json]';

export class CliError extends Error {}

/** Parse argv (without node + script). Throws CliError on bad input; returns { help: true } for -h. */
export function parseArgs(argv) {
  const o = { scenes: [], set: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const val = () => {
      if (i + 1 >= argv.length) throw new CliError(`${a} needs a value`);
      return argv[++i];
    };
    const num = () => {
      const v = Number(val());
      if (!Number.isFinite(v)) throw new CliError(`${a} needs a number`);
      return v;
    };
    switch (a) {
      case 'object':
      case 'yard':
        o.scenes.push(a);
        break;
      case 'all':
        o.scenes.push('object', 'yard');
        break;
      case '--frames': o.frames = Math.round(num()); break;
      case '--fps': o.fps = num(); break;
      case '--duration': o.duration = num(); break;
      case '--size': {
        const m = /^(\d+)x(\d+)$/.exec(val());
        if (!m) throw new CliError('--size must be WxH');
        o.width = Number(m[1]);
        o.height = Number(m[2]);
        break;
      }
      case '--out': o.out = val(); break;
      case '--name': o.name = val(); break;
      case '--seed': o.seed = Math.round(num()); break;
      case '--blur': o.blur = Math.round(num()); break;
      case '--no-noise': o.noNoise = true; break;
      case '--depth-every': o.depthEvery = Math.round(num()); break;
      case '--gt-points': o.gtPoints = Math.round(num()); break;
      case '--gt-focus-points': o.gtFocusPoints = Math.round(num()); break;
      case '--time-offset': o.timeOffsetMs = num(); break;
      case '--texture-scale': o.textureScale = num(); break;
      case '--visibility-every': o.visibilityEvery = Math.round(num()); break;
      case '--supersample': o.supersample = Math.round(num()); break;
      case '--set': o.set.push(val()); break;
      case '-h':
      case '--help':
        o.help = true;
        break;
      default:
        throw new CliError(`unknown argument ${a}`);
    }
  }
  if (o.frames !== undefined && !(o.frames >= 1)) throw new CliError('--frames must be ≥ 1');
  if (o.fps !== undefined && !(o.fps > 0)) throw new CliError('--fps must be > 0');
  if (o.scenes.length === 0) o.scenes = ['object', 'yard'];
  o.scenes = [...new Set(o.scenes)];
  return o;
}

/**
 * generateSynth() config for one scene. Frame-count semantics: --frames alone spreads
 * the frames over the scene's default duration (fps = N / duration); --frames + --fps
 * sets duration = N / fps; --fps alone gives N = fps · duration.
 */
export function sceneConfig(scene, o, warn = () => {}) {
  const d = SCENE_DEFAULTS[scene];
  let { fps, duration, frames } = o;
  if (frames !== undefined && fps === undefined) {
    duration = duration ?? d.duration;
    fps = frames / duration;
  } else if (frames !== undefined && fps !== undefined) {
    duration = duration ?? frames / fps;
  } else {
    fps = fps ?? d.fps;
    duration = duration ?? d.duration;
    frames = Math.max(1, Math.round(fps * duration));
  }
  const cfg = { scene, fps, duration, frames };
  if (o.width) {
    cfg.width = o.width;
    cfg.height = o.height;
    const portrait = scene === 'object';
    if (portrait !== o.height > o.width) warn(`--size ${o.width}x${o.height} does not match the ${portrait ? 'portrait' : 'landscape'} screen angle of '${scene}'`);
  }
  if (o.name) {
    // The generator deletes and recreates <out>/<name>: only allow a plain directory name.
    if (!/^[A-Za-z0-9_][A-Za-z0-9._-]*$/.test(o.name) || o.name.includes('..'))
      throw new CliError(`--name must be a plain directory name ([A-Za-z0-9._-], no path separators or '..'), got '${o.name}'`);
    cfg.name = o.scenes.length > 1 ? `${o.name}-${scene}` : o.name;
  }
  if (o.seed !== undefined) cfg.seed = o.seed;
  if (o.depthEvery !== undefined) cfg.depthEvery = o.depthEvery;
  if (o.gtPoints !== undefined) cfg.gtPoints = o.gtPoints;
  if (o.gtFocusPoints !== undefined) cfg.gtFocusPoints = o.gtFocusPoints;
  if (o.timeOffsetMs !== undefined) cfg.timeOffsetMs = o.timeOffsetMs;
  if (o.textureScale !== undefined) cfg.textureScale = o.textureScale;
  if (o.visibilityEvery !== undefined) cfg.visibilityEvery = o.visibilityEvery;
  const effects = {};
  if (o.blur !== undefined) effects.motionBlurSamples = o.blur;
  if (o.noNoise) effects.noise = false;
  if (o.supersample !== undefined) effects.supersample = o.supersample;
  if (Object.keys(effects).length) cfg.effects = effects;
  for (const kv of o.set) {
    const eq = kv.indexOf('=');
    if (eq < 0) throw new CliError(`--set expects key=json, got ${kv}`);
    const path = kv.slice(0, eq).split('.');
    let value;
    try {
      value = JSON.parse(kv.slice(eq + 1));
    } catch {
      value = kv.slice(eq + 1);
    }
    let tgt = cfg;
    for (const k of path.slice(0, -1)) {
      if (tgt[k] === undefined || tgt[k] === null || typeof tgt[k] !== 'object') tgt[k] = {};
      tgt = tgt[k];
    }
    tgt[path[path.length - 1]] = value;
  }
  return cfg;
}
