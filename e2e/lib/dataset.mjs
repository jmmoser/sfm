// Load a synthetic dataset written by e2e/generate-synth.mjs (Node only).
import { readFile, access } from 'node:fs/promises';
import { join } from 'node:path';

async function exists(p) {
  try {
    await access(p);
    return true;
  } catch {
    return false;
  }
}

async function readTyped(path, Ctor) {
  const b = await readFile(path);
  const copy = new Uint8Array(b.byteLength);
  copy.set(b);
  return new Ctor(copy.buffer, 0, copy.byteLength / Ctor.BYTES_PER_ELEMENT);
}

/**
 * @param {string} dir dataset directory
 * @param {{ points?: boolean }} [opts] points=false skips the (large) GT sample files
 */
export async function loadDataset(dir, opts = {}) {
  const meta = JSON.parse(await readFile(join(dir, 'meta.json'), 'utf8'));
  const frames = JSON.parse(await readFile(join(dir, 'frames.json'), 'utf8'));
  const ds = { dir, meta, frames, gt: null, gtFocus: null };
  if (opts.points === false) return ds;
  const loadSet = async (prefix) => {
    const pf = join(dir, `${prefix}points.bin`);
    if (!(await exists(pf))) return null;
    const set = { points: await readTyped(pf, Float32Array), normals: null, visibility: null };
    const nf = join(dir, `${prefix}normals.i8`);
    if (await exists(nf)) set.normals = await readTyped(nf, Int8Array);
    const vf = join(dir, `${prefix}visibility.u8`);
    if (await exists(vf)) set.visibility = await readTyped(vf, Uint8Array);
    return set;
  };
  ds.gt = await loadSet('gt_');
  ds.gtFocus = await loadSet('gt_focus_');
  return ds;
}

export async function loadJSONIfExists(path) {
  return (await exists(path)) ? JSON.parse(await readFile(path, 'utf8')) : null;
}
