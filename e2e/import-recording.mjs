#!/usr/bin/env node
/**
 * Import a raw capture recording exported from the app (Diagnostics › Raw capture recordings ›
 * Export: ZIP parts of ≤ 200 MB, src/io/recording.ts) as an e2e dataset.
 *
 *   node e2e/import-recording.mjs <part.zip | dir with the parts> [more parts …] [--name NAME]
 *        [--out DIR] [--force]
 *
 * Extracts every part into <out>/<name>/ (default e2e/data/<name>, name = the archive name
 * without "-partNofM", e.g. sfm-rec-20261021-140210-ab12cd34; a plain name that does not start
 * with '.', so `--force` can only ever replace a directory inside <out>), then checks the
 * dataset: meta.json, frames.jsonl and every JPEG it names (rows recorded without pixels,
 * `image: null`, are expected and only counted), and that no part is missing. Replay it with
 * `node e2e/run-e2e.mjs <name>` (verbatim FrameInputs, GT-free scoring; e2e/README.md "Raw capture
 * recordings"). Exit code 0 when the dataset is complete, 1 when parts / frames are missing, 2 for
 * usage errors.
 */
import { mkdir, readdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readStoreZip } from './lib/zip-read.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const USAGE = 'usage: node e2e/import-recording.mjs <part.zip | dir> [more parts …] [--name NAME] [--out DIR] [--force]';

function parseArgs(argv) {
  const o = { inputs: [], name: null, out: join(ROOT, 'e2e', 'data'), force: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${a} needs a value`);
      return argv[++i];
    };
    if (a === '--name') o.name = next();
    else if (a === '--out') o.out = resolve(next());
    else if (a === '--force') o.force = true;
    else if (a === '--help' || a === '-h') o.help = true;
    else if (a.startsWith('--')) throw new Error(`unknown option ${a}`);
    else o.inputs.push(a);
  }
  if (o.name !== null && !isDatasetName(o.name)) throw new Error("--name must be a plain name ([A-Za-z0-9_.-], not starting with '.')");
  return o;
}

/** A dataset directory name: [A-Za-z0-9_.-] only and not starting with '.' (no '.', '..', hidden dirs). */
export function isDatasetName(name) {
  return typeof name === 'string' && /^[\w.-]+$/.test(name) && !name.startsWith('.');
}

/** `dir` is a directory strictly inside `parent` (what --force may delete). */
export function isStrictChild(parent, dir) {
  const rel = relative(resolve(parent), resolve(dir));
  return rel !== '' && rel !== '..' && !rel.startsWith(`..${sep}`) && !isAbsolute(rel);
}

/** "sfm-rec-…-part2of3.zip" → { base: "sfm-rec-…", part: 2, count: 3 }; a single archive → part 1 of 1. */
export function partOf(file) {
  const b = basename(file).replace(/\.zip$/i, '');
  const m = /^(.*)-part(\d+)of(\d+)$/.exec(b);
  return m ? { base: m[1], part: Number(m[2]), count: Number(m[3]) } : { base: b, part: 1, count: 1 };
}

async function expandInputs(inputs) {
  const files = [];
  for (const p of inputs) {
    const st = await stat(p);
    if (st.isDirectory()) {
      for (const f of (await readdir(p)).sort()) if (/\.zip$/i.test(f)) files.push(join(p, f));
    } else files.push(p);
  }
  return files;
}

async function main() {
  let o;
  try {
    o = parseArgs(process.argv.slice(2));
  } catch (e) {
    console.error(`error: ${e.message}\n${USAGE}`);
    process.exit(2);
  }
  if (o.help || o.inputs.length === 0) {
    console.log(USAGE);
    process.exit(o.help ? 0 : 2);
  }
  const files = await expandInputs(o.inputs);
  if (!files.length) {
    console.error('error: no .zip files given');
    process.exit(2);
  }
  const parts = files.map((f) => ({ file: f, ...partOf(f) })).sort((a, b) => a.part - b.part);
  const name = o.name ?? parts[0].base.replace(/[^\w.-]+/g, '-');
  if (!isDatasetName(name)) {
    console.error(`error: cannot derive a dataset name from ${basename(parts[0].file)}: use --name`);
    process.exit(2);
  }
  const dir = join(o.out, name);
  if (!isStrictChild(o.out, dir)) {
    console.error(`error: ${dir} is not a directory inside ${o.out}`);
    process.exit(2);
  }
  const existing = await readdir(dir).catch(() => []);
  if (existing.length && !o.force) {
    console.error(`error: ${dir} exists (use --force to replace it)`);
    process.exit(2);
  }
  if (existing.length) await rm(dir, { recursive: true, force: true });
  await mkdir(dir, { recursive: true });
  let entries = 0;
  let bytes = 0;
  for (const p of parts) {
    const zip = readStoreZip(new Uint8Array(await readFile(p.file)));
    for (const e of zip) {
      const target = join(dir, e.name); // names validated by readStoreZip (no absolute / '..')
      await mkdir(dirname(target), { recursive: true });
      await writeFile(target, e.data);
      entries++;
      bytes += e.data.length;
    }
    console.log(`[import] ${basename(p.file)}: ${zip.length} files`);
  }
  // ---- check ----
  const problems = [];
  const count = Math.max(...parts.map((p) => p.count));
  const have = new Set(parts.map((p) => p.part));
  for (let k = 1; k <= count; k++) if (!have.has(k)) problems.push(`part ${k} of ${count} is missing`);
  let meta = null;
  try {
    meta = JSON.parse(await readFile(join(dir, 'meta.json'), 'utf8'));
  } catch {
    problems.push('meta.json is missing (it is in part 1)');
  }
  let rows = 0;
  let missing = 0;
  let noPixels = 0;
  let notSent = 0;
  try {
    const text = await readFile(join(dir, 'frames.jsonl'), 'utf8');
    for (const line of text.split('\n')) {
      if (!line.trim()) continue;
      rows++;
      const row = JSON.parse(line);
      if (row?.notSent) notSent++;
      // recorded without pixels on the phone (the engine still got it: its IMU is replayed)
      if (row?.image === null) {
        noPixels++;
        continue;
      }
      const file = row?.image?.file;
      if (typeof file !== 'string' || !(await stat(join(dir, file)).catch(() => null))) missing++;
    }
  } catch {
    problems.push('frames.jsonl is missing (it is in part 1)');
  }
  if (missing) problems.push(`${missing} of ${rows} frames have no JPEG`);
  const notes = [noPixels ? `${noPixels} recorded without pixels` : '', notSent ? `${notSent} never reached the engine` : ''].filter(Boolean).join(', ');
  console.log(`[import] ${dir}: ${entries} files, ${(bytes / 1e6).toFixed(1)} MB; ${rows} frames${notes ? ` (${notes})` : ''}${meta ? ` (${meta.mode}, ${meta.quality}, ${meta.duration} s at ${meta.fps} Hz, ${meta.width}×${meta.height}${meta.locationStripped ? ', no location' : ''})` : ''}`);
  for (const p of problems) console.log(`[import] PROBLEM: ${p}`);
  console.log(`[import] replay: node e2e/run-e2e.mjs ${name}${o.out !== join(ROOT, 'e2e', 'data') ? '   (run-e2e reads e2e/data/<name>: import there, or symlink it)' : ''}`);
  process.exit(problems.length ? 1 : 0);
}

const isMain = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) await main();
