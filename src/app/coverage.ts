/**
 * Object-mode capture coverage: which sides and heights around the object already have
 * registered keyframes. Drawn by the capture screen's coverage ring (src/app/components/
 * coverage-ring.ts) with a one-line prompt and the Finish readiness hint; recorded by the e2e
 * harness as `result.live.coverage` (src/harness/engine/pipeline.ts).
 *
 * Model (recomputed from all registered keyframes on every live model update — keyframe rate,
 * O(n): < 1 ms for 280 keyframes):
 *  - centre: the optical-axis convergence point of the keyframes (src/geometry/convergence.ts,
 *    the point `classifyScene` calls an orbit). Trusted (`stable`) once there are ≥ 6 cameras and
 *    they pass its orbit test (they look at the point). It is the least-squares point of ALL
 *    keyframes, so it settles as the orbit grows; there is no smoothing across updates because
 *    the live model's frame can jump (a re-initialised model), which would make a remembered
 *    centre wrong.
 *  - up: `estimateUpVector` (src/viewer/helpers.ts): gravity (`upCam`) where known, the camera
 *    right axes otherwise.
 *  - per keyframe, d = C − centre: azimuth in the plane ⟂ up, referenced to the FIRST keyframe
 *    (12 sectors of 30°, sector 0 centred on it, counter-clockwise seen from above), and
 *    elevation asin(d̂ · up) in three bands: low −10–10° (the object's height: a camera there
 *    scatters around the centre's horizon), mid 10–40° (a level loop: object-real spans 13–35°),
 *    high > 40° (looking down). Views from further below are left out of the bands. A cell is
 *    covered with ≥ 2 keyframes.
 *  - two tiers for the prompt: level (low + mid, merged per sector) and high. Both low and mid
 *    are a loop round the object's sides, at or a little above its centre — an object at eye
 *    level (a bust, a plant, a person) is circled in the low band, and a loop straddling 10° fills
 *    both — while the views from above (> 40°) are the measured weak axis (below). The tier the
 *    user circles in is the one with more keyframes; `primaryBand` is its band with the most (the
 *    ring's target band).
 *  - completeness: covered sectors of the circled tier + of the other tier, / 24. Azimuth only
 *    (`elevation: false`: no motion sensor, so up is a guess): covered sectors of any height / 12.
 *  - prompt (first rule that matches): not stable → 'locating'; in the circled tier an empty run
 *    ≥ 3 sectors → 'other_side', fewer than 10 covered → 'fill_gaps'; the other tier < 10 covered
 *    → 'look_down' (high missing) / 'go_lower' (level missing); else 'done'. Azimuth only: the
 *    same on the merged ring, then 'done'.
 *
 * Why these numbers: height is the measured weak axis — splat PSNR of held-out synthetic views
 * falls from 30 dB to 26.8 / 22.1 dB at +12 / +25 cm above the object-real orbit, which spans
 * 13–35° elevation while those views sit at 27–52° (e2e/README.md "Novel views"); PSNR tracks the
 * angle to the nearest training camera (2–3° → 30 dB, 10° → 26.8, 19° → 22.1 dB). The cell size,
 * the band edges and the ≥ 2 rule are a first guess, to be set from per-view PSNR against the
 * nearest-keyframe angle (e2e/nvs.mjs) and real captures; the readiness never blocks Finish
 * (statues and wall-mounted objects cannot be seen from above).
 */
import type { Pose, Vec3 } from '../core/types';
import { opticalAxisConvergence } from '../geometry/convergence';
import { estimateUpVector } from '../viewer/helpers';
import type { LiveCamera } from '../workers/protocol';

const DEG = Math.PI / 180;

/** Azimuth sectors (30° each) and elevation bands (0 low, 1 mid, 2 high). */
export const COVERAGE_SECTORS = 12;
export const COVERAGE_BANDS = 3;
/** Upper edges of the low and mid bands (rad). */
export const COVERAGE_BAND_EDGES: readonly [number, number] = [10 * DEG, 40 * DEG];
/** Lower edge of the low band (rad): views from further below the centre's horizon are in no band. */
export const COVERAGE_MIN_ELEVATION = -10 * DEG;
/** Registered cameras (that also pass the orbit test) before the centre is trusted. */
export const COVERAGE_MIN_CAMERAS = 6;

/** What the coverage ring asks for next (copy: format.ts `coveragePromptText`). */
export type CoveragePrompt = 'locating' | 'other_side' | 'fill_gaps' | 'look_down' | 'go_lower' | 'done';

export interface CoverageOptions {
  /** Elevation bands (needs gravity). false: azimuth only. Default true. */
  elevation?: boolean;
  /** Keyframes in a cell for it to count as covered. Default 2. */
  minPerCell?: number;
}

/** A viewing position around the object: sector 0–11 and band 0–2 (−1 = below the low band). */
export interface CoverageCell {
  sector: number;
  band: number;
}

export interface Coverage {
  /** The centre is trusted (≥ 6 cameras that pass the orbit test); everything below is empty otherwise. */
  stable: boolean;
  /** Elevation bands in use (false: azimuth only). */
  elevation: boolean;
  /** World up and the object centre (null until stable). */
  up: Vec3 | null;
  centre: Vec3 | null;
  /** Azimuth basis: e1 towards the first keyframe, e2 = up × e1 (null until stable). */
  e1: Vec3 | null;
  e2: Vec3 | null;
  /** counts[band][sector]: keyframes per cell (views below the low band are not counted). */
  counts: number[][];
  /** Keyframes per sector at any height (incl. below the low band) — the azimuth-only ring. */
  sectors: number[];
  /** Covered cells per band (low, mid, high). */
  covered: [number, number, number];
  /** Covered sectors at any height. */
  coveredSectors: number;
  /** Keyframes a cell needs to count as covered (CoverageOptions.minPerCell). */
  minPerCell: number;
  /** The band the user circles in: the band with the most keyframes of the circled tier (see the header). */
  primaryBand: 0 | 1 | 2;
  /**
   * Keyframes per sector of the circled tier (level: low + mid; high) — what 'other_side' /
   * 'fill_gaps' test. Azimuth only: `sectors`.
   */
  loop: number[];
  /** 0–1, see the header. */
  completeness: number;
  prompt: CoveragePrompt;
  /** Cell of the pose passed as `current` (null: none, or not stable). */
  currentCell: CoverageCell | null;
  /** Per input camera: azimuth [0, 2π) and elevation (rad) around the centre; NaN until stable. */
  azimuthRad: Float64Array;
  elevationRad: Float64Array;
}

function zeros(rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () => new Array<number>(cols).fill(0));
}

function centreOf(p: Pose, out: Float64Array): Float64Array {
  const { R, t } = p;
  out[0] = -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]);
  out[1] = -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]);
  out[2] = -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]);
  return out;
}

/** Elevation band of an elevation (rad): 0 low, 1 mid, 2 high, −1 below the low band. */
export function coverageBand(elevationRad: number): number {
  if (!(elevationRad >= COVERAGE_MIN_ELEVATION)) return -1;
  return elevationRad < COVERAGE_BAND_EDGES[0] ? 0 : elevationRad < COVERAGE_BAND_EDGES[1] ? 1 : 2;
}

/** Sector of an azimuth (rad, any range): sector 0 is centred on azimuth 0. */
export function coverageSector(azimuthRad: number): number {
  const s = Math.floor(azimuthRad / ((2 * Math.PI) / COVERAGE_SECTORS) + 0.5);
  return ((s % COVERAGE_SECTORS) + COVERAGE_SECTORS) % COVERAGE_SECTORS;
}

/** Longest circular run of sectors with fewer than `min` keyframes. */
export function largestGap(row: readonly number[], min: number): number {
  const n = row.length;
  let best = 0,
    run = 0;
  for (let i = 0; i < 2 * n; i++) {
    run = row[i % n] < min ? run + 1 : 0;
    best = Math.max(best, run);
  }
  return Math.min(best, n);
}

/** Azimuth / elevation (rad) of a world position around the coverage centre (NaN until stable). */
function aroundCentre(cov: Pick<Coverage, 'centre' | 'up' | 'e1' | 'e2'>, C: ArrayLike<number>): { azimuth: number; elevation: number } {
  const { centre: P, up, e1, e2 } = cov;
  if (!P || !up || !e1 || !e2) return { azimuth: NaN, elevation: NaN };
  const d0 = C[0] - P[0],
    d1 = C[1] - P[1],
    d2 = C[2] - P[2];
  const len = Math.hypot(d0, d1, d2);
  if (!(len > 0)) return { azimuth: NaN, elevation: NaN };
  const u = (d0 * up[0] + d1 * up[1] + d2 * up[2]) / len;
  const x = d0 * e1[0] + d1 * e1[1] + d2 * e1[2];
  const y = d0 * e2[0] + d1 * e2[1] + d2 * e2[2];
  let az = Math.atan2(y, x);
  if (az < 0) az += 2 * Math.PI;
  if (az >= 2 * Math.PI) az = 0; // −1e-19 + 2π
  return { azimuth: az, elevation: Math.asin(Math.max(-1, Math.min(1, u))) };
}

/** Cell of a camera pose (e.g. the newest tracked frame) in a coverage map; null until stable. */
export function coverageCell(cov: Coverage, pose: Pose): CoverageCell | null {
  if (!cov.stable) return null;
  const { azimuth, elevation } = aroundCentre(cov, centreOf(pose, new Float64Array(3)));
  if (!Number.isFinite(azimuth)) return null;
  return { sector: coverageSector(azimuth), band: coverageBand(elevation) };
}

function emptyCoverage(n: number, elevation: boolean, minPerCell: number): Coverage {
  return {
    stable: false,
    elevation,
    up: null,
    centre: null,
    e1: null,
    e2: null,
    counts: zeros(COVERAGE_BANDS, COVERAGE_SECTORS),
    sectors: new Array<number>(COVERAGE_SECTORS).fill(0),
    covered: [0, 0, 0],
    coveredSectors: 0,
    minPerCell,
    primaryBand: 1,
    loop: new Array<number>(COVERAGE_SECTORS).fill(0),
    completeness: 0,
    prompt: 'locating',
    currentCell: null,
    azimuthRad: new Float64Array(n).fill(NaN),
    elevationRad: new Float64Array(n).fill(NaN),
  };
}

/**
 * Coverage of the registered keyframes `cams` (LiveUpdate.cameras: keyframe order, the first is
 * the azimuth reference), and the cell of `current` (LiveUpdate.framePose) when given.
 */
export function computeCoverage(cams: readonly Pick<LiveCamera, 'pose' | 'upCam'>[], opts: CoverageOptions & { current?: Pose | null } = {}): Coverage {
  const n = cams.length;
  const elevation = opts.elevation !== false;
  const minPer = Math.max(1, opts.minPerCell ?? 2);
  const cov = emptyCoverage(n, elevation, minPer);
  if (n < COVERAGE_MIN_CAMERAS) return cov;
  const conv = opticalAxisConvergence(cams.map((c) => c.pose));
  if (!conv.point || !conv.converging) return cov;
  const up = estimateUpVector(cams.map((c) => ({ R: c.pose.R, upCam: c.upCam })));
  if (!up) return cov;
  const P = conv.point;
  // azimuth reference: the first keyframe with a horizontal offset from the centre
  const C = new Float64Array(3);
  let e1: Float64Array | null = null;
  for (const c of cams) {
    centreOf(c.pose, C);
    const d = [C[0] - P[0], C[1] - P[1], C[2] - P[2]];
    const u = d[0] * up[0] + d[1] * up[1] + d[2] * up[2];
    const h = [d[0] - u * up[0], d[1] - u * up[1], d[2] - u * up[2]];
    const hn = Math.hypot(h[0], h[1], h[2]);
    if (hn > 1e-9 * Math.max(1, Math.hypot(d[0], d[1], d[2]))) {
      e1 = Float64Array.of(h[0] / hn, h[1] / hn, h[2] / hn);
      break;
    }
  }
  if (!e1) return cov;
  const e2 = Float64Array.of(up[1] * e1[2] - up[2] * e1[1], up[2] * e1[0] - up[0] * e1[2], up[0] * e1[1] - up[1] * e1[0]);
  cov.stable = true;
  cov.up = up;
  cov.centre = P;
  cov.e1 = e1;
  cov.e2 = e2;
  for (let i = 0; i < n; i++) {
    const { azimuth, elevation: el } = aroundCentre(cov, centreOf(cams[i].pose, C));
    cov.azimuthRad[i] = azimuth;
    cov.elevationRad[i] = el;
    if (!Number.isFinite(azimuth)) continue;
    const s = coverageSector(azimuth);
    cov.sectors[s]++;
    const b = coverageBand(el);
    if (b >= 0) cov.counts[b][s]++;
  }
  for (let b = 0; b < COVERAGE_BANDS; b++) cov.covered[b] = cov.counts[b].filter((k) => k >= minPer).length;
  cov.coveredSectors = cov.sectors.filter((k) => k >= minPer).length;
  if (elevation) {
    const total = (row: readonly number[]): number => row.reduce((a, b) => a + b, 0);
    const coveredOf = (row: readonly number[]): number => row.filter((k) => k >= minPer).length;
    const [low, mid, high] = cov.counts;
    const level = low.map((k, s) => k + mid[s]);
    const fromAbove = total(high) > total(level);
    const loop = fromAbove ? high : level;
    const other = fromAbove ? level : high;
    cov.primaryBand = fromAbove ? 2 : total(low) > total(mid) ? 0 : 1;
    cov.loop = loop.slice();
    const coveredLoop = coveredOf(loop),
      coveredOther = coveredOf(other);
    cov.completeness = (coveredLoop + coveredOther) / (2 * COVERAGE_SECTORS);
    if (largestGap(loop, minPer) >= 3) cov.prompt = 'other_side';
    else if (coveredLoop < 10) cov.prompt = 'fill_gaps';
    else if (coveredOther < 10) cov.prompt = fromAbove ? 'go_lower' : 'look_down';
    else cov.prompt = 'done';
  } else {
    cov.completeness = cov.coveredSectors / COVERAGE_SECTORS;
    cov.loop = cov.sectors.slice();
    if (largestGap(cov.sectors, minPer) >= 3) cov.prompt = 'other_side';
    else if (cov.coveredSectors < 10) cov.prompt = 'fill_gaps';
    else cov.prompt = 'done';
  }
  if (opts.current) cov.currentCell = coverageCell(cov, opts.current);
  return cov;
}
