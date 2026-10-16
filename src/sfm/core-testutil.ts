/**
 * Test-only helpers for the SfM core (not re-exported): exhaustive consistency checks
 * of a `Reconstruction` (track ↔ feature maps, counters, visibility statistics).
 */
import type { CorrespondenceGraph } from './correspondence-graph';
import { VISIBILITY_PYRAMID_LEVELS, type Reconstruction } from './reconstruction';

export interface InvariantOptions {
  /** Require tracks to contain registered images only (mapper invariant; default true). */
  registeredOnly?: boolean;
  /** Require at most one observation per image in a track (default false). */
  oneObservationPerImage?: boolean;
  /** Check the visibility statistics against a brute-force recount (needs the graph). */
  graph?: CorrespondenceGraph;
  /** Images whose visibility is checked (default: all). */
  visibilityImages?: Iterable<number>;
}

/** Brute-force recount of an image's visibility statistics (same definitions as Reconstruction). */
export function bruteForceVisibility(
  rec: Reconstruction,
  g: CorrespondenceGraph,
  id: number,
): { visible: number; score: number; counts: Uint16Array } {
  const img = rec.images.get(id)!;
  const cam = rec.cameras.get(img.cameraId)!;
  const L = VISIBILITY_PYRAMID_LEVELS;
  const D = 1 << L;
  const cells: Set<number>[] = [];
  for (let l = 0; l < L; l++) cells.push(new Set());
  const counts = new Uint16Array(img.point3D.length);
  let visible = 0;
  for (let f = 0; f < img.point3D.length; f++) {
    let cnt = 0;
    for (const c of g.correspondences(id, f)) {
      const o = rec.images.get(c.image);
      if (o && o.point3D[c.feature] >= 0) cnt++;
    }
    counts[f] = cnt;
    if (cnt === 0) continue;
    visible++;
    const kp = img.keypoints;
    let cx = 0,
      cy = 0;
    if (kp) {
      cx = Math.min(D - 1, Math.max(0, Math.floor(((kp[4 * f] + 0.5) * D) / cam.width)));
      cy = Math.min(D - 1, Math.max(0, Math.floor(((kp[4 * f + 1] + 0.5) * D) / cam.height)));
    }
    for (let l = L - 1; l >= 0; l--) {
      cells[l].add(cy * 4096 + cx);
      cx >>= 1;
      cy >>= 1;
    }
  }
  let score = 0;
  for (let l = 0; l < L; l++) score += cells[l].size * (1 << (2 * (l + 1)));
  return { visible, score, counts };
}

/** Returns a list of violated invariants (empty = consistent). */
export function reconstructionInvariantErrors(rec: Reconstruction, opts: InvariantOptions = {}): string[] {
  const errs: string[] = [];
  const push = (s: string): void => {
    if (errs.length < 20) errs.push(s);
  };
  const registeredOnly = opts.registeredOnly ?? true;
  const reg = rec.registeredImageIds();
  const regSet = new Set(reg);
  if (regSet.size !== reg.length) push('duplicate ids in the registration order');
  if (reg.length !== rec.numRegisteredImages) push('numRegisteredImages mismatch');
  for (const img of rec.images.values()) {
    if (img.registered !== regSet.has(img.id)) push(`image ${img.id}: registered flag vs registration order`);
    if (rec.imageLUT()[img.id] !== img) push(`image ${img.id}: LUT entry`);
    if (!rec.cameras.has(img.cameraId)) push(`image ${img.id}: unknown camera`);
    for (let k = 0; k < 9; k++) if (!Number.isFinite(img.pose.R[k])) push(`image ${img.id}: non-finite R`);
    for (let k = 0; k < 3; k++) if (!Number.isFinite(img.pose.t[k])) push(`image ${img.id}: non-finite t`);
  }
  for (const [cid, cam] of rec.cameras)
    for (const k of ['f', 'cx', 'cy', 'k1', 'k2'] as const) if (!Number.isFinite(cam[k])) push(`camera ${cid}: non-finite ${k}`);
  let obs = 0;
  for (const [id, p] of rec.points) {
    if (p.id !== id) push(`point ${id}: id field ${p.id}`);
    if (p.track.length < 2) push(`point ${id}: track length ${p.track.length}`);
    if (!(Number.isFinite(p.xyz[0]) && Number.isFinite(p.xyz[1]) && Number.isFinite(p.xyz[2]))) push(`point ${id}: non-finite xyz`);
    const seen = new Set<string>();
    const imgs = new Set<number>();
    for (const el of p.track) {
      const key = `${el.image}:${el.feature}`;
      if (seen.has(key)) push(`point ${id}: duplicate element ${key}`);
      seen.add(key);
      if (opts.oneObservationPerImage && imgs.has(el.image)) push(`point ${id}: two observations in image ${el.image}`);
      imgs.add(el.image);
      const img = rec.images.get(el.image);
      if (!img) {
        push(`point ${id}: unknown image ${el.image}`);
        continue;
      }
      if (registeredOnly && !img.registered) push(`point ${id}: observation in unregistered image ${el.image}`);
      if (img.point3D[el.feature] !== id) push(`point ${id}: feature ${key} maps to ${img.point3D[el.feature]}`);
    }
    obs += p.track.length;
  }
  if (obs !== rec.numObservations) push(`numObservations ${rec.numObservations} != Σ track lengths ${obs}`);
  for (const img of rec.images.values()) {
    let n = 0;
    const p3 = img.point3D;
    for (let f = 0; f < p3.length; f++) {
      const pid = p3[f];
      if (pid < 0) {
        if (pid !== -1) push(`image ${img.id}: feature ${f} has id ${pid}`);
        continue;
      }
      n++;
      const p = rec.points.get(pid);
      if (!p) push(`image ${img.id}: feature ${f} → deleted point ${pid}`);
      else if (!p.track.some((el) => el.image === img.id && el.feature === f))
        push(`image ${img.id}: feature ${f} → point ${pid} without that element`);
    }
    if (n !== img.numPoints3D) push(`image ${img.id}: numPoints3D ${img.numPoints3D} != ${n}`);
    if (!img.registered && n > 0 && registeredOnly) push(`unregistered image ${img.id} has ${n} points`);
  }
  if (opts.graph) {
    const ids = opts.visibilityImages ? [...opts.visibilityImages] : [...rec.images.keys()];
    for (const id of ids) {
      const ref = bruteForceVisibility(rec, opts.graph, id);
      const img = rec.images.get(id)!;
      if (img.numVisiblePoints3D !== ref.visible) push(`image ${id}: numVisiblePoints3D ${img.numVisiblePoints3D} != ${ref.visible}`);
      if (img.visibilityScore !== ref.score) push(`image ${id}: visibilityScore ${img.visibilityScore} != ${ref.score}`);
    }
  }
  return errs;
}
