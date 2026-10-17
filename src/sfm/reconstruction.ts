/**
 * Sparse reconstruction with track bookkeeping (COLMAP `Reconstruction`, Schönberger &
 * Frahm 2016): cameras, images (pose, feature → point map), 3-D points with tracks.
 *
 * When a `CorrespondenceGraph` is attached (`attachGraph`), the reconstruction also
 * maintains, for every image, how many of its features correspond to triangulated
 * points (`numVisiblePoints3D`) and COLMAP's multi-resolution visibility-pyramid
 * score (`visibilityScore`, 6 levels), which drive next-best-view selection.
 *
 * Invariants: a feature maps to at most one point; every track element belongs to an
 * image of this reconstruction; tracks contain observations of registered images only
 * (enforced by the mapper / triangulator, not here).
 */
import type { CameraIntrinsics, Mat3, MetricInfo, Pose, Vec3 } from '../core/types';
import type { CorrespondenceGraph } from './correspondence-graph';
import { isValidImageId } from './database';

export interface Track {
  image: number;
  feature: number;
}

export interface RecImage {
  id: number;
  cameraId: number;
  /** World-to-camera pose (identity placeholder while unregistered). */
  pose: Pose;
  registered: boolean;
  /** Point id per feature, −1 = none. */
  point3D: Int32Array;
  numPoints3D: number;
  // ---- extensions (not in §4.5) ----
  /** Keypoints (stride 4, full-resolution px) shared with the FeatureDatabase, or null. */
  keypoints: Float32Array | null;
  /** Features with ≥ 1 correspondence to a triangulated point (graph attached only). */
  numVisiblePoints3D: number;
  /** COLMAP visibility-pyramid score of those features (graph attached only). */
  visibilityScore: number;
}

export interface RecPoint {
  id: number;
  xyz: Vec3;
  color: [number, number, number];
  /** Mean reprojection error (px) of the track, −1 if unknown. */
  error: number;
  track: Track[];
}

/** Number of levels of the visibility pyramid (COLMAP kNumPoint3DVisibilityPyramidLevels). */
export const VISIBILITY_PYRAMID_LEVELS = 6;
const PYR_MAX_DIM = 1 << VISIBILITY_PYRAMID_LEVELS;
const PYR_CELLS = ((1 << (2 * (VISIBILITY_PYRAMID_LEVELS + 1))) - 4) / 3; // Σ 4^(l+1)

/** Per-image visibility bookkeeping (graph attached). */
interface VisData {
  /** Per feature: number of correspondences that have a triangulated point. */
  counts: Uint16Array;
  /** Pyramid cell counters, levels concatenated (level l has (2^(l+1))² cells). */
  cells: Int32Array;
  w: number;
  h: number;
}

/** Maximum visibility score of an image (all cells of all levels occupied). */
export function maxVisibilityScore(): number {
  let s = 0;
  for (let l = 0; l < VISIBILITY_PYRAMID_LEVELS; l++) {
    const d = 1 << (l + 1);
    s += d * d * d * d;
  }
  return s;
}

export function defaultMetricInfo(): MetricInfo {
  return {
    metric: false,
    source: 'none',
    relativeUncertainty: null,
    gravityAligned: false,
    northAligned: false,
    originLatLonAlt: null,
  };
}

/** True if the camera's intrinsics are implausible (COLMAP `Camera::HasBogusParams`). */
export function hasBogusParams(
  cam: CameraIntrinsics,
  minFocalRatio = 0.1,
  maxFocalRatio = 10,
  maxExtraParam = 1,
): boolean {
  const ratio = cam.f / Math.max(cam.width, cam.height);
  return (
    !(ratio >= minFocalRatio && ratio <= maxFocalRatio) ||
    !(Math.abs(cam.k1) <= maxExtraParam) ||
    !(Math.abs(cam.k2) <= maxExtraParam)
  );
}

/**
 * Squared pixel reprojection error of world point X observed at (u, v); Infinity if
 * the point is not in front of the camera.
 */
export function reprojectionErrorSq(
  cam: CameraIntrinsics,
  pose: Pose,
  X0: number,
  X1: number,
  X2: number,
  u: number,
  v: number,
): number {
  const R = pose.R,
    t = pose.t;
  const z = R[6] * X0 + R[7] * X1 + R[8] * X2 + t[2];
  if (!(z > 1e-12)) return Infinity;
  const iz = 1 / z;
  const x = (R[0] * X0 + R[1] * X1 + R[2] * X2 + t[0]) * iz;
  const y = (R[3] * X0 + R[4] * X1 + R[5] * X2 + t[1]) * iz;
  const r2 = x * x + y * y;
  const d = 1 + r2 * (cam.k1 + cam.k2 * r2);
  const du = cam.f * d * x + cam.cx - u,
    dv = cam.f * d * y + cam.cy - v;
  return du * du + dv * dv;
}

function centerOf(p: Pose, out: Float64Array, o: number): void {
  const R = p.R,
    t = p.t;
  out[o] = -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]);
  out[o + 1] = -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]);
  out[o + 2] = -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]);
}

/** cos of the angle between rays C1→X and C2→X. */
function cosRayAngle(C: Float64Array, a: number, b: number, X: ArrayLike<number>): number {
  const ax = X[0] - C[a],
    ay = X[1] - C[a + 1],
    az = X[2] - C[a + 2];
  const bx = X[0] - C[b],
    by = X[1] - C[b + 1],
    bz = X[2] - C[b + 2];
  const den = Math.sqrt((ax * ax + ay * ay + az * az) * (bx * bx + by * by + bz * bz));
  return den > 0 ? (ax * bx + ay * by + az * bz) / den : 1;
}

let TRI_SCRATCH = new Float64Array(3 * 64);

export class Reconstruction {
  cameras = new Map<number, CameraIntrinsics>();
  images = new Map<number, RecImage>();
  points = new Map<number, RecPoint>();
  metric: MetricInfo = defaultMetricInfo();
  private nextCameraId = 0;
  private nextPointId = 0;
  /** Registered image ids in registration order. */
  private regOrder: number[] = [];
  private graph: CorrespondenceGraph | null = null;
  private vis = new Map<number, VisData>();
  private numObs = 0;
  /** Dense image lookup by id (hot loops). */
  private lut: (RecImage | undefined)[] = [];

  // ------------------------------------------------------------------ cameras / images

  /** Adds a camera (copied) and returns its id. */
  addCamera(cam: CameraIntrinsics, id?: number): number {
    const cid = id ?? this.nextCameraId;
    if (this.cameras.has(cid)) throw new RangeError(`Reconstruction.addCamera: camera ${cid} exists`);
    this.cameras.set(cid, { ...cam });
    this.nextCameraId = Math.max(this.nextCameraId, cid + 1);
    return cid;
  }

  /**
   * Adds an unregistered image with `numFeatures` features. `keypoints` (stride 4,
   * full-resolution px) are needed for reprojection errors and visibility scores.
   */
  addImage(id: number, cameraId: number, numFeatures: number, keypoints: Float32Array | null = null): RecImage {
    if (!isValidImageId(id)) throw new RangeError(`Reconstruction.addImage: bad image id ${id} (integer in [0, 2^20))`);
    if (this.images.has(id)) throw new RangeError(`Reconstruction.addImage: image ${id} exists`);
    if (!this.cameras.has(cameraId)) throw new RangeError(`Reconstruction.addImage: unknown camera ${cameraId}`);
    const img: RecImage = {
      id,
      cameraId,
      pose: { R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), t: new Float64Array(3) },
      registered: false,
      point3D: new Int32Array(numFeatures).fill(-1),
      numPoints3D: 0,
      keypoints,
      numVisiblePoints3D: 0,
      visibilityScore: 0,
    };
    this.images.set(id, img);
    this.lut[id] = img;
    if (this.graph) this.initVis(img);
    return img;
  }

  /** Registers an image with the given pose (copied). */
  registerImage(id: number, pose: Pose): void {
    const img = this.images.get(id);
    if (!img) throw new RangeError(`Reconstruction.registerImage: unknown image ${id}`);
    img.pose = { R: Float64Array.from(pose.R), t: Float64Array.from(pose.t) };
    if (!img.registered) {
      img.registered = true;
      this.regOrder.push(id);
    }
  }

  /** Deletes all observations of the image (points with < 2 observations vanish) and unregisters it. */
  deregisterImage(id: number): void {
    const img = this.images.get(id);
    if (!img) return;
    const p3 = img.point3D;
    for (let f = 0; f < p3.length; f++) if (p3[f] >= 0) this.deleteObservation(id, f);
    if (img.registered) {
      img.registered = false;
      const k = this.regOrder.indexOf(id);
      if (k >= 0) this.regOrder.splice(k, 1);
    }
  }

  /** Dense lookup table image id → image (undefined if absent); do not mutate. */
  imageLUT(): ReadonlyArray<RecImage | undefined> {
    return this.lut;
  }

  /** Registered image ids in registration order. */
  registeredImageIds(): number[] {
    return this.regOrder.slice();
  }

  get numRegisteredImages(): number {
    return this.regOrder.length;
  }

  get numPoints(): number {
    return this.points.size;
  }

  /** Total number of observations (Σ track lengths). */
  get numObservations(): number {
    return this.numObs;
  }

  // ------------------------------------------------------------------ points

  /** Adds a point with the given track (features must not have a point yet). Returns its id. */
  addPoint(xyz: ArrayLike<number>, track: Track[], color?: ArrayLike<number>): number {
    // validate (features free, no element twice — marked −2 while checking) before mutating
    for (let k = 0; k < track.length; k++) {
      const el = track[k];
      const img = this.images.get(el.image);
      const cur = img ? img.point3D[el.feature] : undefined;
      if (cur !== -1) {
        for (let q = 0; q < k; q++) this.lut[track[q].image]!.point3D[track[q].feature] = -1;
        if (!img) throw new RangeError(`Reconstruction.addPoint: unknown image ${el.image}`);
        if (cur === undefined) throw new RangeError(`Reconstruction.addPoint: feature ${el.feature} out of range in image ${el.image}`);
        if (cur === -2) throw new Error(`Reconstruction.addPoint: element (${el.image}, ${el.feature}) appears twice in the track`);
        throw new Error(`Reconstruction.addPoint: feature ${el.feature} of image ${el.image} already has a point`);
      }
      img!.point3D[el.feature] = -2;
    }
    const id = this.nextPointId++;
    const tr: Track[] = new Array(track.length);
    for (let k = 0; k < track.length; k++) {
      const el = track[k];
      tr[k] = { image: el.image, feature: el.feature };
      const img = this.lut[el.image]!;
      img.point3D[el.feature] = id;
      img.numPoints3D++;
    }
    this.numObs += tr.length;
    this.points.set(id, {
      id,
      xyz: Float64Array.of(xyz[0], xyz[1], xyz[2]),
      color: color ? [color[0], color[1], color[2]] : [128, 128, 128],
      error: -1,
      track: tr,
    });
    if (this.graph) for (const el of tr) this.setObservationTriangulated(el.image, el.feature);
    return id;
  }

  /** Adds an observation to an existing point (the feature must not have a point yet). */
  addObservation(pointId: number, t: Track): void {
    const p = this.points.get(pointId);
    if (!p) throw new RangeError(`Reconstruction.addObservation: unknown point ${pointId}`);
    const img = this.images.get(t.image);
    if (!img) throw new RangeError(`Reconstruction.addObservation: unknown image ${t.image}`);
    if (img.point3D[t.feature] !== -1)
      throw new Error(`Reconstruction.addObservation: feature ${t.feature} of image ${t.image} already has a point`);
    img.point3D[t.feature] = pointId;
    img.numPoints3D++;
    p.track.push({ image: t.image, feature: t.feature });
    this.numObs++;
    if (this.graph) this.setObservationTriangulated(t.image, t.feature);
  }

  /**
   * Removes the observation of (image, feature). As in COLMAP, a point whose track
   * would drop below 2 observations is deleted entirely.
   */
  deleteObservation(image: number, feature: number): void {
    const img = this.images.get(image);
    if (!img) return;
    const pid = img.point3D[feature];
    if (pid < 0) return;
    const p = this.points.get(pid)!;
    if (p.track.length <= 2) {
      this.deletePoint(pid);
      return;
    }
    if (this.graph) this.resetTriObservation(image, feature);
    const tr = p.track;
    for (let k = 0; k < tr.length; k++) {
      if (tr[k].image === image && tr[k].feature === feature) {
        tr.splice(k, 1);
        break;
      }
    }
    img.point3D[feature] = -1;
    img.numPoints3D--;
    this.numObs--;
  }

  deletePoint(id: number): void {
    const p = this.points.get(id);
    if (!p) return;
    if (this.graph) for (const el of p.track) this.resetTriObservation(el.image, el.feature);
    for (const el of p.track) {
      const img = this.lut[el.image]!;
      img.point3D[el.feature] = -1;
      img.numPoints3D--;
    }
    this.numObs -= p.track.length;
    this.points.delete(id);
  }

  /**
   * Merges two points into a new one (position and colour weighted by track length,
   * concatenated track); both inputs are deleted. Returns the new id.
   */
  mergePoints(a: number, b: number): number {
    if (a === b) return a;
    const pa = this.points.get(a),
      pb = this.points.get(b);
    if (!pa || !pb) throw new RangeError('Reconstruction.mergePoints: unknown point');
    const wa = pa.track.length,
      wb = pb.track.length;
    const iw = 1 / (wa + wb);
    const xyz = Float64Array.of(
      (wa * pa.xyz[0] + wb * pb.xyz[0]) * iw,
      (wa * pa.xyz[1] + wb * pb.xyz[1]) * iw,
      (wa * pa.xyz[2] + wb * pb.xyz[2]) * iw,
    );
    const color = [
      Math.round((wa * pa.color[0] + wb * pb.color[0]) * iw),
      Math.round((wa * pa.color[1] + wb * pb.color[1]) * iw),
      Math.round((wa * pa.color[2] + wb * pb.color[2]) * iw),
    ];
    const track = pa.track.concat(pb.track);
    this.deletePoint(a);
    this.deletePoint(b);
    return this.addPoint(xyz, track, color);
  }

  // ------------------------------------------------------------------ visibility (graph)

  /**
   * Attaches the correspondence graph and (re)computes the visibility statistics of
   * every image. Images added later get theirs on `addImage`.
   */
  attachGraph(graph: CorrespondenceGraph | null): void {
    this.graph = graph;
    this.vis.clear();
    if (graph) for (const img of this.images.values()) this.initVis(img);
    else
      for (const img of this.images.values()) {
        img.numVisiblePoints3D = 0;
        img.visibilityScore = 0;
      }
  }

  get correspondenceGraph(): CorrespondenceGraph | null {
    return this.graph;
  }

  /**
   * Recomputes the visibility statistics of one image from scratch (use after the
   * graph gained correspondences of an image that was already added).
   */
  syncVisibility(id: number): void {
    const img = this.images.get(id);
    if (img && this.graph) this.initVis(img);
  }

  private initVis(img: RecImage): void {
    const g = this.graph!;
    const cam = this.cameras.get(img.cameraId)!;
    const n = img.point3D.length;
    const vd: VisData = { counts: new Uint16Array(n), cells: new Int32Array(PYR_CELLS), w: cam.width, h: cam.height };
    this.vis.set(img.id, vd);
    img.numVisiblePoints3D = 0;
    img.visibilityScore = 0;
    const heads = g.heads(img.id);
    if (!heads) return;
    const nx = g.nextOf,
      im = g.imageOf,
      ft = g.featureOf;
    const lim = Math.min(n, heads.length);
    for (let f = 0; f < lim; f++) {
      for (let e = heads[f]; e >= 0; e = nx[e]) {
        const other = this.lut[im[e]];
        if (other && other.point3D[ft[e]] >= 0) this.incrementCorr(img, vd, f);
      }
    }
  }

  private cellIndex(vd: VisData, kp: Float32Array | null, f: number): number {
    if (!kp) return 0;
    let cx = Math.floor(((kp[4 * f] + 0.5) * PYR_MAX_DIM) / vd.w);
    let cy = Math.floor(((kp[4 * f + 1] + 0.5) * PYR_MAX_DIM) / vd.h);
    cx = cx < 0 ? 0 : cx >= PYR_MAX_DIM ? PYR_MAX_DIM - 1 : cx;
    cy = cy < 0 ? 0 : cy >= PYR_MAX_DIM ? PYR_MAX_DIM - 1 : cy;
    return cy * PYR_MAX_DIM + cx;
  }

  private incrementCorr(img: RecImage, vd: VisData, f: number): void {
    if (vd.counts[f]++ === 0) img.numVisiblePoints3D++;
    const c = this.cellIndex(vd, img.keypoints, f);
    let cx = c % PYR_MAX_DIM,
      cy = (c / PYR_MAX_DIM) | 0;
    for (let l = VISIBILITY_PYRAMID_LEVELS - 1; l >= 0; l--) {
      const dim = 1 << (l + 1);
      const off = ((1 << (2 * (l + 1))) - 4) / 3;
      const k = off + cy * dim + cx;
      if (vd.cells[k]++ === 0) img.visibilityScore += dim * dim;
      cx >>= 1;
      cy >>= 1;
    }
  }

  private decrementCorr(img: RecImage, vd: VisData, f: number): void {
    if (vd.counts[f] === 0) return; // stale graph (correspondence added after the count) — clamp
    if (--vd.counts[f] === 0) img.numVisiblePoints3D--;
    const c = this.cellIndex(vd, img.keypoints, f);
    let cx = c % PYR_MAX_DIM,
      cy = (c / PYR_MAX_DIM) | 0;
    for (let l = VISIBILITY_PYRAMID_LEVELS - 1; l >= 0; l--) {
      const dim = 1 << (l + 1);
      const off = ((1 << (2 * (l + 1))) - 4) / 3;
      const k = off + cy * dim + cx;
      if (vd.cells[k] > 0 && --vd.cells[k] === 0) img.visibilityScore -= dim * dim;
      cx >>= 1;
      cy >>= 1;
    }
  }

  private setObservationTriangulated(image: number, feature: number): void {
    const g = this.graph!;
    const heads = g.heads(image);
    if (!heads || feature >= heads.length) return;
    const nx = g.nextOf,
      im = g.imageOf,
      ft = g.featureOf;
    for (let e = heads[feature]; e >= 0; e = nx[e]) {
      const other = this.lut[im[e]];
      if (!other) continue;
      const vd = this.vis.get(other.id);
      if (vd && ft[e] < vd.counts.length) this.incrementCorr(other, vd, ft[e]);
    }
  }

  private resetTriObservation(image: number, feature: number): void {
    const g = this.graph!;
    const heads = g.heads(image);
    if (!heads || feature >= heads.length) return;
    const nx = g.nextOf,
      im = g.imageOf,
      ft = g.featureOf;
    for (let e = heads[feature]; e >= 0; e = nx[e]) {
      const other = this.lut[im[e]];
      if (!other) continue;
      const vd = this.vis.get(other.id);
      if (vd && ft[e] < vd.counts.length) this.decrementCorr(other, vd, ft[e]);
    }
  }

  /** Per-feature counts of correspondences with a triangulated point (graph attached), or undefined. */
  visibleCounts(id: number): Uint16Array | undefined {
    return this.vis.get(id)?.counts;
  }

  // ------------------------------------------------------------------ geometry ops

  /**
   * Applies X' = s R X + t to points and poses (R' = R_i Rᵀ, t' = s t_i − R' t), so
   * every projection is unchanged. s must be finite and > 0 (s ≤ 0 would put the
   * points behind their cameras); R must be a rotation.
   */
  transform(s: number, R: Mat3, t: Vec3): void {
    if (!(s > 0 && s < Infinity)) throw new RangeError(`Reconstruction.transform: scale must be finite and > 0 (got ${s})`);
    for (const p of this.points.values()) {
      const x = p.xyz[0],
        y = p.xyz[1],
        z = p.xyz[2];
      p.xyz[0] = s * (R[0] * x + R[1] * y + R[2] * z) + t[0];
      p.xyz[1] = s * (R[3] * x + R[4] * y + R[5] * z) + t[1];
      p.xyz[2] = s * (R[6] * x + R[7] * y + R[8] * z) + t[2];
    }
    const Rn = new Float64Array(9);
    for (const img of this.images.values()) {
      const Ri = img.pose.R,
        ti = img.pose.t;
      // R' = Ri Rᵀ
      for (let r = 0; r < 3; r++)
        for (let c = 0; c < 3; c++)
          Rn[3 * r + c] = Ri[3 * r] * R[3 * c] + Ri[3 * r + 1] * R[3 * c + 1] + Ri[3 * r + 2] * R[3 * c + 2];
      const t0 = s * ti[0] - (Rn[0] * t[0] + Rn[1] * t[1] + Rn[2] * t[2]);
      const t1 = s * ti[1] - (Rn[3] * t[0] + Rn[4] * t[1] + Rn[5] * t[2]);
      const t2 = s * ti[2] - (Rn[6] * t[0] + Rn[7] * t[1] + Rn[8] * t[2]);
      Ri.set(Rn);
      ti[0] = t0;
      ti[1] = t1;
      ti[2] = t2;
    }
  }

  /** Reprojection error (px) of one observation; Infinity if behind the camera or unknown keypoints. */
  observationError(pointId: number, t: Track): number {
    const p = this.points.get(pointId);
    const img = this.images.get(t.image);
    if (!p || !img || !img.keypoints) return Infinity;
    const cam = this.cameras.get(img.cameraId)!;
    const kp = img.keypoints;
    return Math.sqrt(
      reprojectionErrorSq(cam, img.pose, p.xyz[0], p.xyz[1], p.xyz[2], kp[4 * t.feature], kp[4 * t.feature + 1]),
    );
  }

  /**
   * Mean reprojection error (px) over all observations, recomputed from the current
   * geometry (observations of images without keypoints are skipped; observations
   * behind a camera are skipped). 0 if there are none.
   */
  meanReprojectionError(): number {
    let sum = 0,
      n = 0;
    for (const p of this.points.values()) {
      const X0 = p.xyz[0],
        X1 = p.xyz[1],
        X2 = p.xyz[2];
      for (const el of p.track) {
        const img = this.lut[el.image]!;
        const kp = img.keypoints;
        if (!kp) continue;
        const e2 = reprojectionErrorSq(
          this.cameras.get(img.cameraId)!,
          img.pose,
          X0,
          X1,
          X2,
          kp[4 * el.feature],
          kp[4 * el.feature + 1],
        );
        if (e2 < Infinity) {
          sum += Math.sqrt(e2);
          n++;
        }
      }
    }
    return n > 0 ? sum / n : 0;
  }

  /** Recomputes `error` (mean px reprojection error of the track) of every point. */
  updatePointErrors(): void {
    for (const p of this.points.values()) {
      let s = 0,
        n = 0;
      for (const el of p.track) {
        const e = this.observationError(p.id, el);
        if (e < Infinity) {
          s += e;
          n++;
        }
      }
      p.error = n > 0 ? s / n : -1;
    }
  }

  meanTrackLength(): number {
    return this.points.size > 0 ? this.numObs / this.points.size : 0;
  }

  /** Camera centre of an image. */
  imageCenter(id: number, out: Vec3 = new Float64Array(3)): Vec3 {
    const img = this.images.get(id);
    if (!img) throw new RangeError(`Reconstruction.imageCenter: unknown image ${id}`);
    centerOf(img.pose, out, 0);
    return out;
  }

  // ------------------------------------------------------------------ filtering (COLMAP)

  /**
   * Filters the given points: observations with a reprojection error above
   * `maxReprojErrorPx` (or behind the camera) are removed (the point if < 2 remain),
   * then points whose track has no pair of rays with an angle ≥ `minTriAngle`
   * (radians) are deleted. Returns the number of removed observations.
   */
  filterPoints3D(maxReprojErrorPx: number, minTriAngle: number, pointIds: Iterable<number>): number {
    let n = 0;
    const ids = Array.isArray(pointIds) ? (pointIds as number[]) : [...pointIds];
    n += this.filterLargeReprojError(maxReprojErrorPx, ids);
    n += this.filterSmallTriAngle(minTriAngle, ids);
    return n;
  }

  /** `filterPoints3D` for every point observed by one of the given images. */
  filterPoints3DInImages(maxReprojErrorPx: number, minTriAngle: number, imageIds: Iterable<number>): number {
    const ids = new Set<number>();
    for (const iid of imageIds) {
      const img = this.images.get(iid);
      if (!img) continue;
      const p3 = img.point3D;
      for (let f = 0; f < p3.length; f++) if (p3[f] >= 0) ids.add(p3[f]);
    }
    return this.filterPoints3D(maxReprojErrorPx, minTriAngle, [...ids]);
  }

  /** `filterPoints3D` for all points. */
  filterAllPoints3D(maxReprojErrorPx: number, minTriAngle: number): number {
    return this.filterPoints3D(maxReprojErrorPx, minTriAngle, [...this.points.keys()]);
  }

  /**
   * Flat per-image projection table (R 9, t 3, f, cx, cy, k1, k2 = 17 doubles) for
   * images with keypoints; `slot[id]` = row or −1.
   */
  private projectionTable(): { table: Float64Array; slot: Int32Array } {
    const slot = new Int32Array(this.lut.length).fill(-1);
    const table = new Float64Array(17 * this.images.size);
    let r = 0;
    for (const img of this.images.values()) {
      if (!img.keypoints) continue;
      const cam = this.cameras.get(img.cameraId)!;
      const o = 17 * r;
      table.set(img.pose.R, o);
      table.set(img.pose.t, o + 9);
      table[o + 12] = cam.f;
      table[o + 13] = cam.cx;
      table[o + 14] = cam.cy;
      table[o + 15] = cam.k1;
      table[o + 16] = cam.k2;
      slot[img.id] = r++;
    }
    return { table, slot };
  }

  private filterLargeReprojError(maxErr: number, ids: number[]): number {
    const max2 = maxErr * maxErr;
    let removed = 0;
    const bad: number[] = [];
    const { table: T, slot } = this.projectionTable();
    const lut = this.lut;
    for (const id of ids) {
      const p = this.points.get(id);
      if (!p) continue;
      bad.length = 0;
      let sum = 0,
        good = 0;
      const X0 = p.xyz[0],
        X1 = p.xyz[1],
        X2 = p.xyz[2];
      const tr = p.track;
      for (let k = 0; k < tr.length; k++) {
        const el = tr[k];
        const r = slot[el.image];
        if (r < 0) {
          good++;
          continue;
        }
        const o = 17 * r;
        const z = T[o + 6] * X0 + T[o + 7] * X1 + T[o + 8] * X2 + T[o + 11];
        let e2 = Infinity;
        if (z > 1e-12) {
          const iz = 1 / z;
          const x = (T[o] * X0 + T[o + 1] * X1 + T[o + 2] * X2 + T[o + 9]) * iz;
          const y = (T[o + 3] * X0 + T[o + 4] * X1 + T[o + 5] * X2 + T[o + 10]) * iz;
          const r2 = x * x + y * y;
          const fd = T[o + 12] * (1 + r2 * (T[o + 15] + T[o + 16] * r2));
          const kp = lut[el.image]!.keypoints!;
          const du = fd * x + T[o + 13] - kp[4 * el.feature],
            dv = fd * y + T[o + 14] - kp[4 * el.feature + 1];
          e2 = du * du + dv * dv;
        }
        if (e2 > max2 || !(e2 < Infinity)) bad.push(k);
        else {
          sum += Math.sqrt(e2);
          good++;
        }
      }
      if (bad.length === 0) {
        p.error = good > 0 ? sum / good : -1;
        continue;
      }
      if (tr.length - bad.length <= 1) {
        removed += tr.length;
        this.deletePoint(id);
        continue;
      }
      const els = bad.map((k) => tr[k]);
      for (const el of els) this.deleteObservation(el.image, el.feature);
      removed += els.length;
      p.error = good > 0 ? sum / good : -1;
    }
    return removed;
  }

  private filterSmallTriAngle(minAngle: number, ids: number[]): number {
    const cosMax = Math.cos(minAngle);
    let removed = 0;
    for (const id of ids) {
      const p = this.points.get(id);
      if (!p) continue;
      const n = p.track.length;
      if (TRI_SCRATCH.length < 3 * n) TRI_SCRATCH = new Float64Array(3 * n * 2);
      const C = TRI_SCRATCH;
      for (let k = 0; k < n; k++) centerOf(this.lut[p.track[k].image]!.pose, C, 3 * k);
      let ok = false;
      for (let a = 0; a < n && !ok; a++)
        for (let b = a + 1; b < n; b++)
          if (cosRayAngle(C, 3 * a, 3 * b, p.xyz) <= cosMax) {
            ok = true;
            break;
          }
      if (!ok) {
        removed += n;
        this.deletePoint(id);
      }
    }
    return removed;
  }

  /**
   * Removes observations IN the given images whose reprojection error exceeds
   * maxReprojErrorPx (or that are behind the camera); a point left with < 2
   * observations is deleted. Cheaper than `filterPoints3DInImages` (which checks every
   * observation of the images' points) after a local BA that only moved these poses.
   * Returns the number of removed observations.
   */
  filterObservationsInImages(maxReprojErrorPx: number, imageIds: Iterable<number>): number {
    const max2 = maxReprojErrorPx * maxReprojErrorPx;
    let removed = 0;
    const bad: number[] = [];
    for (const iid of imageIds) {
      const img = this.images.get(iid);
      if (!img || !img.keypoints) continue;
      const cam = this.cameras.get(img.cameraId)!;
      const kp = img.keypoints;
      const p3 = img.point3D;
      bad.length = 0;
      for (let f = 0; f < p3.length; f++) {
        const pid = p3[f];
        if (pid < 0) continue;
        const X = this.points.get(pid)!.xyz;
        const e2 = reprojectionErrorSq(cam, img.pose, X[0], X[1], X[2], kp[4 * f], kp[4 * f + 1]);
        if (!(e2 <= max2)) bad.push(f);
      }
      for (const f of bad) {
        const pid = p3[f];
        if (pid < 0) continue;
        const p = this.points.get(pid)!;
        removed += p.track.length <= 2 ? p.track.length : 1;
        this.deleteObservation(iid, f);
      }
    }
    return removed;
  }

  /** Removes observations of points behind their camera (COLMAP, before global BA). */
  filterObservationsWithNegativeDepth(): number {
    let removed = 0;
    const els: Track[] = [];
    for (const p of this.points.values()) {
      for (const el of p.track) {
        const img = this.lut[el.image]!;
        const R = img.pose.R;
        const z = R[6] * p.xyz[0] + R[7] * p.xyz[1] + R[8] * p.xyz[2] + img.pose.t[2];
        if (!(z > 1e-12)) els.push(el);
      }
    }
    for (const el of els) {
      const img = this.lut[el.image]!;
      if (img.point3D[el.feature] >= 0) {
        this.deleteObservation(el.image, el.feature);
        removed++;
      }
    }
    return removed;
  }

  /**
   * Deregisters registered images without any point or whose camera has bogus
   * intrinsics. Returns their ids.
   */
  filterImages(minFocalRatio: number, maxFocalRatio: number, maxExtraParam: number): number[] {
    const out: number[] = [];
    for (const id of this.regOrder) {
      const img = this.images.get(id)!;
      if (img.numPoints3D === 0 || hasBogusParams(this.cameras.get(img.cameraId)!, minFocalRatio, maxFocalRatio, maxExtraParam))
        out.push(id);
    }
    for (const id of out) this.deregisterImage(id);
    return out;
  }

  /** Number of points shared by each pair of registered images that co-observe one (for diagnostics). */
  numSharedPoints(a: number, b: number): number {
    const ia = this.images.get(a),
      ib = this.images.get(b);
    if (!ia || !ib) return 0;
    let n = 0;
    const p3 = ia.point3D;
    for (let f = 0; f < p3.length; f++) {
      const pid = p3[f];
      if (pid < 0) continue;
      for (const el of this.points.get(pid)!.track)
        if (el.image === b) {
          n++;
          break;
        }
    }
    return n;
  }

  // ------------------------------------------------------------------ copy

  /** Deep copy (keypoint arrays and the attached graph are shared, not copied). */
  clone(): Reconstruction {
    const r = new Reconstruction();
    for (const [id, c] of this.cameras) r.cameras.set(id, { ...c });
    for (const [id, img] of this.images) {
      const c: RecImage = {
        ...img,
        pose: { R: Float64Array.from(img.pose.R), t: Float64Array.from(img.pose.t) },
        point3D: Int32Array.from(img.point3D),
      };
      r.images.set(id, c);
      r.lut[id] = c;
    }
    for (const [id, p] of this.points)
      r.points.set(id, {
        id,
        xyz: Float64Array.from(p.xyz),
        color: [p.color[0], p.color[1], p.color[2]],
        error: p.error,
        track: p.track.map((el) => ({ image: el.image, feature: el.feature })),
      });
    r.metric = { ...this.metric, originLatLonAlt: this.metric.originLatLonAlt ? [...this.metric.originLatLonAlt] : null };
    r.nextCameraId = this.nextCameraId;
    r.nextPointId = this.nextPointId;
    r.regOrder = this.regOrder.slice();
    r.graph = this.graph;
    r.numObs = this.numObs;
    for (const [id, vd] of this.vis)
      r.vis.set(id, { counts: Uint16Array.from(vd.counts), cells: Int32Array.from(vd.cells), w: vd.w, h: vd.h });
    return r;
  }
}
