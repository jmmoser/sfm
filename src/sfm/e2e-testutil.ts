/**
 * Shared helpers of the SfM end-to-end tests (synthetic scenes with the acceptance
 * settings: 0.7 px noise, 25 % outlier matches, focal initialised 8 % off, true
 * k1 = 0.03). Test-only module (not re-exported).
 */
import type { CameraIntrinsics, FeatureSet, FrameInfo, Pose } from '../core/types';
import { projectWorld } from '../geometry/camera';
import { estimateTwoViewGeometry } from '../geometry/twoview';
import { Rng } from '../math/random';
import { rotvecToMat3 } from '../math/rotation';
import { CorrespondenceGraph } from './correspondence-graph';
import { FeatureDatabase } from './database';
import { LiveMapper, type LiveMapperOptions } from './live';
import { buildSyntheticDatabase, generateScene, type SyntheticScene } from './synthetic';

export interface E2EData {
  scene: SyntheticScene;
  db: FeatureDatabase;
  guess: CameraIntrinsics;
  verifyMs: number;
}

export function makeE2EData(kind: 'orbit' | 'walk', numImages: number, seed: number): E2EData {
  const scene = generateScene({ kind, numImages, seed, k1: 0.03, noisePx: 0.7, outlierRatio: 0.25, descriptors: false });
  const guess = { ...scene.camera, f: scene.camera.f * 1.08, k1: 0, k2: 0 };
  const { db, stats } = buildSyntheticDatabase(scene, { camera: guess });
  return { scene, db, guess, verifyMs: stats.timeMs };
}

/**
 * Streams the keyframes in capture order into a LiveMapper the way the engine does:
 * add the image to the graph, add its verified pairs with earlier keyframes, then
 * `addKeyframe`. Returns the mapper and per-keyframe wall times.
 */
export function streamLive(data: E2EData, opts: Partial<LiveMapperOptions> = {}): { live: LiveMapper; ms: number[] } {
  const { db } = data;
  const graph = new CorrespondenceGraph();
  const live = new LiveMapper(db, graph, { intrinsics: data.guess, ...opts });
  const ids = db.frameIds();
  const ms: number[] = [];
  for (const id of ids) {
    graph.addImage(id, db.features.get(id)!.count);
    for (const j of ids) {
      if (j >= id) break;
      const g = db.getTwoView(j, id);
      if (g && g.config !== 'degenerate' && g.inliers.length >= 30) graph.addTwoView(j, id, g.inliers);
    }
    const t0 = performance.now();
    live.addKeyframe(id);
    ms.push(performance.now() - t0);
  }
  return { live, ms };
}

export const ROTATION_ONLY_CAMERA: CameraIntrinsics = { width: 1280, height: 960, f: 900, cx: 639.5, cy: 479.5, k1: 0, k2: 0 };

/** Frames + features for a pure-rotation capture (no baseline). */
export function rotationOnlyDb(): FeatureDatabase {
  const rng = new Rng(4);
  const cam = ROTATION_ONLY_CAMERA;
  const X: number[] = [];
  for (let j = 0; j < 3000; j++) {
    const az = rng.next() * 2 * Math.PI,
      el = (rng.next() - 0.5) * 1.2,
      d = 3 + 20 * rng.next();
    X.push(d * Math.cos(el) * Math.sin(az), -d * Math.sin(el), d * Math.cos(el) * Math.cos(az));
  }
  const db = new FeatureDatabase();
  const vis: Map<number, number>[] = [];
  const uv = new Float64Array(2);
  for (let i = 0; i < 12; i++) {
    const pose: Pose = { R: rotvecToMat3(Float64Array.of(0, 0.12 * i, 0)), t: new Float64Array(3) };
    const kp: number[] = [];
    const v = new Map<number, number>();
    for (let j = 0; j < 3000; j++) {
      const P = X.slice(3 * j, 3 * j + 3);
      const z = pose.R[6] * P[0] + pose.R[7] * P[1] + pose.R[8] * P[2];
      if (z < 0.5 || !projectWorld(cam, pose, P, uv) || uv[0] < 0 || uv[1] < 0 || uv[0] > 1279 || uv[1] > 959) continue;
      v.set(j, kp.length / 4);
      kp.push(uv[0] + 0.5 * rng.gaussian(), uv[1] + 0.5 * rng.gaussian(), 2, 0);
    }
    vis.push(v);
    const n = kp.length / 4;
    const fs: FeatureSet = { count: n, keypoints: Float32Array.from(kp), descriptors: new Uint8Array(128 * n), scores: new Float32Array(n), width: 1280, height: 960 };
    const info: FrameInfo = { id: i, t: i, width: 1280, height: 960, sharpness: 1, R_cam_device: Float64Array.of(1, 0, 0, 0, -1, 0, 0, 0, -1), gravityCam: null, heading: null, geo: null };
    db.addFrame(info, fs);
  }
  for (let i = 0; i < 12; i++)
    for (let j = i + 1; j < Math.min(12, i + 4); j++) {
      const m: number[] = [];
      for (const [p, fa] of vis[i]) {
        const fb = vis[j].get(p);
        if (fb !== undefined) m.push(fa, fb);
      }
      const geom = estimateTwoViewGeometry(db.features.get(i)!.keypoints, db.features.get(j)!.keypoints, Uint32Array.from(m), cam, cam);
      db.setTwoView(i, j, geom);
    }
  return db;
}

