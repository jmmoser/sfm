import { describe, expect, it } from 'vitest';
import {
  cameraCenter,
  evaluateDepthMap,
  groundTruthDepthMap,
  lookAt,
  makeSyntheticDataset,
  orbitPoses,
  pinholeIntrinsics,
  raycast,
  renderDataset,
  renderSynthetic,
  surfaceMask,
  textureLuminance,
  type SynthScene,
} from './synthetic';

describe('synthetic ray caster', () => {
  it('lookAt gives a proper rotation with the target on the optical axis', () => {
    const eye = [1, -4, 2];
    const target = [0.5, 0.5, 0.2];
    const p = lookAt(eye, target);
    const R = p.R;
    // orthonormal, det +1
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const d = R[3 * i] * R[3 * j] + R[3 * i + 1] * R[3 * j + 1] + R[3 * i + 2] * R[3 * j + 2];
        expect(d).toBeCloseTo(i === j ? 1 : 0, 12);
      }
    }
    const det = R[0] * (R[4] * R[8] - R[5] * R[7]) - R[1] * (R[3] * R[8] - R[5] * R[6]) + R[2] * (R[3] * R[7] - R[4] * R[6]);
    expect(det).toBeCloseTo(1, 12);
    const C = cameraCenter(p);
    expect(Array.from(C).map((v) => +v.toFixed(9))).toEqual(eye);
    const xc = [0, 1, 2].map((i) => R[3 * i] * target[0] + R[3 * i + 1] * target[1] + R[3 * i + 2] * target[2] + p.t[i]);
    expect(xc[0]).toBeCloseTo(0, 9);
    expect(xc[1]).toBeCloseTo(0, 9);
    expect(xc[2]).toBeGreaterThan(0);
    // world up (+z) maps to image up (−y)
    expect(R[5]).toBeLessThan(0);
  });

  it('renders analytic depth and normals for a fronto-parallel quad', () => {
    const scene: SynthScene = {
      primitives: [{ kind: 'quad', center: [0, 5, 0], axisU: [1, 0, 0], axisV: [0, 0, 1], halfU: 10, halfV: 10, material: { scale: 1 } }],
    };
    const cam = pinholeIntrinsics(40, 30, 1.2);
    const pose = lookAt([0, 0, 0], [0, 1, 0]);
    const r = renderSynthetic(scene, cam, pose, { supersample: 1 });
    for (const i of [0, 455, 1199]) {
      expect(r.depth[i]).toBeCloseTo(5, 9); // z-depth of a fronto-parallel plane is constant
      expect(r.normal[3 * i]).toBeCloseTo(0, 9);
      expect(r.normal[3 * i + 1]).toBeCloseTo(0, 9);
      expect(r.normal[3 * i + 2]).toBeCloseTo(-1, 9);
      expect(r.primitive[i]).toBe(0);
    }
    // texture actually varies
    let mn = 255;
    let mx = 0;
    for (let i = 0; i < 1200; i++) {
      mn = Math.min(mn, r.image.data[4 * i + 1]);
      mx = Math.max(mx, r.image.data[4 * i + 1]);
    }
    expect(mx - mn).toBeGreaterThan(30);
  });

  it('ray-casts boxes and spheres', () => {
    const scene: SynthScene = {
      primitives: [
        { kind: 'box', center: [0, 5, 0], halfSize: [1, 1, 1], yaw: Math.PI / 4, material: { scale: 1 } },
        { kind: 'sphere', center: [0, 0, 10], radius: 2, material: { scale: 1 } },
      ],
    };
    const hb = raycast(scene, [0, 0, 0], [0, 1, 0])!;
    // yawed 45°: the corner points at the camera, distance 5 − √2
    expect(hb.prim).toBe(0);
    expect(hb.t).toBeCloseTo(5 - Math.SQRT2, 9);
    const hs = raycast(scene, [0, 0, 0], [0, 0, 1])!;
    expect(hs.prim).toBe(1);
    expect(hs.t).toBeCloseTo(8, 9);
    expect(hs.nz).toBeCloseTo(-1, 9);
    expect(raycast(scene, [0, 0, 0], [1, 0, 0])).toBeNull();
  });

  it('texture is a 3D function: multi-view consistent colours', () => {
    const ds = makeSyntheticDataset('near', { width: 96, height: 72, numSources: 2 });
    const ref = ds.views[0];
    const src = ds.views[1];
    let diff = 0;
    let n = 0;
    for (let y = 10; y < 62; y += 3) {
      for (let x = 10; x < 86; x += 3) {
        const i = y * 96 + x;
        const z = ds.gtDepth[0][i];
        if (!(z > 0)) continue;
        // back-project and project into the source
        const xc = ((x - ref.cam.cx) / ref.cam.f) * z;
        const yc = ((y - ref.cam.cy) / ref.cam.f) * z;
        const R = ref.pose.R;
        const t = ref.pose.t;
        const d = [xc - t[0], yc - t[1], z - t[2]];
        const X = [0, 1, 2].map((k) => R[k] * d[0] + R[3 + k] * d[1] + R[6 + k] * d[2]);
        const Rs = src.pose.R;
        const ps = [0, 1, 2].map((k) => Rs[3 * k] * X[0] + Rs[3 * k + 1] * X[1] + Rs[3 * k + 2] * X[2] + src.pose.t[k]);
        const u = Math.round((src.cam.f * ps[0]) / ps[2] + src.cam.cx);
        const v = Math.round((src.cam.f * ps[1]) / ps[2] + src.cam.cy);
        if (u < 0 || v < 0 || u >= 96 || v >= 72) continue;
        const j = v * 96 + u;
        // skip occluded points
        if (Math.abs(ds.gtDepth[1][j] - ps[2]) > 0.02 * ps[2]) continue;
        diff += Math.abs(ref.image.data[4 * i + 1] - src.image.data[4 * j + 1]);
        n++;
      }
    }
    expect(n).toBeGreaterThan(200);
    expect(diff / n).toBeLessThan(25);
  });

  it('band-limits fine octaves at large footprints', () => {
    const m = { scale: 1, octaves: 8, contrast: 1 };
    let vFine = 0;
    let vCoarse = 0;
    for (let i = 0; i < 200; i++) {
      const x = i * 0.0137;
      const a = textureLuminance(m, x, 0.3, 0.7, 0.001) - textureLuminance(m, x + 0.01, 0.3, 0.7, 0.001);
      const b = textureLuminance(m, x, 0.3, 0.7, 0.2) - textureLuminance(m, x + 0.01, 0.3, 0.7, 0.2);
      vFine += a * a;
      vCoarse += b * b;
    }
    expect(vCoarse).toBeLessThan(vFine);
  });

  it('evaluateDepthMap / surfaceMask on ground truth', () => {
    const ds = makeSyntheticDataset('near', { width: 80, height: 60, numSources: 1 });
    const gt = groundTruthDepthMap(80, 60, ds.gtDepth[0], ds.gtNormal[0]);
    const s = evaluateDepthMap(gt, ds.gtDepth[0], ds.gtNormal[0]);
    expect(s.validFraction).toBe(1);
    expect(s.accurateFraction).toBe(1);
    expect(s.medianNormalErrorDeg).toBeLessThan(0.01);
    const mask = surfaceMask(ds.views[0].cam, ds.gtDepth[0], ds.gtNormal[0], { edgeRadius: 3 });
    const nMask = mask.reduce((a, b) => a + b, 0);
    expect(nMask).toBeGreaterThan(0.3 * s.gtPixels);
    expect(nMask).toBeLessThan(s.gtPixels);
    expect(ds.range.min).toBeGreaterThan(0);
    expect(ds.range.max).toBeGreaterThan(ds.range.min);
  });
});

describe('synthetic dataset helpers', () => {
  it('orbitPoses + renderDataset give all-around coverage of an object', () => {
    const scene: SynthScene = { primitives: [{ kind: 'sphere', center: [0, 0, 0.5], radius: 0.5, material: { scale: 0.3 } }] };
    const cam = pinholeIntrinsics(48, 36, 1.0);
    const poses = orbitPoses([0, 0, 0.5], 2, 0.3, 8);
    expect(poses.length).toBe(8);
    const ds = renderDataset(scene, cam, poses, { supersample: 1 });
    expect(ds.views.map((v) => v.id)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    for (const d of ds.gtDepth) {
      // the sphere is centred in every view
      const c = d[18 * 48 + 24];
      expect(c).toBeGreaterThan(1.3);
      expect(c).toBeLessThan(1.6);
    }
    expect(ds.range.min).toBeLessThan(1.25);
  });
});
