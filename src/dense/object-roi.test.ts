import { describe, expect, it } from 'vitest';
import type { ObjectRoi, Pose } from '../core/types';
import { Rng } from '../math/random';
import { objectRoiPivot, proposeObjectRoi, type ObjectRoiInput } from './object-roi';
import { lookAt } from './synthetic';

/**
 * Synthetic sparse models in the layout of the e2e 'object' scene (src/harness/synth/scene-object.ts):
 * a table top at z = 0.75 in a 5 × 5 × 2.6 m room, a 3-part object (pedestal + knot, a globe, a box —
 * connected only through the table), a magazine, and a mug at the camera-ring radius. Sparse points
 * are surface samples with 0.5 mm noise plus 1 % outliers in the room; cameras orbit the object.
 */
type Group = 'table' | 'flat' | 'floor' | 'wall' | 'object' | 'mug' | 'outlier';

class Cloud {
  readonly xyz: number[] = [];
  readonly group: Group[] = [];
  constructor(
    readonly rng: Rng,
    readonly noise = 0.0005,
  ) {}
  add(g: Group, x: number, y: number, z: number): void {
    const s = this.noise;
    this.xyz.push(x + s * this.rng.gaussian(), y + s * this.rng.gaussian(), z + s * this.rng.gaussian());
    this.group.push(g);
  }
  /** Horizontal rectangle centred at (cx, cy) with half sizes hx, hy, rotated by yaw. */
  rect(g: Group, n: number, cx: number, cy: number, z: number, hx: number, hy: number, yaw = 0): void {
    const c = Math.cos(yaw);
    const s = Math.sin(yaw);
    for (let i = 0; i < n; i++) {
      const u = (2 * this.rng.next() - 1) * hx;
      const v = (2 * this.rng.next() - 1) * hy;
      this.add(g, cx + c * u - s * v, cy + s * u + c * v, z);
    }
  }
  /** Box surface (all faces but the bottom). */
  box(g: Group, n: number, c: number[], h: number[], yaw = 0): void {
    const cs = Math.cos(yaw);
    const sn = Math.sin(yaw);
    const area = [h[1] * h[2], h[0] * h[2], h[0] * h[1]];
    const total = 2 * area[0] + 2 * area[1] + area[2];
    for (let i = 0; i < n; i++) {
      let r = this.rng.next() * total;
      const p = [(2 * this.rng.next() - 1) * h[0], (2 * this.rng.next() - 1) * h[1], (2 * this.rng.next() - 1) * h[2]];
      if ((r -= 2 * area[0]) < 0) p[0] = r + area[0] < 0 ? -h[0] : h[0];
      else if ((r -= 2 * area[1]) < 0) p[1] = r + area[1] < 0 ? -h[1] : h[1];
      else p[2] = h[2];
      this.add(g, c[0] + cs * p[0] - sn * p[1], c[1] + sn * p[0] + cs * p[1], c[2] + p[2]);
    }
  }
  /** Ellipsoid surface (radii r, yawed). */
  ellipsoid(g: Group, n: number, c: number[], r: number[], yaw = 0): void {
    const cs = Math.cos(yaw);
    const sn = Math.sin(yaw);
    for (let i = 0; i < n; i++) {
      const v = [this.rng.gaussian(), this.rng.gaussian(), this.rng.gaussian()];
      const l = Math.hypot(v[0], v[1], v[2]);
      const p = [(r[0] * v[0]) / l, (r[1] * v[1]) / l, (r[2] * v[2]) / l];
      this.add(g, c[0] + cs * p[0] - sn * p[1], c[1] + sn * p[0] + cs * p[1], c[2] + p[2]);
    }
  }
  /** Vertical cylinder wall + top rim. */
  cylinder(g: Group, n: number, cx: number, cy: number, z0: number, height: number, radius: number): void {
    for (let i = 0; i < n; i++) {
      const a = 2 * Math.PI * this.rng.next();
      this.add(g, cx + radius * Math.cos(a), cy + radius * Math.sin(a), z0 + height * this.rng.next());
    }
  }
  room(n: number): void {
    this.rect('floor', n, 0, 0, 0, 2.5, 2.5);
    for (const [x, y, hx, hy] of [
      [2.5, 0, 0, 2.5],
      [-2.5, 0, 0, 2.5],
      [0, 2.5, 2.5, 0],
      [0, -2.5, 2.5, 0],
    ]) {
      for (let i = 0; i < n / 4; i++) this.add('wall', x + (2 * this.rng.next() - 1) * hx, y + (2 * this.rng.next() - 1) * hy, 2.6 * this.rng.next());
    }
  }
  outliers(n: number): void {
    for (let i = 0; i < n; i++) this.add('outlier', 5 * this.rng.next() - 2.5, 5 * this.rng.next() - 2.5, 2.6 * this.rng.next());
  }
  /** Rigid transform of every point: p ← R (p − o) + o. */
  rotate(R: number[], o: number[]): void {
    for (let i = 0; i < this.xyz.length; i += 3) {
      const p = [this.xyz[i] - o[0], this.xyz[i + 1] - o[1], this.xyz[i + 2] - o[2]];
      for (let k = 0; k < 3; k++) this.xyz[i + k] = R[3 * k] * p[0] + R[3 * k + 1] * p[1] + R[3 * k + 2] * p[2] + o[k];
    }
  }
}

const TABLE_Z = 0.75;

const log = (p: ReturnType<typeof proposeObjectRoi>): void => {
  const f = (v: number[]): string => `[${v.map((x) => x.toFixed(3)).join(', ')}]`;
  console.log(
    `[object-roi] ${p.roi ? `box ${f(p.roi.min)} – ${f(p.roi.max)}, plane ${p.roi.planeZ.toFixed(4)}, confidence ${p.roi.confidence.toFixed(2)}` : `none (${p.reason})`}; ` +
      `ring ${p.ringRadius.toFixed(3)}, d ${p.distMedian.toFixed(3)}, plane ${p.planePoints}/${p.annulusPoints} pts, tilt ${((p.planeTiltRad * 180) / Math.PI).toFixed(2)}°, σ ${(1000 * p.planeSigma).toFixed(2)} mm, clearance ${(1000 * p.clearance).toFixed(1)} mm, object ${p.objectPoints} pts`,
  );
};

/** The e2e object on the table at the origin (pedestal + knot, globe, box: 3 parts). */
function addObject(c: Cloud, base = TABLE_Z + 0.002): void {
  c.box('object', 400, [0, 0, base + 0.02], [0.07, 0.07, 0.02]);
  c.ellipsoid('object', 700, [0, 0, base + 0.146], [0.09, 0.04, 0.09], 0.3);
  c.ellipsoid('object', 300, [0.15, 0.075, base + 0.045], [0.045, 0.045, 0.045]);
  c.box('object', 300, [-0.145, -0.065, base + 0.05], [0.035, 0.025, 0.05], 0.45);
}

/** Orbit of n cameras (radius r, height h above `target`) looking at it, jittered by 2 cm. */
function orbit(rng: Rng, n: number, r: number, h: number, target: number[], up = [0, 0, 1]): Pose[] {
  const poses: Pose[] = [];
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n + 0.3;
    const j = () => 0.02 * rng.gaussian();
    const eye = [target[0] + r * Math.cos(a) + j(), target[1] + r * Math.sin(a) + j(), target[2] + h + j()];
    poses.push(lookAt(eye, [target[0] + j(), target[1] + j(), target[2] + j()], up));
  }
  return poses;
}

function model(c: Cloud, poses: Pose[]): ObjectRoiInput {
  return { images: poses.map((pose) => ({ pose })), xyz: Float64Array.from(c.xyz), numPoints: c.group.length };
}

/** The e2e layout: table, magazine, mug at the camera radius (0.45 m), room, 3-part object. */
function tableScene(seed: number): { cloud: Cloud; poses: Pose[] } {
  const rng = new Rng(seed);
  const c = new Cloud(rng);
  c.rect('table', 4000, 0, 0, TABLE_Z, 0.6, 0.4);
  c.rect('flat', 200, 0, 0, TABLE_Z + 0.0015, 0.22, 0.16); // placemat
  c.rect('flat', 300, 0.4, 0.21, TABLE_Z + 0.001, 0.14, 0.105, 0.35); // magazine
  c.cylinder('mug', 300, -0.38, -0.24, TABLE_Z, 0.095, 0.04); // r = 0.449 m from the axis
  c.room(2000);
  addObject(c);
  c.outliers(80);
  return { cloud: c, poses: orbit(rng, 30, 0.45, 0.2, [0, 0, 0.86]) };
}

function inside(roi: ObjectRoi, x: number, y: number, z: number): boolean {
  return x >= roi.min[0] && x <= roi.max[0] && y >= roi.min[1] && y <= roi.max[1] && z >= roi.min[2] && z <= roi.max[2];
}

/**
 * Checks of the e2e acceptance: every object sample inside the box (above its bottom cut), the box
 * at most 5 cm wider than the object, table, flats and mug outside. `maxCut`: share of the object
 * samples below the bottom cut (the lowest few mm of its sides).
 */
function expectIsolates(roi: ObjectRoi, c: Cloud, tableZ: number, maxCut = 0.02): void {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  let objOut = 0;
  let objCut = 0;
  let objN = 0;
  let otherIn = 0;
  for (let i = 0; i < c.group.length; i++) {
    const [x, y, z] = [c.xyz[3 * i], c.xyz[3 * i + 1], c.xyz[3 * i + 2]];
    const g = c.group[i];
    if (g === 'object') {
      objN++;
      [x, y, z].forEach((v, k) => {
        lo[k] = Math.min(lo[k], v);
        hi[k] = Math.max(hi[k], v);
      });
      if (z < roi.min[2]) objCut++;
      else if (!inside(roi, x, y, z)) objOut++;
    } else if (g !== 'outlier' && inside(roi, x, y, z)) otherIn++;
  }
  expect(objOut).toBe(0);
  expect(objCut / objN).toBeLessThan(maxCut);
  for (let k = 0; k < 3; k++) {
    expect(roi.min[k]).toBeGreaterThan(lo[k] - 0.05);
    expect(roi.max[k]).toBeLessThan(hi[k] + 0.05);
  }
  expect(otherIn).toBe(0);
  expect(roi.min[2] - tableZ).toBeGreaterThan(0.001);
  expect(roi.min[2] - tableZ).toBeLessThan(0.008);
}

describe('proposeObjectRoi', () => {
  it('table scene: 3-part object inside, table top, magazine and mug (at the camera radius) outside', () => {
    const { cloud, poses } = tableScene(1);
    const p = proposeObjectRoi(model(cloud, poses), { gravityAligned: true });
    log(p);
    expect(p.reason).toBe('');
    const roi = p.roi!;
    expect(roi).not.toBeNull();
    expect(Math.abs(roi.planeZ - TABLE_Z)).toBeLessThan(0.001);
    expect(Math.hypot(roi.centre[0], roi.centre[1], roi.centre[2] - 0.86)).toBeLessThan(0.02);
    expect(p.planeTiltRad).toBeLessThan(0.2 * (Math.PI / 180));
    // the bottom cut removes only the lowest few mm of the pedestal / box sides
    expectIsolates(roi, cloud, TABLE_Z);
    expect(roi.confidence).toBeGreaterThan(0.5);
    expect(roi.confidence).toBeLessThanOrEqual(1);
    // deterministic
    expect(proposeObjectRoi(model(cloud, poses), { gravityAligned: true }).roi).toEqual(roi);
    // the export pivot: bottom centre on the plane
    const piv = objectRoiPivot(roi);
    expect(piv[2]).toBe(roi.planeZ);
    expect(piv[0]).toBeCloseTo(0.5 * (roi.min[0] + roi.max[0]), 12);
  });

  it('minClearance raises the bottom cut (2 TSDF voxels)', () => {
    const { cloud, poses } = tableScene(2);
    const p = proposeObjectRoi(model(cloud, poses), { gravityAligned: true, minClearance: 0.006 });
    expect(p.clearance).toBeCloseTo(0.006, 9);
    expect(p.roi!.min[2] - p.roi!.planeZ).toBeCloseTo(0.006, 9);
    // relative to the orbit distance d (runMesh: 2 voxels of 2 px at d)
    const q = proposeObjectRoi(model(cloud, poses), { gravityAligned: true, minClearance: { perDistance: 0.015 } });
    expect(q.clearance).toBeCloseTo(0.015 * q.distMedian, 9);
  });

  it('is scale-invariant (non-metric model) and absorbs a 2° gravity error', () => {
    const s = 3.7;
    const { cloud, poses } = tableScene(3);
    const scaled = model(cloud, poses);
    const xyz = scaled.xyz as Float64Array;
    for (let i = 0; i < xyz.length; i++) xyz[i] *= s;
    const sp = scaled.images.map(({ pose }) => ({ pose: { R: pose.R, t: pose.t.map((v) => v * s) } }));
    const a = proposeObjectRoi({ ...scaled, images: sp }, { gravityAligned: true }).roi!;
    const b = proposeObjectRoi(model(cloud, poses), { gravityAligned: true }).roi!;
    for (let k = 0; k < 3; k++) {
      expect(a.min[k] / s).toBeCloseTo(b.min[k], 6);
      expect(a.max[k] / s).toBeCloseTo(b.max[k], 6);
    }
    // the world tilted by 2° about x (gravity error): one plane, the object still isolated
    const t = (2 * Math.PI) / 180;
    const R = [1, 0, 0, 0, Math.cos(t), -Math.sin(t), 0, Math.sin(t), Math.cos(t)];
    const o = [0, 0, TABLE_Z];
    const tiltedCloud = tableScene(3).cloud;
    tiltedCloud.rotate(R, o);
    const rot = (v: number[]): number[] => [0, 1, 2].map((k) => R[3 * k] * (v[0] - o[0]) + R[3 * k + 1] * (v[1] - o[1]) + R[3 * k + 2] * (v[2] - o[2]) + o[k]);
    const tp = orbit(new Rng(30), 30, 0.45, 0.2, rot([0, 0, 0.86]), rot([o[0], o[1], o[2] + 1]).map((v, k) => v - o[k]));
    const q = proposeObjectRoi(model(tiltedCloud, tp), { gravityAligned: true });
    expect(q.reason).toBe('');
    expect(Math.abs(q.planeTiltRad - t)).toBeLessThan(0.1 * (Math.PI / 180));
    let objOut = 0;
    for (let i = 0; i < tiltedCloud.group.length; i++) {
      if (tiltedCloud.group[i] !== 'object') continue;
      const [x, y, z] = [tiltedCloud.xyz[3 * i], tiltedCloud.xyz[3 * i + 1], tiltedCloud.xyz[3 * i + 2]];
      if (z > q.roi!.min[2] + 0.008 && !inside(q.roi!, x, y, z)) objOut++;
    }
    expect(objOut).toBe(0);
  });

  it('a flat 3 cm book still gets the table as its plane', () => {
    const rng = new Rng(4);
    const c = new Cloud(rng);
    c.rect('table', 4000, 0, 0, TABLE_Z, 0.6, 0.4);
    c.room(2000);
    c.box('object', 1500, [0, 0, TABLE_Z + 0.015], [0.12, 0.085, 0.015], 0.2);
    c.outliers(60);
    const poses = orbit(rng, 30, 0.4, 0.25, [0, 0, TABLE_Z + 0.015]);
    const p = proposeObjectRoi(model(c, poses), { gravityAligned: true });
    expect(p.reason).toBe('');
    expect(Math.abs(p.roi!.planeZ - TABLE_Z)).toBeLessThan(0.001);
    expect(p.roi!.max[2]).toBeGreaterThan(TABLE_Z + 0.03);
    // 3 cm sides: the few-mm bottom cut is a larger share of the samples
    expectIsolates(p.roi!, c, TABLE_Z, 0.06);
  });

  it('no box: hand-held object (no plane, or only a floor far below it)', () => {
    for (const z of [1.2, 0.45]) {
      const rng = new Rng(5);
      const c = new Cloud(rng);
      c.room(3000);
      addObject(c, z - 0.11);
      c.outliers(60);
      const p = proposeObjectRoi(model(c, orbit(rng, 30, 0.45, 0.2, [0, 0, z])), { gravityAligned: true });
      expect(p.roi).toBeNull();
      expect(p.reason).toMatch(z > 1 ? /no support plane/ : /does not rest/);
    }
  });

  it('no box: model not gravity-aligned, or the support tilted by 15°', () => {
    const { cloud, poses } = tableScene(6);
    const p = proposeObjectRoi(model(cloud, poses), { gravityAligned: false });
    expect(p.roi).toBeNull();
    expect(p.reason).toMatch(/gravity/);
    // table + object tilted by 15° about x; the cameras still orbit the vertical axis
    const t = (15 * Math.PI) / 180;
    const tilted = tableScene(6).cloud;
    tilted.rotate([1, 0, 0, 0, Math.cos(t), -Math.sin(t), 0, Math.sin(t), Math.cos(t)], [0, 0, TABLE_Z]);
    const q = proposeObjectRoi(model(tilted, poses), { gravityAligned: true });
    expect(q.roi).toBeNull();
    expect(q.reason).toMatch(/tilted|no support plane/);
  });

  it('no box: cameras that do not orbit (a sideways walk) or look straight down', () => {
    const { cloud } = tableScene(7);
    const walk: Pose[] = [];
    for (let i = 0; i < 20; i++) walk.push(lookAt([-1 + 0.1 * i, -1.5, 1.4], [-1 + 0.1 * i, 1, 0.9]));
    expect(proposeObjectRoi(model(cloud, walk), { gravityAligned: true }).roi).toBeNull();
    const down = orbit(new Rng(8), 20, 0.1, 0.5, [0, 0, 0.86]); // 79° elevation
    const p = proposeObjectRoi(model(cloud, down), { gravityAligned: true });
    expect(p.roi).toBeNull();
    expect(p.reason).toMatch(/straight down/);
  });
});
