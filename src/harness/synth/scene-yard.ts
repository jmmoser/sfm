/**
 * 'yard' scene: 120 × 120 m ground (multi-scale grass/dirt/gravel), a fenced backyard
 * (≈ 28 × 27 m) with a walking path, patio, picnic table, garden bed, shed, trees and
 * bushes, and 8 houses (≈ 10 × 8 × 6 m + gable roofs) at 18–47 m. Z-up ENU world
 * (x east, y north, z up), metres; the yard itself is flat (z = 0), terrain undulates
 * gently beyond ~16 m.
 */
import {
  BackSide,
  BoxGeometry,
  CylinderGeometry,
  DoubleSide,
  IcosahedronGeometry,
  Mesh,
  PlaneGeometry,
  SphereGeometry,
  Vector3,
  type BufferGeometry,
} from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { SynthRng } from './rng';
import { GeoBuilder, SceneBuilder, skyColors, type SynthScene } from './scene-common';
import { blob } from './scene-object';
import * as T from './textures';
import { yardLoopPoint, type YardTrajectoryOptions } from './trajectory';

const GROUND_HALF = 60;

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

/** Terrain height (m): flat inside the yard, gentle undulation beyond ~16 m. */
export function groundHeight(x: number, y: number): number {
  const w = smoothstep(16, 32, Math.hypot(x, y));
  if (w === 0) return 0;
  return (
    w *
    (0.6 * Math.sin(x / 9 + 0.3) * Math.cos(y / 11 - 0.2) +
      0.35 * Math.sin((x + 0.6 * y) / 17 + 1.0) +
      0.2 * Math.sin(x / 4.3) * Math.sin(y / 5.1))
  );
}

export interface HouseSpec {
  x: number;
  y: number;
  /** Width along local x (ridge direction) and depth along local y (m). */
  w: number;
  d: number;
  wallH: number;
  roofH: number;
  /** Rotation about +Z (rad). Local −Y is the front facade. */
  yaw: number;
  style: T.FacadeStyle;
  color: [number, number, number];
  trim: [number, number, number];
  roofTint: [number, number, number];
  floors: number;
  /** Which facades get a door: [front, right, back, left]. */
  doors: [boolean, boolean, boolean, boolean];
  garage?: boolean;
  chimney?: boolean;
}

export const YARD_HOUSES: HouseSpec[] = [
  { x: 0, y: -18.5, w: 13, d: 9, wallH: 5.8, roofH: 2.6, yaw: 0, style: 'siding', color: [214, 205, 178], trim: [245, 245, 240], roofTint: [0.55, 0.52, 0.5], floors: 2, doors: [true, false, true, false], chimney: true },
  { x: -27, y: -12, w: 10, d: 8, wallH: 5.6, roofH: 2.4, yaw: Math.PI / 2, style: 'brick', color: [150, 70, 50], trim: [235, 230, 220], roofTint: [0.35, 0.33, 0.33], floors: 2, doors: [true, false, false, false], garage: true },
  { x: 28, y: -9, w: 11, d: 8, wallH: 5.9, roofH: 2.8, yaw: -Math.PI / 2, style: 'stucco', color: [222, 200, 160], trim: [120, 70, 50], roofTint: [0.75, 0.42, 0.3], floors: 2, doors: [true, false, true, false] },
  { x: -20, y: 27, w: 10, d: 8, wallH: 5.5, roofH: 2.3, yaw: Math.PI, style: 'siding', color: [120, 150, 170], trim: [250, 250, 250], roofTint: [0.4, 0.4, 0.42], floors: 2, doors: [true, false, false, true], chimney: true },
  { x: 8, y: 31, w: 12, d: 8, wallH: 6.0, roofH: 2.7, yaw: Math.PI + 0.15, style: 'stone', color: [175, 165, 150], trim: [60, 60, 65], roofTint: [0.3, 0.3, 0.3], floors: 2, doors: [true, false, true, false], garage: true },
  { x: 34, y: 21, w: 10, d: 8, wallH: 5.6, roofH: 2.5, yaw: Math.atan2(-34, 21), style: 'brick', color: [170, 95, 65], trim: [240, 240, 230], roofTint: [0.5, 0.45, 0.4], floors: 2, doors: [true, false, false, false] },
  { x: -46, y: 6, w: 11, d: 9, wallH: 6.2, roofH: 2.6, yaw: Math.PI / 2 + 0.1, style: 'siding', color: [190, 190, 160], trim: [90, 60, 40], roofTint: [0.45, 0.35, 0.3], floors: 2, doors: [true, true, false, false], chimney: true },
  { x: 14, y: -44, w: 12, d: 9, wallH: 5.8, roofH: 2.6, yaw: 0.2, style: 'stucco', color: [235, 230, 215], trim: [70, 90, 110], roofTint: [0.6, 0.35, 0.28], floors: 2, doors: [false, false, true, false] },
];

export interface YardSceneOptions {
  seed: number;
  anisotropy: number;
  textureScale: number;
  /** Walking loop (to draw the gravel path under it). */
  trajectory: Partial<YardTrajectoryOptions>;
}

export function buildYardScene(o: Partial<YardSceneOptions> = {}): SynthScene {
  const seed = o.seed ?? 1;
  const ts = o.textureScale ?? 1;
  const px = (n: number) => Math.max(16, Math.round(n * ts));
  const rng = new SynthRng(seed).derive('yard-scene');
  const R = (label: string) => rng.derive(label);
  const b = new SceneBuilder(o.anisotropy ?? 8, new Vector3(-0.35, -0.55, 0.76).normalize(), 0.5);
  const groundAO = (zGround: number) => (z: number) => 0.8 + 0.2 * smoothstep(zGround, zGround + 1.6, z);

  // ---------------- ground ----------------
  {
    const segs = 120;
    const g = new PlaneGeometry(2 * GROUND_HALF, 2 * GROUND_HALF, segs, segs);
    const p = g.getAttribute('position');
    for (let i = 0; i < p.count; i++) p.setZ(i, groundHeight(p.getX(i), p.getY(i)));
    g.computeVertexNormals();
    const size = px(2048);
    const macro = T.groundMacroTexture(size, GROUND_HALF, R('ground-macro'), (ctx, mask) => {
      // gravel path under the walking loop
      for (const c of [ctx, mask]) {
        c.strokeStyle = c === ctx ? 'rgb(150,140,122)' : 'rgb(0,0,0)';
        c.lineWidth = 1.3;
        c.lineJoin = 'round';
        c.beginPath();
        for (let k = 0; k <= 400; k++) {
          const [x, y] = yardLoopPoint((k / 400) * Math.PI * 2, o.trajectory);
          if (k === 0) c.moveTo(x, y);
          else c.lineTo(x, y);
        }
        c.closePath();
        c.stroke();
        // stepping stones to the patio, garden bed soil, a sandbox
        c.fillStyle = c === ctx ? 'rgb(120,96,70)' : 'rgb(0,0,0)';
        c.save();
        c.translate(-9.5, -6);
        c.rotate(0.3);
        c.fillRect(-2.4, -0.9, 4.8, 1.8);
        c.restore();
        c.fillStyle = c === ctx ? 'rgb(214,196,150)' : 'rgb(0,0,0)';
        c.fillRect(7, -9, 2.4, 2.4);
        for (let k = 0; k < 5; k++) {
          c.fillStyle = c === ctx ? 'rgb(160,156,150)' : 'rgb(0,0,0)';
          c.beginPath();
          c.ellipse(0.4 * k - 0.6, -9.6 + 0.85 * k, 0.32, 0.24, 0.3 * k, 0, Math.PI * 2);
          c.fill();
        }
      }
    });
    const macroTex = b.dataTexture(macro, size, size);
    const grass = b.texture(T.grassDetailTexture(px(1024), R('grass')), { linear: true });
    const soil = b.texture(T.soilDetailTexture(px(1024), R('soil')), { linear: true });
    const mat = b.detailMaterial(
      macroTex,
      { tex: grass, scale: (2 * GROUND_HALF) / 2.3, rot: 0.0 },
      { tex: soil, scale: (2 * GROUND_HALF) / 3.7, rot: 0.55 },
      { maskFromAlpha: true, strength: 0.6 },
    );
    b.add(new Mesh(g, mat));
  }

  // ---------------- houses ----------------
  YARD_HOUSES.forEach((h, i) => buildHouse(b, h, R(`house${i}`), px, groundAO));

  // shed
  buildHouse(
    b,
    { x: 10.5, y: 9, w: 3.2, d: 2.6, wallH: 2.3, roofH: 0.9, yaw: Math.atan2(-10.5, 9), style: 'siding', color: [120, 60, 50], trim: [235, 235, 225], roofTint: [0.35, 0.35, 0.35], floors: 1, doors: [true, false, false, false] },
    R('shed'),
    (n) => px(n / 2),
    groundAO,
  );

  // ---------------- fences ----------------
  {
    const fenceH = 1.8;
    const segs: [number, number, number, number][] = [
      [-14, 13, 14, 13], // back (north)
      [-14, -14, -14, 13], // west
      [14, -14, 14, -11], // east, south of the gate
      [14, -9.8, 14, 13], // east, north of the gate
    ];
    const postGeos: BufferGeometry[] = [];
    segs.forEach(([x0, y0, x1, y1], i) => {
      const L = Math.hypot(x1 - x0, y1 - y0);
      const tex = b.texture(T.fenceTexture(L, fenceH, px(100), R(`fence${i}`)));
      const m = new Mesh(new BoxGeometry(L, 0.03, fenceH), b.material(tex));
      m.position.set((x0 + x1) / 2, (y0 + y1) / 2, fenceH / 2);
      m.rotation.z = Math.atan2(y1 - y0, x1 - x0);
      b.add(m, { ao: groundAO(0) });
      // posts on the yard side
      const n = Math.max(1, Math.round(L / 2.4));
      const ux = (x1 - x0) / L,
        uy = (y1 - y0) / L;
      // inward normal: towards the origin
      let nx = -uy,
        ny = ux;
      if (nx * -(x0 + x1) + ny * -(y0 + y1) < 0) {
        nx = -nx;
        ny = -ny;
      }
      for (let k = 0; k <= n; k++) {
        const g = new BoxGeometry(0.1, 0.1, fenceH + 0.1);
        g.rotateZ(Math.atan2(uy, ux));
        g.translate(x0 + ((x1 - x0) * k) / n + nx * 0.065, y0 + ((y1 - y0) * k) / n + ny * 0.065, (fenceH + 0.1) / 2);
        postGeos.push(g);
      }
    });
    const posts = mergeGeometries(postGeos);
    for (const g of postGeos) g.dispose();
    b.add(new Mesh(posts, b.material(b.texture(T.woodTexture(px(256), px(1024), R('posts'), { boards: 1, light: [140, 110, 80], dark: [90, 70, 50] })))), { ao: groundAO(0) });
  }

  // ---------------- patio, picnic table, garden bed, sandbox ----------------
  {
    const patio = new Mesh(new BoxGeometry(6, 4, 0.06), b.material(b.texture(T.paverTexture(px(1200), px(800), 200, R('pavers')))));
    patio.position.set(0, -12, 0.03);
    b.add(patio);
    const wood = b.material(b.texture(T.woodTexture(px(1024), px(512), R('picnic'), { boards: 4, light: [170, 130, 90], dark: [110, 80, 50] })));
    const parts: [number, number, number, number, number, number][] = [
      [1.5, -11.8, 0.74, 1.8, 0.8, 0.05],
      [1.5, -12.55, 0.44, 1.8, 0.28, 0.04],
      [1.5, -11.05, 0.44, 1.8, 0.28, 0.04],
      [0.85, -11.8, 0.36, 0.08, 1.7, 0.7],
      [2.15, -11.8, 0.36, 0.08, 1.7, 0.7],
    ];
    for (const [x, y, z, sx, sy, sz] of parts) {
      const m = new Mesh(new BoxGeometry(sx, sy, sz), wood);
      m.position.set(x, y, z + 0.06);
      b.add(m);
    }
    // garden bed (wooden frame, soil, plants)
    const bedWood = b.material(b.texture(T.woodTexture(px(1024), px(128), R('bed'), { boards: 2 })));
    const frame = new Mesh(new BoxGeometry(4.8, 1.8, 0.3), bedWood);
    frame.position.set(-9.5, -6, 0.15);
    frame.rotation.z = 0.3;
    b.add(frame);
    const soilTop = new Mesh(new PlaneGeometry(4.7, 1.7), b.material(b.texture(T.soilDetailTexture(px(1024), R('bed-soil'))), { color: [0.55, 0.38, 0.25] }));
    soilTop.position.set(-9.5, -6, 0.301);
    soilTop.rotation.z = 0.3;
    b.add(soilTop);
    const plantMat = b.material(b.texture(T.leavesTexture(px(512), R('bed-plants'), 105), { repeat: [2, 2] }));
    for (let k = 0; k < 6; k++) {
      const pr = R(`bedplant${k}`);
      const g = blob(new IcosahedronGeometry(0.28 + 0.12 * pr.next(), 3), pr, 0.25);
      const m = new Mesh(g, plantMat);
      const u = -1.9 + k * 0.76;
      m.position.set(-9.5 + u * Math.cos(0.3), -6 + u * Math.sin(0.3), 0.45);
      b.add(m);
    }
    // sandbox: four boards around a sand surface
    const sbWood = b.material(b.texture(T.woodTexture(px(1024), px(128), R('sandbox'), { boards: 2 })));
    for (const [dx, dy, sx, sy] of [
      [0, 1.15, 2.4, 0.1],
      [0, -1.15, 2.4, 0.1],
      [1.15, 0, 0.1, 2.2],
      [-1.15, 0, 0.1, 2.2],
    ]) {
      const board = new Mesh(new BoxGeometry(sx, sy, 0.25), sbWood);
      board.position.set(8.2 + dx, -7.8 + dy, 0.125);
      b.add(board);
    }
    const sand = new Mesh(new PlaneGeometry(2.2, 2.2), b.material(b.texture(T.soilDetailTexture(px(512), R('sand'))), { color: [0.95, 0.85, 0.62] }));
    sand.position.set(8.2, -7.8, 0.15);
    b.add(sand);
  }

  // ---------------- trees and bushes ----------------
  {
    const leafMats = [100, 115, 85].map((hue, k) => b.material(b.texture(T.leavesTexture(px(1024), R(`leaves${k}`), hue), { repeat: [3, 3] })));
    const barkMats = [0, 1, 2].map((k) => b.material(b.texture(T.barkTexture(px(512), px(1024), R(`bark${k}`)), { repeat: [2, 1] })));
    const tree = (x: number, y: number, s: number, tr: SynthRng) => {
      const z0 = groundHeight(x, y);
      const trunkH = 2.6 * s + 1.2 * tr.next();
      const rr = 0.16 * s + 0.08 * tr.next();
      const trunk = new Mesh(new CylinderGeometry(rr * 0.75, rr, trunkH + 1, 14, 3).rotateX(Math.PI / 2), tr.pick(barkMats));
      trunk.position.set(x, y, z0 + (trunkH + 1) / 2 - 0.3);
      b.add(trunk, { ao: groundAO(z0) });
      const cr = (1.6 + 1.2 * tr.next()) * s;
      const crown = new Mesh(blob(new IcosahedronGeometry(cr, 4), tr, 0.18), tr.pick(leafMats));
      crown.scale.set(1, 1, 0.85 + 0.3 * tr.next());
      crown.position.set(x, y, z0 + trunkH + cr * 0.7);
      b.add(crown);
      if (tr.next() < 0.5) {
        const c2 = new Mesh(blob(new IcosahedronGeometry(cr * 0.7, 3), tr, 0.2), tr.pick(leafMats));
        c2.position.set(x + cr * 0.5 * (tr.next() - 0.5), y + cr * 0.5 * (tr.next() - 0.5), z0 + trunkH + cr * 1.3);
        b.add(c2);
      }
    };
    const near: [number, number, number][] = [
      [-11, 9.5, 1.1],
      [11.8, -3, 0.9],
      [-12, -10, 1.0],
      [4, 10.8, 1.0],
      [-18, 16, 1.2],
      [19, 14, 1.3],
      [22, -20, 1.1],
      [-21, -24, 1.2],
      [-35, 18, 1.4],
      [40, 5, 1.3],
      [-8, 22, 1.2],
      [25, 36, 1.4],
      [-30, 38, 1.3],
      [45, -15, 1.4],
      [-38, -30, 1.3],
      [30, -33, 1.2],
    ];
    near.forEach(([x, y, s], i) => tree(x, y, s, R(`tree${i}`)));
    const ring = R('ring');
    for (let k = 0; k < 40; k++) {
      const a = (k / 40) * Math.PI * 2 + 0.05 * ring.next();
      const r = 52 + 5 * ring.next();
      tree(r * Math.cos(a), r * Math.sin(a), 1.3 + 0.6 * ring.next(), R(`ring${k}`));
    }
    // bushes along the back fence and near the house
    const bushMat = b.material(b.texture(T.leavesTexture(px(1024), R('bush'), 95), { repeat: [2, 2] }));
    for (let k = 0; k < 8; k++) {
      const br = R(`bush${k}`);
      const x = -11 + k * 3.1 + br.next(),
        y = 11.9 - 0.4 * br.next();
      const r = 0.55 + 0.4 * br.next();
      const m = new Mesh(blob(new IcosahedronGeometry(r, 3), br, 0.25), bushMat);
      m.scale.set(1.3, 1, 0.9);
      m.position.set(x, y, r * 0.6);
      b.add(m);
    }
    for (let k = 0; k < 5; k++) {
      const br = R(`hbush${k}`);
      const r = 0.5 + 0.3 * br.next();
      const m = new Mesh(blob(new IcosahedronGeometry(r, 3), br, 0.25), bushMat);
      m.position.set(-5.5 + k * 2.6, -13.4, r * 0.55);
      b.add(m);
    }
  }

  // ---------------- sky ----------------
  {
    const g = new SphereGeometry(900, 48, 24);
    g.rotateX(Math.PI / 2);
    skyColors(g, [205, 218, 232], [95, 140, 205], 900);
    b.add(new Mesh(g, b.material(null, { side: BackSide })), { gt: false, depth: false, shade: false });
  }

  return {
    name: 'yard',
    scene: b.scene,
    surfaceMeshes: b.surfaceMeshes,
    focusMeshes: b.focusMeshes,
    depthHidden: b.depthHidden,
    near: 0.1,
    far: 2000,
    roi: { min: [-50, -50, -3], max: [50, 50, 15] },
    focus: { min: [-14, -14, 0], max: [14, 13, 3] },
    dispose: () => b.dispose(),
  };
}

/**
 * House: four textured walls (one 4×-wide facade atlas, 2 gables), two roof slopes
 * (double-sided, shingles), optional chimney. Walls extend below the terrain.
 */
function buildHouse(b: SceneBuilder, h: HouseSpec, rng: SynthRng, px: (n: number) => number, ao: (z: number) => (z: number) => number): void {
  const { w, d, wallH: H, roofH: rh } = h;
  // ground height under the footprint
  const c = Math.cos(h.yaw),
    s = Math.sin(h.yaw);
  let zMin = Infinity,
    zSum = 0;
  for (const [lx, ly] of [
    [0, 0],
    [-w / 2, -d / 2],
    [w / 2, -d / 2],
    [w / 2, d / 2],
    [-w / 2, d / 2],
  ]) {
    const z = groundHeight(h.x + c * lx - s * ly, h.y + s * lx + c * ly);
    zMin = Math.min(zMin, z);
    zSum += z;
  }
  const zBase = zSum / 5;
  const z0 = zMin - zBase - 0.3; // local wall bottom (below the terrain)

  // atlas: 4 facades (top 640/1024 rows) + 2 gables (bottom-left)
  const AW = px(4096),
    AH = px(1024);
  const fw = AW / 4,
    fh = Math.round(AH * 0.625);
  const { canvas, ctx } = T.makeCanvas(AW, AH);
  const lengths = [w, d, w, d];
  for (let f = 0; f < 4; f++) {
    T.drawFacade(ctx, f * fw, 0, fw, fh, { widthM: lengths[f], heightM: H, style: h.style, color: h.color, trim: h.trim, door: h.doors[f], garage: h.garage && f === 0, floors: h.floors }, rng);
  }
  for (let g = 0; g < 2; g++) T.drawGable(ctx, g * fw, fh, fw, AH - fh, d, rh, h.style, h.color, h.trim, rng);
  T.addGrain(canvas, 0.03, rng, 0.1);
  const wallMat = b.material(b.texture(canvas));
  const vTop = 1,
    vBot = 1 - fh / AH; // facade v range (ground at vBot, wall top at 1)
  const gb = new GeoBuilder();
  const v = (z: number) => vBot + (z / H) * (vTop - vBot);
  const walls: [number, number, number, number, number][] = [
    [-w / 2, -d / 2, 1, 0, w],
    [w / 2, -d / 2, 0, 1, d],
    [w / 2, d / 2, -1, 0, w],
    [-w / 2, d / 2, 0, -1, d],
  ];
  walls.forEach(([x0, y0, rx, ry, L], f) => {
    const u0 = f / 4,
      u1 = (f + 1) / 4;
    gb.quad([x0, y0, z0], [x0 + rx * L, y0 + ry * L, z0], [x0 + rx * L, y0 + ry * L, H], [x0, y0, H], [u0, v(z0)], [u1, v(z0)], [u1, vTop], [u0, vTop]);
  });
  const gv = vBot; // gable region: v ∈ [0, vBot]
  gb.tri([w / 2, -d / 2, H], [w / 2, d / 2, H], [w / 2, 0, H + rh], [0, 0], [0.25, 0], [0.125, gv]);
  gb.tri([-w / 2, d / 2, H], [-w / 2, -d / 2, H], [-w / 2, 0, H + rh], [0.25, 0], [0.5, 0], [0.375, gv]);
  const wallMesh = new Mesh(gb.build(), wallMat);
  wallMesh.position.set(h.x, h.y, zBase);
  wallMesh.rotation.z = h.yaw;
  b.add(wallMesh, { ao: ao(zBase) });

  // roof
  const oh = 0.45;
  const W = w / 2 + oh,
    D = d / 2 + oh;
  const ze = H - oh * (rh / (d / 2)),
    zr = H + rh;
  const slope = Math.hypot(D, zr - ze);
  const ru = rng.next() * 4;
  const rb = new GeoBuilder();
  const U = (x: number) => (x + ru) / 4;
  rb.quad([-W, -D, ze], [W, -D, ze], [W, 0, zr], [-W, 0, zr], [U(-W), 0], [U(W), 0], [U(W), slope / 4], [U(-W), slope / 4]);
  rb.quad([W, D, ze], [-W, D, ze], [-W, 0, zr], [W, 0, zr], [U(W) + 0.37, 0], [U(-W) + 0.37, 0], [U(-W) + 0.37, slope / 4], [U(W) + 0.37, slope / 4]);
  // UVs are in metres / 4 → repeat wrapping (requested through `offset`).
  const roofTex = b.texture(T.shingleTexture(px(1024), rng.derive('shingles')), { offset: [0, 0] });
  const roofMat = b.material(roofTex, { side: DoubleSide, color: h.roofTint.map((x) => Math.min(1, x * 1.6)) as [number, number, number] });
  const roof = new Mesh(rb.build(), roofMat);
  roof.position.copy(wallMesh.position);
  roof.rotation.z = h.yaw;
  b.add(roof);

  if (h.chimney) {
    const ch = new Mesh(new BoxGeometry(0.7, 0.7, rh + 1.6), b.material(b.texture(T.paverTexture(px(256), px(512), 200, rng.derive('chimney'))), { color: [0.8, 0.5, 0.4] }));
    const lx = w * 0.25,
      ly = d * 0.12;
    ch.position.set(h.x + c * lx - s * ly, h.y + s * lx + c * ly, zBase + H + (rh + 1.6) / 2 - 0.2);
    ch.rotation.z = h.yaw;
    b.add(ch);
  }
}
