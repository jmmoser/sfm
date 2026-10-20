/**
 * 'object' scene: a ~25 cm composite object (torus knot on a marble pedestal, a
 * globe sphere, a printed box) on a 1.2 × 0.8 m wooden table with a placemat and a
 * magazine, in a 5 × 5 × 2.6 m room (plank floor, rug, plaster walls with posters,
 * bookshelf, couch, door, window, plant). World: Z-up ENU, metres; the table top is
 * at z = 0.75 and the object centre near (0, 0, 0.86).
 */
import {
  BackSide,
  BoxGeometry,
  CylinderGeometry,
  IcosahedronGeometry,
  Mesh,
  PlaneGeometry,
  SphereGeometry,
  TorusKnotGeometry,
  Vector3,
} from 'three';
import { SynthRng } from './rng';
import { SceneBuilder, atlasBoxUVs, orientPlane, skyColors, type SynthScene } from './scene-common';
import * as T from './textures';

export const TABLE_TOP_Z = 0.75;
export const OBJECT_BASE_Z = TABLE_TOP_Z + 0.002; // on the placemat

export interface ObjectSceneOptions {
  seed: number;
  anisotropy: number;
  /** Texture resolution multiplier (1 = full quality). */
  textureScale: number;
}

export function buildObjectScene(o: Partial<ObjectSceneOptions> = {}): SynthScene {
  const seed = o.seed ?? 1;
  const ts = o.textureScale ?? 1;
  const px = (n: number) => Math.max(16, Math.round(n * ts));
  const rng = new SynthRng(seed).derive('object-scene');
  const b = new SceneBuilder(o.anisotropy ?? 8, new Vector3(0.35, -0.45, 0.82).normalize(), 0.62);
  const R = (label: string) => rng.derive(label);

  // ---------------- room ----------------
  const roomHalf = 2.5,
    roomH = 2.6;
  {
    // floor: planks + tiled fine detail
    const floorMap = b.texture(T.woodTexture(px(2048), px(2048), R('floor'), { boards: 28, stagger: true, light: [178, 132, 88], dark: [110, 70, 40] }));
    const dA = b.texture(T.soilDetailTexture(px(512), R('floor-d1')), { linear: true });
    const dB = b.texture(T.grassDetailTexture(px(512), R('floor-d2')), { linear: true });
    const mat = b.detailMaterial(floorMap, { tex: dA, scale: 5 / 0.6, rot: 0.3 }, { tex: dB, scale: 5 / 0.37, rot: 1.1 }, { maskFromAlpha: false, strength: 0.18 });
    b.add(new Mesh(new PlaneGeometry(2 * roomHalf, 2 * roomHalf), mat));
  }
  {
    const ceil = new Mesh(new PlaneGeometry(2 * roomHalf, 2 * roomHalf), b.material(b.texture(T.plasterTexture(px(1024), px(1024), R('ceiling'), [236, 236, 232]))));
    ceil.position.set(0, 0, roomH);
    orientPlane(ceil, [0, 0, -1], [0, 1, 0]);
    b.add(ceil);
  }
  const wallDefs: { n: [number, number, number]; c: [number, number, number]; color: [number, number, number] }[] = [
    { n: [0, -1, 0], c: [0, roomHalf, roomH / 2], color: [214, 206, 188] },
    { n: [0, 1, 0], c: [0, -roomHalf, roomH / 2], color: [200, 210, 214] },
    { n: [-1, 0, 0], c: [roomHalf, 0, roomH / 2], color: [222, 214, 200] },
    { n: [1, 0, 0], c: [-roomHalf, 0, roomH / 2], color: [206, 214, 196] },
  ];
  wallDefs.forEach((w, i) => {
    const m = new Mesh(new PlaneGeometry(2 * roomHalf, roomH), b.material(b.texture(T.plasterTexture(px(2048), px(1066), R(`wall${i}`), w.color))));
    m.position.set(...w.c);
    orientPlane(m, w.n);
    b.add(m);
    // posters (3 mm in front of the wall)
    const pr = R(`posters${i}`);
    const styles = ['abstract', 'photo', 'text', 'pattern'] as const;
    let x = -roomHalf + 0.35 + pr.next() * 0.3;
    while (x < roomHalf - 0.6) {
      const pw = 0.45 + 0.5 * pr.next();
      const ph = pw * (0.7 + 0.7 * pr.next());
      if (x + pw > roomHalf - 0.2) break;
      // skip the region behind the bookshelf (west wall), the door (east wall, along-wall x =
      // −world y) and the window (north wall)
      const blocked = (i === 3 && x < 1.0 && x + pw > -1.0) || (i === 2 && x + pw > -1.7 && x < -0.6) || (i === 0 && x + pw > -1.95 && x < -0.45);
      if (!blocked) {
        const style = pr.pick(styles);
        const tex = b.texture(T.posterTexture(px(Math.round(1024 * pw)), px(Math.round(1024 * ph)), pr.derive(`p${x}`), style));
        const pm = new Mesh(new PlaneGeometry(pw, ph), b.material(tex));
        const zc = 1.0 + 0.25 * pr.next() + ph / 2;
        // along-wall axis: right = up × normal
        const nx = w.n[0],
          ny = w.n[1];
        const rx = -ny,
          ry = nx; // (0,0,1) × n = (−n_y, n_x, 0)
        const off = 0.003;
        pm.position.set(w.c[0] + nx * off + rx * (x + pw / 2), w.c[1] + ny * off + ry * (x + pw / 2), zc);
        orientPlane(pm, w.n);
        b.add(pm);
      }
      x += pw + 0.25 + 0.4 * pr.next();
    }
  });
  // window on the north wall: frame + outdoor "view"
  {
    const view = new Mesh(new PlaneGeometry(1.2, 1.0), b.material(b.texture(T.posterTexture(px(1200), px(1000), R('window-view'), 'photo'))));
    view.position.set(-1.2, roomHalf - 0.004, 1.55);
    orientPlane(view, [0, -1, 0]);
    b.add(view);
    const frameMat = b.material(b.texture(T.woodTexture(px(512), px(256), R('window-frame'), { light: [240, 238, 230], dark: [205, 200, 190], boards: 2 })));
    for (const [cx, cz, sx, sz] of [
      [-1.2, 2.08, 1.3, 0.07],
      [-1.2, 1.02, 1.3, 0.07],
      [-1.83, 1.55, 0.07, 1.1],
      [-0.57, 1.55, 0.07, 1.1],
      [-1.2, 1.55, 0.04, 1.0],
    ]) {
      const f = new Mesh(new BoxGeometry(sx, 0.05, sz), frameMat);
      f.position.set(cx, roomHalf - 0.025, cz);
      b.add(f);
    }
  }
  // door on the east wall
  {
    const door = new Mesh(new PlaneGeometry(0.9, 2.05), b.material(b.texture(T.woodTexture(px(460), px(1050), R('door'), { boards: 1, light: [150, 100, 64], dark: [90, 55, 30] }))));
    door.position.set(roomHalf - 0.004, 1.15, 1.025);
    orientPlane(door, [-1, 0, 0]);
    b.add(door);
    const knob = new Mesh(new SphereGeometry(0.03, 16, 12), b.material(null, { color: [0.8, 0.65, 0.25] }));
    knob.position.set(roomHalf - 0.04, 0.8, 1.0);
    b.add(knob);
  }
  // bookshelf against the west wall (front face +x)
  {
    const shelfW = 1.6,
      shelfD = 0.35,
      shelfH = 2.0;
    const wood = b.material(b.texture(T.woodTexture(px(512), px(512), R('shelf-wood'), { boards: 3 })));
    const m = new Mesh(new BoxGeometry(shelfD, shelfW, shelfH), wood);
    m.position.set(-roomHalf + shelfD / 2, 0, shelfH / 2);
    b.add(m);
    // Front as an upright plane 1 mm in front of the carcass (BoxGeometry side-face
    // UVs are rotated in a Z-up world).
    const front = new Mesh(new PlaneGeometry(shelfW, shelfH), b.material(b.texture(T.bookshelfTexture(px(1600), px(2000), R('books'), 5))));
    front.position.set(-roomHalf + shelfD + 0.001, 0, shelfH / 2);
    orientPlane(front, [1, 0, 0]);
    b.add(front);
  }
  // couch against the south wall
  {
    const fab = b.material(b.texture(T.fabricTexture(px(1024), px(512), R('couch'), [90, 110, 140])));
    const fab2 = b.material(b.texture(T.fabricTexture(px(1024), px(512), R('couch2'), [80, 100, 130])));
    const parts: [number, number, number, number, number, number][] = [
      [0.6, -roomHalf + 0.5, 0.22, 2.0, 0.8, 0.44],
      [0.6, -roomHalf + 0.12, 0.5, 2.0, 0.24, 1.0],
      [-0.48, -roomHalf + 0.5, 0.33, 0.18, 0.8, 0.66],
      [1.68, -roomHalf + 0.5, 0.33, 0.18, 0.8, 0.66],
    ];
    parts.forEach(([x, y, z, sx, sy, sz], i) => {
      const m = new Mesh(new BoxGeometry(sx, sy, sz), i === 0 ? fab : fab2);
      m.position.set(x, y, z);
      b.add(m);
    });
    // cushions
    for (let k = 0; k < 3; k++) {
      const c = new Mesh(new BoxGeometry(0.42, 0.14, 0.38), b.material(b.texture(T.posterTexture(px(420), px(380), R(`cushion${k}`), 'pattern'))));
      c.position.set(0.05 + k * 0.55, -roomHalf + 0.33, 0.62);
      c.rotation.set(0.25, 0, 0.05 * (k - 1));
      b.add(c);
    }
  }
  // rug under the table
  {
    const rug = new Mesh(new PlaneGeometry(2.4, 1.7), b.material(b.texture(T.rugTexture(px(2048), px(1450), R('rug')))));
    rug.position.set(0, 0, 0.005);
    b.add(rug);
  }
  // potted plant in the north-east corner
  {
    const pot = new Mesh(new CylinderGeometry(0.16, 0.12, 0.35, 32, 1), b.material(b.texture(T.packagingAtlas(1, 1, px(512), R('pot')))));
    pot.rotation.x = Math.PI / 2;
    pot.position.set(1.9, 1.9, 0.175);
    b.add(pot);
    const crown = blob(new IcosahedronGeometry(0.35, 4), R('plant-shape'), 0.22);
    const cm = new Mesh(crown, b.material(b.texture(T.leavesTexture(px(1024), R('plant-leaves'), 120), { repeat: [2, 2] })));
    cm.position.set(1.9, 1.9, 0.75);
    b.add(cm);
  }

  // ---------------- table ----------------
  const tableW = 1.2,
    tableD = 0.8,
    topT = 0.04;
  {
    const topTex = b.texture(T.woodTexture(px(2048), px(1366), R('table-top'), { boards: 5, light: [205, 160, 110], dark: [130, 85, 48] }));
    const edge = b.material(b.texture(T.woodTexture(px(1024), px(64), R('table-edge'), { boards: 1 })));
    const top = new Mesh(new BoxGeometry(tableW, tableD, topT), [edge, edge, edge, edge, b.material(topTex), edge]);
    top.position.set(0, 0, TABLE_TOP_Z - topT / 2);
    b.add(top, { focus: true });
    const legMat = b.material(b.texture(T.woodTexture(px(128), px(1024), R('table-legs'), { boards: 1, light: [150, 105, 70], dark: [90, 60, 35] })));
    for (const sx of [-1, 1])
      for (const sy of [-1, 1]) {
        const leg = new Mesh(new BoxGeometry(0.05, 0.05, TABLE_TOP_Z - topT), legMat);
        leg.position.set(sx * (tableW / 2 - 0.06), sy * (tableD / 2 - 0.06), (TABLE_TOP_Z - topT) / 2);
        b.add(leg);
      }
    // placemat under the object, magazine, mug, pen
    const mat = new Mesh(new PlaneGeometry(0.44, 0.32), b.material(b.texture(T.fabricTexture(px(1100), px(800), R('placemat'), [196, 180, 140]))));
    mat.position.set(0, 0, TABLE_TOP_Z + 0.0015);
    b.add(mat, { focus: true });
    const mag = new Mesh(new PlaneGeometry(0.28, 0.21), b.material(b.texture(T.magazineTexture(px(840), px(630), R('magazine')))));
    mag.position.set(0.4, 0.21, TABLE_TOP_Z + 0.001);
    mag.rotation.z = 0.35;
    b.add(mag, { focus: true });
    const mug = new Mesh(new CylinderGeometry(0.04, 0.037, 0.095, 48, 1), b.material(b.texture(T.packagingAtlas(2, 1, px(512), R('mug')), {})));
    mug.rotation.x = Math.PI / 2;
    mug.position.set(-0.42, -0.2, TABLE_TOP_Z + 0.0475);
    b.add(mug, { focus: true });
    const pen = new Mesh(new CylinderGeometry(0.005, 0.005, 0.14, 12, 1).rotateZ(Math.PI / 2), b.material(null, { color: [0.1, 0.2, 0.7] }));
    pen.rotation.z = 1.1;
    pen.position.set(0.35, -0.25, TABLE_TOP_Z + 0.005);
    b.add(pen, { focus: true });
  }

  // ---------------- the object (~25 cm) ----------------
  {
    const ped = new Mesh(atlasBoxUVs(new BoxGeometry(0.14, 0.14, 0.04), 3, 2), b.material(b.texture(T.marbleTexture(px(1536), px(1024), R('pedestal')))));
    ped.position.set(0, 0, OBJECT_BASE_Z + 0.02);
    b.add(ped, { focus: true });
    // torus knot standing on the pedestal (local extent ±0.09 in its plane, ±0.04 thick)
    const knotGeo = new TorusKnotGeometry(0.06, 0.016, 400, 40, 2, 3);
    const knotTex = b.texture(T.cellsTexture(px(4096), px(512), R('knot'), 900));
    const knot = new Mesh(knotGeo, b.material(knotTex));
    knot.rotation.set(Math.PI / 2, 0.2, 0.3);
    knot.position.set(0, 0, OBJECT_BASE_Z + 0.04 + 0.106);
    b.add(knot, { focus: true });
    const sphere = new Mesh(new SphereGeometry(0.045, 96, 64), b.material(b.texture(T.globeTexture(px(1024), px(512), R('globe')))));
    sphere.position.set(0.15, 0.075, OBJECT_BASE_Z + 0.045);
    sphere.rotation.set(0.3, -0.2, 0.5);
    b.add(sphere, { focus: true });
    const box = new Mesh(atlasBoxUVs(new BoxGeometry(0.07, 0.05, 0.1), 3, 2), b.material(b.texture(T.packagingAtlas(3, 2, px(512), R('box')))));
    box.position.set(-0.145, -0.065, OBJECT_BASE_Z + 0.05);
    box.rotation.z = 0.45;
    b.add(box, { focus: true });
  }

  // Dark "outside" beyond the room is never visible; a dome keeps misses well defined.
  {
    const g = new SphereGeometry(30, 32, 16);
    g.rotateX(Math.PI / 2);
    skyColors(g, [40, 40, 40], [40, 40, 40], 30);
    b.add(new Mesh(g, b.material(null, { side: BackSide })), { gt: false, depth: false, shade: false });
  }

  return {
    name: 'object',
    scene: b.scene,
    surfaceMeshes: b.surfaceMeshes,
    focusMeshes: b.focusMeshes,
    depthHidden: b.depthHidden,
    near: 0.02,
    far: 40,
    roi: { min: [-0.7, -0.5, TABLE_TOP_Z - 0.05], max: [0.7, 0.5, 1.1] },
    focus: { min: [-0.19, -0.12, OBJECT_BASE_Z], max: [0.2, 0.13, OBJECT_BASE_Z + 0.25] },
    dispose: () => b.dispose(),
  };
}

/** Smoothly displace a (non-indexed) polyhedron radially → organic blob. */
export function blob(g: IcosahedronGeometry, rng: SynthRng, amount: number): IcosahedronGeometry {
  const p = g.getAttribute('position');
  const waves: [number, number, number, number, number][] = [];
  for (let k = 0; k < 7; k++) {
    const a = rng.next() * Math.PI * 2,
      c = rng.next() * 2 - 1;
    const s = Math.sqrt(1 - c * c);
    waves.push([s * Math.cos(a), s * Math.sin(a), c, 1.5 + 4 * rng.next(), rng.next() * 6.3]);
  }
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i),
      y = p.getY(i),
      z = p.getZ(i);
    const r = Math.hypot(x, y, z) || 1;
    const ux = x / r,
      uy = y / r,
      uz = z / r;
    let d = 0;
    for (const [wx, wy, wz, f, ph] of waves) d += Math.sin(f * (wx * ux + wy * uy + wz * uz) + ph) / waves.length;
    const k = 1 + amount * d * 1.6;
    p.setXYZ(i, x * k, y * k, z * k);
  }
  g.deleteAttribute('normal');
  g.computeVertexNormals();
  return g;
}
