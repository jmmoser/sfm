import { describe, expect, it } from 'vitest';
import { BoxGeometry, Mesh, Vector3 } from 'three';
import { bakeShading } from './scene-common';

describe('bakeShading', () => {
  const light = new Vector3(0.35, -0.45, 0.82).normalize();
  const ambient = 0.62;

  it('shades every face from its world-space normal, also for meshes placed far from the origin', () => {
    // n·t ≥ 1 for some faces of each box (a point-style transform of the normals flips them,
    // or divides by zero for the top face at z = 1)
    const meshes: Mesh[] = [];
    const add = (pos: [number, number, number], rotZ = 0, scale: [number, number, number] = [1, 1, 1]): void => {
      const m = new Mesh(new BoxGeometry(0.3, 0.8, 0.4));
      m.position.set(...pos);
      m.rotation.z = rotZ;
      m.scale.set(...scale);
      m.updateMatrixWorld(true);
      meshes.push(m);
    };
    add([1.68, -2, 0.33], 0.3);
    add([-2.325, 0, 1.0]);
    add([-9.5, -6, 0.15], 0.3);
    add([5, 12, 0.5], 0, [1.3, 1, 0.9]);
    for (const m of meshes) {
      const g = m.geometry;
      bakeShading(g, m.matrixWorld, light, ambient);
      const col = g.getAttribute('color');
      const pos = g.getAttribute('position');
      const idx = g.getIndex()!;
      const a = new Vector3(),
        b = new Vector3(),
        c = new Vector3();
      for (let tri = 0; tri < idx.count; tri += 3) {
        // world normal of the transformed triangle (counter-clockwise from outside)
        const [ia, ib, ic] = [idx.getX(tri), idx.getX(tri + 1), idx.getX(tri + 2)];
        a.fromBufferAttribute(pos, ia).applyMatrix4(m.matrixWorld);
        b.fromBufferAttribute(pos, ib).applyMatrix4(m.matrixWorld);
        c.fromBufferAttribute(pos, ic).applyMatrix4(m.matrixWorld);
        const nW = b.sub(a).cross(c.sub(a)).normalize();
        const want = Math.min(1, ambient + (1 - ambient) * Math.max(0, 0.15 + 0.85 * nW.dot(light)));
        for (const i of [ia, ib, ic]) {
          expect(Number.isFinite(col.getX(i))).toBe(true);
          expect(col.getX(i)).toBeCloseTo(want, 5);
        }
      }
    }
  });
});
