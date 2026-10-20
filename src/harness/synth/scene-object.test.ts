/**
 * buildObjectScene (Node): the wall posters keep clear of the door and the window. There is no
 * canvas in Node, so every texture is a 1 × 1 stand-in tagged with the function that drew it.
 */
import { describe, expect, it, vi } from 'vitest';
import { Mesh, PlaneGeometry, type MeshBasicMaterial } from 'three';
import { buildObjectScene } from './scene-object';

vi.mock('./textures', async (importOriginal) => {
  const T = await importOriginal<Record<string, unknown>>();
  return Object.fromEntries(Object.entries(T).map(([k, v]) => [k, typeof v === 'function' ? () => ({ width: 1, height: 1, drawnBy: k }) : v]));
});

interface Poster {
  /** World extent along the wall (x on the north / south walls, y on the east / west walls). */
  along: [number, number];
  z: [number, number];
}

/** Posters (3 mm in front of a wall) of the wall at x = wallX or y = wallY. */
function posters(seed: number, wall: { x: number } | { y: number }): Poster[] {
  const s = buildObjectScene({ seed, textureScale: 0.01, anisotropy: 1 });
  const out: Poster[] = [];
  s.scene.traverse((o) => {
    if (!(o instanceof Mesh) || !(o.geometry instanceof PlaneGeometry)) return;
    const map = (o.material as MeshBasicMaterial).map;
    if ((map?.image as { drawnBy?: string } | undefined)?.drawnBy !== 'posterTexture') return;
    const p = o.position;
    const [off, along] = 'x' in wall ? [Math.abs(wall.x - p.x), p.y] : [Math.abs(wall.y - p.y), p.x];
    if (Math.abs(off - 0.003) > 1e-9) return; // e.g. the window view (4 mm)
    const { width, height } = o.geometry.parameters;
    out.push({ along: [along - width / 2, along + width / 2], z: [p.z - height / 2, p.z + height / 2] });
  });
  s.dispose();
  return out;
}

const overlaps = (a: [number, number], b: [number, number]): boolean => a[0] < b[1] && b[0] < a[1];

describe('buildObjectScene posters', () => {
  const seeds = Array.from({ length: 50 }, (_, i) => i + 1);

  it('leave the door on the east wall clear', () => {
    let n = 0;
    for (const seed of seeds) {
      for (const p of posters(seed, { x: 2.5 })) {
        n++;
        // door: y 1.15 ± 0.45, z 0–2.05
        expect(overlaps(p.along, [0.7, 1.6]) && overlaps(p.z, [0, 2.05]), `seed ${seed}: poster y [${p.along}]`).toBe(false);
      }
    }
    expect(n).toBeGreaterThan(seeds.length);
  });

  it('leave the window on the north wall clear', () => {
    let n = 0;
    for (const seed of seeds) {
      for (const p of posters(seed, { y: 2.5 })) {
        n++;
        // window frame: x −1.865 … −0.535, z 0.985 … 2.115
        expect(overlaps(p.along, [-1.865, -0.535]) && overlaps(p.z, [0.985, 2.115]), `seed ${seed}: poster x [${p.along}]`).toBe(false);
      }
    }
    expect(n).toBeGreaterThan(seeds.length);
  });
});
