import { describe, expect, it } from 'vitest';
import type { LiveCamera } from '../../workers/protocol';
import { computeCoverage } from '../coverage';
import { lookAt } from '../mock-scene';
import { createCoverageRing, ringPoint, sectorPath } from './coverage-ring';

const intrinsics = { width: 480, height: 640, f: 500, cx: 239.5, cy: 319.5, k1: 0, k2: 0 };

/** Keyframes on an arc of `deg` degrees (CCW seen from above) at elevation `elevDeg` around the origin. */
function arc(n: number, deg: number, elevDeg: number): LiveCamera[] {
  const out: LiveCamera[] = [];
  const e = (elevDeg * Math.PI) / 180;
  for (let i = 0; i < n; i++) {
    const a = -Math.PI / 2 + (((deg * Math.PI) / 180) * i) / (n - 1);
    out.push({ pose: lookAt([Math.cos(a) * Math.cos(e), Math.sin(a) * Math.cos(e), Math.sin(e)], [0, 0, 0]), intrinsics });
  }
  return out;
}

const cls = (el: Element): string => el.getAttribute('class') ?? '';

describe('coverage ring', () => {
  it('azimuth 0 at the bottom, counter-clockwise on screen', () => {
    const [x0, y0] = ringPoint(10, 0);
    expect([Math.abs(x0), y0]).toEqual([0, 10]);
    const [x1, y1] = ringPoint(10, Math.PI / 2);
    expect(x1).toBeCloseTo(10, 9); // right of the centre (SVG y points down)
    expect(y1).toBeCloseTo(0, 9);
    expect(sectorPath(5, 10, 0, 0.5)).toMatch(/^M0 10A10 10 0 0 0 [\d.]+ [\d.]+L[\d.]+ [\d.]+A5 5 0 0 1 0 5Z$/);
  });

  it('draws bands × sectors, the prompt, the target cells and the phone marker', () => {
    const ring = createCoverageRing();
    document.body.append(ring.el);
    ring.el.hidden = false;
    const cells = (): Element[] => [...ring.el.querySelectorAll('.coverage-cell')];
    const prompt = ring.el.querySelector('[data-testid="coverage-prompt"]') as HTMLElement;
    // empty and dimmed until the object is located
    expect(cells().length).toBe(36);
    expect(cells().every((c) => cls(c).includes('c0'))).toBe(true);
    expect(ring.el.classList.contains('locating')).toBe(true);
    expect(prompt.hidden).toBe(true);

    // half a level loop (25° elevation, 0–180°): sectors 0–6 of the mid band covered
    const half = computeCoverage(arc(20, 180, 25), { elevation: true });
    expect(half.stable).toBe(true);
    ring.update(half);
    expect(ring.el.classList.contains('locating')).toBe(false);
    expect(prompt.hidden).toBe(false);
    expect(prompt.textContent).toBe('Walk to the other side');
    const mid = cells().filter((c) => cls(c).includes('b1'));
    expect(mid.length).toBe(12);
    expect(mid.filter((c) => cls(c).includes('c2')).length).toBe(half.covered[1]);
    // the empty mid cells pulse, the high band does not (yet)
    expect(mid.filter((c) => cls(c).includes('target')).length).toBe(12 - half.covered[1]);
    expect(cells().filter((c) => cls(c).includes('b2') && cls(c).includes('target')).length).toBe(0);

    const marker = ring.el.querySelector('.coverage-marker')!;
    ring.setCurrent({ sector: 3, band: 1 });
    expect(cls(marker)).toContain('shown');
    expect(Number(marker.getAttribute('cx'))).toBeCloseTo(29, 1); // 90°: right, mid band radius (23 + 35) / 2
    expect(Math.abs(Number(marker.getAttribute('cy')))).toBeLessThan(0.01);
    ring.setCurrent({ sector: 3, band: -1 }); // below the low band: no band to mark
    expect(cls(marker)).not.toContain('shown');

    // half a loop at the object's height: the low band is the one circled, its empty cells pulse
    const eye = computeCoverage(arc(20, 180, 0));
    ring.update(eye);
    expect([eye.primaryBand, eye.prompt]).toEqual([0, 'other_side']);
    const low = cells().filter((c) => cls(c).includes('b0'));
    expect(low.filter((c) => cls(c).includes('c2')).length).toBe(eye.covered[0]);
    expect(low.filter((c) => cls(c).includes('target')).length).toBe(12 - eye.covered[0]);
    expect(cells().filter((c) => !cls(c).includes('b0') && cls(c).includes('target')).length).toBe(0);
    // the same keyframes split between the low and mid bands: a sector either band covers is no gap
    const split = computeCoverage([...arc(10, 180, 3), ...arc(10, 180, 15)]);
    ring.update(split);
    const loopGaps = split.loop.filter((k) => k < 2).length;
    expect(split.prompt).toBe('other_side');
    // neither band alone covers the half loop
    expect(Math.max(split.covered[0], split.covered[1])).toBeLessThan(12 - loopGaps);
    expect(cells().filter((c) => cls(c).includes('target')).length).toBe(loopGaps);

    // a full level loop: the high band is what is missing
    const loop = computeCoverage(arc(40, 360 - 9, 25));
    ring.update(loop);
    expect(loop.prompt).toBe('look_down');
    expect(prompt.textContent).toBe('Raise the phone, loop looking down');
    expect(cells().filter((c) => cls(c).includes('b2') && cls(c).includes('target')).length).toBe(12);
    expect(ring.el.dataset.prompt).toBe('look_down');

    // azimuth only (no motion sensor): one ring of 12
    ring.update(computeCoverage(arc(40, 360 - 9, 25), { elevation: false }));
    expect(cells().length).toBe(12);
    expect(ring.el.classList.contains('flat')).toBe(true);
    expect(prompt.textContent).toBe('All sides covered — you can finish');
    ring.el.remove();
  });
});
