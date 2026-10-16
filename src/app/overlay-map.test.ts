import { describe, expect, it } from 'vitest';
import { fitTransform, mapNormalizedPoint, mapOverlay, rotateNormalizedPoint, unmapPoint } from './overlay-map';

describe('object-fit: cover mapping', () => {
  it('landscape frame in a portrait element crops left/right symmetrically', () => {
    // 1920×1080 video in a 393×852 element: scale = 852/1080, width 1514.7 px, offset −560.8 px.
    const t = fitTransform(1920, 1080, 393, 852);
    expect(t.scale).toBeCloseTo(852 / 1080, 12);
    expect(t.height).toBeCloseTo(852, 9);
    expect(t.width).toBeCloseTo(1920 * (852 / 1080), 9);
    expect(t.offsetY).toBeCloseTo(0, 9);
    expect(t.offsetX).toBeCloseTo((393 - t.width) / 2, 9);
    // The frame centre maps to the element centre, the frame's top edge to the element's top.
    expect(mapNormalizedPoint(0.5, 0.5, t)).toEqual({ x: 196.5, y: 426 });
    expect(mapNormalizedPoint(0.5, 0, t).y).toBeCloseTo(0, 9);
    // A point at u = 0.25 lies outside the visible area (cropped).
    expect(mapNormalizedPoint(0.25, 0.5, t).x).toBeLessThan(0);
  });

  it('portrait frame in a landscape element crops top/bottom', () => {
    const t = fitTransform(1440, 1920, 800, 400);
    expect(t.scale).toBeCloseTo(800 / 1440, 12);
    expect(t.offsetX).toBeCloseTo(0, 9);
    expect(t.offsetY).toBeCloseTo((400 - 1920 * (800 / 1440)) / 2, 9);
  });

  it('same aspect: identity up to scale', () => {
    const t = fitTransform(1920, 1440, 960, 720);
    expect(t).toEqual({ scale: 0.5, width: 960, height: 720, offsetX: 0, offsetY: 0 });
    expect(mapNormalizedPoint(1, 1, t)).toEqual({ x: 960, y: 720 });
  });

  it('contain letterboxes instead', () => {
    const t = fitTransform(1920, 1080, 393, 852, 'contain');
    expect(t.width).toBeCloseTo(393, 9);
    expect(t.offsetX).toBeCloseTo(0, 9);
    expect(t.offsetY).toBeGreaterThan(0);
  });

  it('unmapPoint inverts mapNormalizedPoint', () => {
    const t = fitTransform(1280, 720, 393, 852);
    for (const [u, v] of [
      [0.5, 0.5],
      [0.41, 0.07],
      [0.6, 0.93],
    ]) {
      const p = mapNormalizedPoint(u, v, t);
      const q = unmapPoint(p.x, p.y, t);
      expect(q.u).toBeCloseTo(u, 12);
      expect(q.v).toBeCloseTo(v, 12);
    }
  });

  it('degenerate sizes give an empty transform', () => {
    expect(fitTransform(0, 1080, 393, 852).scale).toBe(0);
    expect(fitTransform(1920, 1080, 0, 852).width).toBe(0);
  });
});

describe('quarter-turn rotation (clockwise as displayed, like rotateRGBA90)', () => {
  it('maps the corners', () => {
    // k = 1: the top-left corner goes to the top-right.
    expect(rotateNormalizedPoint(0, 0, 1)).toEqual({ u: 1, v: 0 });
    expect(rotateNormalizedPoint(1, 0, 1)).toEqual({ u: 1, v: 1 });
    expect(rotateNormalizedPoint(0, 0, 2)).toEqual({ u: 1, v: 1 });
    expect(rotateNormalizedPoint(0, 0, 3)).toEqual({ u: 0, v: 1 });
    expect(rotateNormalizedPoint(0.3, 0.2, -1)).toEqual(rotateNormalizedPoint(0.3, 0.2, 3));
  });
  it('four quarter turns are the identity', () => {
    let p = { u: 0.17, v: 0.71 };
    for (let i = 0; i < 4; i++) p = rotateNormalizedPoint(p.u, p.v, 1);
    expect(p.u).toBeCloseTo(0.17, 12);
    expect(p.v).toBeCloseTo(0.71, 12);
  });
  it('matches the pixel rule dst(x′, y′) = src(y′, H−1−x′) in the continuous limit', () => {
    // A pixel at (x, y) of a W×H frame goes to (H−1−y, x) of the H×W rotated frame.
    const W = 40;
    const H = 30;
    const x = 7;
    const y = 4;
    const r = rotateNormalizedPoint((x + 0.5) / W, (y + 0.5) / H, 1);
    expect(r.u * H - 0.5).toBeCloseTo(H - 1 - y, 9);
    expect(r.v * W - 0.5).toBeCloseTo(x, 9);
  });
});

describe('mapOverlay', () => {
  it('maps, drops cropped points and NaNs', () => {
    const pts = new Float32Array([0.5, 0.5, 0.1, 0.5, NaN, 0.2, 0.45, 0.9]);
    const xy = mapOverlay(pts, 1920, 1080, 393, 852);
    expect(xy.length).toBe(4); // (0.1, 0.5) is cropped away, the NaN dropped
    expect(xy[0]).toBeCloseTo(196.5, 3);
    expect(xy[1]).toBeCloseTo(426, 3);
  });

  it('rotated frames swap the aspect before fitting', () => {
    // A portrait-captured frame (1080×1920) shown after the UI rotated: k = 1.
    // (margin: the corner lies in the cover-cropped band and would be dropped otherwise)
    const xy = mapOverlay(new Float32Array([0.5, 0.5, 0, 0]), 1080, 1920, 852, 393, { quarterTurns: 1, margin: 100 });
    expect(xy[0]).toBeCloseTo(426, 3);
    expect(xy[1]).toBeCloseTo(196.5, 3);
    // (0, 0) → (1, 0) after the turn: the top-right corner of the (cover-cropped) frame.
    const t = fitTransform(1920, 1080, 852, 393);
    expect(xy[2]).toBeCloseTo(t.offsetX + t.width, 3);
    expect(xy[3]).toBeCloseTo(t.offsetY, 3);
  });

  it('margin keeps points just outside the element', () => {
    const pts = new Float32Array([0.5, -0.001]);
    expect(mapOverlay(pts, 100, 100, 100, 100).length).toBe(0);
    expect(mapOverlay(pts, 100, 100, 100, 100, { margin: 1 }).length).toBe(2);
  });
});
