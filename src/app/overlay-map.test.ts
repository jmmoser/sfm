import { describe, expect, it } from 'vitest';
import { OrientationTracker } from '../engine/frame-ingest';
import { TestRng } from '../features/testdata';
import { cameraFromDevice, canonicalRotation, rotateImagePoint } from '../sensors/frame-orientation';
import { FrameOrientationTracker, fitTransform, mapNormalizedPoint, mapOverlay, rotateNormalizedPoint, unmapPoint } from './overlay-map';

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

describe('raw sensor buffers (FrameOrientationTracker)', () => {
  it('classifies like the engine: landscape bitmap in a portrait UI = raw, upright for θ = 90', () => {
    const o = new FrameOrientationTracker();
    expect(o.observe({ width: 1920, height: 1440, screenAngle: 0, t: 0 })).toBe(90);
    expect(o.rawDelivery).toBe(true);
    // Raw delivery is a session property: landscape bitmaps of a landscape-270 UI are raw too.
    expect(o.observe({ width: 1920, height: 1440, screenAngle: 270, t: 2000 })).toBe(90);
    // A portrait bitmap in a landscape UI is dropped by the engine: no overlay.
    expect(o.observe({ width: 1440, height: 1920, screenAngle: 90, t: 4000 })).toBeNull();
    const up = new FrameOrientationTracker();
    expect(up.observe({ width: 1440, height: 1920, screenAngle: 0, t: 0 })).toBe(0);
    expect(up.rawDelivery).toBe(false);
    expect(up.observe({ width: 1920, height: 1440, screenAngle: 270, t: 2000 })).toBe(270);
    expect(up.observe({ width: 1920, height: 1440, screenAngle: 90, t: 4000 })).toBe(90);
  });

  it('frames within 500 ms of a UI rotation do not decide the delivery path', () => {
    const o = new FrameOrientationTracker();
    o.observe({ width: 1920, height: 1440, screenAngle: 90, t: 0 });
    // Transient mis-rotated frame right after turning to portrait: not evidence of raw delivery.
    expect(o.observe({ width: 1920, height: 1440, screenAngle: 0, t: 100 })).toBe(90);
    expect(o.rawDelivery).toBeNull();
    expect(o.observe({ width: 1440, height: 1920, screenAngle: 0, t: 800 })).toBe(0);
    expect(o.rawDelivery).toBe(false);
  });

  it('quarter turns onto the preview: the engine rule (raw is upright for 90)', () => {
    expect(FrameOrientationTracker.overlayTurns(90, 0)).toBe(1); // raw buffer, portrait UI
    expect(FrameOrientationTracker.overlayTurns(90, 180)).toBe(3); // upside-down portrait (iPad)
    expect(FrameOrientationTracker.overlayTurns(90, 270)).toBe(2); // raw delivery, landscape-270 UI
    expect(FrameOrientationTracker.overlayTurns(90, 90)).toBe(0);
    for (const t of [0, 90, 180, 270]) expect(FrameOrientationTracker.overlayTurns(t, t)).toBe(0); // upright frames
  });

  it('agrees with the engine OrientationTracker on random frame sequences', () => {
    const rng = new TestRng(11);
    for (let run = 0; run < 40; run++) {
      const app = new FrameOrientationTracker(500);
      const eng = new OrientationTracker(500);
      let t = 0;
      let ui = [0, 90, 180, 270][Math.floor(rng.next() * 4)];
      const rawPath = rng.next() < 0.5;
      for (let i = 0; i < 60; i++) {
        t += 50 + rng.next() * 400;
        if (rng.next() < 0.08) ui = [0, 90, 180, 270][Math.floor(rng.next() * 4)];
        const portraitUI = ui === 0 || ui === 180;
        // Delivery path: raw buffers are always landscape; upright frames follow the UI (plus rare glitches).
        let landscape = rawPath ? true : !portraitUI;
        if (rng.next() < 0.05) landscape = !landscape;
        const f = { width: landscape ? 1920 : 1440, height: landscape ? 1440 : 1920, screenAngle: ui, t };
        const a = app.observe(f);
        const e = eng.classify(f);
        if (!e.ok) continue; // dropped by the engine: no overlay arrives for it
        // Ambiguous landscape-270 start (engine re-interprets θ₀ later): the as-submitted pixels are the same.
        expect(a, `run ${run} frame ${i} ui ${ui} ${f.width}x${f.height}`).toBe(e.effectiveTheta);
        expect(app.rawDelivery).toBe(eng.rawDelivery);
      }
    }
  });

  it('a keypoint of a raw buffer lands where the scene point is on the upright portrait preview', () => {
    // Raw 1920×1440 buffer (upright for θ = 90) while the UI is portrait (θ = 0); the preview shows
    // the upright 1440×1920 frame in a 720×960 element (same aspect: no crop, scale ½).
    const W = 1920;
    const H = 1440;
    const o = new FrameOrientationTracker();
    const upright = o.observe({ width: W, height: H, screenAngle: 0, t: 0 })!;
    const k = FrameOrientationTracker.overlayTurns(upright, 0);
    for (const [xr, yr] of [
      [100, 200],
      [1500, 1300],
      [960, 10],
    ]) {
      const xy = mapOverlay(new Float32Array([(xr + 0.5) / W, (yr + 0.5) / H]), W, H, 720, 960, { quarterTurns: k });
      // The engine's canonical rotation of pixels (rotateImagePoint, = rotateRGBA90).
      const p = rotateImagePoint(xr, yr, W, H, canonicalRotation(0, upright));
      expect(xy[0]).toBeCloseTo(((p[0] + 0.5) / H) * 720, 3);
      expect(xy[1]).toBeCloseTo(((p[1] + 0.5) / W) * 960, 3);
    }
  });

  it('device directions: the top edge appears up and the right edge right on the preview', () => {
    // Project device-frame directions into the RAW buffer with R_cam_device(90) (ARCHITECTURE §2),
    // then map the overlay onto the portrait preview: the device top must be up, its right edge right.
    const W = 1920;
    const H = 1440;
    const f = 1500;
    const R = cameraFromDevice(90);
    const project = (dx: number, dy: number): [number, number] => {
      // optical axis (−z_device) tilted slightly towards (dx, dy) in the device frame
      const v = [dx * 0.2, dy * 0.2, -1];
      const c = [R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2], R[6] * v[0] + R[7] * v[1] + R[8] * v[2]];
      return [((f * c[0]) / c[2] + (W - 1) / 2 + 0.5) / W, ((f * c[1]) / c[2] + (H - 1) / 2 + 0.5) / H];
    };
    const o = new FrameOrientationTracker();
    const k = FrameOrientationTracker.overlayTurns(o.observe({ width: W, height: H, screenAngle: 0 })!, 0);
    const el = { w: 393, h: 852 };
    const map = (uv: [number, number]) => mapOverlay(new Float32Array(uv), W, H, el.w, el.h, { quarterTurns: k, margin: 1e4 });
    const top = map(project(0, 1));
    const right = map(project(1, 0));
    expect(top[1]).toBeLessThan(el.h / 2 - 50); // above the centre
    expect(Math.abs(top[0] - el.w / 2)).toBeLessThan(1);
    expect(right[0]).toBeGreaterThan(el.w / 2 + 50); // right of the centre
    expect(Math.abs(right[1] - el.h / 2)).toBeLessThan(1);
    // Without the turn (the old behaviour) the top edge would be drawn on the left.
    const wrong = mapOverlay(new Float32Array(project(0, 1)), W, H, el.w, el.h, { margin: 1e4 });
    expect(wrong[0]).toBeLessThan(el.w / 2 - 50);
  });
});
