import { describe, expect, it } from 'vitest';
import { hasCaptureArtefacts, mergeConfigInput, OBJECT_REAL_SHAKE, resolveConfig, SYNTH_PRESETS, trajectoryFor, YARD_REAL_SHAKE } from './config';
import { projectRadial, rCamDevice } from './conventions';
import { expSO3, m3Mul, m3Vec } from './mat';
import {
  blurSampleCount,
  centerFocusDistance,
  DEFAULT_FOCUS_BREATHING,
  focusBreathingSeries,
  imageMotionPx,
  readoutBands,
  readoutCoordinate,
  readoutGeometry,
  readoutLine,
  readoutTimeOffsetMs,
  rsBandCount,
  subframeOffsetsMs,
  whiteBalanceFn,
} from './realism';
import type { WindowDepth } from './sampling';
import { angularSpeedStats, createObjectTrajectory, createYardTrajectory, panPeakRate, trajectoryPose } from './trajectory';

describe('rolling shutter readout geometry', () => {
  it('portrait θ=0 reads image columns right → left; landscape θ=90 reads rows top → bottom', () => {
    const p = readoutGeometry(0, 720, 960);
    expect(p).toEqual({ axis: 'cols', sign: -1, lines: 720 });
    expect(readoutLine(p, 719, 500)).toBe(0); // right column first
    expect(readoutLine(p, 0, 500)).toBe(719);
    expect(readoutCoordinate(p, 719, 0)).toBe(0);
    expect(readoutCoordinate(p, 0, 959)).toBe(1);
    const l = readoutGeometry(90, 960, 720);
    expect(l).toEqual({ axis: 'rows', sign: 1, lines: 720 });
    expect(readoutLine(l, 500, 0)).toBe(0); // top row first
    expect(readoutCoordinate(l, 0, 719)).toBe(1);
    expect(readoutGeometry(180, 720, 960)).toEqual({ axis: 'cols', sign: 1, lines: 720 });
    expect(readoutGeometry(270, 960, 720)).toEqual({ axis: 'rows', sign: -1, lines: 720 });
    expect(readoutGeometry(0, 720, 960, true).sign).toBe(1);
  });

  it('the sweep is device-fixed: the first line always images the device +x side', () => {
    // For every θ, the camera ray through the first-read line's pixel centre has a positive device-x
    // component and the last line's a negative one (R_cam_device maps device → camera axes).
    for (const theta of [0, 90, 180, 270]) {
      const W = theta % 180 === 0 ? 72 : 96,
        H = theta % 180 === 0 ? 96 : 72;
      const g = readoutGeometry(theta, W, H);
      const Rcd = rCamDevice(theta);
      const devX = (u: number, v: number) => {
        // ray (x, y, 1) in camera axes → device axes: v_dev = R_cdᵀ v_cam; take its x
        const x = u - (W - 1) / 2,
          y = v - (H - 1) / 2;
        return Rcd[0] * x + Rcd[3] * y + Rcd[6] * 1e-9;
      };
      const firstPix = g.axis === 'cols' ? [g.sign === 1 ? 0 : W - 1, H / 2] : [W / 2, g.sign === 1 ? 0 : H - 1];
      const lastPix = g.axis === 'cols' ? [g.sign === 1 ? W - 1 : 0, H / 2] : [W / 2, g.sign === 1 ? H - 1 : 0];
      expect(readoutLine(g, firstPix[0], firstPix[1])).toBe(0);
      expect(devX(firstPix[0], firstPix[1])).toBeGreaterThan(0);
      expect(devX(lastPix[0], lastPix[1])).toBeLessThan(0);
    }
  });

  it('line timing: first line at −readout/2, last at +readout/2, frame time = mid-readout', () => {
    const g = readoutGeometry(90, 960, 720);
    expect(readoutTimeOffsetMs(g, 22, 0, 0)).toBeCloseTo(-11, 12);
    expect(readoutTimeOffsetMs(g, 22, 0, 719)).toBeCloseTo(11, 12);
    expect(readoutTimeOffsetMs(g, 22, 0, 359.5)).toBeCloseTo(0, 12);
    // line delay = readout / (L − 1)
    expect(readoutTimeOffsetMs(g, 22, 0, 1) - readoutTimeOffsetMs(g, 22, 0, 0)).toBeCloseTo(22 / 719, 12);
  });

  for (const [theta, W, H, n] of [
    [0, 720, 960, 32],
    [90, 960, 720, 32],
    [0, 721, 961, 24],
    [270, 160, 120, 7],
  ] as const) {
    it(`bands θ=${theta} ${W}x${H}/${n}: partition the image, mean-line timing, monotone in readout order`, () => {
      const g = readoutGeometry(theta, W, H);
      const bands = readoutBands(g, W, H, n, 22);
      expect(bands.length).toBe(n);
      const owner = new Int32Array(W * H).fill(-1);
      let overlaps = 0,
        wrongLine = 0;
      for (const b of bands) {
        for (let v = b.rect.v0; v <= b.rect.v1; v++)
          for (let u = b.rect.u0; u <= b.rect.u1; u++) {
            if (owner[v * W + u] !== -1) overlaps++;
            owner[v * W + u] = b.index;
            // every pixel of the band belongs to its lines
            const i = readoutLine(g, u, v);
            if (i < b.line0 || i >= b.line1) wrongLine++;
          }
        // band time = mean of its lines' times
        let sum = 0;
        for (let i = b.line0; i < b.line1; i++) sum += (i / (g.lines - 1) - 0.5) * 22;
        expect(b.dtMs).toBeCloseTo(sum / (b.line1 - b.line0), 10);
        const sizes = b.line1 - b.line0;
        expect(sizes === Math.floor(g.lines / n) || sizes === Math.ceil(g.lines / n)).toBe(true);
      }
      expect([overlaps, wrongLine]).toEqual([0, 0]);
      expect(owner.every((x) => x >= 0)).toBe(true);
      for (let k = 1; k < n; k++) {
        expect(bands[k].dtMs).toBeGreaterThan(bands[k - 1].dtMs);
        expect(bands[k].line0).toBe(bands[k - 1].line1);
      }
      // timing error of any line vs its band ≤ half a band of line delays
      const half = (Math.ceil(g.lines / n) / 2) * (22 / (g.lines - 1));
      for (const b of bands) for (const i of [b.line0, b.line1 - 1]) expect(Math.abs((i / (g.lines - 1) - 0.5) * 22 - b.dtMs)).toBeLessThanOrEqual(half + 1e-12);
      // symmetric about mid-readout overall
      expect(bands[0].dtMs).toBeCloseTo(-bands[n - 1].dtMs, 0);
    });
  }
});

describe('adaptive band count', () => {
  it('keeps the step between bands ≤ stepPx (relaxed to 10 % of the blur), within [min, max]', () => {
    expect(rsBandCount(4.1, 2, 48, 0.3)).toBe(14);
    expect(rsBandCount(0.2, 0, 48, 0.3)).toBe(4); // minimum
    expect(rsBandCount(30, 0, 48, 0.3)).toBe(48); // maximum
    expect(rsBandCount(23, 12.6, 48, 0.3)).toBe(19); // a blurred pan frame hides 1.26 px steps
    expect(rsBandCount(23, 12.6, 32, 0)).toBe(32); // fixed count
    expect(rsBandCount(5, 0, 2, 0.3)).toBe(2);
    for (const [rs, blur] of [[1, 0], [8.85, 4.8], [12, 1]]) {
      const n = rsBandCount(rs, blur, 48, 0.3);
      expect(rs / n).toBeLessThanOrEqual(Math.max(0.3, 0.1 * blur) + 1e-12);
    }
  });
});

describe('motion blur sub-frames', () => {
  it('offsets and adaptive counts', () => {
    expect(subframeOffsetsMs(1, 12, 'midpoints')).toEqual([0]);
    expect(subframeOffsetsMs(3, 12, 'endpoints')).toEqual([-6, 0, 6]);
    expect(subframeOffsetsMs(4, 12, 'midpoints')).toEqual([-4.5, -1.5, 1.5, 4.5]);
    expect(blurSampleCount(0.1, 1, 12)).toBe(1);
    expect(blurSampleCount(0.6, 1, 12)).toBe(2);
    expect(blurSampleCount(3.2, 1, 12)).toBe(5);
    expect(blurSampleCount(40, 1, 12)).toBe(12);
    expect(blurSampleCount(40, 0, 3)).toBe(3); // fixed count
  });

  it('imageMotionPx: pure rotation moves the image by ≈ f·angle at the centre, independent of depth', () => {
    const cam = { width: 640, height: 480, f: 500, cx: 319.5, cy: 239.5, k1: 0, k2: 0 };
    const a = { R: expSO3([0, 0, 0]), t: Float64Array.of(0, 0, 0) };
    const b = { R: expSO3([0, 0.01, 0]), t: Float64Array.of(0, 0, 0) };
    const d1 = imageMotionPx(cam, a, b, [[319.5, 239.5]], 1);
    const d2 = imageMotionPx(cam, a, b, [[319.5, 239.5]], 50);
    expect(d1).toBeCloseTo(500 * Math.tan(0.01), 6);
    expect(d2).toBeCloseTo(d1, 9);
    // translation: parallax ∝ 1/depth
    const c = { R: expSO3([0, 0, 0]), t: Float64Array.of(-0.01, 0, 0) };
    expect(imageMotionPx(cam, a, c, [[319.5, 239.5]], 1)).toBeCloseTo(5, 9);
    expect(imageMotionPx(cam, a, c, [[319.5, 239.5]], 10)).toBeCloseTo(0.5, 9);
  });
});

describe('focus breathing', () => {
  const cfg = { ...DEFAULT_FOCUS_BREATHING, amplitude: 0.006, lagS: 0.25 };

  it('spans exactly ±amplitude, near focus → longer focal, linear in diopters', () => {
    const times = Array.from({ length: 200 }, (_, i) => i / 8);
    const dist = times.map((t) => 0.45 + 0.1 * Math.sin(t / 2));
    const s = focusBreathingSeries(dist, times, cfg);
    expect(Math.max(...s.b)).toBeCloseTo(0.006, 12);
    expect(Math.min(...s.b)).toBeCloseTo(-0.006, 12);
    // monotone relation b ↔ AF diopters (linear)
    const k = (s.b[10] - s.b[50]) / (s.diopters[10] - s.diopters[50]);
    for (let i = 0; i < 200; i += 7) expect(s.b[i] - s.b[50]).toBeCloseTo(k * (s.diopters[i] - s.diopters[50]), 12);
    expect(k).toBeGreaterThan(0); // more diopters (nearer) → larger focal
  });

  it('first-order AF lag: exact exponential step response at the frame times', () => {
    const times = Array.from({ length: 40 }, (_, i) => i / 10);
    const dist = times.map((t) => (t < 1 ? 1 : 0.5)); // 1 D → 2 D step at t = 1 s
    const s = focusBreathingSeries(dist, times, cfg);
    for (let i = 0; i < 40; i++) {
      const t = times[i];
      // the follower sees the new target from the first frame at/after the step (t = 1.0): one
      // frame interval (0.1 s) of response has accumulated there
      if (t < 1 - 1e-9) expect(s.diopters[i]).toBeCloseTo(1, 12);
      else expect(s.diopters[i]).toBeCloseTo(2 - Math.exp(-(t - 0.9) / 0.25), 10);
    }
    // smooth: no frame-to-frame jump larger than the step's first increment
    for (let i = 1; i < 40; i++) expect(Math.abs(s.b[i] - s.b[i - 1])).toBeLessThanOrEqual(0.012 * (1 - Math.exp(-0.1 / 0.25)) * (1 + 1e-4)); // the range stops 6e-6 D short of the 2 D target
  });

  it('tiny distance changes are not amplified (minHalfRangeD)', () => {
    const times = [0, 1, 2, 3];
    const s = focusBreathingSeries([20, 21, 22, 20], times, cfg);
    for (const b of s.b) expect(Math.abs(b)).toBeLessThan(0.006 * 0.2);
  });

  it('centre AF distance: median depth of the central region; sky → skyM', () => {
    const cam = { width: 100, height: 80, f: 100, cx: 49.5, cy: 39.5, k1: 0, k2: 0 };
    const W = 50,
      H = 40;
    const data = new Float32Array(W * H).fill(10);
    // window covers x, y ∈ [−0.5, 0.5] × [−0.4, 0.4]; the centre ±0.15 / ±0.12 has depth 2
    for (let r = 0; r < H; r++)
      for (let c = 0; c < W; c++) {
        const x = -0.5 + (c + 0.5) / W,
          y = -0.4 + ((r + 0.5) * 0.8) / H;
        if (Math.abs(x) < 0.12 && Math.abs(y) < 0.1) data[r * W + c] = 2;
      }
    const wd: WindowDepth = { data, width: W, height: H, window: { xmin: -0.5, xmax: 0.5, ymin: -0.4, ymax: 0.4 } };
    expect(centerFocusDistance(wd, cam)).toBe(2);
    expect(centerFocusDistance({ ...wd, data: new Float32Array(W * H) }, cam)).toBe(DEFAULT_FOCUS_BREATHING.skyM);
  });
});

describe('white balance drift', () => {
  it('off → unity; on → bounded, slow, warm/cool swing with opposite red / blue', () => {
    expect(whiteBalanceFn(1, 0)(3)).toEqual([1, 1, 1]);
    const wb = whiteBalanceFn(1, 0.03);
    let maxStep = 0;
    let prev = wb(0);
    for (let i = 1; i <= 400; i++) {
      const g = wb(i * 0.1);
      for (const c of g) expect(Math.abs(c - 1)).toBeLessThanOrEqual(0.03 + 1e-12);
      expect(g[0] - 1).toBeCloseTo(1 - g[2], 12);
      maxStep = Math.max(maxStep, Math.abs(g[0] - prev[0]));
      prev = g;
    }
    expect(maxStep).toBeLessThan(0.006); // < 0.6 % per 0.1 s: slow
  });
});

describe('hand-held shake (trajectories)', () => {
  it('off by default: trajectories unchanged', () => {
    const a = createObjectTrajectory(),
      b = createObjectTrajectory({ shake: { tremorDeg: 0, tremorM: 0, pans: [] } });
    for (const t of [0, 3.3, 12.1]) {
      expect(Array.from(trajectoryPose(a, t).R)).toEqual(Array.from(trajectoryPose(b, t).R));
      expect(Array.from(a.evaluate(t).omegaDev)).toEqual(Array.from(b.evaluate(t).omegaDev));
    }
  });

  for (const [name, mk, shake] of [
    ['object', createObjectTrajectory, OBJECT_REAL_SHAKE],
    ['yard', createYardTrajectory, YARD_REAL_SHAKE],
  ] as const) {
    it(`${name}: pans reach 1–1.5 rad/s, tremor is 8–12 Hz, derivatives stay exact`, () => {
      const traj = mk({ shake });
      for (const p of shake.pans) {
        expect(panPeakRate(p)).toBeGreaterThan(1.0);
        expect(panPeakRate(p)).toBeLessThan(1.5);
      }
      const st = angularSpeedStats(traj, 0, traj.duration, 0.002);
      expect(st.max).toBeGreaterThan(1.2);
      expect(st.max).toBeLessThan(1.6);
      // ω matches the finite-difference body rate (exact jets incl. Gaussian bumps)
      for (const t of [shake.pans[0].t - shake.pans[0].sigmaS, shake.pans[1].t + 0.05, 2.0]) {
        const h = 1e-5;
        const Rm = traj.evaluate(t - h).R_wd,
          Rp = traj.evaluate(t + h).R_wd;
        // ω ≈ vee(R(t)ᵀ (R(t+h) − R(t−h)) / 2h)
        const R = traj.evaluate(t).R_wd;
        const D = Rp.map((x, i) => (x - Rm[i]) / (2 * h));
        const O = [0, 0, 0, 0, 0, 0, 0, 0, 0];
        for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) for (let k = 0; k < 3; k++) O[3 * r + c] += R[3 * k + r] * D[3 * k + c];
        const w = traj.evaluate(t).omegaDev;
        expect(Math.abs(O[7] - w[0])).toBeLessThan(1e-6);
        expect(Math.abs(O[2] - w[1])).toBeLessThan(1e-6);
        expect(Math.abs(O[3] - w[2])).toBeLessThan(1e-6);
      }
      // tremor alone (no pans): 8–12 Hz rate, small amplitude
      const tremorOnly = mk({ shake: { ...shake, pans: [] } });
      const base = mk();
      const n = 2048,
        dt = 1 / 200;
      const diff: number[] = [];
      for (let k = 0; k < n; k++) {
        const t = 1 + k * dt;
        diff.push(tremorOnly.evaluate(t).omegaDev[0] - base.evaluate(t).omegaDev[0]);
      }
      // dominant frequency by a direct DFT scan over 1–30 Hz
      let best = 0,
        bestF = 0;
      for (let f = 1; f <= 30; f += 0.1) {
        let re = 0,
          im = 0;
        for (let k = 0; k < n; k++) {
          re += diff[k] * Math.cos(2 * Math.PI * f * k * dt);
          im += diff[k] * Math.sin(2 * Math.PI * f * k * dt);
        }
        const p = re * re + im * im;
        if (p > best) {
          best = p;
          bestF = f;
        }
      }
      expect(bestF).toBeGreaterThanOrEqual(8);
      expect(bestF).toBeLessThanOrEqual(12);
      const rms = Math.sqrt(diff.reduce((a, b) => a + b * b, 0) / n);
      expect(rms).toBeGreaterThan(0.01);
      expect(rms).toBeLessThan(0.15);
    });
  }
});

describe('real-capture presets', () => {
  it('default configs have no artefacts; presets turn every one on', () => {
    for (const scene of ['object', 'yard'] as const) {
      const c = resolveConfig({ scene });
      expect(hasCaptureArtefacts(c)).toBe(false);
      expect([c.rsReadoutMs, c.whiteBalanceDrift, c.effects.motionBlurPxPerSample]).toEqual([0, 0, 0]);
      expect(c.focusBreathing).toBeNull();
      expect(c.preset).toBeNull();
    }
    const o = resolveConfig({ scene: 'object', preset: 'object-real' });
    expect(o.name).toBe('object-real');
    expect([o.width, o.height, o.fps, o.duration, o.frames]).toEqual([720, 960, 8, 25, 200]);
    expect([o.rsReadoutMs, o.rsBands, o.rsBandStepPx, o.exposureJitter, o.whiteBalanceDrift, o.jpegQuality]).toEqual([22, 48, 0.3, 0.15, 0.03, 0.85]);
    expect(o.focusBreathing?.amplitude).toBe(0.006);
    expect(o.effects.motionBlurPxPerSample).toBe(1.5);
    expect(o.effects.exposureMs).toBe(12);
    expect(o.trajectory.shake).toEqual(OBJECT_REAL_SHAKE);
    expect(hasCaptureArtefacts(o)).toBe(true);
    const y = resolveConfig({ scene: 'yard', preset: 'yard-real', rsReadoutMs: 0, trajectory: { shake: { pans: [] } } });
    expect([y.width, y.height, y.fps, y.frames]).toEqual([960, 720, 6, 240]);
    expect(y.rsReadoutMs).toBe(0); // explicit override wins
    expect(y.trajectory.shake?.pans).toEqual([]);
    expect(y.trajectory.shake?.tremorDeg).toBe(YARD_REAL_SHAKE.tremorDeg); // deep merge keeps the rest
    expect(y.gps).not.toBeNull();
    expect(trajectoryFor(y).screenAngle).toBe(90);
    expect(() => resolveConfig({ scene: 'yard', preset: 'object-real' })).toThrow();
    // the preset table itself is not mutated by resolution
    expect(SYNTH_PRESETS['yard-real'].rsReadoutMs).toBe(22);
  });

  it('mergeConfigInput: deep for objects, replace for arrays / null', () => {
    expect(mergeConfigInput({ a: { b: 1, c: [1, 2] }, d: 1 }, { a: { c: [3] }, d: null })).toEqual({ a: { b: 1, c: [3] }, d: null });
  });
});

describe('rolling-shutter projection sanity (pure)', () => {
  it('a point seen during a fast pan lands where its band pose projects it, several px from the GS projection', () => {
    // camera rotating about its y axis at 1.3 rad/s; portrait θ = 0 → column readout
    const cam = { width: 720, height: 960, f: 753.4, cx: 359.5, cy: 479.5, k1: -0.03, k2: 0.005 };
    const R0 = m3Mul(expSO3([0.1, 0.2, 0]), expSO3([0, 0, 0]));
    const pose = (tMs: number) => {
      const Rk = expSO3([0, 1.3 * (tMs / 1000), 0]);
      return { R: m3Mul(Rk, R0), t: m3Vec(Rk, Float64Array.of(0.1, 0, 0.3)) };
    };
    const g = readoutGeometry(0, cam.width, cam.height);
    const bands = readoutBands(g, cam.width, cam.height, 32, 22);
    const X = (() => {
      // a point that projects to u ≈ 600 at t = 0
      const p = pose(0);
      const xc = [(600 - cam.cx) / cam.f * 2, 0, 2];
      const Rt = [p.R[0], p.R[3], p.R[6], p.R[1], p.R[4], p.R[7], p.R[2], p.R[5], p.R[8]];
      const C = [-(p.R[0] * p.t[0] + p.R[3] * p.t[1] + p.R[6] * p.t[2]), -(p.R[1] * p.t[0] + p.R[4] * p.t[1] + p.R[7] * p.t[2]), -(p.R[2] * p.t[0] + p.R[5] * p.t[1] + p.R[8] * p.t[2])];
      return [0, 1, 2].map((r) => Rt[3 * r] * xc[0] + Rt[3 * r + 1] * xc[1] + Rt[3 * r + 2] * xc[2] + C[r]);
    })();
    const out = new Float64Array(3);
    projectRadial(cam, pose(0), X, out);
    const gs = [out[0], out[1]];
    // fixed point: band of the pixel ↔ pose of the band
    let k = bands.findIndex((b) => gs[0] >= b.rect.u0 - 0.5 && gs[0] <= b.rect.u1 + 0.5);
    for (let it = 0; it < 10; it++) {
      projectRadial(cam, pose(bands[k].dtMs), X, out);
      const k2 = bands.findIndex((b) => out[0] >= b.rect.u0 - 0.5 && out[0] <= b.rect.u1 + 0.5);
      if (k2 === k) break;
      k = k2;
    }
    const shift = Math.hypot(out[0] - gs[0], out[1] - gs[1]);
    // right side of a portrait frame is read first (dt < 0)
    expect(bands[k].dtMs).toBeLessThan(-5);
    expect(shift).toBeGreaterThan(5);
  });
});
