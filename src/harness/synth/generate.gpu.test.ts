/**
 * End-to-end dataset generation in the browser (tiny settings): both scenes build,
 * render textured frames, and every output file is produced with consistent metadata.
 */
import { describe, expect, it } from 'vitest';
import { generateDataset, type SynthSink } from './generate';

function memorySink() {
  const files = new Map<string, Uint8Array[]>();
  const text = new Map<string, string>();
  const sink: SynthSink = {
    async write(path, data, append = false) {
      if (typeof data === 'string') {
        text.set(path, append ? (text.get(path) ?? '') + data : data);
        return;
      }
      const u8 = data instanceof Blob ? new Uint8Array(await data.arrayBuffer()) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
      if (!append || !files.has(path)) files.set(path, [u8]);
      else files.get(path)!.push(u8);
    },
    log() {},
  };
  const bytes = (p: string) => {
    const parts = files.get(p) ?? [];
    const out = new Uint8Array(parts.reduce((a, b) => a + b.length, 0));
    let o = 0;
    for (const q of parts) {
      out.set(q, o);
      o += q.length;
    }
    return out;
  };
  return { sink, files, text, bytes };
}

async function imageStats(jpeg: Uint8Array): Promise<{ mean: number; std: number; w: number; h: number }> {
  const bmp = await createImageBitmap(new Blob([jpeg as BlobPart], { type: 'image/jpeg' }));
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = c.getContext('2d')!;
  ctx.drawImage(bmp, 0, 0);
  const d = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
  let s = 0,
    s2 = 0;
  const n = d.length / 4;
  for (let i = 0; i < n; i++) {
    const l = 0.2126 * d[4 * i] + 0.7152 * d[4 * i + 1] + 0.0722 * d[4 * i + 2];
    s += l;
    s2 += l * l;
  }
  const mean = s / n;
  return { mean, std: Math.sqrt(s2 / n - mean * mean), w: bmp.width, h: bmp.height };
}

describe('generateDataset (tiny)', () => {
  for (const scene of ['object', 'yard'] as const) {
    it(`${scene}: frames, sensors, depth and GT files`, async () => {
      const m = memorySink();
      const W = scene === 'object' ? 120 : 160,
        H = scene === 'object' ? 160 : 120;
      const summary = await generateDataset(
        { scene, width: W, height: H, frames: 3, fps: 1, duration: 3, gtPoints: 20000, gtFocusPoints: 5000, textureScale: 0.125, depthEvery: 2 },
        m.sink,
      );
      expect(summary.frames).toBe(3);
      for (let i = 0; i < 3; i++) {
        const st = await imageStats(m.bytes(`frames/00000${i}.jpg`));
        expect([st.w, st.h]).toEqual([W, H]);
        expect(st.std).toBeGreaterThan(12); // textured, not blank
        expect(st.mean).toBeGreaterThan(30);
      }
      const meta = JSON.parse(m.text.get('meta.json')!);
      expect(meta.screenAngle).toBe(scene === 'object' ? 0 : 90);
      expect(meta.intrinsics.k1).toBe(-0.03);
      expect(meta.timeOffsetMs).toBe(25);
      const frames = JSON.parse(m.text.get('frames.json')!);
      expect(frames[1].t - frames[1].tTrue).toBe(25);
      expect(frames[0].depthFile).toBe('depth/000000.f32');
      const imu = JSON.parse(m.text.get('imu.json')!);
      expect(imu.length).toBe(181); // 60 Hz (iOS DeviceMotion) over [−0.5, 2.5] s
      expect(imu[0].t).toBe(9500);
      expect(m.text.has('geo.json')).toBe(scene === 'yard');
      const depth = new Float32Array(m.bytes('depth/000002.f32').buffer);
      expect(depth.length).toBe(W * H);
      let valid = 0;
      for (const v of depth) if (v > 0) valid++;
      expect(valid / depth.length).toBeGreaterThan(scene === 'object' ? 0.99 : 0.4);
      expect(m.bytes('gt_points.bin').length).toBe(20000 * 12);
      expect(m.bytes('gt_normals.i8').length).toBe(20000 * 3);
      expect(m.bytes('gt_visibility.u8').length).toBe(20000);
      expect(meta.gt.visible).toBeGreaterThan(100);
      if (scene === 'object') {
        expect(m.bytes('gt_focus_points.bin').length).toBe(5000 * 12);
        expect(meta.gt.focus.visible).toBeGreaterThan(500);
      } else expect(meta.gt.focus).toBeNull();
      // default: no real-capture artefacts, frames.json keeps its legacy fields
      expect(meta.captureArtefacts).toBeNull();
      expect(meta.preset).toBeNull();
      expect('f' in frames[0]).toBe(false);
    });
  }

  it('object-real preset (tiny): rolling shutter, focus breathing, WB, adaptive blur metadata', async () => {
    const m = memorySink();
    // frames around the first fast pan (t = 5.5 s) of the preset trajectory
    const summary = await generateDataset(
      { scene: 'object', preset: 'object-real', width: 120, height: 160, frames: 4, fps: 4, duration: 6, startTimeMs: 10000, gtPoints: 5000, gtFocusPoints: 2000, textureScale: 0.125, depthEvery: 0, rsBands: 8 },
      m.sink,
    );
    expect(summary.frames).toBe(4);
    const meta = JSON.parse(m.text.get('meta.json')!);
    expect(meta.preset).toBe('object-real');
    const ca = meta.captureArtefacts;
    expect(ca.rollingShutter.axis).toBe('cols');
    expect(ca.rollingShutter.firstLine).toBe('right column (u = W-1)');
    expect(ca.rollingShutter.readoutMs).toBe(22);
    expect(ca.rollingShutter.bands).toBe(8);
    expect(ca.focusBreathing.f0).toBe(meta.intrinsics.f);
    expect(ca.focusBreathing.fMax / ca.focusBreathing.f0 - 1).toBeCloseTo(0.006, 6);
    expect(ca.shake.panPeakRates.length).toBe(3);
    const frames = JSON.parse(m.text.get('frames.json')!);
    for (const f of frames) {
      expect(f.f / meta.intrinsics.f - 1).toBeGreaterThanOrEqual(-0.006 - 1e-9);
      expect(f.f / meta.intrinsics.f - 1).toBeLessThanOrEqual(0.006 + 1e-9);
      expect(f.focusDistance).toBeGreaterThan(0.2);
      expect(f.focusDistance).toBeLessThan(1.5);
      expect(f.wb.length).toBe(3);
      expect(f.rsBands).toBeGreaterThanOrEqual(4); // adaptive band count, ≤ rsBands
      expect(f.rsBands).toBeLessThanOrEqual(8);
      expect(f.renders).toBeGreaterThanOrEqual(f.rsBands); // ≥ one render per band
      expect(f.t - f.tTrue).toBe(25);
    }
    // frame 3 (t = 4.5 s) is on the rise of the pan: faster motion → more blur sub-frames than frame 0
    expect(frames.map((f: { omega: number }) => f.omega).some((w: number) => w > 0.5)).toBe(true);
    const st = await imageStats(m.bytes('frames/000003.jpg'));
    expect([st.w, st.h]).toEqual([120, 160]);
    expect(st.std).toBeGreaterThan(8);
  });
});
