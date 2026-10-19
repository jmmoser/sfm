import { describe, expect, it } from 'vitest';
import { analyzeLensProbe, hfovFrom35mm, parseJpeg, stallIn, type LensProbeInput, type PhotoShot } from './lens-probe';

type Entry = { tag: number; type: number; count: number; value: number[] | string };

/** A TIFF (EXIF) block with IFD0 and an Exif IFD; values that do not fit 4 bytes go after the IFDs. */
function tiff(le: boolean, ifd0: Entry[], exif: Entry[]): Uint8Array {
  const buf = new Uint8Array(1024);
  const dv = new DataView(buf.buffer);
  buf.set(le ? [0x49, 0x49] : [0x4d, 0x4d]);
  dv.setUint16(2, 42, le);
  dv.setUint32(4, 8, le);
  const exifAt = 8 + 2 + (ifd0.length + 1) * 12 + 4;
  let data = exifAt + 2 + exif.length * 12 + 4;
  const writeIfd = (at: number, entries: Entry[]): void => {
    dv.setUint16(at, entries.length, le);
    entries.forEach((e, i) => {
      const o = at + 2 + i * 12;
      dv.setUint16(o, e.tag, le);
      dv.setUint16(o + 2, e.type, le);
      dv.setUint32(o + 4, e.count, le);
      const size = { 2: 1, 3: 2, 4: 4, 5: 8 }[e.type as 2 | 3 | 4 | 5] * e.count;
      const put = (p: number): void => {
        if (typeof e.value === 'string') for (let k = 0; k < e.value.length; k++) buf[p + k] = e.value.charCodeAt(k);
        else if (e.type === 3) dv.setUint16(p, e.value[0], le);
        else if (e.type === 4) dv.setUint32(p, e.value[0], le);
        else {
          dv.setUint32(p, e.value[0], le);
          dv.setUint32(p + 4, e.value[1], le);
        }
      };
      if (size <= 4) put(o + 8);
      else {
        dv.setUint32(o + 8, data, le);
        put(data);
        data += size + (size % 2);
      }
    });
  };
  writeIfd(8, [...ifd0, { tag: 0x8769, type: 4, count: 1, value: [exifAt] }]);
  writeIfd(exifAt, exif);
  return buf.subarray(0, data);
}

function jpeg(t: Uint8Array | null, w: number, h: number): Uint8Array {
  const parts: number[] = [0xff, 0xd8];
  if (t) {
    const len = 2 + 6 + t.length;
    parts.push(0xff, 0xe1, len >> 8, len & 255, ...'Exif'.split('').map((c) => c.charCodeAt(0)), 0, 0, ...t);
  }
  parts.push(0xff, 0xe0, 0, 4, 1, 2); // an APP0 to skip
  parts.push(0xff, 0xc0, 0, 17, 8, h >> 8, h & 255, w >> 8, w & 255, 3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1);
  parts.push(0xff, 0xda, 0, 2, 0xff, 0xd9);
  return new Uint8Array(parts);
}

const APPLE_IFD0: Entry[] = [
  { tag: 0x010f, type: 2, count: 6, value: 'Apple\0' },
  { tag: 0x0110, type: 2, count: 18, value: 'iPhone 16 Pro Max\0' },
  { tag: 0x0112, type: 3, count: 1, value: [6] },
];
const APPLE_EXIF: Entry[] = [
  { tag: 0x829a, type: 5, count: 1, value: [1, 120] },
  { tag: 0x8827, type: 3, count: 1, value: [64] },
  { tag: 0x920a, type: 5, count: 1, value: [6765, 1000] },
  { tag: 0xa405, type: 3, count: 1, value: [24] },
  { tag: 0xa434, type: 2, count: 24, value: 'iPhone back camera 6.765\0' },
  { tag: 0xa002, type: 4, count: 1, value: [4032] },
  { tag: 0xa003, type: 4, count: 1, value: [3024] },
];

describe('parseJpeg', () => {
  for (const le of [false, true]) {
    it(`reads the SOF size and the EXIF fields (${le ? 'little' : 'big'}-endian)`, () => {
      const r = parseJpeg(jpeg(tiff(le, APPLE_IFD0, APPLE_EXIF), 4032, 3024));
      expect(r.width).toBe(4032);
      expect(r.height).toBe(3024);
      expect(r.exif).toEqual({
        make: 'Apple',
        model: 'iPhone 16 Pro Max',
        orientation: 6,
        exposureTime: 1 / 120,
        iso: 64,
        focalLength: 6.765,
        focalLength35: 24,
        lensModel: 'iPhone back camera 6.765',
        pixelWidth: 4032,
        pixelHeight: 3024,
      });
    });
  }

  it('copes with no EXIF, a truncated header and non-JPEG bytes', () => {
    expect(parseJpeg(jpeg(null, 1920, 1440))).toEqual({ width: 1920, height: 1440, exif: null });
    const full = jpeg(tiff(true, APPLE_IFD0, APPLE_EXIF), 4032, 3024);
    expect(() => parseJpeg(full.subarray(0, 60))).not.toThrow();
    expect(parseJpeg(new Uint8Array([0x89, 0x50, 0x4e, 0x47]))).toEqual({ width: null, height: null, exif: null });
    expect(parseJpeg(new Uint8Array([0xff, 0xd9, 0, 0]))).toEqual({ width: null, height: null, exif: null });
  });

  it('reads the size of a PNG', () => {
    const png = new Uint8Array(33);
    png.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13, 0x49, 0x48, 0x44, 0x52]);
    new DataView(png.buffer).setUint32(16, 640);
    new DataView(png.buffer).setUint32(20, 480);
    expect(parseJpeg(png)).toEqual({ width: 640, height: 480, exif: null });
  });
});

describe('hfovFrom35mm', () => {
  it('uses the diagonal definition of the 35 mm equivalent', () => {
    // 4:3: the long side spans 0.8 of the 43.27 mm diagonal
    expect(hfovFrom35mm(26)).toBeCloseTo((2 * Math.atan(34.616 / 52) * 180) / Math.PI, 2);
    expect(hfovFrom35mm(24)).toBeGreaterThan(hfovFrom35mm(26));
    expect(hfovFrom35mm(13)).toBeGreaterThan(100); // ultra-wide
    expect(hfovFrom35mm(26, 3024, 4032)).toBeCloseTo(hfovFrom35mm(26, 4032, 3024), 6); // long side either way
  });
});

describe('stallIn', () => {
  const at30 = (from: number, to: number): number[] => Array.from({ length: Math.round((to - from) / 33.3) }, (_, i) => from + i * 33.3);
  it('finds no stall in a steady stream', () => {
    const r = stallIn(at30(0, 2000), 500, 1000, 33.3);
    expect(r.maxGapMs).toBeCloseTo(33.3, 0);
    expect(r.dropped).toBe(0);
  });
  it('measures a gap inside the window and the frames missing in it', () => {
    const f = [...at30(0, 600), ...at30(900, 2000)];
    const r = stallIn(f, 500, 1000, 33.3);
    expect(r.maxGapMs).toBeGreaterThan(300);
    expect(r.dropped).toBeGreaterThanOrEqual(8);
  });
  it('counts a stream that stopped as a stall to the end of the window', () => {
    const r = stallIn(at30(0, 600), 500, 1500, 33.3);
    expect(r.maxGapMs).toBeGreaterThan(900);
  });
});

function shot(t0: number, latency: number, o: Partial<PhotoShot> = {}): PhotoShot {
  return { requested: null, t0, t1: t0 + latency, bytes: 3_000_000, type: 'image/jpeg', width: 4032, height: 3024, exif: { focalLength: 6.765, focalLength35: 24 }, ...o };
}

function input(o: { gapAround?: number; secondFrames?: boolean; firstStops?: boolean } = {}): LensProbeInput {
  const frames: number[] = [];
  for (let t = 0; t < 20_000; t += 33.3) {
    if (o.gapAround !== undefined && t > o.gapAround && t < o.gapAround + 400) continue;
    if (o.firstStops && t > 12_000 && t < 17_000) continue;
    frames.push(t);
  }
  const secondFrames: number[] = [];
  if (o.secondFrames) for (let t = 12_000; t < 15_000; t += 33.3) secondFrames.push(t);
  return {
    lenses: [
      { label: 'Back Ultra Wide Camera', lens: 'ultrawide', facing: 'back', zoom: { min: 1, max: 10 }, focusDistanceMinM: 0.02, maxWidth: 4032, maxHeight: 3024, maxFrameRate: 60 },
      { label: 'Back Camera', lens: 'wide', facing: 'back', zoom: { min: 1, max: 10 }, focusDistanceMinM: 0.15, maxWidth: 4032, maxHeight: 3024, maxFrameRate: 60 },
      { label: 'Front Camera', lens: 'front', facing: 'front', zoom: null, focusDistanceMinM: null, maxWidth: null, maxHeight: null, maxFrameRate: null },
    ],
    firstCamera: { label: 'Back Camera', lens: 'wide', width: 1920, height: 1440, frameRate: 30 },
    frames,
    baseline: { t0: 0, t1: 2000 },
    photo: {
      available: true,
      error: null,
      capabilities: { imageWidth: { min: 640, max: 4032 }, imageHeight: { min: 480, max: 3024 } },
      settings: null,
      shots: [shot(2500, 250), shot(4000, 200), shot(5500, 300), shot(7000, 220), shot(8500, 260, { requested: { imageWidth: 4032, imageHeight: 3024 } })],
      grabFrame: { ms: 12, width: 1920, height: 1440 },
      trackBefore: { width: 1920, height: 1440, frameRate: 30 },
      trackAfter: { width: 1920, height: 1440, frameRate: 30 },
    },
    concurrent: {
      second: { label: 'Back Ultra Wide Camera', lens: 'ultrawide' },
      error: null,
      openMs: 400,
      t0: 12_000,
      t1: 15_000,
      secondFrames,
      events: o.firstStops ? [{ t: 12_050, track: 'first', type: 'mute' }] : [],
      firstAfter: { readyState: 'live', muted: false },
      after: { t0: 17_500, t1: 19_000 },
    },
    perLens: [
      { label: 'Back Ultra Wide Camera', lens: 'ultrawide', error: null, videoWidth: 1920, videoHeight: 1440, zoom: null, focusDistanceMinM: 0.02, photo: { width: 4032, height: 3024, latencyMs: 300, exif: { focalLength: 2.22, focalLength35: 13 } } },
      { label: 'Back Telephoto Camera', lens: 'tele', error: 'getUserMedia() timed out after 6000 ms', videoWidth: null, videoHeight: null, zoom: null, focusDistanceMinM: null, photo: null },
    ],
  };
}

describe('analyzeLensProbe', () => {
  it('reports photo size, latency, EXIF and a video that keeps running', () => {
    const r = analyzeLensProbe(input({ secondFrames: true }));
    expect(r.photo.sizes).toEqual(['4032x3024']);
    expect(r.photo.latencyMs).toEqual({ n: 4, p50: 220, max: 300 }); // the largest-size shot is not in it
    expect(r.photo.largerOnRequest).toBe(false);
    expect(r.photo.trackChanged).toBeNull();
    expect(Math.max(...r.photo.shots.map((s) => s.videoMaxGapMs!))).toBeLessThan(40);
    expect(r.summary.join('\n')).toMatch(/the video keeps running/);
    expect(r.summary.join('\n')).toMatch(/24 mm equivalent/);
    expect(r.perLens[0].hfovDeg).toBeGreaterThan(100);
    expect(r.perLens[1].hfovDeg).toBeNull();
    expect(r.summary.join('\n')).toMatch(/Back Telephoto Camera: getUserMedia\(\) timed out/);
  });

  it('flags a video stall around a photo', () => {
    const r = analyzeLensProbe(input({ gapAround: 4050, secondFrames: true }));
    expect(r.photo.shots[1].videoMaxGapMs).toBeGreaterThan(380);
    expect(r.photo.shots[1].videoDropped).toBeGreaterThan(5);
    expect(r.summary.join('\n')).toMatch(/STALLS/);
  });

  it('two cameras: both ran, the first stopped, the second gave nothing, the second failed', () => {
    const both = analyzeLensProbe(input({ secondFrames: true })).concurrent;
    expect(both.verdict).toBe('both-ran');
    expect(both.secondFps).toBeGreaterThan(25);
    expect(both.firstFpsDuring).toBeGreaterThan(25);

    const stopped = analyzeLensProbe(input({ secondFrames: true, firstStops: true })).concurrent;
    expect(stopped.verdict).toBe('first-stopped');
    expect(stopped.events).toEqual(['50 ms first mute']);
    expect(stopped.firstFpsAfter).toBeGreaterThan(25);

    expect(analyzeLensProbe(input()).concurrent.verdict).toBe('second-no-frames');

    const failedIn = input();
    failedIn.concurrent = { ...failedIn.concurrent, error: 'NotReadableError: busy', t0: null, t1: null, openMs: null, firstAfter: null, after: null };
    const failed = analyzeLensProbe(failedIn);
    expect(failed.concurrent.verdict).toBe('second-failed');
    expect(failed.summary.join('\n')).toMatch(/failed \(NotReadableError: busy\)/);

    const none = input();
    none.concurrent = { ...none.concurrent, second: null, t0: null, t1: null, firstAfter: null, after: null };
    expect(analyzeLensProbe(none).concurrent.verdict).toBe('not-run');
  });

  it('notices a larger photo on request and a changed video track', () => {
    const inp = input({ secondFrames: true });
    inp.photo.shots[4] = { ...inp.photo.shots[4], width: 8064, height: 6048 };
    inp.photo.trackAfter = { width: 1920, height: 1080, frameRate: 30 };
    const r = analyzeLensProbe(inp);
    expect(r.photo.largerOnRequest).toBe(true);
    expect(r.photo.trackChanged).toBe('1920x1440 @ 30 fps → 1920x1080 @ 30 fps');
  });

  it('without ImageCapture', () => {
    const inp = input({ secondFrames: true });
    inp.photo = { ...inp.photo, available: false, error: 'ImageCapture is not available', shots: [], grabFrame: null };
    const r = analyzeLensProbe(inp);
    expect(r.photo.latencyMs).toBeNull();
    expect(r.summary[1]).toBe('takePhoto(): unavailable — ImageCapture is not available.');
  });
});
