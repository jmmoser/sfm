import { describe, expect, it } from 'vitest';
import type { FrameInput } from '../workers/protocol';
import { MemorySink, header, pumpFrame } from './recording-testdata';
import {
  RecordingWriter,
  buildDatasetFiles,
  datasetIndices,
  decodeFrameRow,
  decodeSensorRow,
  encodeFrameRow,
  encodeSensorRow,
  frameMeta,
  parseJsonl,
  partFileName,
  planZipParts,
  recordedSize,
  zipEntryBytes,
  type EncodedImage,
  type RecordedFrame,
  type RecordedSensorRow,
  type RecorderFrameMessage,
  type RecordingHeader,
} from './recording';
import { createZip, readZip } from './zip';

const msg = (f: FrameInput, seq: number): RecorderFrameMessage => ({ seq, sentAt: 5000 + seq, input: frameMeta(f), kind: 'RGBAImage', srcWidth: 64, srcHeight: 48 });
const enc = (n: number, w = 64, h = 48): EncodedImage => ({ jpeg: new Uint8Array(n).fill(7), width: w, height: h });

describe('frames.jsonl rows', () => {
  it('round-trip every FrameInput field exactly (typed arrays, −∞, −0, NaN, null vs absent)', () => {
    for (let i = 0; i < 6; i++) {
      const f = pumpFrame(i);
      const row: RecordedFrame = { seq: i, sentAt: 1.5, input: frameMeta(f), image: { file: 'frames/000000.jpg', width: 64, height: 48, srcWidth: 64, srcHeight: 48, kind: 'RGBAImage', bytes: 10 } };
      const back = decodeFrameRow(encodeFrameRow(row));
      expect(back).toEqual(row);
      const { bitmap: _b, ...expected } = f;
      expect(back.input).toStrictEqual(expected);
      expect(back.input.gravityDevice).toBeInstanceOf(Float64Array);
      expect(Object.is(back.input.imu[0].gyro[1], -0)).toBe(true);
      expect(back.input.imu[1].gyro[2]).toBeNaN();
      if (i < 3) expect(back.input.screenAngleSince).toBe(-Infinity);
    }
    // a frame without the optional FramePump fields keeps them absent
    const bare = decodeFrameRow(encodeFrameRow({ seq: 0, sentAt: 0, input: { t: 1, screenAngle: 0, imu: [] }, image: null, error: 'x' }));
    expect(Object.keys(bare.input).sort()).toEqual(['imu', 'screenAngle', 't']);
    expect(frameMeta(pumpFrame(0))).not.toHaveProperty('bitmap');
  });

  it('parseJsonl skips a truncated last line (an app killed mid-write)', () => {
    const lines = [0, 1].map((i) => encodeFrameRow({ seq: i, sentAt: 0, input: frameMeta(pumpFrame(i)), image: null }));
    const p = parseJsonl(`${lines.join('\n')}\n${lines[1].slice(0, 40)}`, decodeFrameRow);
    expect(p.rows.map((r) => r.seq)).toEqual([0, 1]);
    expect(p.bad).toBe(1);
  });

  it('dataset indices: FrameInput.index when distinct integers, else seq', () => {
    const rows = [0, 1, 2].map((i) => ({ seq: i, sentAt: 0, input: { ...frameMeta(pumpFrame(i)), index: 10 + i }, image: null }) as RecordedFrame);
    expect(datasetIndices(rows)).toEqual([10, 11, 12]);
    rows[2].input.index = 10;
    expect(datasetIndices(rows)).toEqual([0, 1, 2]);
  });

  it('recordedSize fits the long side like the engine and never enlarges', () => {
    expect(recordedSize(1920, 1440, 1280)).toEqual({ width: 1280, height: 960 });
    expect(recordedSize(1440, 1920, 1280)).toEqual({ width: 960, height: 1280 });
    expect(recordedSize(640, 480, 1280)).toEqual({ width: 640, height: 480 });
    expect(recordedSize(640, 480, 0)).toEqual({ width: 640, height: 480 });
  });
});

describe('RecordingWriter', () => {
  it('writes frames in seq order although their encodes finish out of order, image before row', async () => {
    const sink = new MemorySink();
    const w = new RecordingWriter(sink, 2);
    let release0!: (e: EncodedImage) => void;
    const slow = new Promise<EncodedImage>((r) => (release0 = r));
    const p0 = w.writeFrame(msg(pumpFrame(0), 0), slow);
    const p1 = w.writeFrame(msg(pumpFrame(1), 1), enc(5));
    await Promise.resolve();
    expect(sink.lines['frames.jsonl']).toHaveLength(0);
    release0(enc(9, 32, 24));
    const [r0, r1] = await Promise.all([p0, p1]);
    expect(r0.image).toMatchObject({ file: 'frames/000000.jpg', width: 32, height: 24, srcWidth: 64, srcHeight: 48, kind: 'RGBAImage', bytes: 9, store: { key: '0' } });
    expect(r1.image?.bytes).toBe(5);
    expect(sink.lines['frames.jsonl'].map((l) => decodeFrameRow(l).seq)).toEqual([0, 1]);
    expect(sink.images.get('0')?.length).toBe(9);
    expect(sink.flushes).toBe(1); // every 2 frames
    expect(w.stats).toMatchObject({ frames: 2, imageBytes: 14, failedImages: 0 });
  });

  it('a frame whose pixels failed keeps its row (no image, the error), the recording goes on', async () => {
    const sink = new MemorySink();
    const w = new RecordingWriter(sink);
    const r0 = await w.writeFrame(msg(pumpFrame(0), 0), Promise.reject(new Error('decode failed')));
    const r1 = await w.writeFrame({ ...msg(pumpFrame(1), 1), error: 'the pixels could not be copied' }, null);
    const r2 = await w.writeFrame(msg(pumpFrame(2), 2), enc(3));
    expect(r0.image).toBeNull();
    expect(r0.error).toMatch(/decode failed/);
    expect(r1.error).toBe('the pixels could not be copied');
    expect(r2.image?.bytes).toBe(3);
    expect(w.stats).toMatchObject({ frames: 3, failedImages: 2, imageBytes: 3 });
  });

  it('a sink failure rejects that frame and every later call', async () => {
    const sink = new MemorySink();
    const w = new RecordingWriter(sink);
    await w.writeFrame(msg(pumpFrame(0), 0), enc(3));
    sink.failOn = 'putImage';
    await expect(w.writeFrame(msg(pumpFrame(1), 1), enc(3))).rejects.toMatchObject({ name: 'QuotaExceededError' });
    sink.failOn = null;
    await expect(w.writeFrame(msg(pumpFrame(2), 2), enc(3))).rejects.toMatchObject({ name: 'QuotaExceededError' });
    await expect(w.writeLive({ seq: 2, index: 2, ms: 1, waitMs: 0, update: null, error: null })).rejects.toBeTruthy();
    // close still writes the final header and closes the sink, then reports the failure
    await expect(w.close(header({ state: 'complete' }))).rejects.toMatchObject({ name: 'QuotaExceededError' });
    expect(JSON.parse(sink.header).state).toBe('complete');
    expect(sink.closed).toBe(true);
  });

  it('writeSensors appends a batch as one write of JSON-safe lines', async () => {
    const sink = new MemorySink();
    const w = new RecordingWriter(sink);
    const rows: RecordedSensorRow[] = [
      { k: 'motion', t: 1.5, ts: 1.25, interval: 16, rot: [1, -0, NaN], acc: null, accG: [0, -9.8, 0] },
      { k: 'geo', t: 2, fix: { t: 1.9, lat: 47.6, lon: -122.3, alt: null, accuracy: 5, altitudeAccuracy: null } },
    ];
    await w.writeSensors(rows);
    await w.writeSensors([]);
    expect(sink.lines['sensors.jsonl']).toHaveLength(1);
    const back = parseJsonl(sink.text('sensors.jsonl'), decodeSensorRow).rows;
    expect(back).toEqual(rows);
    expect(Object.is((back[0] as Extract<RecordedSensorRow, { k: 'motion' }>).rot![1], -0)).toBe(true);
    expect(() => decodeSensorRow('{"t":1}')).toThrow();
  });

  it('close writes the final header with the writer counts (frames still queued at stop included)', async () => {
    const sink = new MemorySink();
    const w = new RecordingWriter(sink);
    await w.writeHeader(header());
    void w.writeFrame(msg(pumpFrame(0), 0), enc(4));
    void w.writeFrame(msg(pumpFrame(1), 1), enc(6));
    await w.close(header({ state: 'complete', summary: { frames: 1, imageBytes: 4, failedImages: 0, waitMs: 0, durationMs: 100, pump: null } }));
    const h = JSON.parse(sink.header) as RecordingHeader;
    expect(h.state).toBe('complete');
    expect(h.summary).toMatchObject({ frames: 2, imageBytes: 10, failedImages: 0 });
    expect(sink.closed).toBe(true);
  });
});

async function recorded(n: number): Promise<MemorySink> {
  const sink = new MemorySink();
  const w = new RecordingWriter(sink);
  for (let i = 0; i < n; i++) {
    const f = pumpFrame(i);
    await w.writeFrame(msg(f, i), i === 2 ? null : enc(100 + i));
    await w.writeLive({ seq: i, index: i, ms: 50, waitMs: 0, update: { frameIndex: i, decision: i % 2 ? 'skip' : 'keyframe', guidance: 'good', tracking: 'tracking', numKeyframes: 1 + (i >> 1), numRegistered: i >> 1, numPoints: 10 * i, timings: { total: 50 } }, error: null });
  }
  return sink;
}

describe('buildDatasetFiles', () => {
  it('builds the e2e layout: meta without GT, frames.json, verbatim frames.jsonl, derived imu / geo / heading', async () => {
    const sink = await recorded(6);
    const b = buildDatasetFiles(header(), sink.text('frames.jsonl'), sink.text('live.jsonl'));
    const names = b.texts.map((t) => t.name);
    expect(names).toEqual(['meta.json', 'recording.json', 'frames.json', 'frames.jsonl', 'imu.json', 'geo.json', 'heading.json', 'live.jsonl']);
    const m = b.meta;
    expect(m).toMatchObject({ scene: 'recording', source: 'sfm-raw-capture', mode: 'object', quality: 'balanced', hfovPriorDeg: 67.3, gt: null, intrinsics: null, numFrames: 5, missingImages: 1, notSent: 0, badLines: 0, width: 64, height: 48, screenAngle: 0, locationStripped: false });
    expect(m.originLatLonAlt).toEqual([47.6, -122.3, 56.5]);
    expect(m.duration).toBeCloseTo(0.5, 6);
    const text = (n: string): string => b.texts.find((t) => t.name === n)!.text;
    const frames = JSON.parse(text('frames.json')) as { index: number; file: string; t: number }[];
    expect(frames.map((f) => f.index)).toEqual([0, 1, 3, 4, 5]);
    expect(frames[2]).toEqual({ index: 3, file: 'frames/000003.jpg', t: pumpFrame(3).t });
    // frames.jsonl: every row (the one without pixels too), verbatim inputs, the store reference dropped
    const rows = parseJsonl(text('frames.jsonl'), decodeFrameRow).rows;
    expect(rows.map((r) => r.seq)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(rows[0].image).not.toHaveProperty('store');
    expect(rows[2]).toMatchObject({ image: null, error: 'no image' });
    const { bitmap: _b, ...in3 } = pumpFrame(3);
    expect(rows[3].input).toStrictEqual(in3);
    // derived streams: what the engine got — frame 2's IMU and fix too (its pixels are what is missing)
    expect((JSON.parse(text('imu.json')) as unknown[]).length).toBe(12);
    expect((JSON.parse(text('geo.json')) as { t: number }[]).map((g) => g.t)).toEqual([0, 2, 4].map((i) => pumpFrame(i).geo!.t));
    expect(JSON.parse(text('heading.json'))[0]).toEqual({ t: pumpFrame(0).t, heading: 123.4, accuracy: 12, north: 'magnetic', declinationDeg: null });
    expect(b.images.map((i) => i.name)).toEqual(frames.map((f) => f.file));
    expect(b.images[0].image.store).toEqual({ key: '0' });
  });

  it('frames live.jsonl marks notSent are flagged and give nothing to the derived files', async () => {
    const sink = await recorded(4);
    sink.lines['live.jsonl'][1] = JSON.stringify({ seq: 1, index: 1, ms: 0, waitMs: 0, update: null, error: 'EngineCrashedError: Engine disposed', notSent: true });
    const b = buildDatasetFiles(header(), sink.text('frames.jsonl'), sink.text('live.jsonl'));
    const text = (n: string): string => b.texts.find((t) => t.name === n)!.text;
    const rows = parseJsonl(text('frames.jsonl'), decodeFrameRow).rows;
    expect(rows.map((r) => [r.seq, !!r.notSent])).toEqual([
      [0, false],
      [1, true],
      [2, false],
      [3, false],
    ]);
    expect(rows[0]).not.toHaveProperty('notSent');
    expect((JSON.parse(text('frames.json')) as { index: number }[]).map((f) => f.index)).toEqual([0, 3]); // 1 not sent, 2 without pixels
    expect((JSON.parse(text('imu.json')) as unknown[]).length).toBe(6);
    expect(b.meta).toMatchObject({ numFrames: 2, missingImages: 1, notSent: 1 });
    // its JPEG is still exported (nothing recorded is dropped)
    expect(b.images.map((i) => i.name)).toEqual(['frames/000000.jpg', 'frames/000001.jpg', 'frames/000003.jpg']);
  });

  it('sensors.jsonl is exported as recorded, without its GPS rows when the location is stripped', () => {
    const rows: RecordedSensorRow[] = [
      { k: 'motion', t: 1, ts: 1, interval: 16, rot: [1, 2, 3], acc: null, accG: [0, -9.8, 0] },
      { k: 'geo', t: 2, fix: { t: 2, lat: 47.6, lon: -122.3, alt: 5, accuracy: 4, altitudeAccuracy: null } },
    ];
    const sensorsText = rows.map(encodeSensorRow).join('\n');
    const keep = buildDatasetFiles(header(), '', null, { sensorsText });
    expect(keep.texts.find((t) => t.name === 'sensors.jsonl')!.text).toBe(`${sensorsText}\n`);
    const strip = buildDatasetFiles(header(), '', null, { sensorsText, stripLocation: true });
    const out = strip.texts.find((t) => t.name === 'sensors.jsonl')!.text;
    expect(parseJsonl(out, decodeSensorRow).rows).toEqual([rows[0]]);
    expect(out).not.toContain('47.6');
    expect(buildDatasetFiles(header(), '', null, { sensorsText: '' }).texts.map((t) => t.name)).not.toContain('sensors.jsonl');
  });

  it('stripLocation drops every GPS fix and the origin', async () => {
    const sink = await recorded(4);
    const b = buildDatasetFiles(header(), sink.text('frames.jsonl'), null, { stripLocation: true });
    expect(b.meta.originLatLonAlt).toBeNull();
    expect(b.meta.locationStripped).toBe(true);
    expect(b.texts.map((t) => t.name)).not.toContain('geo.json');
    expect(b.texts.map((t) => t.name)).not.toContain('live.jsonl');
    const rows = parseJsonl(b.texts.find((t) => t.name === 'frames.jsonl')!.text, decodeFrameRow).rows;
    expect(rows.every((r) => r.input.geo === null)).toBe(true);
    expect(b.texts.find((t) => t.name === 'frames.jsonl')!.text).not.toContain('47.6');
  });
});

describe('ZIP parts', () => {
  it('planned sizes are the exact archive sizes, ≤ the limit, text files first', async () => {
    const sink = await recorded(6);
    const b = buildDatasetFiles(header(), sink.text('frames.jsonl'), sink.text('live.jsonl'));
    const enc8 = new TextEncoder();
    const entries = [...b.texts.map((t) => ({ name: t.name, size: enc8.encode(t.text).length })), ...b.images.map((i) => ({ name: i.name, size: i.image.bytes }))];
    const total = entries.reduce((s, e) => s + zipEntryBytes(e.name, e.size), 22);
    const limit = Math.max(Math.ceil(total / 3), ...entries.map((e) => zipEntryBytes(e.name, e.size) + 22));
    const plan = planZipParts(entries, limit);
    expect(plan.length).toBeGreaterThanOrEqual(2);
    expect(plan[0].names.slice(0, 3)).toEqual(['meta.json', 'recording.json', 'frames.json']);
    expect(plan.flatMap((p) => p.names)).toEqual(entries.map((e) => e.name));
    for (const p of plan) {
      expect(p.bytes).toBeLessThanOrEqual(limit);
      const zip = createZip(p.names.map((n) => ({ name: n, data: b.texts.find((t) => t.name === n)?.text ?? sink.images.get(String(Number(n.slice(7, 13))))! })));
      expect(zip.length).toBe(p.bytes);
      expect(readZip(zip).map((e) => e.name)).toEqual(p.names);
    }
    // an entry larger than the limit gets a part of its own
    expect(planZipParts([{ name: 'a', size: 10 }, { name: 'b', size: 1000 }, { name: 'c', size: 10 }], 200).map((p) => p.names)).toEqual([['a'], ['b'], ['c']]);
  });

  it('part file names', () => {
    expect(partFileName('sfm-rec-20261021-140210-ab12cd34', 2, 3)).toBe('sfm-rec-20261021-140210-ab12cd34-part2of3.zip');
    expect(partFileName('Object · Oct 21, 14:02', 1, 1)).toBe('Object-Oct-21-14-02.zip');
  });
});
