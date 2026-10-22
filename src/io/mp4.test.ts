/**
 * MP4 writer (src/io/mp4.ts): synthetic chunks with a fake avcC parsed back — a valid box tree
 * (every size inside its parent, faststart order), stsz holds N samples, the stts total is
 * N · timescale / 30, the avcC is byte-identical, stss lists exactly the key chunks, the samples sit
 * where stco / stsz say; reordered timestamps get ctts + an edit list; Annex B input is converted.
 */
import { describe, expect, it } from 'vitest';
import { Rng } from '../math/random';
import { Mp4Writer, avcConfigFromParameterSets, parseMp4Boxes, readMp4, splitAnnexB, type Mp4Box } from './mp4';

const FAKE_AVCC = Uint8Array.from([1, 0x64, 0x00, 0x28, 0xff, 0xe1, 0x00, 0x04, 0x67, 0x64, 0x00, 0x28, 0x01, 0x00, 0x02, 0x68, 0xee]);

/** Synthetic AVCC chunk: one NAL (type 5 for keys, 1 otherwise) of `len` payload bytes tagged with `i`. */
function chunk(i: number, key: boolean, len: number): Uint8Array {
  const b = new Uint8Array(4 + 1 + len);
  new DataView(b.buffer).setUint32(0, 1 + len);
  b[4] = key ? 0x65 : 0x41;
  for (let k = 0; k < len; k++) b[5 + k] = (i * 31 + k) & 0xff;
  return b;
}

async function bytesOf(blob: Blob): Promise<Uint8Array> {
  return new Uint8Array(await blob.arrayBuffer());
}

/** Every box lies inside its parent and the top level covers the file exactly. */
function checkTree(boxes: Mp4Box[], start: number, end: number): void {
  let o = start;
  for (const b of boxes) {
    expect(b.start).toBe(o);
    expect(b.start + b.size).toBeLessThanOrEqual(end);
    o += b.size;
  }
  if (boxes.length) expect(o).toBe(end);
}

describe('Mp4Writer', () => {
  it('writes a faststart single-track H.264 file that parses back exactly', async () => {
    const rng = new Rng(5);
    const n = 180;
    const w = new Mp4Writer({ width: 720, height: 1280, fps: 30 });
    w.setDecoderConfig(FAKE_AVCC);
    const chunks: Uint8Array[] = [];
    const keyIdx: number[] = [];
    for (let i = 0; i < n; i++) {
      const key = i % 60 === 0 || i === 97; // forced key frames plus one the encoder chose
      const c = chunk(i, key, key ? 4000 + rng.int(500) : 200 + rng.int(900));
      chunks.push(c);
      if (key) keyIdx.push(i + 1);
      w.addChunk(c, { key, timestampUs: Math.round((i * 1e6) / 30) });
    }
    expect(w.sampleCount).toBe(n);
    const file = await bytesOf(w.finish());
    const boxes = parseMp4Boxes(file);
    expect(boxes.map((b) => b.type)).toEqual(['ftyp', 'moov', 'mdat']);
    checkTree(boxes, 0, file.length);
    const walk = (bs: Mp4Box[]): void => {
      for (const b of bs)
        if (b.children) {
          const hdr = b.type === 'stsd' ? 16 : b.type === 'avc1' ? 86 : 8;
          checkTree(b.children, b.start + hdr, b.start + b.size);
          walk(b.children);
        }
    };
    walk(boxes);
    const info = readMp4(file);
    expect([info.width, info.height]).toEqual([720, 1280]);
    expect(info.timescale).toBe(30000);
    expect(info.samples.length).toBe(n);
    expect(info.stts).toEqual([[n, 1000]]);
    expect(info.stts.reduce((s, [c, d]) => s + c * d, 0)).toBe((n * info.timescale) / 30);
    expect(info.duration).toBe((n * info.timescale) / 30);
    expect(Array.from(info.avcC)).toEqual(Array.from(FAKE_AVCC));
    expect(info.codec).toBe('avc1.640028');
    expect(info.stss).toEqual(keyIdx);
    expect(info.samples.filter((s) => s.key).length).toBe(keyIdx.length);
    // no reordering: no ctts / edit list, pts = dts
    const names = JSON.stringify(boxes);
    expect(names.includes('"ctts"')).toBe(false);
    expect(names.includes('"elst"')).toBe(false);
    info.samples.forEach((s, i) => {
      expect(s.pts).toBe(i * 1000);
      expect(Array.from(file.subarray(s.offset, s.offset + s.size))).toEqual(Array.from(chunks[i]));
    });
    // the mdat payload starts right after its header; the moov precedes it (faststart)
    const mdat = boxes[2];
    expect(info.samples[0].offset).toBe(mdat.start + 8);
    expect(mdat.size).toBe(8 + chunks.reduce((s, c) => s + c.length, 0));
    // mvhd duration in ms
    const mvhd = boxes[1].children!.find((b) => b.type === 'mvhd')!;
    const dv = new DataView(file.buffer);
    expect(dv.getUint32(mvhd.start + 20)).toBe(1000);
    expect(dv.getUint32(mvhd.start + 24)).toBe(6000);
  });

  it('reordered presentation times (B-frames) get ctts offsets and an edit list', async () => {
    // decode order I P B B P B B …: presentation 0, 3, 1, 2, 6, 4, 5
    const order = [0, 3, 1, 2, 6, 4, 5];
    const w = new Mp4Writer({ width: 64, height: 48, fps: 30 });
    w.setDecoderConfig(FAKE_AVCC);
    order.forEach((p, i) => w.addChunk(chunk(i, i === 0, 10), { key: i === 0, timestampUs: Math.round((p * 1e6) / 30) }));
    const info = readMp4(await bytesOf(w.finish()));
    expect(info.samples.map((s) => s.pts / 1000)).toEqual(order);
    expect(info.samples.map((s) => s.dts / 1000)).toEqual([0, 1, 2, 3, 4, 5, 6]);
    expect(info.stss).toEqual([1]);
  });

  it('converts Annex B chunks without a description: avcC from the SPS / PPS, AVCC samples', async () => {
    const sps = Uint8Array.from([0x67, 0x42, 0xe0, 0x1f, 0xaa, 0xbb]);
    const pps = Uint8Array.from([0x68, 0xce, 0x3c, 0x80]);
    const idr = Uint8Array.from([0x65, 1, 2, 3, 0, 0, 3, 1]);
    const p = Uint8Array.from([0x41, 9, 8, 7]);
    const sc4 = [0, 0, 0, 1];
    const sc3 = [0, 0, 1];
    const key = Uint8Array.from([...sc4, 0x09, 0xf0, ...sc4, ...sps, ...sc4, ...pps, ...sc3, ...idr]);
    expect(splitAnnexB(key).map((n) => n[0] & 0x1f)).toEqual([9, 7, 8, 5]);
    const w = new Mp4Writer({ width: 320, height: 240, fps: 30 });
    w.addChunk(key, { key: true, timestampUs: 0 });
    w.addChunk(Uint8Array.from([...sc4, ...p]), { key: false, timestampUs: 33333 });
    const file = await bytesOf(w.finish());
    const info = readMp4(file);
    expect(Array.from(info.avcC)).toEqual(Array.from(avcConfigFromParameterSets(sps, pps)));
    expect(info.codec).toBe('avc1.42e01f');
    const s0 = file.subarray(info.samples[0].offset, info.samples[0].offset + info.samples[0].size);
    // the AUD / SPS / PPS are moved out of the sample: one length-prefixed IDR NAL
    expect(Array.from(s0)).toEqual([0, 0, 0, idr.length, ...idr]);
    const s1 = file.subarray(info.samples[1].offset, info.samples[1].offset + info.samples[1].size);
    expect(Array.from(s1)).toEqual([0, 0, 0, p.length, ...p]);
  });

  it('refuses a file without frames, a decoder configuration or a leading key frame', () => {
    expect(() => new Mp4Writer({ width: 16, height: 16, fps: 29.97 })).toThrow(/fps/);
    const a = new Mp4Writer({ width: 16, height: 16, fps: 30 });
    a.setDecoderConfig(FAKE_AVCC);
    expect(() => a.finish()).toThrow(/no frames/);
    const b = new Mp4Writer({ width: 16, height: 16, fps: 30 });
    b.addChunk(chunk(0, true, 4), { key: true, timestampUs: 0 });
    expect(() => b.finish()).toThrow(/decoder configuration/);
    const c = new Mp4Writer({ width: 16, height: 16, fps: 30 });
    c.setDecoderConfig(FAKE_AVCC);
    c.addChunk(chunk(0, false, 4), { key: false, timestampUs: 0 });
    expect(() => c.finish()).toThrow(/key frame/);
    expect(() => c.setDecoderConfig(Uint8Array.from([0, 1, 2]))).toThrow(/AVCDecoderConfigurationRecord/);
  });

  it('parseMp4Boxes rejects a box that leaves its parent', () => {
    const bad = Uint8Array.from([0, 0, 0, 40, ...'ftyp'.split('').map((c) => c.charCodeAt(0)), 0, 0, 0, 0]);
    expect(() => parseMp4Boxes(bad)).toThrow(/beyond its parent/);
  });
});
