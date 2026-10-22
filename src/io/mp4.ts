/**
 * Minimal single-track H.264 MP4 (ISO/IEC 14496-12 / -14 / -15) for the fly-through video:
 * `Mp4Writer` takes WebCodecs `EncodedVideoChunk` bytes in decode order and writes a
 * "faststart" file (ftyp, moov, then one mdat), which plays in Photos, Messages, QuickTime and
 * browsers. `readMp4` parses such a file back (tests, the e2e harness).
 *
 * Boxes: ftyp (isom, iso2, avc1, mp41); moov → mvhd, trak → tkhd, [edts → elst], mdia → mdhd, hdlr
 * 'vide', minf → vmhd, dinf → dref → 'url ' (self-contained), stbl → stsd → avc1 (avcC, pasp 1:1),
 * stts, [ctts], stss, stsc, stsz, stco. All samples sit in one chunk, so stsc has one entry and
 * stco one offset (the mdat payload start, known once the moov size is known). There is no
 * fragmenting: the encoded chunks are kept until `finish` (≈ 6 Mbit/s × 10 s ≈ 7.5 MB). Raw frames
 * are never kept.
 *
 * Timing: constant frame rate. The media timescale is 1000 · fps, so every sample lasts 1000 ticks
 * (stts has one entry). Presentation times come from the chunk timestamps (µs). An encoder that
 * reorders frames (B-frames: presentation ≠ decode order) gets ctts offsets made non-negative by a
 * decoder delay that an edit list (elst media_time) skips. Without reordering neither box is
 * written. stss lists every key sample, even when all are keys.
 *
 * Sample format: avc1 takes AVCC samples (4-byte NAL lengths) and an AVCDecoderConfigurationRecord
 * (the avcC box), which VideoEncoder hands over as `decoderConfig.description` when configured with
 * `avc: { format: 'avc' }` (the default). If an encoder still emits Annex B (start codes) without a
 * description, the samples are converted to AVCC, the SPS / PPS / AUD NAL units are moved out of
 * them, and the avcC is built from the first SPS and PPS (without the High-profile chroma
 * extension, which decoders do not need).
 *
 * Limits: one video track, 32-bit sizes and offsets (files < 4 GiB).
 */

export interface Mp4WriterOptions {
  width: number;
  height: number;
  /** Frames per second (integer; every sample lasts 1/fps s). */
  fps: number;
}

export interface Mp4ChunkInfo {
  /** Key (sync) frame. */
  key: boolean;
  /** Presentation timestamp (µs), as `EncodedVideoChunk.timestamp`. */
  timestampUs: number;
}

const TICKS_PER_FRAME = 1000;
const MATRIX = [0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000];

function u8(...v: number[]): Uint8Array {
  return Uint8Array.from(v);
}

function u16(v: number): Uint8Array {
  return u8((v >>> 8) & 0xff, v & 0xff);
}

function u32(v: number): Uint8Array {
  return u8((v >>> 24) & 0xff, (v >>> 16) & 0xff, (v >>> 8) & 0xff, v & 0xff);
}

function u32s(vs: ArrayLike<number>): Uint8Array {
  const out = new Uint8Array(4 * vs.length);
  const dv = new DataView(out.buffer);
  for (let i = 0; i < vs.length; i++) dv.setUint32(4 * i, vs[i] >>> 0);
  return out;
}

function ascii(s: string): Uint8Array {
  return Uint8Array.from(s, (c) => c.charCodeAt(0));
}

function concat(parts: Uint8Array[]): Uint8Array {
  let n = 0;
  for (const p of parts) n += p.length;
  const out = new Uint8Array(n);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

function box(type: string, ...parts: Uint8Array[]): Uint8Array {
  const body = concat(parts);
  return concat([u32(8 + body.length), ascii(type), body]);
}

function fullBox(type: string, version: number, flags: number, ...parts: Uint8Array[]): Uint8Array {
  return box(type, u8(version, (flags >>> 16) & 0xff, (flags >>> 8) & 0xff, flags & 0xff), ...parts);
}

/** NAL units of an Annex B byte stream (start codes 00 00 01 / 00 00 00 01). */
export function splitAnnexB(data: Uint8Array): Uint8Array[] {
  const starts: { at: number; len: number }[] = [];
  for (let i = 0; i + 2 < data.length; i++) {
    if (data[i] === 0 && data[i + 1] === 0) {
      if (data[i + 2] === 1) {
        starts.push({ at: i, len: 3 });
        i += 2;
      } else if (data[i + 2] === 0 && i + 3 < data.length && data[i + 3] === 1) {
        starts.push({ at: i, len: 4 });
        i += 3;
      }
    }
  }
  const out: Uint8Array[] = [];
  for (let k = 0; k < starts.length; k++) {
    const b = starts[k].at + starts[k].len;
    let e = k + 1 < starts.length ? starts[k + 1].at : data.length;
    // trailing zero bytes belong to the next start code (00 00 00 01) or are padding
    while (e > b && data[e - 1] === 0) e--;
    if (e > b) out.push(data.subarray(b, e));
  }
  return out;
}

/** True when `data` starts with an Annex B start code. */
function isAnnexB(data: Uint8Array): boolean {
  return data.length >= 4 && data[0] === 0 && data[1] === 0 && (data[2] === 1 || (data[2] === 0 && data[3] === 1));
}

/** AVCDecoderConfigurationRecord of one SPS and one PPS (4-byte NAL lengths). */
export function avcConfigFromParameterSets(sps: Uint8Array, pps: Uint8Array): Uint8Array {
  if (sps.length < 4) throw new Error('MP4: SPS too short');
  return concat([u8(1, sps[1], sps[2], sps[3], 0xff, 0xe1), u16(sps.length), sps, u8(1), u16(pps.length), pps]);
}

export class Mp4Writer {
  private readonly samples: Uint8Array[] = [];
  private readonly keys: number[] = [];
  private readonly pts: number[] = [];
  private avcC: Uint8Array | null = null;
  private sps: Uint8Array | null = null;
  private pps: Uint8Array | null = null;
  private payload = 0;
  private finished = false;

  constructor(readonly opts: Mp4WriterOptions) {
    if (!(opts.width > 0 && opts.height > 0 && opts.width < 65536 && opts.height < 65536)) throw new Error('MP4: bad frame size');
    if (!(Number.isInteger(opts.fps) && opts.fps > 0 && opts.fps <= 240)) throw new Error('MP4: fps must be an integer in 1…240');
  }

  /** Media timescale (ticks per second): 1000 · fps. */
  get timescale(): number {
    return TICKS_PER_FRAME * this.opts.fps;
  }

  get sampleCount(): number {
    return this.samples.length;
  }

  /** Bytes of encoded video so far. */
  get bytes(): number {
    return this.payload;
  }

  /** The AVCDecoderConfigurationRecord (`EncodedVideoChunkMetadata.decoderConfig.description`). */
  setDecoderConfig(description: AllowSharedBufferSource): void {
    const b = ArrayBuffer.isView(description)
      ? new Uint8Array(description.buffer, description.byteOffset, description.byteLength)
      : new Uint8Array(description as ArrayBuffer);
    if (b.length < 7 || b[0] !== 1) throw new Error('MP4: decoder description is not an AVCDecoderConfigurationRecord');
    this.avcC = b.slice();
  }

  /** Appends one encoded frame (decode order). Annex B data is converted to AVCC. */
  addChunk(data: Uint8Array, info: Mp4ChunkInfo): void {
    if (this.finished) throw new Error('MP4: already finished');
    let sample = data;
    if (!this.avcC && isAnnexB(data)) {
      const nals = splitAnnexB(data).filter((n) => {
        const t = n[0] & 0x1f;
        if (t === 7 && !this.sps) this.sps = n.slice();
        if (t === 8 && !this.pps) this.pps = n.slice();
        return t !== 7 && t !== 8 && t !== 9;
      });
      sample = concat(nals.flatMap((n) => [u32(n.length), n]));
    } else sample = data.slice();
    this.samples.push(sample);
    this.pts.push(info.timestampUs);
    if (info.key) this.keys.push(this.samples.length);
    this.payload += sample.length;
  }

  /** The finished file. */
  finish(): Blob {
    const avcC = this.avcC ?? (this.sps && this.pps ? avcConfigFromParameterSets(this.sps, this.pps) : null);
    if (!avcC) throw new Error('MP4: no decoder configuration (SPS / PPS)');
    const n = this.samples.length;
    if (n === 0) throw new Error('MP4: no frames');
    if (!this.keys.length || this.keys[0] !== 1) throw new Error('MP4: the first frame must be a key frame');
    if (this.payload + 8 > 0xffffffff) throw new Error('MP4: file too large');
    this.finished = true;
    const ftyp = box('ftyp', ascii('isom'), u32(512), ascii('isom'), ascii('iso2'), ascii('avc1'), ascii('mp41'));
    let moov = this.moov(avcC, 0);
    moov = this.moov(avcC, ftyp.length + moov.length + 8);
    const mdatHeader = concat([u32(8 + this.payload), ascii('mdat')]);
    return new Blob([ftyp, moov, mdatHeader, ...this.samples] as BlobPart[], { type: 'video/mp4' });
  }

  /** Composition offsets (ticks, ≥ 0) per sample and the decoder delay; null without reordering. */
  private compositionOffsets(): { offsets: number[]; delay: number } | null {
    const ts = this.timescale;
    const base = Math.min(...this.pts);
    const c = this.pts.map((p, i) => Math.round(((p - base) * ts) / 1e6) - i * TICKS_PER_FRAME);
    // timestamps of a constant-rate stream round to whole frames: anything else is reordering
    if (c.every((v) => Math.abs(v) <= TICKS_PER_FRAME / 100)) return null;
    const delay = Math.max(0, -Math.min(...c));
    return { offsets: c.map((v) => v + delay), delay };
  }

  private moov(avcC: Uint8Array, mdatPayloadOffset: number): Uint8Array {
    const { width, height } = this.opts;
    const n = this.samples.length;
    const ts = this.timescale;
    const mediaDuration = n * TICKS_PER_FRAME;
    const movieTimescale = 1000;
    const movieDuration = Math.round((mediaDuration * movieTimescale) / ts);
    const comp = this.compositionOffsets();

    const mvhd = fullBox('mvhd', 0, 0, u32(0), u32(0), u32(movieTimescale), u32(movieDuration), u32(0x00010000), u16(0x0100), u16(0), u32(0), u32(0), u32s(MATRIX), u32s([0, 0, 0, 0, 0, 0]), u32(2));
    const tkhd = fullBox('tkhd', 0, 3, u32(0), u32(0), u32(1), u32(0), u32(movieDuration), u32(0), u32(0), u16(0), u16(0), u16(0), u16(0), u32s(MATRIX), u32(width * 0x10000), u32(height * 0x10000));
    const edts = comp ? box('edts', fullBox('elst', 0, 0, u32(1), u32(movieDuration), u32(comp.delay), u16(1), u16(0))) : new Uint8Array(0);
    const mdhd = fullBox('mdhd', 0, 0, u32(0), u32(0), u32(ts), u32(mediaDuration), u16(0x55c4) /* 'und' */, u16(0));
    const hdlr = fullBox('hdlr', 0, 0, u32(0), ascii('vide'), u32(0), u32(0), u32(0), ascii('VideoHandler\0'));
    const vmhd = fullBox('vmhd', 0, 1, u16(0), u16(0), u16(0), u16(0));
    const dinf = box('dinf', fullBox('dref', 0, 0, u32(1), fullBox('url ', 0, 1)));
    const compressor = new Uint8Array(32);
    const avc1 = box(
      'avc1',
      new Uint8Array(6),
      u16(1), // data_reference_index
      u16(0),
      u16(0),
      u32s([0, 0, 0]),
      u16(width),
      u16(height),
      u32(0x00480000),
      u32(0x00480000),
      u32(0),
      u16(1), // frame_count
      compressor,
      u16(0x0018),
      u16(0xffff),
      box('avcC', avcC),
      box('pasp', u32(1), u32(1)),
    );
    const stsd = fullBox('stsd', 0, 0, u32(1), avc1);
    const stts = fullBox('stts', 0, 0, u32(1), u32(n), u32(TICKS_PER_FRAME));
    let ctts: Uint8Array = new Uint8Array(0);
    if (comp) {
      const runs: number[] = [];
      for (const o of comp.offsets) {
        if (runs.length && runs[runs.length - 1] === o) runs[runs.length - 2]++;
        else runs.push(1, o);
      }
      ctts = fullBox('ctts', 0, 0, u32(runs.length / 2), u32s(runs));
    }
    const stss = fullBox('stss', 0, 0, u32(this.keys.length), u32s(this.keys));
    const stsc = fullBox('stsc', 0, 0, u32(1), u32(1), u32(n), u32(1));
    const stsz = fullBox('stsz', 0, 0, u32(0), u32(n), u32s(this.samples.map((s) => s.length)));
    const stco = fullBox('stco', 0, 0, u32(1), u32(mdatPayloadOffset));
    const stbl = box('stbl', stsd, stts, ctts, stss, stsc, stsz, stco);
    const minf = box('minf', vmhd, dinf, stbl);
    const mdia = box('mdia', mdhd, hdlr, minf);
    const trak = box('trak', tkhd, edts, mdia);
    return box('moov', mvhd, trak);
  }
}

// ------------------------------------------------------------------------------- reading

export interface Mp4Box {
  type: string;
  /** Offset of the box (its size field) in the file. */
  start: number;
  size: number;
  children?: Mp4Box[];
}

export interface Mp4Sample {
  offset: number;
  size: number;
  key: boolean;
  /** Decode / presentation time (media ticks; presentation after the edit list). */
  dts: number;
  pts: number;
}

export interface Mp4Info {
  boxes: Mp4Box[];
  width: number;
  height: number;
  timescale: number;
  /** Media duration (ticks). */
  duration: number;
  /** The avcC payload (AVCDecoderConfigurationRecord). */
  avcC: Uint8Array;
  /** Codec string derived from the avcC (avc1.PPCCLL). */
  codec: string;
  stts: [number, number][];
  stss: number[];
  samples: Mp4Sample[];
}

const CONTAINERS = new Set(['moov', 'trak', 'mdia', 'minf', 'stbl', 'dinf', 'edts']);

/** Box tree of `data[start, end)`; throws on a size that leaves its parent. */
export function parseMp4Boxes(data: Uint8Array, start = 0, end = data.length): Mp4Box[] {
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const out: Mp4Box[] = [];
  let o = start;
  while (o < end) {
    if (o + 8 > end) throw new Error(`MP4: truncated box header at ${o}`);
    const size = dv.getUint32(o);
    const type = String.fromCharCode(data[o + 4], data[o + 5], data[o + 6], data[o + 7]);
    if (size < 8 || o + size > end) throw new Error(`MP4: box ${type} at ${o} has size ${size} beyond its parent`);
    const b: Mp4Box = { type, start: o, size };
    if (CONTAINERS.has(type)) b.children = parseMp4Boxes(data, o + 8, o + size);
    else if (type === 'stsd') b.children = parseMp4Boxes(data, o + 16, o + size);
    else if (type === 'avc1') b.children = parseMp4Boxes(data, o + 86, o + size);
    out.push(b);
    o += size;
  }
  return out;
}

function findBox(boxes: Mp4Box[], path: string[]): Mp4Box | null {
  let cur: Mp4Box | undefined;
  let list = boxes;
  for (const t of path) {
    cur = list.find((b) => b.type === t);
    if (!cur) return null;
    list = cur.children ?? [];
  }
  return cur ?? null;
}

/** Parses a single-track H.264 MP4 (as written by `Mp4Writer`): sizes, timing and samples. */
export function readMp4(data: Uint8Array): Mp4Info {
  const boxes = parseMp4Boxes(data);
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const need = (path: string[]): Mp4Box => {
    const b = findBox(boxes, path);
    if (!b) throw new Error(`MP4: missing ${path.join('/')}`);
    return b;
  };
  const stbl = ['moov', 'trak', 'mdia', 'minf', 'stbl'];
  const mdhd = need(['moov', 'trak', 'mdia', 'mdhd']);
  const timescale = dv.getUint32(mdhd.start + 20);
  const duration = dv.getUint32(mdhd.start + 24);
  const avc1 = need([...stbl, 'stsd', 'avc1']);
  const width = dv.getUint16(avc1.start + 32);
  const height = dv.getUint16(avc1.start + 34);
  const avcCBox = need([...stbl, 'stsd', 'avc1', 'avcC']);
  const avcC = data.slice(avcCBox.start + 8, avcCBox.start + avcCBox.size);
  const hex = (v: number): string => v.toString(16).padStart(2, '0');
  const codec = `avc1.${hex(avcC[1])}${hex(avcC[2])}${hex(avcC[3])}`;
  const table = (b: Mp4Box, width: number): number[][] => {
    const count = dv.getUint32(b.start + 12);
    const rows: number[][] = [];
    for (let i = 0; i < count; i++) {
      const r: number[] = [];
      for (let k = 0; k < width; k++) r.push(dv.getUint32(b.start + 16 + 4 * (width * i + k)));
      rows.push(r);
    }
    return rows;
  };
  const stts = table(need([...stbl, 'stts']), 2) as [number, number][];
  const stssBox = findBox(boxes, [...stbl, 'stss']);
  const stss = stssBox ? table(stssBox, 1).map((r) => r[0]) : [];
  const stsz = need([...stbl, 'stsz']);
  const fixedSize = dv.getUint32(stsz.start + 12);
  const n = dv.getUint32(stsz.start + 16);
  const sizes = Array.from({ length: n }, (_, i) => (fixedSize ? fixedSize : dv.getUint32(stsz.start + 20 + 4 * i)));
  const stsc = table(need([...stbl, 'stsc']), 3);
  const stco = table(need([...stbl, 'stco']), 1).map((r) => r[0]);
  const cttsBox = findBox(boxes, [...stbl, 'ctts']);
  const ctts = cttsBox ? table(cttsBox, 2) : [];
  const elst = findBox(boxes, ['moov', 'trak', 'edts', 'elst']);
  const mediaStart = elst ? dv.getUint32(elst.start + 20) : 0;
  // sample offsets: chunks per stsc run
  const offsets: number[] = [];
  for (let c = 0, s = 0; c < stco.length; c++) {
    let run = stsc[0];
    for (const r of stsc) if (r[0] <= c + 1) run = r;
    let o = stco[c];
    for (let k = 0; k < run[1] && s < n; k++, s++) {
      offsets.push(o);
      o += sizes[s];
    }
  }
  const dts: number[] = [];
  for (const [count, delta] of stts) for (let k = 0; k < count; k++) dts.push((dts.length ? dts[dts.length - 1] : -delta) + delta);
  const comp: number[] = [];
  for (const [count, off] of ctts) for (let k = 0; k < count; k++) comp.push(off);
  const keys = new Set(stss);
  const samples: Mp4Sample[] = sizes.map((size, i) => ({ offset: offsets[i], size, key: stssBox ? keys.has(i + 1) : true, dts: dts[i], pts: dts[i] + (comp[i] ?? 0) - mediaStart }));
  return { boxes, width, height, timescale, duration, avcC, codec, stts, stss, samples };
}
