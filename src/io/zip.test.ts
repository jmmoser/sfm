import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { crc32, createZip, createZipBlob, normalizeZipPath, readZip, ZipError, type ZipEntryInput } from './zip';

const dir = mkdtempSync(join(tmpdir(), 'sfm-zip-test-'));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

function randomBytes(n: number, seed: number): Uint8Array {
  const out = new Uint8Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    out[i] = s >>> 24;
  }
  return out;
}

// Python verifier: test CRCs, list entries (name, size, crc, flag bits, method, date_time), extract.
const PY = `
import json, sys, zipfile, os
path, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(path) as z:
    bad = z.testzip()
    infos = [dict(name=i.filename, size=i.file_size, csize=i.compress_size, crc=i.CRC,
                  flag=i.flag_bits, method=i.compress_type, dt=list(i.date_time)) for i in z.infolist()]
    z.extractall(out)
print(json.dumps(dict(bad=bad, infos=infos, comment=z.comment.decode('utf-8'))))
`;

function pythonInspect(zipBytes: Uint8Array, name: string): {
  bad: string | null;
  infos: { name: string; size: number; csize: number; crc: number; flag: number; method: number; dt: number[] }[];
  comment: string;
} {
  const zipPath = join(dir, `${name}.zip`);
  const outDir = join(dir, `${name}-out`);
  writeFileSync(zipPath, zipBytes);
  const stdout = execFileSync('python3', ['-I', '-c', PY, zipPath, outDir], { encoding: 'utf-8' });
  return JSON.parse(stdout);
}

describe('crc32', () => {
  it('matches the standard check value and supports chaining', () => {
    const s = new TextEncoder().encode('123456789');
    expect(crc32(s)).toBe(0xcbf43926);
    expect(crc32(new Uint8Array(0))).toBe(0);
    const data = randomBytes(10_007, 1);
    const whole = crc32(data);
    expect(crc32(data.subarray(4000), crc32(data.subarray(0, 4000)))).toBe(whole);
    // Tail lengths 0..7 exercise the byte-wise loop after slicing-by-8.
    for (let n = 0; n < 17; n++) {
      let c = 0xffffffff;
      for (let i = 0; i < n; i++) {
        c ^= data[i];
        for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      }
      expect(crc32(data.subarray(0, n))).toBe((c ^ 0xffffffff) >>> 0);
    }
  });
});

describe('createZip', () => {
  const when = new Date(2024, 4, 17, 13, 45, 22);
  const files: ZipEntryInput[] = [
    { name: 'sparse/0/cameras.txt', data: '# Camera list\n1 RADIAL 640 480 500 320 240 0 0\n', lastModified: when },
    { name: 'images/frame_0001.jpg', data: randomBytes(300_001, 7), lastModified: when },
    { name: 'empty.bin', data: new Uint8Array(0), lastModified: when },
    { name: 'ünïcødé/点云.ply', data: randomBytes(1234, 9).buffer as ArrayBuffer, lastModified: when },
  ];

  it('produces an archive Python zipfile accepts, with correct CRCs, sizes, UTF-8 names and dates', () => {
    const zip = createZip(files, { comment: 'sfm export' });
    const res = pythonInspect(zip, 'basic');
    expect(res.bad).toBeNull();
    expect(res.comment).toBe('sfm export');
    expect(res.infos.map((i) => i.name)).toEqual(files.map((f) => f.name));
    for (let i = 0; i < files.length; i++) {
      const info = res.infos[i];
      const d = files[i].data;
      const bytes = typeof d === 'string' ? new TextEncoder().encode(d) : d instanceof Uint8Array ? d : new Uint8Array(d);
      expect(info.size).toBe(bytes.length);
      expect(info.csize).toBe(bytes.length);
      expect(info.method).toBe(0);
      expect(info.flag & 0x800).toBe(0x800);
      expect(info.crc).toBe(crc32(bytes));
      expect(info.dt).toEqual([2024, 5, 17, 13, 45, 22]);
      const extracted = readFileSync(join(dir, 'basic-out', files[i].name));
      expect(Buffer.compare(extracted, Buffer.from(bytes))).toBe(0);
    }
  });

  it('round-trips through readZip', () => {
    const zip = createZip(files);
    const entries = readZip(zip);
    expect(entries.map((e) => e.name)).toEqual(files.map((f) => f.name));
    expect(new TextDecoder().decode(entries[0].data)).toBe(files[0].data);
    expect(entries[1].data).toEqual(files[1].data as Uint8Array);
    expect(entries[2].data.length).toBe(0);
    expect(entries[1].lastModified.getTime()).toBe(when.getTime());
  });

  it('detects corruption in readZip', () => {
    const zip = createZip(files);
    const bad = zip.slice();
    bad[30 + files[0].name.length + 3] ^= 0xff; // flip a data byte of entry 0
    expect(() => readZip(bad)).toThrow(/CRC mismatch/);
    expect(() => readZip(new Uint8Array(10))).toThrow(ZipError);
  });

  it('normalises and validates names', () => {
    expect(normalizeZipPath('a\\b\\c.txt')).toBe('a/b/c.txt');
    expect(normalizeZipPath('./x.txt')).toBe('x.txt');
    expect(() => normalizeZipPath('../x')).toThrow(/traversal/);
    expect(() => normalizeZipPath('/etc/passwd')).toThrow(/Absolute/);
    expect(() => normalizeZipPath('')).toThrow();
    expect(() => createZip([{ name: 'a', data: 'x' }, { name: './a', data: 'y' }])).toThrow(/Duplicate/);
  });

  it('clamps pre-1980 dates', () => {
    const zip = createZip([{ name: 'old.txt', data: 'x', lastModified: 0 }]);
    const res = pythonInspect(zip, 'old');
    expect(res.infos[0].dt).toEqual([1980, 1, 1, 0, 0, 0]);
  });

  it('createZipBlob is byte-identical to createZip and streams Blob inputs', async () => {
    const blobFiles = files.map((f, i) => (i === 1 ? { ...f, data: new Blob([f.data as Uint8Array<ArrayBuffer>]) } : f));
    const blob = await createZipBlob(blobFiles, { comment: 'c' });
    expect(blob.type).toBe('application/zip');
    const bytes = new Uint8Array(await blob.arrayBuffer());
    expect(bytes).toEqual(createZip(files, { comment: 'c' }));
    const res = pythonInspect(bytes, 'blob');
    expect(res.bad).toBeNull();
    expect(res.infos.length).toBe(files.length);
  });

  it('handles many entries', () => {
    const many = Array.from({ length: 2000 }, (_, i) => ({ name: `f/${i}.txt`, data: `file ${i}` }));
    const zip = createZip(many);
    const res = pythonInspect(zip, 'many');
    expect(res.bad).toBeNull();
    expect(res.infos.length).toBe(2000);
    expect(readZip(zip)[1999].name).toBe('f/1999.txt');
  });
});
