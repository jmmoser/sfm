import { describe, expect, it } from 'vitest';
import { normalizeSessionMeta } from './storage';

describe('normalizeSessionMeta (records of older versions / damaged records)', () => {
  it('keeps a valid record as it is', () => {
    const thumb = new Blob(['x'], { type: 'image/jpeg' });
    const rec = { id: 'a', name: 'Vase', createdAt: 1, updatedAt: 2, mode: 'scene', status: 'dense', numFrames: 42, stats: { quality: 'max', keyframes: 40 }, thumbnail: thumb };
    expect(normalizeSessionMeta(rec)).toEqual(rec);
    expect(normalizeSessionMeta(rec)!.thumbnail).toBe(thumb);
  });

  it('coerces every field to its declared type', () => {
    const m = normalizeSessionMeta({ id: 'b', name: 12, createdAt: 'yesterday', updatedAt: 5, mode: 'drone', status: 'textured', numFrames: '7', stats: { quality: 'fast', bad: { x: 1 }, nan: NaN, ok: 3 }, thumbnail: { $blobBuffer: 1 } })!;
    expect(m).toEqual({ id: 'b', name: '12', createdAt: 5, updatedAt: 5, mode: 'auto', status: 'capturing', numFrames: 0, stats: { quality: 'fast', ok: 3 } });
    const empty = normalizeSessionMeta({ id: 'c' })!;
    expect(empty).toEqual({ id: 'c', name: 'Untitled scan', createdAt: 0, updatedAt: 0, mode: 'auto', status: 'capturing', numFrames: 0 });
    expect(normalizeSessionMeta({ id: 'd', numFrames: 7.9, stats: [1, 2] })).toMatchObject({ numFrames: 7 });
    expect(normalizeSessionMeta({ id: 'd', stats: [1, 2] })!.stats).toBeUndefined();
    expect(normalizeSessionMeta({ id: 'e', numFrames: -3, name: '   ' })).toMatchObject({ numFrames: 0, name: 'Untitled scan' });
  });

  it('a numeric key stays addressable; non-records are dropped', () => {
    expect(normalizeSessionMeta({ id: 5, name: 'n' })!.id).toBe(5);
    for (const bad of [null, undefined, 'x', 3, [], [{ id: 'a' }], { id: {} }, { id: NaN }, { name: 'no id' }]) expect(normalizeSessionMeta(bad)).toBeNull();
  });

  it('the UI helpers work on every normalised record (no throw on damaged data)', async () => {
    const { exportFilename, formatRelativeDate, plural, statusInfo, modeLabel } = await import('../app/format');
    const { photoCount } = await import('../app/screens/home');
    const { routeForSession } = await import('../app/router');
    for (const raw of [{ id: 'x' }, { id: 'y', name: null, createdAt: Infinity, numFrames: Infinity, stats: { keyframes: 'many' } }]) {
      const m = normalizeSessionMeta(raw)!;
      expect(() => [exportFilename(m.name, 'ply'), formatRelativeDate(m.createdAt), plural(photoCount(m), 'photo'), statusInfo(m.status), modeLabel(m.mode), routeForSession(m.id, m.status, photoCount(m))]).not.toThrow();
      expect(plural(photoCount(m), 'photo')).toBe('0 photos');
    }
  });
});
