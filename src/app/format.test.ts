import { describe, expect, it } from 'vitest';
import {
  defaultSessionName,
  escapeHtml,
  exportFilename,
  formatAccuracy,
  formatBytes,
  formatClock,
  formatCount,
  formatDuration,
  formatEta,
  formatPercent,
  formatRelativeDate,
  guidanceInfo,
  plural,
  statusInfo,
} from './format';
import type { Guidance } from '../workers/protocol';

describe('formatBytes', () => {
  it('uses SI units with sensible precision', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(999)).toBe('999 B');
    expect(formatBytes(1000)).toBe('1.00 KB');
    expect(formatBytes(1234)).toBe('1.23 KB');
    expect(formatBytes(34_500_000)).toBe('34.5 MB');
    expect(formatBytes(512_000_000)).toBe('512 MB');
    expect(formatBytes(1_250_000_000)).toBe('1.25 GB');
  });
  it('rolls over at unit boundaries after rounding', () => {
    expect(formatBytes(999_960)).toBe('1.00 MB');
    expect(formatBytes(999_400)).toBe('999 KB');
  });
  it('handles invalid input', () => {
    expect(formatBytes(NaN)).toBe('–');
    expect(formatBytes(-1)).toBe('–');
    expect(formatBytes(null)).toBe('–');
  });
});

describe('durations', () => {
  it('formatClock', () => {
    expect(formatClock(0)).toBe('0:00');
    expect(formatClock(5_400)).toBe('0:05');
    expect(formatClock(83_000)).toBe('1:23');
    expect(formatClock(3_723_000)).toBe('1:02:03');
    expect(formatClock(-5)).toBe('0:00');
  });
  it('formatDuration', () => {
    expect(formatDuration(420)).toBe('0.4 s');
    expect(formatDuration(12_000)).toBe('12 s');
    expect(formatDuration(185_000)).toBe('3 min 5 s');
    expect(formatDuration(180_000)).toBe('3 min');
    expect(formatDuration(3_840_000)).toBe('1 h 4 min');
    expect(formatDuration(NaN)).toBe('–');
  });
  it('formatEta', () => {
    expect(formatEta(null)).toBe('Estimating time…');
    expect(formatEta(2000)).toBe('Almost done');
    expect(formatEta(40_000)).toBe('Less than a minute left');
    expect(formatEta(5 * 60_000)).toBe('About 5 min left');
    expect(formatEta(125 * 60_000)).toBe('About 2 h 5 min left');
  });
});

describe('counts and numbers', () => {
  it('formatCount', () => {
    expect(formatCount(950)).toBe('950');
    expect(formatCount(3400)).toBe('3.4k');
    expect(formatCount(3000)).toBe('3k');
    expect(formatCount(12_345)).toBe('12k');
    expect(formatCount(1_250_000)).toBe('1.25M');
    expect(formatCount(42_000_000)).toBe('42M');
    expect(formatCount(undefined)).toBe('–');
    // the unit follows the rounded value: no '1000k'
    expect(formatCount(999_499)).toBe('999k');
    expect(formatCount(999_500)).toBe('1M');
    expect(formatCount(999_999)).toBe('1M');
    expect(formatCount(-999_999)).toBe('-1M');
  });
  it('plural', () => {
    expect(plural(1, 'photo')).toBe('1 photo');
    expect(plural(0, 'photo')).toBe('0 photos');
    expect(plural(1200, 'point')).toBe('1,200 points');
  });
  it('formatAccuracy', () => {
    expect(formatAccuracy(0.5)).toBe('±0.5 m');
    expect(formatAccuracy(4.2)).toBe('±4 m');
    expect(formatAccuracy(35)).toBe('±35 m');
    expect(formatAccuracy(1200)).toBe('±1.2 km');
    expect(formatAccuracy(null)).toBe('–');
  });
  it('formatPercent clamps', () => {
    expect(formatPercent(0.425)).toBe('43%');
    expect(formatPercent(1.5)).toBe('100%');
    expect(formatPercent(-1)).toBe('0%');
  });
});

describe('formatRelativeDate (local time)', () => {
  const now = new Date(2026, 9, 16, 14, 30).getTime();
  it('recent', () => {
    expect(formatRelativeDate(now - 20_000, now)).toBe('Just now');
    expect(formatRelativeDate(now - 5 * 60_000, now)).toBe('5 min ago');
  });
  it('today / yesterday / older', () => {
    expect(formatRelativeDate(new Date(2026, 9, 16, 9, 5).getTime(), now)).toBe('Today, 09:05');
    expect(formatRelativeDate(new Date(2026, 9, 15, 22, 0).getTime(), now)).toBe('Yesterday, 22:00');
    expect(formatRelativeDate(new Date(2026, 9, 2, 8, 0).getTime(), now)).toBe('Oct 2');
    expect(formatRelativeDate(new Date(2025, 11, 31, 8, 0).getTime(), now)).toBe('Dec 31, 2025');
  });
});

describe('labels', () => {
  it('session names and file names', () => {
    const t = new Date(2026, 9, 16, 14, 5).getTime();
    expect(defaultSessionName('object', t)).toBe('Object · Oct 16, 14:05');
    expect(defaultSessionName('scene', t)).toBe('Area · Oct 16, 14:05');
    expect(exportFilename('Object · Oct 16, 14:05', 'ply', 'points')).toBe('Object_Oct_16_14_05_points.ply');
    expect(exportFilename('Café / Garten <1>', '.glb')).toBe('Cafe_Garten_1.glb');
    expect(exportFilename('***', 'zip')).toBe('scan.zip');
  });
  it('export file names keep non-Latin scripts (were all replaced by "scan")', () => {
    expect(exportFilename('花瓶', 'ply')).toBe('花瓶.ply');
    expect(exportFilename('Кухня 2', 'glb')).toBe('Кухня_2.glb');
    expect(exportFilename('がっこう', 'zip')).toBe('がっこう.zip'); // dakuten survive (NFC)
    expect(exportFilename('한국어 스캔', 'ply', 'mesh')).toBe('한국어_스캔_mesh.ply');
    expect(exportFilename('Küche 🍰 #1', 'glb')).toBe('Kuche_1.glb');
    expect(exportFilename('🍰🍰', 'glb')).toBe('scan.glb');
    expect(Array.from(exportFilename('长'.repeat(100), 'ply').replace(/\.ply$/, '')).length).toBe(60);
  });
  it('status chips', () => {
    expect(statusInfo('meshed')).toEqual({ label: 'Mesh', tone: 'good' });
    expect(statusInfo('capturing').tone).toBe('warn');
  });
  it('every guidance code has friendly copy', () => {
    const codes: Guidance[] = ['good', 'move_slower', 'more_texture', 'translate_more', 'tracking_lost', 'initializing'];
    for (const c of codes) {
      for (const mode of ['object', 'scene', 'auto'] as const) {
        const g = guidanceInfo(c, mode);
        expect(g.title.length).toBeGreaterThan(3);
        expect(g.detail.length).toBeGreaterThan(10);
      }
    }
    expect(guidanceInfo('tracking_lost').tone).toBe('bad');
    expect(guidanceInfo('translate_more', 'object').title).not.toBe(guidanceInfo('translate_more', 'scene').title);
  });
  it('escapeHtml', () => {
    expect(escapeHtml(`<a href="x">'&'</a>`)).toBe('&lt;a href=&quot;x&quot;&gt;&#39;&amp;&#39;&lt;/a&gt;');
  });
});
