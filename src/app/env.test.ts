import { describe, expect, it } from 'vitest';
import { canScan, environmentIssues, type EnvInfo } from './env';

const ok: EnvInfo = {
  secureContext: true,
  apple: true,
  standalone: false,
  indexedDB: true,
  camera: true,
  webgpu: 'available',
  webgl: true,
  motion: true,
  share: true,
  lockdownSuspected: false,
};

describe('environmentIssues', () => {
  it('no issues on a capable device', () => {
    expect(environmentIssues(ok)).toEqual([]);
    expect(canScan(ok)).toBe(true);
  });

  it('WebGPU missing → CPU mode warning only', () => {
    for (const webgpu of ['unsupported', 'no-adapter'] as const) {
      const issues = environmentIssues({ ...ok, webgpu });
      expect(issues.map((i) => i.id)).toEqual(['webgpu']);
      expect(issues[0].severity).toBe('warning');
      expect(issues[0].title).toMatch(/CPU mode/);
    }
    expect(environmentIssues({ ...ok, webgpu: 'unknown' })).toEqual([]);
    expect(canScan({ ...ok, webgpu: 'unsupported' })).toBe(true);
  });

  it('missing camera or IndexedDB blocks scanning', () => {
    expect(canScan({ ...ok, camera: false })).toBe(false);
    expect(canScan({ ...ok, indexedDB: false })).toBe(false);
    expect(environmentIssues({ ...ok, indexedDB: false })[0]).toMatchObject({ id: 'storage', severity: 'blocking' });
  });

  it('insecure context explains HTTPS (and not a missing camera)', () => {
    const ids = environmentIssues({ ...ok, secureContext: false, camera: false }).map((i) => i.id);
    expect(ids).toContain('insecure');
    expect(ids).not.toContain('camera');
  });

  it('Lockdown Mode replaces the individual issues', () => {
    const issues = environmentIssues({ ...ok, lockdownSuspected: true, camera: false, indexedDB: false, webgpu: 'unsupported', webgl: false });
    expect(issues).toHaveLength(1);
    expect(issues[0].id).toBe('lockdown');
    expect(issues[0].body).toMatch(/Configure Web Browsing/);
  });

  it('no WebGL is a warning (capture still works)', () => {
    expect(environmentIssues({ ...ok, webgl: false })).toMatchObject([{ id: 'webgl', severity: 'warning' }]);
  });
});
