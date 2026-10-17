import { describe, expect, it } from 'vitest';
import { persistWithoutPrompt, serviceWorkerUrl } from './pwa';

describe('serviceWorkerUrl', () => {
  it('versions the worker by the content-hashed entry chunk name, relative to the base URL', () => {
    const u = serviceWorkerUrl('https://user.github.io/sfm/', 'https://user.github.io/sfm/assets/main-tEZoFa8O.js');
    expect(u.href).toBe('https://user.github.io/sfm/sw.js?v=main-tEZoFa8O');
    expect(serviceWorkerUrl('https://h.example/app/index.html?mock=1', 'https://h.example/app/assets/main-x_Y-1.mjs').href).toBe('https://h.example/app/sw.js?v=main-x_Y-1');
  });

  it('falls back to one fixed version for unusable entry URLs', () => {
    expect(serviceWorkerUrl('https://h.example/', undefined).href).toBe('https://h.example/sw.js?v=app');
    expect(serviceWorkerUrl('https://h.example/', 'not a url').href).toBe('https://h.example/sw.js?v=app');
    expect(serviceWorkerUrl('https://h.example/', 'https://h.example/assets/%3Cx%3E.js').searchParams.get('v')).toBe('3Cx3E');
  });
});

describe('persistWithoutPrompt', () => {
  const perms = (state: PermissionState | Error) => ({
    query: async () => {
      if (state instanceof Error) throw state;
      return { state } as PermissionStatus;
    },
  });

  it('never calls persist() at launch where it would show a prompt', async () => {
    expect(await persistWithoutPrompt(perms('prompt'))).toBe(false);
    expect(await persistWithoutPrompt(perms('granted'))).toBe(true);
    expect(await persistWithoutPrompt(perms('denied'))).toBe(true);
  });

  it('Safari (unknown permission name) and browsers without the Permissions API: silent, allowed', async () => {
    expect(await persistWithoutPrompt(perms(new TypeError('unknown permission')))).toBe(true);
    expect(await persistWithoutPrompt(undefined)).toBe(true);
  });
});
