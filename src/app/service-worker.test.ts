/**
 * public/sw.js, run in a simulated ServiceWorkerGlobalScope (fake CacheStorage / fetch): precache
 * of the app shell and every chunk reachable from it, versioned + scope-prefixed caches, pruning,
 * navigation caching (only a 200 app shell, one key, never error pages / redirects), offline
 * fallbacks, cache-first assets, and requests the worker must leave alone.
 */
import { describe, expect, it } from 'vitest';
import swSource from '../../public/sw.js?raw';

const ORIGIN = 'https://user.example.io';
const SCOPE = `${ORIGIN}/sfm/`;

class FakeResponse {
  constructor(
    readonly body: string,
    readonly status = 200,
    readonly type: 'basic' | 'opaqueredirect' | 'cors' | 'opaque' = 'basic',
    readonly redirected = false,
  ) {}
  get ok(): boolean {
    return this.status >= 200 && this.status < 300;
  }
  clone(): FakeResponse {
    return new FakeResponse(this.body, this.status, this.type, this.redirected);
  }
  async text(): Promise<string> {
    return this.body;
  }
}

type ReqLike = string | { url: string };
const keyOf = (r: ReqLike): string => (typeof r === 'string' ? new URL(r, SCOPE).href : r.url);

class FakeCache {
  readonly map = new Map<string, FakeResponse>();
  async match(req: ReqLike): Promise<FakeResponse | undefined> {
    return this.map.get(keyOf(req))?.clone();
  }
  async put(req: ReqLike, res: FakeResponse): Promise<void> {
    this.map.set(keyOf(req), res.clone());
  }
}

class FakeCacheStorage {
  readonly stores = new Map<string, FakeCache>();
  async open(name: string): Promise<FakeCache> {
    let c = this.stores.get(name);
    if (!c) this.stores.set(name, (c = new FakeCache()));
    return c;
  }
  async keys(): Promise<string[]> {
    return [...this.stores.keys()];
  }
  async delete(name: string): Promise<boolean> {
    return this.stores.delete(name);
  }
  async match(req: ReqLike): Promise<FakeResponse | undefined> {
    for (const c of this.stores.values()) {
      const r = await c.match(req);
      if (r) return r;
    }
    return undefined;
  }
}

/** The deployed site: path → response (anything else 404). */
type Site = Record<string, FakeResponse | (() => FakeResponse)>;

const SHELL_HTML = `<!doctype html><html><head>
<link rel="manifest" href="./manifest.webmanifest" />
<link rel="icon" type="image/png" sizes="192x192" href="./icons/icon-192.png" />
<script type="module" crossorigin src="./assets/main-AAAA1111.js"></script>
<link rel="modulepreload" crossorigin href="./assets/stages-BBBB2222.js">
<link rel="stylesheet" crossorigin href="./assets/main-CCCC3333.css">
<link rel="help" href="https://third.party.example/x.js">
<link rel="alternate" href="/other-app/index.html">
</head><body></body></html>`;

function deployedSite(): Site {
  return {
    '/sfm/': new FakeResponse(SHELL_HTML),
    '/sfm/manifest.webmanifest': new FakeResponse('{}'),
    '/sfm/icons/icon-192.png': new FakeResponse('png'),
    '/sfm/assets/main-AAAA1111.js': new FakeResponse(
      'import{a}from"./stages-BBBB2222.js";const w=new URL(`engine.worker-DDDD4444.js`,import.meta.url);' +
        'const v=()=>import(`./viewer-EEEE5555.js`);const m=["assets/mock-engine-FFFF6666.js"];' +
        '//# sourceMappingURL=main-AAAA1111.js.map\nconst gone="./missing-GGGG7777.js";',
    ),
    '/sfm/assets/stages-BBBB2222.js': new FakeResponse('export const a=1;'),
    '/sfm/assets/main-CCCC3333.css': new FakeResponse('body{}'),
    '/sfm/assets/engine.worker-DDDD4444.js': new FakeResponse('import"./extractor-HHHH8888.js";'),
    '/sfm/assets/extractor-HHHH8888.js': new FakeResponse('export{}'),
    '/sfm/assets/viewer-EEEE5555.js': new FakeResponse('import"./main-AAAA1111.js";'),
    '/sfm/assets/mock-engine-FFFF6666.js': new FakeResponse('export{}'),
    '/sfm/assets/main-AAAA1111.js.map': new FakeResponse('{"version":3}'),
  };
}

interface FakeEvent {
  request?: { method: string; url: string; mode: string; cache: string; headers: Headers };
  responded: Promise<FakeResponse> | null;
  lifetime: Promise<unknown>[];
  respondWith(p: Promise<FakeResponse>): void;
  waitUntil(p: Promise<unknown>): void;
}

function makeEvent(request?: FakeEvent['request']): FakeEvent {
  return {
    request,
    responded: null,
    lifetime: [],
    respondWith(p) {
      this.responded = p;
    },
    waitUntil(p) {
      this.lifetime.push(p);
    },
  };
}

function boot(opts: { version?: string; caches?: FakeCacheStorage; site?: Site } = {}) {
  const listeners = new Map<string, (e: FakeEvent) => void>();
  const caches = opts.caches ?? new FakeCacheStorage();
  let site: Site | null = opts.site ?? deployedSite();
  const requested: string[] = [];
  const fetchImpl = async (input: ReqLike): Promise<FakeResponse> => {
    const url = new URL(keyOf(input));
    requested.push(url.href);
    if (!site) throw new TypeError('Failed to fetch (offline)');
    if (url.origin !== ORIGIN) throw new Error(`unexpected third-party request ${url.href}`);
    const r = site[url.pathname];
    return r === undefined ? new FakeResponse('not found', 404) : typeof r === 'function' ? r() : r.clone();
  };
  let claimed = false;
  const self = {
    location: { href: `${ORIGIN}/sfm/sw.js${opts.version === undefined ? '?v=main-AAAA1111' : opts.version}` },
    registration: { scope: SCOPE },
    addEventListener: (type: string, fn: (e: FakeEvent) => void) => listeners.set(type, fn),
    skipWaiting: () => undefined,
    clients: {
      claim: async () => {
        claimed = true;
      },
    },
  };
  new Function('self', 'caches', 'fetch', swSource)(self, caches, fetchImpl);
  const dispatch = async (type: string, e: FakeEvent): Promise<FakeEvent> => {
    listeners.get(type)!(e);
    await Promise.all(e.lifetime);
    return e;
  };
  const request = (path: string, init: { mode?: string; method?: string; headers?: Record<string, string>; cache?: string } = {}) =>
    makeEvent({ method: init.method ?? 'GET', url: new URL(path, SCOPE).href, mode: init.mode ?? 'cors', cache: init.cache ?? 'default', headers: new Headers(init.headers) });
  /** Dispatch a fetch event; the response (or null when the worker did not respond). */
  const fetchEvent = async (path: string, init?: Parameters<typeof request>[1]): Promise<FakeResponse | null> => {
    const e = request(path, init);
    listeners.get('fetch')!(e);
    if (!e.responded) return null;
    const res = await e.responded;
    await Promise.all(e.lifetime);
    return res;
  };
  return {
    caches,
    requested,
    install: () => dispatch('install', makeEvent()),
    activate: () => dispatch('activate', makeEvent()),
    fetchEvent,
    setSite: (s: Site | null) => {
      site = s;
    },
    claimed: () => claimed,
  };
}

const keysOf = (c: FakeCache | undefined): string[] => [...(c?.map.keys() ?? [])].map((k) => k.replace(ORIGIN, '')).sort();

describe('service worker (public/sw.js)', () => {
  it('precaches the shell and every chunk reachable from it, into a scope- and version-named cache', async () => {
    const sw = boot();
    await sw.install();
    expect(await sw.caches.keys()).toEqual(['sfm:/sfm/:main-AAAA1111']);
    expect(keysOf(sw.caches.stores.get('sfm:/sfm/:main-AAAA1111'))).toEqual([
      '/sfm/',
      '/sfm/assets/engine.worker-DDDD4444.js',
      '/sfm/assets/extractor-HHHH8888.js',
      '/sfm/assets/main-AAAA1111.js',
      '/sfm/assets/main-CCCC3333.css',
      '/sfm/assets/mock-engine-FFFF6666.js',
      '/sfm/assets/stages-BBBB2222.js',
      '/sfm/assets/viewer-EEEE5555.js',
      '/sfm/icons/icon-192.png',
      '/sfm/manifest.webmanifest',
    ]);
    // never a third party, another app on the origin, or source maps; a missing file is skipped
    expect(sw.requested.every((u) => u.startsWith(SCOPE))).toBe(true);
    expect(sw.requested.some((u) => u.endsWith('.map'))).toBe(false);
    expect(sw.requested).toContain(`${SCOPE}assets/missing-GGGG7777.js`);
  });

  it('a failed precache (offline install) does not fail the install', async () => {
    const sw = boot();
    sw.setSite(null);
    await expect(sw.install()).resolves.toBeDefined();
  });

  it('activate keeps the current and the previous version of its scope, and never touches other caches', async () => {
    const caches = new FakeCacheStorage();
    for (const n of ['other-app', 'sfm-v1', 'sfm:/sfm/:main-old1', 'sfm:/sfm-staging/:main-x', 'sfm:/sfm/:main-old2']) await caches.open(n);
    const sw = boot({ caches });
    await sw.install();
    await sw.activate();
    expect(await caches.keys()).toEqual(['other-app', 'sfm:/sfm-staging/:main-x', 'sfm:/sfm/:main-old2', 'sfm:/sfm/:main-AAAA1111']);
    expect(sw.claimed()).toBe(true);
    // first update from the original worker: its single cache is the "previous" version, kept once
    const legacy = new FakeCacheStorage();
    await legacy.open('sfm-v1');
    const sw2 = boot({ caches: legacy });
    await sw2.install();
    await sw2.activate();
    expect(await legacy.keys()).toEqual(['sfm-v1', 'sfm:/sfm/:main-AAAA1111']);
  });

  it('a version parameter is sanitised; without one the cache is "dev"', async () => {
    const a = boot({ version: '?v=main-AB%3Cscript%3E' });
    await a.install();
    expect(await a.caches.keys()).toEqual(['sfm:/sfm/:main-ABscript']);
    const b = boot({ version: '' });
    await b.install();
    expect(await b.caches.keys()).toEqual(['sfm:/sfm/:dev']);
  });

  it('navigations: network first; only a 200 app shell is stored, under one key', async () => {
    const sw = boot();
    const site = deployedSite();
    site['/sfm/'] = new FakeResponse('<html>v2</html>');
    sw.setSite(site);
    const r = await sw.fetchEvent('./?mock=1&mockFail=refine:Call%20support', { mode: 'navigate' });
    expect(await r!.text()).toBe('<html>v2</html>');
    const cache = sw.caches.stores.get('sfm:/sfm/:main-AAAA1111');
    expect(keysOf(cache)).toEqual(['/sfm/']); // not one entry per query string
    expect(await (await cache!.match(SCOPE))!.text()).toBe('<html>v2</html>');
    // index.html is the shell too
    await sw.fetchEvent('./index.html', { mode: 'navigate' });
    expect(keysOf(cache)).toEqual(['/sfm/']);
  });

  it('error pages, redirects and other documents are never cached (no cache poisoning)', async () => {
    const sw = boot();
    await sw.install();
    const cache = sw.caches.stores.get('sfm:/sfm/:main-AAAA1111')!;
    const site = deployedSite();
    site['/sfm/'] = new FakeResponse('<h1>502 Bad Gateway</h1>', 502);
    site['/sfm/redirect'] = new FakeResponse('', 0, 'opaqueredirect');
    site['/sfm/harness.html'] = new FakeResponse('<html>test page</html>');
    sw.setSite(site);
    expect((await sw.fetchEvent('./', { mode: 'navigate' }))!.status).toBe(502); // passed through…
    expect((await sw.fetchEvent('./redirect', { mode: 'navigate' }))!.type).toBe('opaqueredirect');
    expect((await sw.fetchEvent('./harness.html', { mode: 'navigate' }))!.status).toBe(200);
    expect(await (await cache.match(SCOPE))!.text()).toBe(SHELL_HTML); // …but the good shell stays
    expect(keysOf(cache).filter((k) => !k.startsWith('/sfm/assets/') && !k.startsWith('/sfm/icons/'))).toEqual(['/sfm/', '/sfm/manifest.webmanifest']);
    // a redirected 200 (served for a navigation, Safari rejects it) is not stored either
    site['/sfm/'] = new FakeResponse('<html>elsewhere</html>', 200, 'basic', true);
    await sw.fetchEvent('./', { mode: 'navigate' });
    expect(await (await cache.match(SCOPE))!.text()).toBe(SHELL_HTML);
  });

  it('offline: any navigation gets the cached shell (current version first); nothing cached → network error', async () => {
    const caches = new FakeCacheStorage();
    (await caches.open('sfm:/sfm/:main-old')).map.set(SCOPE, new FakeResponse('<html>old</html>'));
    const sw = boot({ caches });
    await sw.install();
    sw.setSite(null);
    expect(await (await sw.fetchEvent('./?screen=diagnostics', { mode: 'navigate' }))!.text()).toBe(SHELL_HTML);
    expect(await (await sw.fetchEvent('./anything', { mode: 'navigate' }))!.text()).toBe(SHELL_HTML);
    const empty = boot();
    empty.setSite(null);
    await expect(empty.fetchEvent('./', { mode: 'navigate' })).rejects.toThrow(/offline/);
  });

  it('assets: cache first (works offline after install); misses are cached only when 200', async () => {
    const sw = boot();
    await sw.install();
    sw.setSite(null);
    expect(await (await sw.fetchEvent('./assets/engine.worker-DDDD4444.js'))!.text()).toBe('import"./extractor-HHHH8888.js";');
    const site = deployedSite();
    site['/sfm/assets/bench.worker-IIII9999.js'] = new FakeResponse('bench');
    sw.setSite(site);
    expect((await sw.fetchEvent('./assets/bench.worker-IIII9999.js'))!.status).toBe(200);
    expect((await sw.fetchEvent('./assets/gone-JJJJ0000.js'))!.status).toBe(404);
    expect((await sw.fetchEvent('./assets/main-AAAA1111.js.map'))!.status).toBe(200);
    const keys = keysOf(sw.caches.stores.get('sfm:/sfm/:main-AAAA1111'));
    expect(keys).toContain('/sfm/assets/bench.worker-IIII9999.js');
    expect(keys).not.toContain('/sfm/assets/gone-JJJJ0000.js');
    expect(keys).not.toContain('/sfm/assets/main-AAAA1111.js.map');
    // an old page still running the previous version finds its chunks in the previous cache
    const prev = new FakeCacheStorage();
    (await prev.open('sfm:/sfm/:main-old')).map.set(`${SCOPE}assets/viewer-OLD00000.js`, new FakeResponse('old viewer'));
    const sw2 = boot({ caches: prev });
    await sw2.install();
    await sw2.activate();
    sw2.setSite(deployedSite()); // the old file is gone from the server
    expect(await (await sw2.fetchEvent('./assets/viewer-OLD00000.js'))!.text()).toBe('old viewer');
  });

  it('other files (manifest): network first, the cache only offline', async () => {
    const sw = boot();
    await sw.install();
    const site = deployedSite();
    site['/sfm/manifest.webmanifest'] = new FakeResponse('{"name":"v2"}');
    sw.setSite(site);
    expect(await (await sw.fetchEvent('./manifest.webmanifest'))!.text()).toBe('{"name":"v2"}');
    sw.setSite(null);
    expect(await (await sw.fetchEvent('./manifest.webmanifest'))!.text()).toBe('{}');
  });

  it('leaves non-GET, cross-origin, out-of-scope and range requests to the browser', async () => {
    const sw = boot();
    expect(await sw.fetchEvent('./assets/main-AAAA1111.js', { method: 'POST' })).toBeNull();
    expect(await sw.fetchEvent('https://third.party.example/font.woff2')).toBeNull();
    expect(await sw.fetchEvent('/other-app/assets/app-12345678.js')).toBeNull();
    expect(await sw.fetchEvent('/e2e/data/object/meta.json')).toBeNull();
    expect(await sw.fetchEvent('./assets/video-12345678.mp4', { headers: { range: 'bytes=0-' } })).toBeNull();
    expect(await sw.fetchEvent('./assets/main-AAAA1111.js', { cache: 'only-if-cached', mode: 'no-cors' })).toBeNull();
    expect(sw.requested).toEqual([]);
  });
});
