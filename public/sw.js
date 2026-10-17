/*
 * Service worker: offline support for the single-page app. Same-origin only — nothing is ever
 * requested from (or cached for) a third party.
 *
 * Versioning: the page registers `sw.js?v=<entry chunk file name>` (src/main.ts). Vite puts a
 * content hash into that name, and it changes whenever any JS chunk changes (the entry references
 * the others by hashed name), so a deploy installs a new worker with its own cache. Cache names
 * carry the scope path: on a shared origin (GitHub Pages user sites host every project under one
 * origin) other apps' caches are never read or deleted. On activate, all caches of this scope
 * except the current and the previous version are deleted (storage is shared with the scans); the
 * previous one stays so a page still running the old version can load its lazy chunks.
 *
 * Install: precache the app shell (index.html) and every hashed asset reachable from it
 * (HTML → entry chunk / CSS / icons → lazy chunks and worker scripts named in the JS), so the
 * first offline launch works even if no scan was made online (the engine worker is otherwise only
 * fetched at the first scan). Best effort: a file that fails is fetched (and cached) on use.
 *
 * Fetch:
 *  - navigations: network first (a deploy lands at once). Only a successful (200, same-origin,
 *    not redirected) response for the app shell is stored, under ONE key (the scope URL): error
 *    pages, redirects and other documents never end up in the cache, and query strings do not
 *    grow it. Offline: the cached shell.
 *  - assets/ (content-hashed, immutable) and icons/: cache first; cached on first use (200 only).
 *  - anything else in scope (manifest…): network first, the cache as offline fallback.
 */
const SCOPE = new URL(self.registration && self.registration.scope ? self.registration.scope : './', self.location.href);
const VERSION = (new URL(self.location.href).searchParams.get('v') || 'dev').replace(/[^\w.-]/g, '').slice(0, 64) || 'dev';
const PREFIX = `sfm:${SCOPE.pathname}:`;
/** Cache name of the first service worker version (one cache for every deploy, never pruned). */
const LEGACY_CACHE = 'sfm-v1';
const CACHE = PREFIX + VERSION;
const SHELL_URL = SCOPE.href;
const ASSETS_PATH = `${SCOPE.pathname}assets/`;
const ICONS_PATH = `${SCOPE.pathname}icons/`;
const MAX_PRECACHE = 80;

/** Hashed chunk names inside JS: "./x-Hash1234.js", `x-Hash1234.js`, "assets/x-Hash1234.js". */
const JS_REF = /["'`](?:\.\/|assets\/|\.\.\/assets\/)?([\w.-]+-[\w-]{6,}\.(?:js|css|wasm))["'`]/g;
/** src / href attributes of the shell HTML. */
const HTML_REF = /\s(?:src|href)\s*=\s*["']([^"'#?]+)[^"']*["']/g;

/** waitUntil, tolerating engines that refuse it once the event has settled. */
function keepAlive(event, promise) {
  try {
    event.waitUntil(promise);
  } catch {
    /* the work still runs; only the lifetime extension is lost */
  }
}

function isCacheable(res) {
  return !!res && res.status === 200 && res.type === 'basic' && !res.redirected;
}

function inScope(url) {
  return url.origin === SCOPE.origin && url.pathname.startsWith(SCOPE.pathname);
}

function isShell(url) {
  return url.origin === SCOPE.origin && (url.pathname === SCOPE.pathname || url.pathname === `${SCOPE.pathname}index.html`);
}

function isImmutable(url) {
  return (url.pathname.startsWith(ASSETS_PATH) && !url.pathname.endsWith('.map')) || url.pathname.startsWith(ICONS_PATH);
}

/** Shell + everything reachable from it (breadth first, bounded). Never throws. */
async function precache() {
  const cache = await caches.open(CACHE);
  const shell = await fetch(SHELL_URL, { cache: 'no-cache', credentials: 'same-origin' });
  if (!isCacheable(shell)) return;
  const html = await shell.clone().text();
  await cache.put(SHELL_URL, shell);
  const seen = new Set([SHELL_URL]);
  let level = [];
  const add = (list, ref, base) => {
    let u;
    try {
      u = new URL(ref, base);
    } catch {
      return;
    }
    u.hash = '';
    u.search = '';
    if (!inScope(u) || u.pathname.endsWith('.map') || seen.has(u.href) || seen.size >= MAX_PRECACHE) return;
    seen.add(u.href);
    list.push(u.href);
  };
  for (const m of html.matchAll(HTML_REF)) add(level, m[1], SHELL_URL);
  while (level.length) {
    const next = [];
    await Promise.all(
      level.map(async (url) => {
        try {
          const res = await fetch(url, { credentials: 'same-origin' });
          if (!isCacheable(res)) return;
          if (/\.m?js$/.test(new URL(url).pathname)) {
            const text = await res.clone().text();
            // chunk references are relative to the referencing chunk (all live in assets/)
            for (const m of text.matchAll(JS_REF)) add(next, m[1], url);
          }
          await cache.put(url, res);
        } catch {
          /* best effort: fetched on use */
        }
      }),
    );
    level = next;
  }
}

self.addEventListener('install', (event) => {
  event.waitUntil(precache().catch(() => undefined));
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => {
        // CacheStorage keys are in creation order: keep the newest other version of this scope.
        const old = keys.filter((k) => k !== CACHE && (k.startsWith(PREFIX) || k === LEGACY_CACHE));
        return Promise.all(old.slice(0, -1).map((k) => caches.delete(k)));
      })
      .then(() => self.clients.claim()),
  );
});

async function cachedShell() {
  const own = await (await caches.open(CACHE)).match(SHELL_URL);
  if (own) return own;
  // previous version (or the legacy cache) — never another scope's caches
  for (const k of (await caches.keys()).reverse()) {
    if (!k.startsWith(PREFIX) && k !== LEGACY_CACHE) continue;
    const r = await (await caches.open(k)).match(SHELL_URL);
    if (r) return r;
  }
  return undefined;
}

async function navigate(event, url) {
  try {
    const res = await fetch(event.request);
    if (isShell(url) && isCacheable(res)) {
      const copy = res.clone();
      keepAlive(
        event,
        caches
          .open(CACHE)
          .then((c) => c.put(SHELL_URL, copy))
          .catch(() => undefined),
      );
    }
    return res;
  } catch (e) {
    const shell = await cachedShell();
    if (shell) return shell;
    throw e;
  }
}

async function matchOwn(req) {
  for (const k of (await caches.keys()).reverse()) {
    if (!k.startsWith(PREFIX) && k !== LEGACY_CACHE) continue;
    const r = await (await caches.open(k)).match(req);
    if (r) return r;
  }
  return undefined;
}

async function cacheFirst(event) {
  const req = event.request;
  const hit = await matchOwn(req);
  if (hit) return hit;
  const res = await fetch(req);
  if (isCacheable(res)) {
    const copy = res.clone();
    keepAlive(
      event,
      caches
        .open(CACHE)
        .then((c) => c.put(req, copy))
        .catch(() => undefined),
    );
  }
  return res;
}

async function networkFirst(event) {
  try {
    return await fetch(event.request);
  } catch (e) {
    const hit = await matchOwn(event.request);
    if (hit) return hit;
    throw e;
  }
}

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (!inScope(url)) return; // other origins / other apps on this origin: browser default
  if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
  if (req.mode === 'navigate') {
    event.respondWith(navigate(event, url));
    return;
  }
  if (req.headers.has('range')) return; // partial content is never cached
  event.respondWith(isImmutable(url) ? cacheFirst(event) : networkFirst(event));
});
