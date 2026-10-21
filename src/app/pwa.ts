/**
 * Service worker registration (production builds only, never on localhost) and persistent storage
 * at launch. See public/sw.js for the caching strategy.
 *
 * Deployment: the service worker's caches are kept per scope, but IndexedDB (scans, photos, GPS
 * fixes) is per ORIGIN — on a shared origin (`<user>.github.io/<repo>/`) every other project there
 * can read it. Deploy on a dedicated origin for real use.
 */

/**
 * Script URL of the service worker: `sw.js?v=<entry chunk file name>`. Vite content-hashes the
 * entry chunk (`assets/main-<hash>.js`), and that name changes with every code change (it names
 * the other chunks), so each deploy registers a new worker version with its own cache — stale
 * versions are pruned by the worker. Without a usable entry URL the version is 'app' (one cache).
 */
export function serviceWorkerUrl(baseURI: string, entryUrl: string | undefined): URL {
  let v = '';
  try {
    if (entryUrl) v = (new URL(entryUrl).pathname.split('/').pop() ?? '').replace(/\.[cm]?js$/, '');
  } catch {
    v = '';
  }
  v = v.replace(/[^\w.-]/g, '').slice(0, 64) || 'app';
  const url = new URL('sw.js', baseURI);
  url.searchParams.set('v', v);
  return url;
}

function isLocalHost(host: string): boolean {
  return host === 'localhost' || host === '[::1]' || host.endsWith('.localhost') || /^127\./.test(host);
}

export function registerServiceWorker(entryUrl: string): void {
  if (!import.meta.env.PROD || !('serviceWorker' in navigator)) return;
  if (isLocalHost(location.hostname)) return;
  const register = (): void => {
    navigator.serviceWorker.register(serviceWorkerUrl(document.baseURI, entryUrl), { scope: './' }).catch((e) => console.warn('[sw] registration failed', e));
  };
  // boot() may run after 'load' (it awaits the MockEngine chunk with ?mock=1): never miss it.
  if (document.readyState === 'complete') register();
  else window.addEventListener('load', register, { once: true });
}

/**
 * Whether `persist()` may be called without a user gesture: true unless the Permissions API
 * reports 'prompt' (Firefox then shows a permission prompt; Chromium reports 'prompt' for a
 * not-yet-granted origin too — the Start tap asks there). Safari decides silently and does not
 * know the 'persistent-storage' permission name (the query throws): allowed.
 */
export async function persistWithoutPrompt(permissions: Pick<Permissions, 'query'> | undefined): Promise<boolean> {
  if (!permissions || typeof permissions.query !== 'function') return true;
  try {
    const st = await permissions.query({ name: 'persistent-storage' as PermissionName });
    return st.state !== 'prompt';
  } catch {
    return true;
  }
}

/**
 * In a Home Screen app `persist()` is granted (ios-platform.md §2.2) — ask on every launch, but
 * never in a way that shows a permission prompt without a user action (the Start tap asks again).
 */
export function requestPersistenceIfInstalled(): void {
  const nav = navigator as Navigator & { standalone?: boolean };
  const standalone = nav.standalone === true || matchMedia('(display-mode: standalone)').matches;
  if (!standalone || !nav.storage?.persist) return;
  void persistWithoutPrompt(nav.permissions)
    .then((ok) => (ok ? nav.storage.persist() : false))
    .catch(() => false);
}
