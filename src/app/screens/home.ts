/**
 * Home: hero + "New scan", saved sessions (thumbnail, name, date, status, frames) with
 * open / rename / delete, storage usage + "Install to Home Screen" hint, environment banners
 * (Lockdown Mode, HTTPS, camera / storage blocking, WebGPU → CPU mode), Diagnostics link.
 *
 * "Run interrupted" card (testId 'run-interrupted'): the run journal (src/app/run-journal.ts) found
 * a long run of a stored scan that never ended — iOS killed the app while it was open, or evicted
 * it in the background. It names the scan and the step; Resume opens the scan exactly like its list
 * item (OPEN_SESSION: Processing for a captured scan, else the viewer with its dense / photoreal
 * resume card), which acknowledges the card (the kill count stays: the retry runs in light mode).
 * Shown once the session list loaded; dropped when the scan no longer exists.
 */
import type { SessionMeta, StorageEstimateInfo } from '../../io/storage';
import type { App, Screen } from '../app';
import { banner, button, chip, iconButton, section, spinner } from '../components/controls';
import { confirmDialog, promptDialog } from '../components/dialog';
import { actionSheet } from '../components/sheet';
import { toast } from '../components/toast';
import { Disposer, h, replaceChildren } from '../dom';
import { canScan, environmentIssues, type EnvInfo } from '../env';
import { formatBytes, formatRelativeDate, modeLabel, plural, statusInfo } from '../format';
import { icon } from '../icons';
import { MIN_KEYFRAMES, routeForSession } from '../router';
import { interruptionCopy } from '../run-journal';

const INSTALL_HINT_KEY = 'sfm.hideInstallHint';

export function createHomeScreen(app: App): Screen {
  const d = new Disposer();
  /** Thumbnail object URLs by session id + blob (reused across re-renders, revoked on leave). */
  const thumbUrls = new Map<string, { blob: Blob; url: string }>();
  let closed = false;
  const banners = h('div', { class: 'banners' });
  const interruptHost = h('div', { class: 'interrupt-host' });
  const listHost = h('div', { class: 'session-list-host' });
  const storageHost = h('div', { class: 'storage-host' });
  let sessions: SessionMeta[] = [];

  const newScan = button({
    label: 'New scan',
    icon: 'plus',
    variant: 'primary',
    size: 'xl',
    block: true,
    testId: 'new-scan',
    onClick: () => app.navigate({ type: 'NEW_SCAN' }),
  });

  const header = h(
    'header',
    { class: 'home-header' },
    h('div', { class: 'brand' }, h('span', { class: 'brand-mark', 'aria-hidden': 'true' }, icon('cube', { size: 20 })), h('span', { class: 'brand-name', text: '3D Scan' })),
    iconButton('activity', 'Diagnostics', () => app.navigate({ type: 'OPEN_DIAGNOSTICS' }), { variant: 'glass', testId: 'open-diagnostics' }),
  );

  const hero = h(
    'section',
    { class: 'hero' },
    h('div', { class: 'hero-art', html: HERO_SVG, 'aria-hidden': 'true' }),
    h('h1', { class: 'hero-title', text: 'Scan anything in 3D' }),
    h('p', { class: 'hero-text', text: 'Walk around an object or through a space. Everything is reconstructed right on your phone — nothing is uploaded.' }),
    newScan,
  );

  const el = h(
    'div',
    { class: 'screen-scroll home' },
    h('div', { class: 'content' }, header, banners, interruptHost, hero, listHost, storageHost, footer(app)),
  );

  function renderBanners(env: EnvInfo): void {
    const issues = environmentIssues(env);
    replaceChildren(
      banners,
      ...issues.map((i) =>
        banner({
          tone: i.severity === 'blocking' ? 'bad' : 'warn',
          icon: i.id === 'lockdown' ? 'lock' : i.id === 'webgpu' ? 'cpu' : 'warning',
          title: i.title,
          body: i.body,
          testId: `env-${i.id}`,
        }),
      ),
    );
    const ok = canScan(env);
    newScan.disabled = !ok;
    newScan.title = ok ? '' : 'Scanning is not available on this device (see above)';
  }

  function thumb(s: SessionMeta): HTMLElement {
    if (s.thumbnail instanceof Blob && s.thumbnail.size > 0) {
      let t = thumbUrls.get(s.id);
      if (!t || t.blob !== s.thumbnail) {
        if (t) URL.revokeObjectURL(t.url);
        t = { blob: s.thumbnail, url: URL.createObjectURL(s.thumbnail) };
        thumbUrls.set(s.id, t);
      }
      return h('img', { class: 'session-thumb', src: t.url, alt: '', loading: 'lazy', decoding: 'async' });
    }
    return h('span', { class: 'session-thumb session-thumb-empty' }, icon(s.mode === 'scene' ? 'area' : 'cube', { size: 26 }));
  }

  function openSession(s: SessionMeta): void {
    const n = photoCount(s);
    if (!routeForSession(s.id, s.status, n)) {
      toast(`This capture was interrupted with ${plural(n, 'photo')} — at least ${MIN_KEYFRAMES} are needed.`, { tone: 'warn' });
      return;
    }
    // opening the interrupted scan (its card's Resume, or its list item) answers the card
    if (app.journal.interruption()?.sessionId === s.id) app.journal.acknowledge();
    app.navigate({ type: 'OPEN_SESSION', sessionId: s.id, status: s.status, numFrames: n });
  }

  /** The run journal's "run interrupted" card (see the header), once the list is known. */
  function renderInterruption(listed: boolean): void {
    const it = app.journal.interruption();
    const s = it ? sessions.find((x) => x.id === it.sessionId) : undefined;
    // the scan is gone (deleted): nothing to resume
    if (it && !s && listed) app.journal.acknowledge();
    if (!it || !s) {
      replaceChildren(interruptHost);
      return;
    }
    const copy = interruptionCopy({ ...it, name: s.name });
    const card = banner({
      tone: 'warn',
      title: copy.title,
      body: copy.body,
      actions: [button({ label: 'Resume', icon: 'refresh', variant: 'primary', size: 'md', testId: 'run-interrupted-resume', onClick: () => openSession(s) })],
      onDismiss: () => app.journal.acknowledge(),
      testId: 'run-interrupted',
    });
    card.classList.add('run-interrupted');
    card.dataset.kind = it.kind;
    card.dataset.step = it.step;
    replaceChildren(interruptHost, card);
  }

  async function rename(s: SessionMeta): Promise<void> {
    const name = await promptDialog({ title: 'Rename scan', value: s.name, confirmLabel: 'Rename' });
    if (!name || name === s.name) return;
    await app.updateSession(s.id, { name });
    if (app.results.sessionId === s.id) app.results.name = name;
    await loadSessions();
  }

  async function remove(s: SessionMeta): Promise<void> {
    const ok = await confirmDialog({
      title: `Delete “${s.name}”?`,
      message: 'The photos and models of this scan will be removed from this device. This cannot be undone.',
      confirmLabel: 'Delete',
      destructive: true,
    });
    if (!ok) return;
    if (await app.deleteSession(s.id)) {
      toast('Scan deleted', { tone: 'neutral', icon: 'trash' });
      await loadSessions();
      void loadStorage();
    }
  }

  function more(s: SessionMeta): void {
    const canOpen = !!routeForSession(s.id, s.status, photoCount(s));
    actionSheet(
      s.name,
      [
        { label: canOpen ? (s.status === 'captured' || s.status === 'capturing' ? 'Process' : 'Open') : 'Open', icon: 'cube', disabled: !canOpen, onSelect: () => openSession(s), testId: 'session-open' },
        { label: 'Rename', icon: 'pencil', onSelect: () => void rename(s), testId: 'session-rename' },
        { label: 'Delete', icon: 'trash', destructive: true, onSelect: () => void remove(s), testId: 'session-delete' },
      ],
      `${modeLabel(s.mode)} · ${formatRelativeDate(s.createdAt)}`,
    );
  }

  function renderList(): void {
    if (closed) return;
    // Drop the URLs of sessions that are gone (deleted).
    const ids = new Set(sessions.map((s) => s.id));
    for (const [id, t] of thumbUrls) {
      if (!ids.has(id)) {
        URL.revokeObjectURL(t.url);
        thumbUrls.delete(id);
      }
    }
    if (!sessions.length) {
      replaceChildren(
        listHost,
        section(
          'Your scans',
          h(
            'div',
            { class: 'empty-state' },
            icon('object', { size: 34 }),
            h('p', { class: 'empty-title', text: 'No scans yet' }),
            h('p', { class: 'empty-text', text: 'Your scans are stored on this device and appear here.' }),
          ),
        ),
      );
      return;
    }
    const list = h('ul', { class: 'session-list', role: 'list' });
    for (const s of sessions) {
      const st = statusInfo(s.status);
      const main = h(
        'button',
        { type: 'button', class: 'session-main', 'aria-label': `Open ${s.name}`, 'data-testid': 'session-item' },
        thumb(s),
        h(
          'span',
          { class: 'session-text' },
          h('span', { class: 'session-name', text: s.name }),
          h('span', { class: 'session-meta', text: `${formatRelativeDate(s.createdAt)} · ${plural(photoCount(s), 'photo')}` }),
          h('span', { class: 'session-chips' }, chip(st.label, st.tone), chip(modeLabel(s.mode), 'neutral')),
        ),
      );
      main.addEventListener('click', () => openSession(s));
      const moreBtn = iconButton('more', `More actions for ${s.name}`, () => more(s), { className: 'session-more', testId: 'session-more' });
      list.appendChild(h('li', { class: 'session' }, main, moreBtn));
    }
    replaceChildren(listHost, section(`Your scans`, h('div', { class: 'section-count', text: plural(sessions.length, 'scan') }), list));
  }

  async function loadSessions(): Promise<void> {
    const store = await app.sessions();
    if (!store) {
      replaceChildren(listHost);
      return;
    }
    let listed = true;
    try {
      sessions = await store.listSessions({ orderBy: 'createdAt' });
    } catch (e) {
      console.warn('[home] listSessions failed', e);
      sessions = [];
      listed = false;
    }
    // `closed`, not isConnected: the element stays in the DOM during the exit transition, and
    // object URLs created after leave() would never be revoked.
    if (closed) return;
    renderList();
    renderInterruption(listed);
  }

  async function loadStorage(): Promise<void> {
    const store = await app.sessions();
    const est: StorageEstimateInfo | null = store ? await store.estimate() : null;
    const env = app.env;
    const showInstall = env.apple && !env.standalone && !hiddenInstallHint();
    const parts: HTMLElement[] = [];
    if (est && est.quota > 0) {
      const frac = Math.min(1, est.usage / est.quota);
      parts.push(
        h(
          'div',
          { class: 'storage-card', 'data-testid': 'storage' },
          h('div', { class: 'storage-head' }, icon('storage', { size: 18 }), h('span', { text: 'On-device storage' }), est.persisted ? chip('Persistent', 'good') : chip('Not persistent', 'warn')),
          h('div', { class: 'storage-bar', role: 'meter', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': String(Math.round(frac * 100)), 'aria-label': 'Storage used' }, h('div', { class: 'storage-fill', style: { transform: `scaleX(${Math.max(0.01, frac)})` } })),
          h('div', { class: 'storage-text', text: `${formatBytes(est.usage)} used · ${formatBytes(est.available)} available` }),
        ),
      );
    }
    if (showInstall) {
      const card = h(
        'div',
        { class: 'install-card', 'data-testid': 'install-hint' },
        h('span', { class: 'install-icon' }, icon('addSquare', { size: 22 })),
        h(
          'div',
          { class: 'install-text' },
          h('strong', { text: 'Install to Home Screen' }),
          h('span', { text: 'Tap ' }),
          icon('share', { size: 15, className: 'inline-icon' }),
          h('span', { text: ' Share, then “Add to Home Screen”. Installed, your scans are stored persistently — Safari may delete website data after 7 days without use. Scans made in Safari do not move to the installed app.' }),
        ),
      );
      const close = iconButton('close', 'Hide this tip', () => {
        try {
          localStorage.setItem(INSTALL_HINT_KEY, '1');
        } catch {
          /* ignore */
        }
        card.remove();
      });
      close.classList.add('install-close');
      card.appendChild(close);
      parts.push(card);
    }
    if (closed) return;
    replaceChildren(storageHost, ...parts);
  }

  return {
    el,
    enter() {
      renderBanners(app.env);
      d.add(app.store.select((s) => s.env, renderBanners));
      replaceChildren(listHost, h('div', { class: 'list-loading' }, spinner('md', 'Loading scans')));
      void loadSessions();
      void loadStorage();
      // Refresh relative dates now and then.
      d.interval(() => {
        if (sessions.length) renderList();
      }, 60_000);
    },
    leave() {
      closed = true;
      d.dispose();
      for (const t of thumbUrls.values()) URL.revokeObjectURL(t.url);
      thumbUrls.clear();
    },
  };
}

/** Stored frames, or the keyframe count recorded at capture time (engines that store none). */
export function photoCount(s: Pick<SessionMeta, 'numFrames' | 'stats'>): number {
  const kf = s.stats?.keyframes;
  return Math.max(s.numFrames, typeof kf === 'number' && Number.isFinite(kf) ? kf : 0);
}

function hiddenInstallHint(): boolean {
  try {
    return localStorage.getItem(INSTALL_HINT_KEY) === '1';
  } catch {
    return false;
  }
}

function footer(app: App): HTMLElement {
  const diag = h('button', { type: 'button', class: 'link-btn', 'data-testid': 'footer-diagnostics' }, icon('activity', { size: 16 }), h('span', { text: 'Diagnostics' }));
  diag.addEventListener('click', () => app.navigate({ type: 'OPEN_DIAGNOSTICS' }));
  return h(
    'footer',
    { class: 'home-footer' },
    diag,
    h('span', { class: 'footer-note', text: app.mock ? 'Demo engine (mock=1)' : 'Processing happens on this device' }),
  );
}

/** Hero illustration: a point-cloud object inside an orbit of camera frustums. */
const HERO_SVG = `
<svg viewBox="0 0 320 170" width="100%" height="100%" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="hg" cx="50%" cy="55%" r="55%"><stop offset="0" stop-color="#36e0a6" stop-opacity=".22"/><stop offset="1" stop-color="#36e0a6" stop-opacity="0"/></radialGradient>
    <linearGradient id="hc" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#7cf3c8"/><stop offset="1" stop-color="#2bb589"/></linearGradient>
  </defs>
  <ellipse cx="160" cy="92" rx="150" ry="76" fill="url(#hg)"/>
  <ellipse cx="160" cy="118" rx="118" ry="30" stroke="#ffffff" stroke-opacity=".14" stroke-dasharray="3 6"/>
  <g fill="url(#hc)">
    ${heroDots()}
  </g>
  <g stroke="#e8edf5" stroke-opacity=".85" stroke-width="1.6" stroke-linejoin="round">
    <path d="M44 112l16-9v18z" fill="#e8edf5" fill-opacity=".08"/>
    <path d="M262 104l16 9-16 9z" fill="#e8edf5" fill-opacity=".08"/>
    <path d="M118 142l10-14 10 14z" fill="#e8edf5" fill-opacity=".08"/>
    <path d="M206 140l10-14 10 14z" fill="#e8edf5" fill-opacity=".08"/>
  </g>
  <circle cx="60" cy="112" r="3" fill="#36e0a6"/>
</svg>`;

function heroDots(): string {
  // Deterministic "vase" point cloud.
  const out: string[] = [];
  let seed = 12345;
  const rnd = (): number => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  for (let i = 0; i < 230; i++) {
    const t = rnd();
    const y = 30 + t * 82;
    const r = 18 + 20 * Math.sin(Math.PI * Math.min(1, t * 1.1)) + 4 * Math.cos(t * 9);
    const a = rnd() * Math.PI * 2;
    const x = 160 + r * Math.cos(a);
    const depth = (Math.sin(a) + 1) / 2;
    const op = (0.35 + 0.65 * depth).toFixed(2);
    const rad = (0.9 + 1.3 * depth).toFixed(2);
    out.push(`<circle cx="${x.toFixed(1)}" cy="${(y + Math.sin(a) * 3).toFixed(1)}" r="${rad}" fill-opacity="${op}"/>`);
  }
  for (let i = 0; i < 70; i++) {
    const a = rnd() * Math.PI * 2;
    const rr = 30 + rnd() * 70;
    out.push(`<circle cx="${(160 + rr * Math.cos(a)).toFixed(1)}" cy="${(116 + rr * 0.26 * Math.sin(a)).toFixed(1)}" r="1" fill-opacity="${(0.25 + rnd() * 0.3).toFixed(2)}"/>`);
  }
  return out.join('');
}
