/**
 * Hand a generated file to the user.
 *
 * Strategy (mode 'auto', the default):
 *  1. If `navigator.canShare({ files: [file] })` is true (iOS/iPadOS Safari,
 *     Android, some desktop Chromes for image/text types), open the native
 *     share sheet with only the file (adding title/text on iOS makes some
 *     targets receive an extra text item). "Save to Files" lives there on iOS.
 *     - User dismisses the sheet → AbortError → resolves 'cancelled'.
 *     - NotAllowedError (the transient user activation expired, e.g. because
 *       building the file took several seconds), TypeError or any other failure
 *       → fall back to step 2 — EXCEPT in an iOS Home Screen web app
 *       (`navigator.standalone`), where `<a download>` is broken: the share
 *       error is rethrown so the UI can ask for another tap.
 *  2. `<a download>` with an object URL (revoked after `revokeDelayMs`, default
 *     60 s — iOS opens a download confirmation and reads the URL lazily).
 *
 * iOS requires a user gesture for navigator.share: call saveFile directly from
 * a click/tap handler, with the Blob already prepared (e.g. build the export,
 * then show a "Save" button).
 *
 * Window scope only (needs `document` for the download fallback).
 */

export type SaveFileResult = 'shared' | 'downloaded' | 'cancelled';

export interface SaveFileOptions {
  /** 'auto' (share when possible, else download), 'share' (share or throw), 'download'. */
  mode?: 'auto' | 'share' | 'download';
  /** Override the MIME type (default: blob.type, else inferred from the extension). */
  mimeType?: string;
  /** Delay before revoking the object URL (ms). Default 60 000. */
  revokeDelayMs?: number;
}

const MIME_BY_EXT: Record<string, string> = {
  ply: 'application/octet-stream',
  glb: 'model/gltf-binary',
  gltf: 'model/gltf+json',
  usdz: 'model/vnd.usdz+zip',
  obj: 'model/obj',
  zip: 'application/zip',
  json: 'application/json',
  txt: 'text/plain',
  csv: 'text/csv',
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
  png: 'image/png',
  webp: 'image/webp',
  mp4: 'video/mp4',
  bin: 'application/octet-stream',
};

/** MIME type for a file name by extension (application/octet-stream if unknown). */
export function mimeTypeForFilename(filename: string): string {
  const dot = filename.lastIndexOf('.');
  const ext = dot >= 0 ? filename.slice(dot + 1).toLowerCase() : '';
  return MIME_BY_EXT[ext] ?? 'application/octet-stream';
}

const MAX_FILENAME = 200;

/** First `n` UTF-16 units of `s`, without splitting a surrogate pair. */
function truncateUnits(s: string, n: number): string {
  if (s.length <= n) return s;
  const c = s.charCodeAt(n - 1);
  return s.slice(0, c >= 0xd800 && c <= 0xdbff ? n - 1 : n);
}

/**
 * Replace characters that are invalid in file names on iOS / Windows. Long
 * names are shortened to 200 UTF-16 units KEEPING the extension (the share
 * sheet, Files and download managers infer the type from it) and never split
 * a surrogate pair.
 */
export function sanitizeFilename(name: string): string {
  // eslint-disable-next-line no-control-regex
  const s = name.replace(/[\u0000-\u001f<>:"/\\|?*]+/g, '_').replace(/^[\s.]+|[\s.]+$/g, '');
  if (!s.length) return 'download';
  if (s.length <= MAX_FILENAME) return s;
  const ext = /\.[^.\s]{1,16}$/.exec(s)?.[0] ?? '';
  const base = truncateUnits(s.slice(0, s.length - ext.length), MAX_FILENAME - ext.length).replace(/[\s.]+$/, '');
  return (base || 'download') + ext;
}

/**
 * iOS/iPadOS Home Screen web app (`navigator.standalone` exists only on Apple
 * WebKit and is true only in standalone mode). There `<a download>` with a
 * Blob URL is broken (preview sheet / stuck "navigation-less" state, WebKit
 * 275288, 290847 — docs/research/ios-platform.md §3), so it is never used as
 * an automatic fallback.
 */
function isIOSStandalone(): boolean {
  const nav = (globalThis as { navigator?: Navigator & { standalone?: boolean } }).navigator;
  return nav?.standalone === true;
}

function errorName(e: unknown): string | undefined {
  return e && typeof e === 'object' ? (e as { name?: string }).name : undefined;
}

/** True if the Web Share API can share this file in the current context. */
export function canShareFile(file: File): boolean {
  const nav = (globalThis as { navigator?: Navigator }).navigator;
  if (!nav || typeof nav.share !== 'function' || typeof nav.canShare !== 'function') return false;
  try {
    return nav.canShare({ files: [file] });
  } catch {
    return false;
  }
}

/**
 * Share or download `data` as `filename`. Resolves 'shared' when the share
 * sheet completed, 'cancelled' when the user dismissed it, 'downloaded' when
 * the <a download> path was taken.
 */
export async function saveFile(data: Blob, filename: string, opts: SaveFileOptions = {}): Promise<SaveFileResult> {
  const mode = opts.mode ?? 'auto';
  const name = sanitizeFilename(filename);
  const type = opts.mimeType ?? (data.type || mimeTypeForFilename(name));
  const file = data instanceof File && data.name === name && data.type === type ? data : new File([data], name, { type });

  if (mode !== 'download') {
    if (canShareFile(file)) {
      try {
        await navigator.share({ files: [file] });
        return 'shared';
      } catch (e) {
        if (errorName(e) === 'AbortError') return 'cancelled';
        // In an iOS Home Screen app the download fallback is broken: rethrow
        // (typically NotAllowedError — ask the user to tap "Save" again).
        if (mode === 'share' || isIOSStandalone()) throw e;
        // NotAllowedError (no user activation), InvalidStateError (another share
        // in progress), DataError… → fall back to a download.
      }
    } else if (mode === 'share') {
      throw new DOMException('Web Share with files is not supported here', 'NotSupportedError');
    }
  }

  if (typeof document === 'undefined') throw new Error('saveFile: download fallback requires a Window (document)');
  const url = URL.createObjectURL(file);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.rel = 'noopener';
  a.style.display = 'none';
  document.body.appendChild(a);
  try {
    a.click();
  } finally {
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), opts.revokeDelayMs ?? 60_000);
  }
  return 'downloaded';
}
