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
 *       → fall back to step 2.
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

/** Replace characters that are invalid in file names on iOS / Windows. */
export function sanitizeFilename(name: string): string {
  // eslint-disable-next-line no-control-regex
  const s = name.replace(/[\u0000-\u001f<>:"/\\|?*]+/g, '_').replace(/^[\s.]+|[\s.]+$/g, '');
  return s.length ? s.slice(0, 200) : 'download';
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
        if (mode === 'share') throw e;
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
