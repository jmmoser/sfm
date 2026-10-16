import { afterEach, describe, expect, it, vi } from 'vitest';
import { canShareFile, mimeTypeForFilename, sanitizeFilename, saveFile } from './download';

type ShareFn = (data: ShareData) => Promise<void>;

function installShare(share: ShareFn, canShare: (d: ShareData) => boolean): void {
  Object.defineProperty(navigator, 'share', { value: share, configurable: true, writable: true });
  Object.defineProperty(navigator, 'canShare', { value: canShare, configurable: true, writable: true });
}

function removeShare(): void {
  // Own properties shadow the prototype (if any) — deleting restores the original.
  delete (navigator as unknown as Record<string, unknown>).share;
  delete (navigator as unknown as Record<string, unknown>).canShare;
}

function captureDownloads(): { clicks: { href: string; download: string }[]; restore: () => void } {
  const clicks: { href: string; download: string }[] = [];
  const spy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    clicks.push({ href: this.href, download: this.download });
  });
  return { clicks, restore: () => spy.mockRestore() };
}

afterEach(() => {
  removeShare();
  vi.restoreAllMocks();
});

describe('saveFile', () => {
  it('shares via the Web Share API when canShare({files}) is true', async () => {
    const calls: ShareData[] = [];
    installShare(async (d) => void calls.push(d), (d) => !!d.files?.length);
    const dl = captureDownloads();
    const res = await saveFile(new Blob(['ply data']), 'scan.glb');
    expect(res).toBe('shared');
    expect(calls.length).toBe(1);
    const f = calls[0].files![0];
    expect(f.name).toBe('scan.glb');
    expect(f.type).toBe('model/gltf-binary');
    expect(await f.text()).toBe('ply data');
    expect(calls[0].title).toBeUndefined();
    expect(dl.clicks.length).toBe(0);
  });

  it("returns 'cancelled' when the user dismisses the share sheet (AbortError)", async () => {
    installShare(async () => {
      throw new DOMException('Share canceled', 'AbortError');
    }, () => true);
    const dl = captureDownloads();
    expect(await saveFile(new Blob(['x']), 'a.zip')).toBe('cancelled');
    expect(dl.clicks.length).toBe(0);
  });

  it('falls back to <a download> when share fails with NotAllowedError (expired activation)', async () => {
    installShare(async () => {
      throw new DOMException('Must be handling a user gesture', 'NotAllowedError');
    }, () => true);
    const dl = captureDownloads();
    expect(await saveFile(new Blob(['x']), 'a.zip')).toBe('downloaded');
    expect(dl.clicks.length).toBe(1);
    expect(dl.clicks[0].download).toBe('a.zip');
  });

  it("downloads via an object URL when sharing files is unsupported, and revokes it later", async () => {
    installShare(async () => {}, () => false);
    const revoke = vi.spyOn(URL, 'revokeObjectURL');
    const dl = captureDownloads();
    const res = await saveFile(new Blob(['abc'], { type: 'text/plain' }), 'notes/v1:final.txt', { revokeDelayMs: 10 });
    expect(res).toBe('downloaded');
    expect(dl.clicks.length).toBe(1);
    expect(dl.clicks[0].href.startsWith('blob:')).toBe(true);
    expect(dl.clicks[0].download).toBe('notes_v1_final.txt');
    expect(document.querySelectorAll('a[download]').length).toBe(0); // anchor removed
    expect(revoke).not.toHaveBeenCalled();
    await new Promise((r) => setTimeout(r, 30));
    expect(revoke).toHaveBeenCalledWith(dl.clicks[0].href);
  });

  it("honours mode 'download' and mode 'share'", async () => {
    const share = vi.fn(async () => {});
    installShare(share, () => true);
    const dl = captureDownloads();
    expect(await saveFile(new Blob(['x']), 'x.ply', { mode: 'download', revokeDelayMs: 0 })).toBe('downloaded');
    expect(share).not.toHaveBeenCalled();
    installShare(share, () => false);
    await expect(saveFile(new Blob(['x']), 'x.ply', { mode: 'share' })).rejects.toMatchObject({ name: 'NotSupportedError' });
    expect(dl.clicks.length).toBe(1);
  });

  it('works without any Web Share API (headless desktop Chromium)', async () => {
    removeShare();
    const dl = captureDownloads();
    if (typeof navigator.share !== 'function') expect(canShareFile(new File(['x'], 'x.txt'))).toBe(false);
    const res = await saveFile(new Blob(['x']), 'model.glb', { mode: 'download', revokeDelayMs: 0 });
    expect(res).toBe('downloaded');
    expect(dl.clicks.length).toBe(1);
  });
});

describe('helpers', () => {
  it('infers MIME types and sanitises names', () => {
    expect(mimeTypeForFilename('a.GLB')).toBe('model/gltf-binary');
    expect(mimeTypeForFilename('a.zip')).toBe('application/zip');
    expect(mimeTypeForFilename('a.ply')).toBe('application/octet-stream');
    expect(mimeTypeForFilename('noext')).toBe('application/octet-stream');
    expect(sanitizeFilename('  ..evil/../name?.ply ')).toBe('evil_.._name_.ply');
    expect(sanitizeFilename('')).toBe('download');
  });
});
