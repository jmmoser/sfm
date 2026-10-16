/**
 * Screen Wake Lock with automatic re-acquisition. iOS Safari 16.4+ (Home Screen web apps from
 * 18.4): the lock is released by the system whenever the page is hidden, so it is requested
 * again on `visibilitychange` → visible while the caller still wants it. Acquire it from the
 * "Start capture" tap. Never throws: failures are reported in `lastError`.
 */
export class WakeLock {
  static isSupported(): boolean {
    return typeof navigator !== 'undefined' && 'wakeLock' in navigator && !!navigator.wakeLock;
  }

  private sentinel: WakeLockSentinel | null = null;
  private wanted = false;
  private pending: Promise<void> | null = null;
  private listening = false;
  /** Name/message of the last failed request (null after a success). */
  lastError: string | null = null;

  /** True while the screen lock is held. */
  get active(): boolean {
    return !!this.sentinel && !this.sentinel.released;
  }

  /** Keep the screen on until release() (re-acquired after the page becomes visible again). */
  async acquire(): Promise<void> {
    this.wanted = true;
    this.listen(true);
    await this.request();
  }

  async release(): Promise<void> {
    this.wanted = false;
    this.listen(false);
    const s = this.sentinel;
    this.sentinel = null;
    if (s) {
      s.removeEventListener('release', this.onRelease);
      if (!s.released) {
        try {
          await s.release();
        } catch {
          /* already released */
        }
      }
    }
  }

  private listen(on: boolean): void {
    if (typeof document === 'undefined' || on === this.listening) return;
    if (on) document.addEventListener('visibilitychange', this.onVisibility);
    else document.removeEventListener('visibilitychange', this.onVisibility);
    this.listening = on;
  }

  private request(): Promise<void> {
    if (!WakeLock.isSupported()) {
      this.lastError = 'unsupported';
      return Promise.resolve();
    }
    if (this.active || (typeof document !== 'undefined' && document.visibilityState === 'hidden')) return Promise.resolve();
    if (this.pending) return this.pending;
    this.pending = (async () => {
      try {
        const s = await navigator.wakeLock.request('screen');
        if (!this.wanted) {
          await s.release().catch(() => undefined);
          return;
        }
        this.sentinel?.removeEventListener('release', this.onRelease);
        this.sentinel = s;
        s.addEventListener('release', this.onRelease);
        this.lastError = null;
      } catch (e) {
        this.lastError = (e as { name?: string })?.name ?? String(e);
      } finally {
        this.pending = null;
      }
    })();
    return this.pending;
  }

  private readonly onRelease = (e: Event): void => {
    const s = e.target as WakeLockSentinel | null;
    s?.removeEventListener('release', this.onRelease);
    // iOS may dispatch the 'release' of the previous sentinel only after the page is visible
    // again, i.e. after a new lock was obtained: ignore it (it must not orphan the new lock).
    if (s && s !== this.sentinel) return;
    this.sentinel = null;
    if (this.wanted && typeof document !== 'undefined' && document.visibilityState === 'visible') void this.request();
  };

  private readonly onVisibility = (): void => {
    if (this.wanted && document.visibilityState === 'visible') void this.request();
  };
}
