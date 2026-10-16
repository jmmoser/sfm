/**
 * Feature pipeline that uses the WebGPU implementations (SiftGPU, MatcherGPU) when
 * available (and their start-up self-test passes) and the CPU reference
 * (extractSiftCPU, matchCPU, matchGuidedCPU) otherwise. A GPU failure at run time
 * (device loss, validation / OOM errors, failed readbacks — GPUFeatureError) switches
 * the pipeline permanently to the CPU: the failed call is redone on the CPU, a warning
 * is logged once and `gpu` becomes false. GPU work dropped by the OS (page hidden,
 * GPUWorkDroppedError after the internal retries) only redoes that call on the CPU.
 * Input errors (e.g. image data too short) are thrown as-is by both backends.
 */

import type { FeatureSet, GrayImage, MatchPairs } from '../core/types';
import { getGPU } from '../gpu/device';
import type { GPUContext } from '../gpu/device';
import { GPUFeatureError, GPUWorkDroppedError } from './gpu-common';
import { rgbaToGray } from './image';
import { matchCPU, matchGuidedCPU } from './match';
import type { MatchGuide, MatchOptions } from './match';
import { MatcherGPU } from './match-gpu';
import type { MatcherGPUConfig } from './match-gpu';
import type { SiftOptions } from './sift-common';
import { extractSiftCPU } from './sift-cpu';
import { SiftGPU } from './sift-gpu';
import type { SiftBitmapSource, SiftGPUConfig, SiftSelfTestResult } from './sift-gpu';
import type { MatcherSelfTestResult } from './match-gpu';

export interface FeaturePipelineOptions {
  /** Use WebGPU when available (default true). false = CPU only. */
  preferGPU?: boolean;
  /** SIFT options (both backends). */
  sift?: Partial<SiftOptions>;
  /** GPU context to use instead of getGPU() (null = behave as if WebGPU were unavailable). */
  context?: GPUContext | null;
  /** Extra SiftGPU configuration (cached sizes, submit budget). */
  siftGPU?: SiftGPUConfig;
  /** Extra MatcherGPU configuration (descriptor cache budget, submit budget). */
  matcherGPU?: MatcherGPUConfig;
  /** Called once when the pipeline falls back to the CPU at run time. */
  onFallback?: (reason: string) => void;
  /**
   * Compare the GPU implementations with the CPU reference on small synthetic inputs
   * at creation and use the CPU if they disagree (guards against driver / compiler bugs
   * on untested devices). Default true; costs a few ms on a phone GPU.
   */
  selfTest?: boolean;
}

export interface FeaturePipeline {
  /** True while the WebGPU backend is in use. */
  readonly gpu: boolean;
  /** Why the CPU backend is used (null while on the GPU). */
  readonly fallbackReason: string | null;
  extract(img: GrayImage, fullWidth: number, fullHeight: number): Promise<FeatureSet>;
  /**
   * (Extension) Extract from an ImageBitmap / canvas / ImageData. On the GPU the luma
   * conversion and resampling happen on the GPU (no CPU pixel work); on the CPU the
   * bitmap is drawn into an OffscreenCanvas and converted with rgbaToGray.
   */
  extractFromBitmap(bitmap: SiftBitmapSource, fullWidth?: number, fullHeight?: number): Promise<FeatureSet>;
  match(a: FeatureSet, b: FeatureSet, opts?: MatchOptions): Promise<MatchPairs>;
  matchGuided(a: FeatureSet, b: FeatureSet, guide: MatchGuide, opts?: MatchOptions): Promise<MatchPairs>;
  destroy(): void;
}

/** RGBA → gray on the CPU for bitmap sources (OffscreenCanvas readback). */
export function bitmapToGrayCPU(bitmap: SiftBitmapSource): GrayImage {
  if (typeof ImageData !== 'undefined' && bitmap instanceof ImageData) {
    if (bitmap.data.length < 4 * bitmap.width * bitmap.height) {
      throw new Error(`extractFromBitmap: ImageData pixels are detached or shorter than ${bitmap.width}x${bitmap.height}x4`);
    }
    return rgbaToGray(bitmap);
  }
  const w = bitmap.width;
  const h = bitmap.height;
  if (typeof OffscreenCanvas === 'undefined') throw new Error('extractFromBitmap: OffscreenCanvas unavailable for the CPU path');
  const canvas = new OffscreenCanvas(w, h);
  const c2d = canvas.getContext('2d', { willReadFrequently: true });
  if (!c2d) throw new Error('extractFromBitmap: 2D context unavailable');
  c2d.drawImage(bitmap as CanvasImageSource, 0, 0);
  return rgbaToGray(c2d.getImageData(0, 0, w, h));
}

/** Outcome of the GPU-vs-CPU start-up self-tests. */
type SelfTestVerdict = { ok: true } | { ok: false; transient: boolean; reason: string };

async function runSelfTests(sift: SiftGPU, matcher: MatcherGPU): Promise<SelfTestVerdict> {
  const [ts, tm]: [SiftSelfTestResult, MatcherSelfTestResult] = await Promise.all([sift.selfTest(), matcher.selfTest()]);
  if (ts.ok && tm.ok) return { ok: true };
  const reason =
    `GPU self-test failed (SIFT ok=${ts.ok} gpu/cpu ${ts.gpuCount}/${ts.cpuCount} pos ${ts.posFrac.toFixed(3)} cos ${ts.medianCos.toFixed(4)}` +
    `${ts.error ? ` ${ts.error}` : ''}; matcher ok=${tm.ok} overlap ${tm.overlap.toFixed(3)}${tm.error ? ` ${tm.error}` : ''})`;
  // Transient = every failing test failed only because its GPU work was dropped (page hidden).
  const transient = (ts.ok || !!ts.transient) && (tm.ok || !!tm.transient);
  return { ok: false, transient, reason };
}

class FeaturePipelineImpl implements FeaturePipeline {
  private sift: SiftGPU | null;
  private matcher: MatcherGPU | null;
  private reason: string | null;
  private destroyed = false;
  private warnedTransient = false;
  /**
   * The start-up self-test could not run (GPU work dropped, e.g. the app was backgrounded
   * while the pipeline was created): it is re-run before the next GPU use instead of
   * selecting the CPU for the whole session.
   */
  private verifyPending: boolean;
  private verifying: Promise<void> | null = null;

  constructor(
    sift: SiftGPU | null,
    matcher: MatcherGPU | null,
    reason: string | null,
    private readonly siftOpts: Partial<SiftOptions>,
    private readonly onFallback: ((reason: string) => void) | undefined,
    device: GPUDevice | null,
    verifyPending = false,
  ) {
    this.sift = sift;
    this.matcher = matcher;
    this.reason = reason;
    this.verifyPending = verifyPending && !!(sift && matcher);
    if (device && sift && matcher) {
      device.lost.then((info) => {
        if (!this.destroyed && this.sift) this.fallback(`GPU device lost (${info.reason}): ${info.message}`);
      });
    }
  }

  get gpu(): boolean {
    return !!(this.sift && this.matcher);
  }

  get fallbackReason(): string | null {
    return this.reason;
  }

  private fallback(reason: string, err?: unknown): void {
    if (!this.sift && !this.matcher) return;
    this.reason = reason;
    console.warn('[features] WebGPU feature pipeline failed; falling back to the CPU:', reason, err ?? '');
    try {
      this.sift?.destroy();
      this.matcher?.destroy();
    } catch {
      // ignore: the device may be gone
    }
    this.sift = null;
    this.matcher = null;
    this.onFallback?.(reason);
  }

  /** Runs the deferred start-up self-test (once; concurrent calls share it). */
  private async verify(): Promise<void> {
    if (!this.verifyPending) return;
    const s = this.sift;
    const m = this.matcher;
    if (!s || !m) return;
    this.verifying ??= runSelfTests(s, m)
      .then((v) => {
        if (v.ok) this.verifyPending = false;
        else if (!v.transient) {
          this.verifyPending = false;
          this.fallback(v.reason);
        }
        // still transient: stays pending, this call runs on the CPU
      })
      .finally(() => {
        this.verifying = null;
      });
    await this.verifying;
  }

  private async withGPU<T>(gpuOp: ((s: SiftGPU, m: MatcherGPU) => Promise<T>) | null, cpuOp: () => T): Promise<T> {
    if (this.destroyed) throw new Error('FeaturePipeline: used after destroy()');
    if (gpuOp && this.sift && this.matcher) await this.verify();
    const s = this.sift;
    const m = this.matcher;
    if (gpuOp && s && m && !this.verifyPending && !this.destroyed) {
      try {
        return await gpuOp(s, m);
      } catch (e) {
        if (!(e instanceof GPUFeatureError)) throw e;
        if (e instanceof GPUWorkDroppedError) {
          if (!this.warnedTransient) console.warn('[features] GPU work was dropped (page hidden?); this call runs on the CPU:', e.message);
          this.warnedTransient = true;
        } else {
          this.fallback(e.message, e);
        }
      }
    }
    return cpuOp();
  }

  extract(img: GrayImage, fullWidth: number, fullHeight: number): Promise<FeatureSet> {
    return this.withGPU((s) => s.extract(img, fullWidth, fullHeight), () => extractSiftCPU(img, this.siftOpts, fullWidth, fullHeight));
  }

  extractFromBitmap(bitmap: SiftBitmapSource, fullWidth?: number, fullHeight?: number): Promise<FeatureSet> {
    // fullWidth / fullHeight are passed through unchanged: with only one of them given,
    // extractSiftCPU (like SiftGPU) derives the other from the image aspect. (Defaulting the
    // missing one to the bitmap size gave an anisotropic frame on the CPU path only.)
    return this.withGPU((s) => s.extractFromBitmap(bitmap, fullWidth, fullHeight), () =>
      extractSiftCPU(bitmapToGrayCPU(bitmap), this.siftOpts, fullWidth, fullHeight),
    );
  }

  match(a: FeatureSet, b: FeatureSet, opts?: MatchOptions): Promise<MatchPairs> {
    return this.withGPU((_s, m) => m.match(a, b, opts), () => matchCPU(a, b, opts));
  }

  matchGuided(a: FeatureSet, b: FeatureSet, guide: MatchGuide, opts?: MatchOptions): Promise<MatchPairs> {
    return this.withGPU((_s, m) => m.matchGuided(a, b, guide, opts), () => matchGuidedCPU(a, b, guide, opts));
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.sift?.destroy();
    this.matcher?.destroy();
    this.sift = null;
    this.matcher = null;
  }
}

/**
 * Creates the feature pipeline: WebGPU (SiftGPU + MatcherGPU, pipelines compiled up
 * front) when `preferGPU` (default) and a device is available, the CPU reference
 * otherwise. Never rejects because of the GPU.
 */
export async function createFeaturePipeline(opts: FeaturePipelineOptions = {}): Promise<FeaturePipeline> {
  const siftOpts = opts.sift ?? {};
  let sift: SiftGPU | null = null;
  let matcher: MatcherGPU | null = null;
  let reason: string | null = null;
  let device: GPUDevice | null = null;
  let verifyPending = false;
  if (opts.preferGPU === false) {
    reason = 'CPU requested (preferGPU: false)';
  } else {
    let ctx: GPUContext | null = null;
    try {
      ctx = opts.context !== undefined ? opts.context : await getGPU();
    } catch (e) {
      reason = `WebGPU initialisation failed: ${(e as Error)?.message ?? e}`;
    }
    if (!ctx) {
      reason ??= 'WebGPU unavailable';
    } else {
      const [rs, rm] = await Promise.allSettled([SiftGPU.create(ctx, siftOpts, opts.siftGPU), MatcherGPU.create(ctx, opts.matcherGPU)]);
      if (rs.status === 'fulfilled' && rm.status === 'fulfilled') {
        sift = rs.value;
        matcher = rm.value;
        device = ctx.device;
        if (opts.selfTest ?? true) {
          const v = await runSelfTests(sift, matcher);
          if (!v.ok && v.transient) {
            // The tests' GPU work was dropped (page hidden?): no verdict; re-run before first use.
            verifyPending = true;
            console.info('[features] GPU self-test postponed (GPU work dropped):', v.reason);
          } else if (!v.ok) {
            reason = v.reason;
            sift.destroy();
            matcher.destroy();
            sift = null;
            matcher = null;
            device = null;
          }
        }
      } else {
        if (rs.status === 'fulfilled') rs.value.destroy();
        if (rm.status === 'fulfilled') rm.value.destroy();
        const err = rs.status === 'rejected' ? rs.reason : (rm as PromiseRejectedResult).reason;
        reason = `WebGPU feature pipeline unavailable: ${(err as Error)?.message ?? err}`;
      }
    }
    if (reason) console.info('[features] using the CPU feature pipeline:', reason);
  }
  return new FeaturePipelineImpl(sift, matcher, reason, siftOpts, opts.onFallback, device, verifyPending);
}
