/**
 * What a harness run ran on (`HarnessResult.device`): the user agent, cores, screen and the page's
 * WebGPU adapter. The adapter is only requested (no device is created and nothing is submitted, so
 * the engine worker's §2.1 budget is untouched); its `maxBufferSize` gives the iPhone RAM class
 * (docs/research/ios-platform.md §1.1). The engine's own adapter string, from the realm it runs
 * in, is `result.engine.init.adapter`. Never throws: a missing or failing adapter is recorded in
 * `adapterError`.
 */
import type { HarnessDevice } from './types';

const LIMITS = ['maxBufferSize', 'maxStorageBufferBindingSize', 'maxComputeWorkgroupStorageSize', 'maxComputeInvocationsPerWorkgroup', 'maxTextureDimension2D'] as const;

export async function deviceInfo(timeoutMs = 5000): Promise<HarnessDevice> {
  const nav = (typeof navigator !== 'undefined' ? navigator : {}) as Partial<Navigator> & { gpu?: GPU };
  const scr = (globalThis as { screen?: { width: number; height: number } }).screen;
  const d: HarnessDevice = {
    userAgent: nav.userAgent ?? '',
    platform: nav.platform || null,
    hardwareConcurrency: nav.hardwareConcurrency ?? null,
    devicePixelRatio: (globalThis as { devicePixelRatio?: number }).devicePixelRatio ?? 1,
    screen: scr ? [scr.width, scr.height] : null,
    secureContext: (globalThis as { isSecureContext?: boolean }).isSecureContext ?? false,
    adapter: null,
  };
  if (!nav.gpu) {
    d.adapterError = 'navigator.gpu is missing (WebGPU off, or not a secure context)';
    return d;
  }
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const timeout = new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error(`requestAdapter() did not answer in ${timeoutMs} ms`)), timeoutMs);
    });
    const a = await Promise.race([nav.gpu.requestAdapter(), timeout]);
    if (!a) {
      d.adapterError = 'requestAdapter() returned null';
      return d;
    }
    const info = (a.info ?? {}) as Partial<GPUAdapterInfo>;
    const limits: Record<string, number> = {};
    for (const k of LIMITS) {
      const v = (a.limits as unknown as Record<string, unknown>)[k];
      if (typeof v === 'number') limits[k] = v;
    }
    d.adapter = {
      vendor: info.vendor ?? '',
      architecture: info.architecture ?? '',
      device: info.device ?? '',
      description: info.description ?? '',
      features: [...a.features].map(String).sort(),
      limits,
    };
  } catch (e) {
    d.adapterError = (e as Error)?.message ?? String(e);
  } finally {
    clearTimeout(timer);
  }
  return d;
}
