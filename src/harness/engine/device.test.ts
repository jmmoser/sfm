/**
 * deviceInfo (./device.ts) with a stubbed navigator / screen: the adapter's info, sorted features
 * and the limits it keeps, and every way the adapter can be missing (no navigator.gpu, a null
 * adapter, a rejected or hanging requestAdapter) recorded in adapterError instead of thrown.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { deviceInfo } from './device';

function stubNavigator(gpu?: { requestAdapter(): Promise<unknown> }): void {
  vi.stubGlobal('navigator', { userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X)', platform: 'iPhone', hardwareConcurrency: 4, ...(gpu ? { gpu } : {}) });
  vi.stubGlobal('screen', { width: 393, height: 852 });
  vi.stubGlobal('devicePixelRatio', 3);
  vi.stubGlobal('isSecureContext', true);
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('deviceInfo', () => {
  it('records the adapter: info, sorted features and the known numeric limits', async () => {
    const adapter = {
      info: { vendor: 'apple', architecture: 'common-3', device: '', description: 'Apple GPU' },
      features: new Set(['texture-compression-astc', 'float32-filterable']),
      limits: { maxBufferSize: 1073741824, maxStorageBufferBindingSize: 1073741824, maxComputeWorkgroupStorageSize: 32768, maxTextureDimension2D: 16384, maxBindGroups: 4 },
    };
    stubNavigator({ requestAdapter: async () => adapter });
    const d = await deviceInfo();
    expect(d).toEqual({
      userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X)',
      platform: 'iPhone',
      hardwareConcurrency: 4,
      devicePixelRatio: 3,
      screen: [393, 852],
      secureContext: true,
      adapter: {
        vendor: 'apple',
        architecture: 'common-3',
        device: '',
        description: 'Apple GPU',
        features: ['float32-filterable', 'texture-compression-astc'],
        // maxComputeInvocationsPerWorkgroup absent on this adapter, maxBindGroups not one of the kept limits
        limits: { maxBufferSize: 1073741824, maxStorageBufferBindingSize: 1073741824, maxComputeWorkgroupStorageSize: 32768, maxTextureDimension2D: 16384 },
      },
    });
    expect(JSON.parse(JSON.stringify(d))).toEqual(d); // plain JSON (goes into result.json)
  });

  it('no navigator.gpu, a null adapter, a rejection or a hang: adapter null with the reason', async () => {
    stubNavigator();
    expect(await deviceInfo()).toMatchObject({ adapter: null, adapterError: expect.stringMatching(/navigator\.gpu is missing/), userAgent: expect.stringMatching(/iPhone/) });
    stubNavigator({ requestAdapter: async () => null });
    expect(await deviceInfo()).toMatchObject({ adapter: null, adapterError: 'requestAdapter() returned null' });
    stubNavigator({ requestAdapter: async () => Promise.reject(new Error('GPU process lost')) });
    expect(await deviceInfo()).toMatchObject({ adapter: null, adapterError: 'GPU process lost' });
    stubNavigator({ requestAdapter: () => new Promise(() => {}) });
    expect(await deviceInfo(20)).toMatchObject({ adapter: null, adapterError: 'requestAdapter() did not answer in 20 ms' });
  });
});
