/**
 * harness.html query parsing (./query.ts): the manual keys, the new splat / splatIterations /
 * engineConfig keys, the whole-config `config` JSON a phone run sends, malformed values throwing
 * instead of falling back, and the bridge switch.
 */
import { describe, expect, it } from 'vitest';
import { harnessPath } from '../../../e2e/lib/e2e-bridge.mjs';
import { bridgeFromQuery, configFromQuery } from './query';
import type { HarnessConfig } from './types';

const q = (s: string): URLSearchParams => new URLSearchParams(s);

describe('configFromQuery', () => {
  it('no dataset → null; the documented manual keys', () => {
    expect(configFromQuery(q('engine=mock'))).toBeNull();
    const c = configFromQuery(q('dataset=object&engine=mock&maxFrames=40&densify=1&stride=2&imu=0&quality=balanced&mode=scene&headingNorth=app&mockSpeed=8'))!;
    expect(c).toEqual({
      dataset: 'object',
      engine: 'mock',
      maxFrames: 40,
      densify: true,
      frameStride: 2,
      useImu: false,
      quality: 'balanced',
      mode: 'scene',
      headingNorth: 'app',
      mockSpeed: 8,
    });
    // absent keys stay absent (pipeline defaults apply)
    expect(configFromQuery(q('dataset=yard'))).toEqual({ dataset: 'yard', engine: 'worker' });
    expect(configFromQuery(q('dataset=yard&engine=inline&headingNorth=true&mode=auto&quality=max&fps=7.5'))).toEqual({ dataset: 'yard', engine: 'inline', headingNorth: 'true', mode: 'auto', quality: 'max', fps: 7.5 });
  });

  it('a typo throws instead of running another configuration', () => {
    expect(() => configFromQuery(q('dataset=yard&engine=gpu'))).toThrow('harness query: engine=gpu is not one of worker, inline, mock');
    expect(() => configFromQuery(q('dataset=yard&quality=ultra'))).toThrow(/quality=ultra is not one of fast, balanced, max/);
    expect(() => configFromQuery(q('dataset=yard&mode=room'))).toThrow(/mode=room is not one of/);
    expect(() => configFromQuery(q('dataset=yard&headingNorth=grid'))).toThrow(/headingNorth=grid is not one of/);
    expect(() => configFromQuery(q('dataset=yard&engine='))).toThrow(/engine= is not one of/);
    expect(() => configFromQuery(q('dataset=yard&maxFrames=4o'))).toThrow('harness query: maxFrames=4o is not a number');
    expect(() => configFromQuery(q('dataset=yard&stride='))).toThrow(/stride= is not a number/);
    expect(() => configFromQuery(q('dataset=yard&mockSpeed=Infinity'))).toThrow(/mockSpeed=Infinity is not a number/);
    // the JSON config is not re-validated here (the runner built it), but a bad single key on top still throws
    expect(() => configFromQuery(q(`config=${encodeURIComponent('{"dataset":"object"}')}&engine=Mock`))).toThrow(/engine=Mock/);
  });

  it('splat, splatIterations (implies splat) and engineConfig JSON', () => {
    expect(configFromQuery(q('dataset=object-real&splat=1'))!.splat).toBe(true);
    expect(configFromQuery(q('dataset=object-real&splat=0'))!.splat).toBe(false);
    expect(configFromQuery(q('dataset=object-real&splatIterations=500'))!.splat).toEqual({ iterations: 500 });
    expect(configFromQuery(q('dataset=object-real&splat=0&splatIterations=500'))!.splat).toEqual({ iterations: 500 });
    expect(() => configFromQuery(q('dataset=object-real&splatIterations=0'))).toThrow(/splatIterations=0 must be an integer ≥ 1/);
    expect(() => configFromQuery(q('dataset=object-real&splatIterations=2.5'))).toThrow(/splatIterations=2.5/);
    const ec = configFromQuery(q(`dataset=object&engineConfig=${encodeURIComponent('{"featureBackend":"cpu","mapping":"inline"}')}`))!;
    expect(ec.engineConfig).toEqual({ featureBackend: 'cpu', mapping: 'inline' });
    expect(() => configFromQuery(q('dataset=object&engineConfig={bad'))).toThrow(/engineConfig is not JSON/);
    expect(() => configFromQuery(q('dataset=object&engineConfig=[1]'))).toThrow(/JSON object/);
  });

  it('config JSON carries the whole HarnessConfig; single keys override it', () => {
    const cfg: HarnessConfig = {
      dataset: 'object-real',
      engine: 'worker',
      quality: 'fast',
      densify: { maxImageSize: 480 },
      mesh: true,
      splat: { iterations: 300, maxGaussians: 50_000 },
      exports: ['ply-splat'],
      engineConfig: { featureBackend: 'cpu' },
      timeouts: { frameMs: 600_000 },
      logFrames: false,
      output: { prefix: 'object-real-phone/' },
    };
    const back = configFromQuery(new URL(`https://x${harnessPath('abcdef0123456789', cfg)}`).searchParams);
    expect(back).toEqual(cfg);
    const over = configFromQuery(q(`config=${encodeURIComponent(JSON.stringify(cfg))}&engine=mock&splatIterations=50&engineConfig=${encodeURIComponent('{"mapping":"inline"}')}`))!;
    expect(over.engine).toBe('mock');
    expect(over.splat).toEqual({ iterations: 50, maxGaussians: 50_000 });
    expect(over.engineConfig).toEqual({ featureBackend: 'cpu', mapping: 'inline' });
    expect(over.densify).toEqual({ maxImageSize: 480 });
    // a dataset key wins over the JSON's
    expect(configFromQuery(q(`config=${encodeURIComponent(JSON.stringify(cfg))}&dataset=yard`))!.dataset).toBe('yard');
    expect(() => configFromQuery(q('config=nope'))).toThrow(/config is not JSON/);
    expect(configFromQuery(q('config={}'))).toBeNull();
  });
});

describe('bridgeFromQuery', () => {
  it('bridge=1 needs a run id', () => {
    expect(bridgeFromQuery(q('dataset=object'))).toBeNull();
    expect(bridgeFromQuery(q('bridge=0&run=0123456789abcdef'))).toBeNull();
    expect(bridgeFromQuery(q('bridge=1&run=0123456789abcdef'))).toEqual({ run: '0123456789abcdef' });
    expect(() => bridgeFromQuery(q('bridge=1'))).toThrow(/run=<id>/);
    expect(() => bridgeFromQuery(q('bridge=1&run=../x'))).toThrow(/run=<id>/);
    const p = new URL(`https://x${harnessPath('feedfacecafebeef', { dataset: 'object' })}`).searchParams;
    expect(bridgeFromQuery(p)).toEqual({ run: 'feedfacecafebeef' });
  });
});
