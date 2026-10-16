import { describe, expect, it } from 'vitest';
import { CliError, parseArgs, sceneConfig } from '../../../e2e/lib/synth-cli.mjs';
import { resolveConfig, type SynthConfigInput } from './config';

describe('generate-synth CLI', () => {
  it('scene selection', () => {
    expect(parseArgs([]).scenes).toEqual(['object', 'yard']);
    expect(parseArgs(['yard']).scenes).toEqual(['yard']);
    expect(parseArgs(['all', 'object']).scenes).toEqual(['object', 'yard']);
    expect(() => parseArgs(['--frames'])).toThrow(CliError);
    expect(() => parseArgs(['--bogus'])).toThrow(CliError);
    expect(() => parseArgs(['--size', '12'])).toThrow(CliError);
  });

  it('frame-count semantics', () => {
    // --frames alone: spread over the default duration
    let c = sceneConfig('object', parseArgs(['object', '--frames', '40']));
    expect([c.frames, c.duration, c.fps]).toEqual([40, 25, 1.6]);
    // --frames + --fps: duration = N / fps
    c = sceneConfig('yard', parseArgs(['yard', '--frames', '30', '--fps', '6']));
    expect([c.frames, c.duration, c.fps]).toEqual([30, 5, 6]);
    // --fps alone: N = fps · default duration
    c = sceneConfig('yard', parseArgs(['yard', '--fps', '2']));
    expect([c.frames, c.duration, c.fps]).toEqual([80, 40, 2]);
    // defaults
    c = sceneConfig('object', parseArgs(['object']));
    expect([c.frames, c.duration, c.fps]).toEqual([200, 25, 8]);
  });

  it('options map onto SynthConfig (incl. --set dotted paths) and resolve cleanly', () => {
    const warnings: string[] = [];
    const o = parseArgs(['all', '--name', 'x', '--size', '960x720', '--blur', '3', '--no-noise', '--set', 'imu.gyroBias=[0,0,0]', '--set', 'gps=null', '--seed', '9']);
    const c = sceneConfig('object', o, (m) => warnings.push(m));
    expect(c.name).toBe('x-object');
    expect(warnings.length).toBe(1); // landscape size for the portrait scene
    const r = resolveConfig(c as unknown as SynthConfigInput);
    expect(r.effects.motionBlurSamples).toBe(3);
    expect(r.effects.noise).toBe(false);
    expect(r.imu.gyroBias).toEqual([0, 0, 0]);
    expect(r.imu.gyroNoise).toBe(0.004);
    expect(r.seed).toBe(9);
    const y = resolveConfig(sceneConfig('yard', o) as unknown as SynthConfigInput);
    expect(y.gps).toBeNull();
    expect(y.width).toBe(960);
  });
});
