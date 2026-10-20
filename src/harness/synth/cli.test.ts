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
    // --duration 0 would give fps = Infinity (every frame at t = 0), a negative one a single frame
    expect(parseArgs(['--duration', '10']).duration).toBe(10);
    expect(() => parseArgs(['object', '--frames', '50', '--duration', '0'])).toThrow(CliError);
    expect(() => parseArgs(['object', '--duration', '-5'])).toThrow(CliError);
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

  it('real-capture presets and artefact flags', () => {
    expect(parseArgs(['real']).scenes).toEqual(['object-real', 'yard-real']);
    expect(parseArgs(['object-real', 'yard']).scenes).toEqual(['object-real', 'yard']);
    let c = sceneConfig('object-real', parseArgs(['object-real']));
    expect([c.scene, c.preset, c.name, c.frames, c.fps, c.duration]).toEqual(['object', 'object-real', 'object-real', 200, 8, 25]);
    let r = resolveConfig(c as unknown as SynthConfigInput);
    expect([r.rsReadoutMs, r.jpegQuality, r.width, r.height]).toEqual([22, 0.85, 720, 960]);
    c = sceneConfig('yard-real', parseArgs(['real', '--name', 'abl', '--rs-readout', '0', '--focus-breathing', '0', '--set', 'trajectory.shake.pans=[]']));
    expect(c.name).toBe('abl-yard-real');
    r = resolveConfig(c as unknown as SynthConfigInput);
    expect([r.scene, r.rsReadoutMs, r.focusBreathing, r.whiteBalanceDrift, r.fps, r.frames]).toEqual(['yard', 0, null, 0.03, 6, 240]);
    expect(r.trajectory.shake?.pans).toEqual([]);
    expect(r.trajectory.shake?.tremorDeg).toBeGreaterThan(0);
    // flags on a clean scene
    c = sceneConfig('object', parseArgs(['object', '--rs-readout', '30', '--rs-bands', '24', '--focus-breathing', '0.004', '--wb-drift', '0.02', '--exposure-jitter', '0.1', '--jpeg-quality', '0.8', '--adaptive-blur', '1.5', '--blur', '8']));
    r = resolveConfig(c as unknown as SynthConfigInput);
    expect([r.preset, r.rsReadoutMs, r.rsBands, r.focusBreathing?.amplitude, r.whiteBalanceDrift, r.exposureJitter, r.jpegQuality]).toEqual([null, 30, 24, 0.004, 0.02, 0.1, 0.8]);
    expect([r.effects.motionBlurPxPerSample, r.effects.motionBlurSamples]).toEqual([1.5, 8]);
    expect(() => parseArgs(['--rs-readout', '-1'])).toThrow(CliError);
    expect(() => parseArgs(['--jpeg-quality', '2'])).toThrow(CliError);
  });
});
