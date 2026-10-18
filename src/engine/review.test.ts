/**
 * Integration-review regressions (engine), the parts that need no reconstruction: raw sensor
 * delivery as a session property (incl. a session that starts in a landscape-270 UI), the
 * feature pipeline hand-over, the DENSE_PRESETS precedence of flat fields and the
 * low-excitation IMU scale acceptance. The engine runs are in review.slow.test.ts.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import type { FeaturePipeline } from '../features/extractor';
import type { WorldAlignment } from '../sfm/world';
import { createCpuFeaturePipeline } from './cpu-pipeline';
import { PhotogrammetryEngine, pipelineKey } from './engine';
import { OrientationTracker } from './frame-ingest';
import { MemoryEngineStore } from './persistence';
import { parseDensePresetEntry } from './presets';
import { lowExcitationRetry, lowExcitationSigma } from './refine';
import type { EngineConfig } from './types';

let pipeline: FeaturePipeline;
beforeAll(() => {
  pipeline = createCpuFeaturePipeline({ maxFeatures: 2048, maxImageSize: 1024 });
});

describe('raw sensor delivery is a property of the session', () => {
  it('landscape frames of a landscape-270 UI are raw once raw delivery was seen (upright for 90, not 270)', () => {
    const o = new OrientationTracker(500);
    // portrait UI, landscape bitmap: raw buffer
    expect(o.classify({ width: 640, height: 480, screenAngle: 0, t: 0 })).toMatchObject({ ok: true, effectiveTheta: 90, k: 0, raw: true });
    expect(o.rawDelivery).toBe(true);
    // the user turns the phone to landscape-270: the buffer still has the sensor orientation
    o.classify({ width: 640, height: 480, screenAngle: 270, t: 100 });
    expect(o.classify({ width: 640, height: 480, screenAngle: 270, t: 700 })).toMatchObject({ ok: true, effectiveTheta: 90, k: 0, raw: true });
    // landscape-90 is upright for 90 either way
    o.classify({ width: 640, height: 480, screenAngle: 90, t: 800 });
    expect(o.classify({ width: 640, height: 480, screenAngle: 90, t: 1400 })).toMatchObject({ ok: true, effectiveTheta: 90, k: 0 });
  });

  it('a session started in landscape-270 is re-interpreted as θ₀ = 90 when raw buffers show up in portrait', () => {
    const o = new OrientationTracker(500);
    expect(o.classify({ width: 640, height: 480, screenAngle: 270, t: 0 })).toMatchObject({ ok: true, effectiveTheta: 270, k: 0, raw: false });
    expect(o.thetaSession).toBe(270);
    expect(o.sessionAmbiguous).toBe(true);
    o.classify({ width: 640, height: 480, screenAngle: 0, t: 100 });
    // raw buffer in a portrait UI: the first (unrotated) frames were raw too ⇒ θ₀ = 90, no turn
    expect(o.classify({ width: 640, height: 480, screenAngle: 0, t: 700 })).toMatchObject({ ok: true, effectiveTheta: 90, k: 0, raw: true });
    expect(o.thetaSession).toBe(90);
    expect(o.sessionAmbiguous).toBe(false);
  });

  it('upright delivery seen first: landscape-270 frames stay upright for 270; θ₀ = 270 stays when upright portrait frames follow', () => {
    const a = new OrientationTracker(500);
    a.classify({ width: 480, height: 640, screenAngle: 0, t: 0 });
    expect(a.rawDelivery).toBe(false);
    a.classify({ width: 640, height: 480, screenAngle: 270, t: 100 });
    expect(a.classify({ width: 640, height: 480, screenAngle: 270, t: 700 })).toMatchObject({ ok: true, effectiveTheta: 270, k: 3, raw: false });
    const b = new OrientationTracker(500);
    b.classify({ width: 640, height: 480, screenAngle: 270, t: 0 });
    b.classify({ width: 480, height: 640, screenAngle: 0, t: 100 });
    expect(b.classify({ width: 480, height: 640, screenAngle: 0, t: 700 })).toMatchObject({ ok: true, effectiveTheta: 0, k: 1, raw: false });
    expect(b.thetaSession).toBe(270);
  });
});

describe('feature pipeline hand-over (worker re-init)', () => {
  it('releasePipeline reports the SIFT configuration the pipeline was created for, not the loaded session’s quality', async () => {
    const store = new MemoryEngineStore('quality');
    await store.ensureSession({ mode: 'object', quality: 'balanced' });
    await store.putValue('engine/state', {
      version: 1,
      mode: 'object',
      quality: 'balanced',
      hfovPriorDeg: null,
      thetaSession: 0,
      cameraTheta: 0,
      cameraThetaVerified: false,
      resolvedMode: null,
      nextKeyframeId: 0,
      frameCounter: 0,
      imuChunks: 0,
      captureFinished: true,
      vocabTrainedOn: null,
      hasSparse: false,
      hasDense: false,
      hasMesh: false,
      timeOffsetMs: null,
      updatedAt: 0,
    });
    // the worker hands a spare pipeline created for 'fast' to the engine it builds for loadSession
    const config: EngineConfig = { pipeline, store };
    const e = new PhotogrammetryEngine({ mode: 'auto', quality: 'fast', persist: true }, config);
    await e.loadSession('quality');
    expect(e.options.quality).toBe('balanced');
    // re-processing the session adopts the pipeline (refine fails later: no keyframes)
    await expect(e.refine()).rejects.toThrow(/at least 2 keyframes/);
    const rel = e.releasePipeline();
    expect(rel?.pipeline).toBe(pipeline);
    expect(rel?.key).toBe(pipelineKey('fast', true, config));
    e.dispose();
  });
});

describe('DENSE_PRESETS entry parsing', () => {
  it('flat fields win over values derived from the nested blocks, whatever the key order', () => {
    // src/dense/presets.ts order: flat fields first, option blocks last
    const q = parseDensePresetEntry({ fusionMinViewsObject: 3, fusionMinViewsScene: 2, iterations: 4, patchMatch: { iterations: 9 }, fusion: { minViews: 2 } });
    expect(q).toMatchObject({ fusionMinViewsObject: 3, fusionMinViewsScene: 2, iterations: 4 });
    const p = parseDensePresetEntry({
      fusion: { minViews: 2, maxNeighbours: 10, maxNormalErrorDeg: 15 },
      fusionMinViewsObject: 3,
      fusionMinViewsScene: 2,
      patchMatch: { iterations: 9 },
      iterations: 4,
      tsdf: { maxBlocks: 123 },
    });
    expect(p.fusionMinViewsObject).toBe(3);
    expect(p.fusionMinViewsScene).toBe(2);
    expect(p.iterations).toBe(4);
    // only nested: taken
    expect(p.fusionMaxNeighbours).toBe(10);
    expect(p.tsdfMaxBlocks).toBe(123);
    expect(p.fusion).toEqual({ minViews: 2, maxNeighbours: 10, maxNormalErrorDeg: 15 });
    // nested-only entry
    expect(parseDensePresetEntry({ fusion: { minViews: 2 } })).toMatchObject({ fusionMinViewsObject: 2, fusionMinViewsScene: 2 });
  });
});

describe('low-excitation IMU scale acceptance', () => {
  const alignment = (excitationRms: number, r2: number, relUncertainty: number, reason = `insufficient excitation (${excitationRms} m/s² < 0.08)`): WorldAlignment =>
    ({
      metric: { metric: false, source: 'none', relativeUncertainty: null, gravityAligned: true, northAligned: false, originLatLonAlt: null },
      details: { imuScale: { reliable: false, reason, excitationRms, r2, relUncertainty, scale: 0.4 } },
    }) as unknown as WorldAlignment;

  it('R² decides at keyframe-sampling-limited excitation; σ covers the attenuation bound 1 − R²', () => {
    // e2e object, every 3rd frame (53 keyframes): excitation 0.029, R² 0.977, σ 1.2 % (error 1.8 %)
    expect(lowExcitationRetry(alignment(0.029, 0.977, 0.0123))).toMatchObject({ excitation: 0.029, r2: 0.977 });
    // the full-rate case keeps working
    expect(lowExcitationRetry(alignment(0.058, 0.988, 0.0078))).not.toBeNull();
    // R² 0.9 only with ≥ 0.03 m/s²
    expect(lowExcitationRetry(alignment(0.029, 0.93, 0.0123))).toBeNull();
    expect(lowExcitationRetry(alignment(0.031, 0.93, 0.0123))).not.toBeNull();
    // never below 0.015 m/s², never with σ > 3 % or another reason
    expect(lowExcitationRetry(alignment(0.012, 0.99, 0.01))).toBeNull();
    expect(lowExcitationRetry(alignment(0.029, 0.99, 0.035))).toBeNull();
    expect(lowExcitationRetry(alignment(0.029, 0.99, 0.01, 'insufficient excitation (0.029 < 0.08); poor fit (R² 0.2)'))).toBeNull();
    expect(lowExcitationSigma(0.0123, 0.977)).toBeCloseTo(0.0246, 6);
    expect(lowExcitationSigma(0.005, 0.96)).toBeCloseTo(0.04, 6);
    expect(lowExcitationSigma(0.003, 0.995)).toBe(0.02);
  });
});
