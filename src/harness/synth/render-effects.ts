/** Realism effects applied by the frame renderer (render.ts). Pure data. */
export interface RenderEffects {
  /** Render-target resolution multiplier and sub-pixel samples per axis (≥ 1). Default 2. */
  supersample: number;
  /** Sensor noise on/off. */
  noise: boolean;
  /** Shot-noise variance per unit linear signal. Default 1.2e-4. */
  shotNoise: number;
  /** Read-noise variance (linear units²). Default 4e-6. */
  readNoise: number;
  /** Corner light fall-off (0 = none). Default 0.18 (corners at 82 %). */
  vignetting: number;
  /**
   * Number of sub-frame renders averaged for motion blur (1 = off). With
   * motionBlurPxPerSample > 0 it is the MAXIMUM of the adaptive count.
   */
  motionBlurSamples: number;
  /**
   * Adaptive motion blur: sub-frames per frame (or rolling-shutter band) = 1 + ⌈image motion
   * over the exposure / this many px⌉ (1 below 0.25 px), at most motionBlurSamples; the
   * sub-frames sit at the midpoints of equal slices of the exposure. 0 = fixed count
   * (default; sub-frames at both ends and evenly in between).
   */
  motionBlurPxPerSample: number;
  /** Exposure time (ms) spanned by the motion-blur sub-frames. */
  exposureMs: number;
}

export const DEFAULT_RENDER_EFFECTS: RenderEffects = {
  supersample: 2,
  noise: true,
  shotNoise: 1.2e-4,
  readNoise: 4e-6,
  vignetting: 0.18,
  motionBlurSamples: 1,
  motionBlurPxPerSample: 0,
  exposureMs: 8,
};
