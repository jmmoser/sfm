/**
 * Compass heading simulation.
 *
 * `heading`: degrees clockwise from TRUE north (declination already removed) of the
 * camera viewing direction (optical axis, +z_cam) projected onto the horizontal plane,
 * in [0, 360). This is what iOS `webkitCompassHeading` approximates while the phone is
 * held upright (it follows the rear camera direction). Noise: white Gaussian σ =
 * `sigmaDeg` plus an optional constant bias; `accuracy` = sigmaDeg (like
 * `webkitCompassAccuracy`, ± degrees).
 * `headingDeviceTop`: same for the device +y axis (top edge), the flat-phone
 * definition; null when that axis is within ~11.5° of vertical.
 */
import { headingDegOfDirection, poseFromDevice } from './conventions';
import { SynthRng } from './rng';
import type { Trajectory } from './trajectory';

export interface HeadingSample {
  t: number;
  heading: number;
  accuracy: number;
  headingDeviceTop: number | null;
  /** Noise-free camera heading (ground truth, not available on a device). */
  trueHeading: number;
}

export interface HeadingConfig {
  rateHz: number;
  sigmaDeg: number;
  biasDeg: number;
  seed: number;
}

export const DEFAULT_HEADING_CONFIG: HeadingConfig = { rateHz: 10, sigmaDeg: 5, biasDeg: 0, seed: 13 };

function wrap360(d: number): number {
  d %= 360;
  return d < 0 ? d + 360 : d;
}

export function simulateHeadings(
  traj: Trajectory,
  startS: number,
  endS: number,
  t0Ms: number,
  config: Partial<HeadingConfig> = {},
): HeadingSample[] {
  const c = { ...DEFAULT_HEADING_CONFIG, ...config };
  const rng = new SynthRng(c.seed).derive('heading');
  const out: HeadingSample[] = [];
  const n = Math.floor((endS - startS) * c.rateHz + 1e-9) + 1;
  for (let k = 0; k < n; k++) {
    const tau = startS + k / c.rateHz;
    const s = traj.evaluate(tau);
    const pose = poseFromDevice(s.R_wd, s.position, traj.screenAngle);
    const fwd = [pose.R[6], pose.R[7], pose.R[8]];
    const trueHeading = headingDegOfDirection(fwd, 1e-3);
    const top = [s.R_wd[1], s.R_wd[4], s.R_wd[7]]; // device +y in world = column 1 of R_wd
    const topHeading = headingDegOfDirection(top, 0.2);
    const noise = c.biasDeg + c.sigmaDeg * rng.gaussian();
    if (!Number.isFinite(trueHeading)) continue;
    out.push({
      t: t0Ms + tau * 1000,
      heading: wrap360(trueHeading + noise),
      accuracy: c.sigmaDeg,
      headingDeviceTop: Number.isFinite(topHeading) ? wrap360(topHeading + noise) : null,
      trueHeading,
    });
  }
  return out;
}
