import { it } from 'vitest';
import { Rng } from '../math/random';
import { cameraFromDevice, orbitTrajectory, simulateImu, simulateKeyframes } from './extras-testutil';
import { estimateInertialAlignment } from './inertial';

it('explore', () => {
  const variants: [string, any, any, any][] = [
    ['base', {}, {}, {}],
    ['noAccBias', { zeroBias: true }, {}, {}],
    ['noVisNoise', {}, { posNoise: 0, rotNoise: 0 }, {}],
    ['noImuNoise', { accelNoise: 0, gyroNoise: 0 }, {}, {}],
    ['allClean', { zeroBias: true, accelNoise: 0, gyroNoise: 0, gyroBias: [0,0,0] }, { posNoise: 0, rotNoise: 0 }, {}],
    ['bigJitter', { jit: [5, 12] }, {}, {}],
    ['prior0.1', {}, {}, { deviceBiasSigma: 0.1 }],
  ];
  for (const [name, io, ko, so] of variants) {
    const errs: number[] = [], z: number[] = [], ex: number[] = [];
    for (let seed = 100; seed < 120; seed++) {
      const rng = new Rng(seed);
      const traj = orbitTrajectory(rng, io.jit ? { posJitterMm: io.jit } : {});
      const Rcd = cameraFromDevice(0);
      let ab: [number, number, number] = [0.05 * rng.gaussian(), 0.05 * rng.gaussian(), 0.05 * rng.gaussian()];
      if (io.zeroBias) ab = [0, 0, 0];
      const imu = simulateImu(traj, Rcd, rng, { ...io, accelBias: ab, withAccel: seed % 2 === 0 });
      const kf = simulateKeyframes(traj, Rcd, rng, { offsetMs: -30, s: rng.uniform(0.2, 5), posNoise: 0.0005, ...ko });
      const al = estimateInertialAlignment(kf.keyframes, imu, { scale: { estimateDeviceBias: true, ...so } });
      const e = al.scale!.scale * kf.s - 1;
      errs.push(e); z.push(e / al.scale!.relUncertainty); ex.push(al.scale!.excitationRms);
    }
    const rmsE = Math.sqrt(errs.reduce((a, b) => a + b * b, 0) / errs.length);
    const maxE = Math.max(...errs.map(Math.abs));
    const meanE = errs.reduce((a, b) => a + b, 0) / errs.length;
    const rmsZ = Math.sqrt(z.reduce((a, b) => a + b * b, 0) / z.length);
    console.log(name, 'mean%', (100 * meanE).toFixed(2), 'rms%', (100 * rmsE).toFixed(2), 'max%', (100 * maxE).toFixed(2), 'rmsZ', rmsZ.toFixed(2), 'exc', (ex.reduce((a,b)=>a+b,0)/ex.length).toFixed(3));
  }
});
