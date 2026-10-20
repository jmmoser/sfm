// CPU helper pool (browser: real workers): the verification and quantisation jobs give exactly the
// results of the engine thread, and a disposed pool rejects its jobs.
import { describe, expect, it } from 'vitest';
import { estimateTwoViewGeometry } from '../geometry/twoview';
import { buildSyntheticDatabase, generateScene } from '../sfm/synthetic';
import { VocabularyIndex } from '../sfm/vocab';
import { Rng } from '../math/random';
import { CpuPool } from './cpu-pool';

describe('CpuPool', () => {
  it('verifies pairs and quantises keyframes exactly like the engine thread', async () => {
    const pool = CpuPool.create(2);
    expect(pool).not.toBeNull();
    try {
      // two-view verification of synthetic putative matches (25 % outliers)
      const scene = generateScene({ kind: 'orbit', numImages: 6, seed: 3, noisePx: 0.5, outlierRatio: 0.25, descriptors: false });
      const { db } = buildSyntheticDatabase(scene, { verify: false });
      const cam = scene.camera;
      const jobs: Promise<void>[] = [];
      let n = 0;
      for (const [a, b, g] of db.pairs()) {
        const kp1 = db.features.get(a)!.keypoints,
          kp2 = db.features.get(b)!.keypoints;
        const opts = { maxIterations: 500 };
        const here = estimateTwoViewGeometry(kp1, kp2, g.inliers, cam, cam, opts);
        jobs.push(
          pool!.verify(kp1, kp2, g.inliers, cam, cam, opts).then(({ geometry, ms }) => {
            expect(ms).toBeGreaterThanOrEqual(0);
            expect(geometry.config).toBe(here.config);
            expect(Array.from(geometry.inliers)).toEqual(Array.from(here.inliers));
            if (here.R) expect(Array.from(geometry.R!)).toEqual(Array.from(here.R));
          }),
        );
        n++;
      }
      expect(n).toBeGreaterThan(3);
      await Promise.all(jobs);
      // visual words
      const rng = new Rng(7);
      const sets = Array.from({ length: 5 }, () => {
        const count = 300;
        const descriptors = Uint8Array.from({ length: 128 * count }, () => Math.floor(rng.next() * 120));
        return { count, descriptors, keypoints: new Float32Array(4 * count), scores: Float32Array.from({ length: count }, () => rng.next()), width: 640, height: 480 };
      });
      const vocab = VocabularyIndex.train(sets, { numWords: 32, maxFeaturesPerImage: 200 });
      const words = await Promise.all(sets.map((fs) => pool!.quantize(vocab, fs)));
      sets.forEach((fs, k) => expect(Array.from(words[k])).toEqual(Array.from(vocab.quantize(fs))));
    } finally {
      pool!.dispose();
    }
    expect(pool!.size).toBe(0);
    await expect(pool!.quantize(VocabularyIndex.train([{ count: 40, descriptors: new Uint8Array(40 * 128).fill(3) }], { numWords: 4 }), { count: 1, descriptors: new Uint8Array(128), keypoints: new Float32Array(4), scores: new Float32Array(1), width: 1, height: 1 })).rejects.toThrow();
  });
});
