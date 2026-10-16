import { test } from 'vitest';
import { generateScene, buildSyntheticDatabase, evaluateReconstruction } from './synthetic';
import { runIncrementalSfM } from './mapper';

for (const kind of ['orbit'] as const) {
test('explore hard ' + kind, async () => {
  const s = generateScene({ kind, numImages: kind === 'orbit' ? 60 : 100, seed: 3, k1: 0.03, noisePx: 0.7, outlierRatio: 0.25, descriptors: false });
  const guess = { ...s.camera, f: s.camera.f * 1.08, k1: 0 };
  const { db, stats } = buildSyntheticDatabase(s, { verify: true, camera: guess });
  console.log(JSON.stringify(stats));
  const t0 = performance.now();
  const models = await runIncrementalSfM(db, { intrinsics: guess, onModel: (r, t) => console.log(JSON.stringify(t)) });
  const t1 = performance.now();
  console.log(kind, 'models', models.length, 'time', (t1 - t0).toFixed(0));
  for (const m of models) {
    const ev = evaluateReconstruction(m, s);
    console.log('reg', m.numRegisteredImages, 'pts', m.numPoints, 'mre', m.meanReprojectionError().toFixed(3), 'mtl', m.meanTrackLength().toFixed(2), JSON.stringify({ ...ev, sim: undefined }));
  }
});
}
