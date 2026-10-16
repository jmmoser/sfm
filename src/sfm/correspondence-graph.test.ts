import { describe, expect, it } from 'vitest';
import { Rng } from '../math/random';
import { CorrespondenceGraph } from './correspondence-graph';

/** Brute-force reference: list of (imageA, featA, imageB, featB) for all pairs. */
function reference(pairs: Map<string, Uint32Array>, image: number, feature: number): string[] {
  const out: string[] = [];
  for (const [key, m] of pairs) {
    const [a, b] = key.split(',').map(Number);
    for (let k = 0; k < m.length; k += 2) {
      if (a === image && m[k] === feature) out.push(`${b}:${m[k + 1]}`);
      if (b === image && m[k + 1] === feature) out.push(`${a}:${m[k]}`);
    }
  }
  return out.sort();
}

describe('CorrespondenceGraph', () => {
  it('matches a brute-force reference under incremental growth, replacement and removal', () => {
    const rng = new Rng(7);
    const g = new CorrespondenceGraph();
    const nf = new Map<number, number>();
    const pairs = new Map<string, Uint32Array>(); // key "a,b" oriented as added
    const images: number[] = [];
    for (let step = 0; step < 60; step++) {
      // add an image
      const id = 3 * step + 1;
      const n = 20 + rng.int(30);
      g.addImage(id, n);
      nf.set(id, n);
      images.push(id);
      // connect it to a few previous images (both orientations)
      for (let q = 0; q < 3 && images.length > 1; q++) {
        const other = images[rng.int(images.length - 1)];
        const [a, b] = rng.next() < 0.5 ? [id, other] : [other, id];
        const m: number[] = [];
        const usedA = new Set<number>(),
          usedB = new Set<number>();
        const cnt = 1 + rng.int(15);
        for (let k = 0; k < cnt; k++) {
          const fa = rng.int(nf.get(a)!),
            fb = rng.int(nf.get(b)!);
          if (usedA.has(fa) || usedB.has(fb)) continue;
          usedA.add(fa);
          usedB.add(fb);
          m.push(fa, fb);
        }
        // remove previous entry of the same pair (any orientation) from the reference
        pairs.delete(`${a},${b}`);
        pairs.delete(`${b},${a}`);
        pairs.set(`${a},${b}`, Uint32Array.from(m));
        g.addTwoView(a, b, Uint32Array.from(m));
      }
      // occasionally remove a pair
      if (step % 7 === 3 && pairs.size > 0) {
        const keys = [...pairs.keys()];
        const key = keys[rng.int(keys.length)];
        const [a, b] = key.split(',').map(Number);
        pairs.delete(key);
        g.removeTwoView(b, a);
      }
    }
    const outI = new Int32Array(64),
      outF = new Int32Array(64);
    let total = 0;
    for (const id of images) {
      let nCorr = 0,
        nObs = 0;
      const heads = g.heads(id)!;
      for (let f = 0; f < nf.get(id)!; f++) {
        const ref = reference(pairs, id, f);
        const got = g
          .correspondences(id, f)
          .map((c) => `${c.image}:${c.feature}`)
          .sort();
        expect(got).toEqual(ref);
        // typed-array API
        const c = g.getCorrespondences(id, f, outI, outF);
        expect(c).toBe(ref.length);
        const got2: string[] = [];
        for (let k = 0; k < c; k++) got2.push(`${outI[k]}:${outF[k]}`);
        expect(got2.sort()).toEqual(ref);
        // raw iteration
        let it = 0;
        for (let e = heads[f]; e >= 0; e = g.nextOf[e]) it++;
        expect(it).toBe(ref.length);
        expect(g.numCorrespondencesOf(id, f)).toBe(ref.length);
        expect(g.isTwoViewObservation(id, f)).toBe(ref.length === 1);
        nCorr += ref.length;
        if (ref.length > 0) nObs++;
      }
      expect(g.numCorrespondences(id)).toBe(nCorr);
      expect(g.numObservations(id)).toBe(nObs);
      total += nCorr;
    }
    // pair statistics
    let sum = 0;
    for (const [a, b, cnt] of g.imagePairs()) {
      expect(a).toBeLessThan(b);
      const m = pairs.get(`${a},${b}`) ?? pairs.get(`${b},${a}`)!;
      expect(cnt).toBe(m.length / 2);
      expect(g.numCorrespondencesBetween(b, a)).toBe(cnt);
      sum += 2 * cnt;
      expect(g.neighbours(a).get(b)).toBe(cnt);
      expect(g.neighbours(b).get(a)).toBe(cnt);
    }
    expect(sum).toBe(total);
    expect(g.numPairs).toBe(pairs.size);
  });

  it('matchesBetween is oriented', () => {
    const g = new CorrespondenceGraph();
    g.addImage(4, 10);
    g.addImage(2, 10);
    g.addTwoView(4, 2, Uint32Array.of(1, 7, 3, 9));
    expect(Array.from(g.matchesBetween(4, 2))).toEqual([1, 7, 3, 9]);
    expect(Array.from(g.matchesBetween(2, 4))).toEqual([7, 1, 9, 3]);
    expect(g.matchesBetween(2, 5).length).toBe(0);
    expect(g.correspondences(2, 7)).toEqual([{ image: 4, feature: 1 }]);
    expect(g.correspondences(99, 0)).toEqual([]);
    expect(g.correspondences(2, 100)).toEqual([]);
    expect(g.getCorrespondences(2, 7, new Int32Array(0), new Int32Array(0))).toBe(1);
  });

  it('validates input', () => {
    const g = new CorrespondenceGraph();
    g.addImage(0, 5);
    g.addImage(1, 5);
    g.addImage(0, 5); // idempotent
    expect(() => g.addImage(0, 6)).toThrow(RangeError);
    expect(() => g.addTwoView(0, 2, Uint32Array.of(0, 0))).toThrow(RangeError);
    expect(() => g.addTwoView(0, 0, Uint32Array.of(0, 0))).toThrow(RangeError);
    expect(() => g.addTwoView(0, 1, Uint32Array.of(5, 0))).toThrow(RangeError);
    expect(() => g.addTwoView(0, 1, Uint32Array.of(0, 5))).toThrow(RangeError);
    g.addTwoView(0, 1, new Uint32Array(0));
    expect(g.numPairs).toBe(0);
    expect(g.imageIds()).toEqual([0, 1]);
  });

  it('grows its pool without invalidating content (large graph)', () => {
    const g = new CorrespondenceGraph();
    const n = 3000;
    for (let i = 0; i < 20; i++) g.addImage(i, n);
    const m = new Uint32Array(2 * n);
    for (let k = 0; k < n; k++) {
      m[2 * k] = k;
      m[2 * k + 1] = (k * 7) % n;
    }
    for (let i = 0; i < 19; i++) g.addTwoView(i, i + 1, m);
    expect(g.numCorrespondences(5)).toBe(2 * n);
    // image 5 feature 10: from (4,5) 7k ≡ 10 (mod 3000) ⇒ k = 430; from (5,6) → 70
    expect(g.correspondences(5, 10).map((c) => `${c.image}:${c.feature}`).sort()).toEqual(['4:430', '6:70']);
    // feature 10 of image 6 ← feature k of 5 with 7k ≡ 10 (mod 3000): k = 430 since 7·430 = 3010
    expect(g.correspondences(6, 10).some((c) => c.image === 5 && c.feature === 430)).toBe(true);
  });
});
