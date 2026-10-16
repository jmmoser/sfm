export class KdTree {
  constructor(points: Float32Array | Float64Array);
  readonly n: number;
  nearest(x: number, y: number, z: number, maxDist?: number): { index: number; dist: number };
}
