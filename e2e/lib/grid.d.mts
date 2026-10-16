export class HashGrid {
  constructor(points: Float32Array | Float64Array, cellSize: number);
  readonly n: number;
  readonly cell: number;
  nearest(x: number, y: number, z: number, maxDist: number): { index: number; dist: number };
  hasWithin(x: number, y: number, z: number, r: number): boolean;
}
