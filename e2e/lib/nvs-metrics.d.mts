type RGBA8 = Uint8Array | Uint8ClampedArray;
type Mask = Uint8Array | null;
export const SSIM_RADIUS: number;
export const SSIM_SIGMA: number;
export const AFFINE_IDENTITY: number[];
export function gaussianKernel(radius?: number, sigma?: number): Float64Array;
export function ssim(a: RGBA8, b: RGBA8, w: number, h: number, mask?: Mask, cache?: Map<Mask, unknown> | null): number;
export function mse(a: RGBA8, b: RGBA8, n: number, mask?: Mask): { mse: number; n: number };
export function psnrFromMse(mse: number): number;
export function psnr(a: RGBA8, b: RGBA8, n: number, mask?: Mask): number;
export function fitAffineColour(src: RGBA8, dst: RGBA8, n: number, mask?: Mask): number[];
export function applyAffineColour<T extends RGBA8>(img: T, n: number, M: ArrayLike<number>, mask?: Mask): T;
export function unpremultiply(render: RGBA8, n: number): { straight: Uint8ClampedArray; alpha: Float32Array; covered: Uint8Array };
export function compositeOver(straight: RGBA8, alpha: Float32Array, n: number, bg?: number[]): Uint8ClampedArray;
export function validMask(gt: RGBA8, n: number): Uint8Array;
export interface PreparedGroundTruth { data: RGBA8; valid: Uint8Array; nValid: number; ssimCache: Map<Mask, unknown> }
export function prepareGroundTruth(gt: RGBA8, w: number, h: number): PreparedGroundTruth;
export function alignmentShift(render: RGBA8, renderMask: Mask, gt: RGBA8, gtMask: Mask, w: number, h: number, radius?: number): { dx: number; dy: number } | null;
export interface Score { psnr: number; ssim: number | null }
export interface ViewScore {
  coverage: number;
  validPixels: number;
  full: Score;
  covered: Score;
  cc: { full: Score; covered: Score; affine: number[] };
  shift: { dx: number; dy: number } | null;
  images?: { full: Uint8ClampedArray; ccFull: Uint8ClampedArray };
}
export function scoreView(
  render: RGBA8,
  gt: RGBA8 | PreparedGroundTruth,
  w: number,
  h: number,
  opts?: { background?: number[]; ssim?: boolean; shift?: boolean; keepImages?: boolean },
): ViewScore;
export function triptych(gt: RGBA8, full: RGBA8, w: number, h: number, gain?: number): Uint8ClampedArray;
export interface Stat { mean: number | null; median: number | null; min: number | null; max: number | null; n: number }
export interface ScoreSummary {
  views: number;
  coverage: Stat;
  full: { psnr: Stat; ssim: Stat };
  covered: { psnr: Stat; ssim: Stat };
  cc: { full: { psnr: Stat; ssim: Stat }; covered: { psnr: Stat; ssim: Stat } };
  shiftPx: Stat;
}
export function aggregateScores(scores: ViewScore[]): ScoreSummary;
