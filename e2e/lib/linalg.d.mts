export type M3 = number[];
export type V3 = number[];
export interface Sim3 { s: number; R: M3; t: V3 }
export function mat3Mul(a: ArrayLike<number>, b: ArrayLike<number>): M3;
export function mat3T(a: ArrayLike<number>): M3;
export function mat3Vec(a: ArrayLike<number>, v: ArrayLike<number>): V3;
export function mat3Det(a: ArrayLike<number>): number;
export function cross(a: ArrayLike<number>, b: ArrayLike<number>): V3;
export function norm(a: ArrayLike<number>): number;
export function rotationAngleBetween(a: ArrayLike<number>, b: ArrayLike<number>): number;
export function axisAngle(axis: ArrayLike<number>, angle: number): M3;
export function svd3(A: ArrayLike<number>): { U: M3; S: number[]; V: M3 };
export function projectToSO3(M: ArrayLike<number>): M3;
export function chordalMean(Rs: ArrayLike<number>[]): M3;
export function umeyama(src: ArrayLike<number>[], dst: ArrayLike<number>[], withScale?: boolean): Sim3 | null;
export function applySim3(sim: Sim3, p: ArrayLike<number>): V3;
export function percentile(values: ArrayLike<number>, p: number): number;
export function mean(values: ArrayLike<number>): number;
export function rms(values: ArrayLike<number>): number;
