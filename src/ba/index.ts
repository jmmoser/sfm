/**
 * Bundle adjustment (docs/ARCHITECTURE.md §4.3): Levenberg–Marquardt with analytic
 * Jacobians of the RADIAL camera model, robust losses, Schur complement on points,
 * dense (envelope Cholesky) or block-Jacobi PCG reduced camera system, shared
 * intrinsics, gauge masks and pose priors.
 */
export * from './problem';
export * from './loss';
export * from './camera-model';
export * from './priors';
export * from './linear';
export * from './solver';
