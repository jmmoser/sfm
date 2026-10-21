/**
 * Geometry module: RADIAL camera model, poses, LO-MSAC, minimal and linear solvers
 * (5-point E, 7/8-point F, DLT H, Lambda Twist P3P, EPnP), triangulation, two-view
 * verification and robust Sim3 alignment. See docs/ARCHITECTURE.md §4.2.
 *
 * Solvers use module-level scratch buffers: all functions are synchronous and
 * non-reentrant (fine in workers — each worker has its own module instance).
 * Accuracy / timing measurements: docs/research/solvers.md.
 */
export * from './camera';
export * from './pose';
export * from './ransac';
export * from './essential';
export * from './fundamental';
export * from './homography';
export * from './pnp';
export * from './triangulation';
export * from './twoview';
export * from './align';
export { eigenvaluesNonsymmetric, eigenvectorInverseIteration } from './eigen';
