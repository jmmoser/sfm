/**
 * Structure-from-Motion (docs/ARCHITECTURE.md §4.5): feature database, incremental
 * correspondence graph, reconstruction with track bookkeeping, COLMAP-style
 * incremental triangulator / mapper, live keyframe mapper.
 *
 * `./synthetic` (deterministic synthetic SfM scenes for tests) is intentionally not
 * re-exported — import it directly from tests.
 */
// ---- sfm core (database, graph, reconstruction, triangulation, mapping)
export * from './database';
export * from './correspondence-graph';
export * from './reconstruction';
export * from './triangulator';
export * from './mapper';
export * from './live';

// ---- sfm extras (keyframes, pairs, vocab, inertial, geo, world) — other module owner appends below
// (append-only block: add `export * from './keyframes';` etc. here)
export * from './keyframes';
export * from './pairs';
export * from './vocab';
export * from './inertial';
export * from './geo';
export * from './world';
