// Path confinement for files the browser asks Node to write (pure; unit-tested in
// src/harness/engine/e2e-bridge.test.ts). Used by the Playwright bindings (./e2e-browser.mjs,
// e2e/nvs.mjs) and the HTTP bridge (./e2e-bridge.mjs).
import { isAbsolute, resolve, sep } from 'node:path';

/** Resolve `rel` inside `root`; throws for absolute paths and `..` escapes. */
export function safeJoin(root, rel) {
  if (typeof rel !== 'string' || !rel || isAbsolute(rel) || rel.split(/[\\/]/).includes('..')) throw new Error(`bad path ${rel}`);
  const file = resolve(root, rel);
  if (!file.startsWith(resolve(root) + sep)) throw new Error(`bad path ${rel}`);
  return file;
}
