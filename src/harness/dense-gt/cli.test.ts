import { describe, expect, it } from 'vitest';
import { loadSubsetVisibility, parseArgs } from '../../../e2e/run-dense-gt.mjs';

// Node built-ins through an untyped dynamic import (the main tsconfig has no @types/node; see src/io/zip.test.ts).
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const nodeImport = (m: string): Promise<any> => import(/* @vite-ignore */ m);
const { mkdtempSync, rmSync, writeFileSync } = (await nodeImport('node:fs')) as {
  mkdtempSync(prefix: string): string;
  rmSync(path: string, opts: { recursive: boolean; force: boolean }): void;
  writeFileSync(path: string, data: Uint8Array): void;
};
const { tmpdir } = (await nodeImport('node:os')) as { tmpdir(): string };
const { join } = (await nodeImport('node:path')) as { join(...parts: string[]): string };

describe('run-dense-gt CLI', () => {
  it("'all' expands in place and keeps the other datasets", () => {
    expect(parseArgs([]).scenes).toEqual(['object', 'yard']);
    expect(parseArgs(['all']).scenes).toEqual(['object', 'yard']);
    expect(parseArgs(['all', 'object-real', 'yard-real']).scenes).toEqual(['object', 'yard', 'object-real', 'yard-real']);
    expect(parseArgs(['yard', 'all']).scenes).toEqual(['yard', 'object']);
  });

  it('--frames takes distinct frame indices ≥ 0', () => {
    expect(parseArgs(['object', '--frames', '0,20,40']).cfg.frames).toEqual([0, 20, 40]);
    expect(parseArgs(['object', '--frames', '3, 7']).cfg.frames).toEqual([3, 7]);
    for (const bad of ['0,20,40,', '5,,10', '3,3,10', 'a,1', '-1,2', '1.5,2', '5', '']) expect(() => parseArgs(['object', '--frames', bad]), bad).toThrow();
  });

  it('subset visibility is only read when this run wrote it, with one entry per GT sample', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'dense-gt-cli-'));
    try {
      const file = join(dir, 'object.compvis.u8');
      writeFileSync(file, Uint8Array.from([1, 0, 1, 1]));
      // left over from an earlier run: this run produced none
      expect(await loadSubsetVisibility(file, false, 4)).toBeNull();
      // another GT sample count (e.g. a regenerated dataset)
      expect(await loadSubsetVisibility(file, true, 6)).toBeNull();
      expect(Array.from((await loadSubsetVisibility(file, true, 4)) ?? [])).toEqual([1, 0, 1, 1]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
