/** e2e/import-recording.mjs: "sfm-rec-…-part2of3.zip" → { base, part, count } (a single archive: part 1 of 1). */
export function partOf(file: string): { base: string; part: number; count: number };
/** A dataset directory name: [A-Za-z0-9_.-] only, not starting with '.'. */
export function isDatasetName(name: string): boolean;
/** `dir` is a directory strictly inside `parent`. */
export function isStrictChild(parent: string, dir: string): boolean;
