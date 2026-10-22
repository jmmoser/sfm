/** Connect middleware (node:http IncomingMessage / ServerResponse; untyped: the main tsconfig has no @types/node). */
export type Middleware = (req: unknown, res: unknown, next: () => void) => void;

export interface RunEnvelope {
  result: unknown;
  error: string | null;
}

export interface BridgeRun {
  id: string;
  outDir: string;
  prefix: string;
  /** Relative harness URL (bridge=1, run id, config JSON). */
  path: string;
  info: Record<string, unknown> | null;
  startedAt: number | null;
  lastSeen: number | null;
  abortReason: string | null;
  settled: boolean;
  files: Map<string, number>;
  started: Promise<Record<string, unknown>>;
  result: Promise<RunEnvelope>;
  log: (msg: string) => void;
  finish(envelope: Partial<RunEnvelope>): boolean;
  abort(reason: string): void;
  close(): void;
  wait(opts: {
    timeoutMs: number;
    startTimeoutMs?: number;
    remindMs?: number;
    staleMs?: number;
    graceMs?: number;
    tickMs?: number;
    remind?: () => void;
  }): Promise<RunEnvelope & { waitMs: number; runMs: number }>;
}

export interface Bridge {
  runs: Map<string, BridgeRun>;
  readonly active: BridgeRun | null;
  expectRun(opts: { outDir: string; prefix?: string; config: unknown; log?: (msg: string) => void; id?: string }): BridgeRun;
  middleware: Middleware;
  plugin(): { name: string; apply: 'serve'; configureServer: (server: unknown) => void; configurePreviewServer: (server: unknown) => void };
}

export function createBridge(opts?: { now?: () => number; maxRunBytes?: number }): Bridge;
export function bridgePlugin(middleware: Middleware): ReturnType<Bridge['plugin']>;
export function harnessPath(id: string, config: unknown): string;
export function resolveWritePath(outDir: string, prefix: string, path: unknown): string;
