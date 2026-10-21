/**
 * Run journal: a crash-safe marker of the long engine run in progress, so that the next launch can
 * tell that iOS stopped it (a memory kill, a crash, or eviction in the background) instead of
 * restarting without a word, and the retry can shed the page's own memory.
 *
 * Stored in localStorage `sfm.run.v1` (RunRecord, one scan at a time). Writes: at the start of each
 * long operation (refine / densify / mesh on the Processing screen, the viewer's photoreal build —
 * `begin`), at each stage change and at most every BEAT_MS on progress (`RunHandle.stage`), and on
 * visibilitychange while a run is open (`setVisible`). The run is closed (`RunHandle.end`) on
 * success, on cancel and on any error the engine reported; a kill runs nothing, so the open entry
 * survives. `recover` (App start) turns an open entry into an interruption — 'killed' when the page
 * was visible at the last write (likely memory or a crash; closing a visible tab looks the same, so
 * the copy says "stopped", never "out of memory"), 'evicted' when it was in the background — and
 * counts it in `kills[step]`. The Home card reports it until it is resumed or dismissed
 * (`acknowledge`, which keeps the counts) or the next run starts.
 *
 * Light mode: while `kills[step] ≥ 1` for the scan, its next densify / mesh / photoreal build runs
 * with a lighter page (the displayed cloud cut to LIGHT_MODE_MAX_POINTS, no mesh in the 3D view,
 * prepared export files dropped); the engine's memory plan is unchanged, so splat checkpoints stay
 * valid. A clean run of a step resets its count; a cancel or an error keeps it; a run of another
 * scan starts with no counts.
 *
 * The page's own unload (reload, tab close) fires `pagehide` (not persisted) and then a
 * visibilitychange to hidden: that hidden is not recorded (`pageHide`; until `pageshow` or the page
 * is visible again), so an unload of a visible page still reads as 'killed'. A second tab of the app
 * that launches while another one runs would report that run as interrupted (no cross-tab lock).
 *
 * Every storage access is in try/catch: unavailable, throwing or corrupt storage gives no card and
 * no light mode. Pure helpers (parse / begin / end / recover / copy) are unit-tested in
 * run-journal.test.ts.
 */
import type { QualityPreset } from '../workers/protocol';

export const RUN_JOURNAL_KEY = 'sfm.run.v1';

/** The long operations the journal follows: Processing's steps and the viewer's photoreal build. */
export type RunStep = 'refine' | 'densify' | 'mesh' | 'splat';
export const RUN_STEPS: readonly RunStep[] = ['refine', 'densify', 'mesh', 'splat'];

/** 'killed': stopped while the page was visible; 'evicted': stopped while it was in the background. */
export type InterruptionKind = 'killed' | 'evicted';

/** At most this many points of the dense cloud are displayed during a light-mode build. */
export const LIGHT_MODE_MAX_POINTS = 1_500_000;

/** A run's stage is written when it changes; otherwise at most this often on progress (`lastBeat`). */
export const BEAT_MS = 5000;

/** What is stored under RUN_JOURNAL_KEY. */
export interface RunRecord {
  v: 1;
  sessionId: string;
  /** Scan name when the run started (the Home card prefers the stored session's current name). */
  name: string;
  /** The run in progress; null once it ended (success, cancel, reported error) or was recovered at launch. */
  step: RunStep | null;
  /** Stage of the (last) run: Processing's StageId or a SPLAT_STAGES value. */
  stage: string;
  quality: QualityPreset;
  /** The page was visible at the last write. */
  visible: boolean;
  /** Epoch ms. */
  startedAt: number;
  /** Epoch ms of the last write while the run was open. */
  lastBeat: number;
  /** Runs of this scan that never ended (killed / evicted), per step. */
  kills: Partial<Record<RunStep, number>>;
  /** Set by `recover`: the step whose run never ended, reported on Home until resumed / dismissed / the next run. */
  interrupted?: RunStep | null;
}

/** An interrupted run as the Home card reports it. */
export interface RunInterruption {
  sessionId: string;
  name: string;
  step: RunStep;
  stage: string;
  quality: QualityPreset;
  kind: InterruptionKind;
  startedAt: number;
  lastBeat: number;
  /** Interrupted runs of this step of the scan so far, this one included. */
  kills: number;
}

export interface RunStart {
  sessionId: string;
  name: string;
  step: RunStep;
  stage: string;
  quality: QualityPreset;
  visible: boolean;
}

/** The subset of the Web Storage API the journal uses. */
export interface JournalStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

const isStep = (v: unknown): v is RunStep => typeof v === 'string' && (RUN_STEPS as readonly string[]).includes(v);
const isQuality = (v: unknown): v is QualityPreset => v === 'fast' || v === 'balanced' || v === 'max';
const isTime = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= 0;

/** The stored record, or null when absent or not a valid version-1 record (never throws). */
export function parseRunRecord(raw: string | null): RunRecord | null {
  if (!raw) return null;
  let o: unknown;
  try {
    o = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!o || typeof o !== 'object') return null;
  const r = o as Record<string, unknown>;
  const { sessionId, name, step, stage, quality, visible, startedAt, lastBeat, interrupted } = r;
  if (r.v !== 1 || typeof sessionId !== 'string' || !sessionId || typeof name !== 'string') return null;
  if (!(step === null || isStep(step)) || typeof stage !== 'string' || !isQuality(quality) || typeof visible !== 'boolean') return null;
  if (!isTime(startedAt) || !isTime(lastBeat)) return null;
  if (!(interrupted === undefined || interrupted === null || isStep(interrupted))) return null;
  const kills: Partial<Record<RunStep, number>> = {};
  if (r.kills !== undefined) {
    if (!r.kills || typeof r.kills !== 'object') return null;
    for (const s of RUN_STEPS) {
      const n = (r.kills as Record<string, unknown>)[s];
      if (n === undefined) continue;
      if (typeof n !== 'number' || !Number.isInteger(n) || n < 0) return null;
      if (n > 0) kills[s] = n;
    }
  }
  return { v: 1, sessionId, name, step, stage, quality, visible, startedAt, lastBeat, kills, ...(interrupted ? { interrupted } : {}) };
}

/** The record of a run starting now: the kill counts of the same scan are kept, an unreported interruption is dropped. */
export function beginRecord(prev: RunRecord | null, run: RunStart, now: number): RunRecord {
  const kills = prev && prev.sessionId === run.sessionId ? { ...prev.kills } : {};
  return { v: 1, sessionId: run.sessionId, name: run.name, step: run.step, stage: run.stage, quality: run.quality, visible: run.visible, startedAt: now, lastBeat: now, kills };
}

/** The record once its run ended: 'done' resets the step's kill count, 'stopped' (cancel / error) keeps it. */
export function endRecord(rec: RunRecord, outcome: 'done' | 'stopped'): RunRecord {
  const kills = { ...rec.kills };
  if (outcome === 'done' && rec.step) delete kills[rec.step];
  return { ...rec, step: null, kills };
}

/**
 * At launch: an open run never ended, so the page was killed or evicted while it ran — it becomes
 * the record's `interrupted` step and counts in `kills`. An interruption not yet acknowledged (the
 * app was closed again before Resume) is reported again without being counted twice.
 */
export function recoverRecord(rec: RunRecord | null): { record: RunRecord | null; interruption: RunInterruption | null } {
  if (!rec) return { record: null, interruption: null };
  if (rec.step === null) return { record: rec, interruption: interruptionOf(rec) };
  const step = rec.step;
  const record: RunRecord = { ...rec, step: null, interrupted: step, kills: { ...rec.kills, [step]: (rec.kills[step] ?? 0) + 1 } };
  return { record, interruption: interruptionOf(record) };
}

/** The interruption a record reports (null: none). */
export function interruptionOf(rec: RunRecord | null): RunInterruption | null {
  if (!rec || !rec.interrupted) return null;
  const step = rec.interrupted;
  return {
    sessionId: rec.sessionId,
    name: rec.name,
    step,
    stage: rec.stage,
    quality: rec.quality,
    kind: rec.visible ? 'killed' : 'evicted',
    startedAt: rec.startedAt,
    lastBeat: rec.lastBeat,
    kills: rec.kills[step] ?? 1,
  };
}

/** The next `step` of scan `sessionId` runs in light mode: a run of that step was killed / evicted. */
export function lightModeFor(rec: RunRecord | null, sessionId: string, step: RunStep): boolean {
  return !!rec && rec.sessionId === sessionId && (rec.kills[step] ?? 0) >= 1;
}

const STEP_PHRASE: Record<RunStep, string> = {
  refine: 'Reconstructing the scan',
  densify: 'Building the dense cloud',
  mesh: 'Building the mesh',
  splat: 'Building the photoreal model',
};

const RESUME_TEXT: Record<RunStep, string> = {
  refine: 'Resume processes it again from the saved photos.',
  densify: 'Resume continues from the saved depth maps, with a lighter 3D view meanwhile to save memory.',
  mesh: 'Resume opens the scan to build the mesh again, with a lighter 3D view meanwhile to save memory.',
  splat: 'Resume continues from the saved training, with a lighter 3D view meanwhile to save memory.',
};

/** Title and text of the Home card ("stopped", never a claimed cause: closing a visible tab looks like a kill). */
export function interruptionCopy(it: Pick<RunInterruption, 'name' | 'step' | 'kind'>): { title: string; body: string } {
  const where =
    it.kind === 'evicted'
      ? 'stopped while the app was in the background — iOS closes background apps to free memory, so keep the app open while it processes.'
      : 'stopped while the app was open — iOS may have run out of memory, or the app was closed.';
  return { title: `Processing of “${it.name}” stopped`, body: `${STEP_PHRASE[it.step]} ${where} ${RESUME_TEXT[it.step]}` };
}

/** The open run of this page: stage changes / beats while it runs, `end` when it stopped. */
export interface RunHandle {
  stage(stage: string): void;
  end(outcome: 'done' | 'stopped'): void;
}

const NO_RUN: RunHandle = { stage: () => undefined, end: () => undefined };

/** The journal over a storage (localStorage in the app: `RunJournal.local()`; null: a no-op journal). */
export class RunJournal {
  /** The run this page opened and has not ended (other handles are stale: no writes). */
  private current: { handle: RunHandle; sessionId: string; step: RunStep; startedAt: number; stage: string; wroteAt: number } | null = null;
  /** The page is being unloaded (pagehide, not persisted): its last visibilitychange is not recorded. */
  private unloading = false;

  constructor(
    private readonly storage: JournalStorage | null,
    private readonly now: () => number = Date.now,
  ) {}

  /** Over window.localStorage (null when the accessor throws, e.g. blocked storage). */
  static local(): RunJournal {
    let s: JournalStorage | null = null;
    try {
      s = globalThis.localStorage ?? null;
    } catch {
      s = null;
    }
    return new RunJournal(s);
  }

  read(): RunRecord | null {
    try {
      return parseRunRecord(this.storage?.getItem(RUN_JOURNAL_KEY) ?? null);
    } catch {
      return null;
    }
  }

  private write(rec: RunRecord | null): void {
    try {
      if (rec) this.storage?.setItem(RUN_JOURNAL_KEY, JSON.stringify(rec));
      else this.storage?.removeItem(RUN_JOURNAL_KEY);
    } catch {
      /* storage full / unavailable: the journal is best effort */
    }
  }

  /** App start: classify a run that never ended (see recoverRecord) and return what Home reports. */
  recover(): RunInterruption | null {
    const prev = this.read();
    const { record, interruption } = recoverRecord(prev);
    if (record && record !== prev) this.write(record);
    return interruption;
  }

  /** The interruption to report on Home (null: none). */
  interruption(): RunInterruption | null {
    return interruptionOf(this.read());
  }

  /** The interruption was resumed or dismissed: no more card; the kill counts stay. */
  acknowledge(): void {
    const rec = this.read();
    if (rec?.interrupted) this.write({ ...rec, interrupted: null });
  }

  /** The scan was deleted: drop its record. */
  forget(sessionId: string): void {
    if (this.read()?.sessionId === sessionId) this.write(null);
    if (this.current?.sessionId === sessionId) this.current = null;
  }

  /** The next `step` of this scan runs in light mode (lightModeFor). */
  lightMode(sessionId: string, step: RunStep): boolean {
    return lightModeFor(this.read(), sessionId, step);
  }

  /** A long operation starts (no storage: a handle that does nothing). */
  begin(run: Omit<RunStart, 'visible'> & { visible?: boolean }): RunHandle {
    if (!this.storage) return NO_RUN;
    const visible = run.visible ?? !(typeof document !== 'undefined' && document.hidden);
    const now = this.now();
    this.write(beginRecord(this.read(), { ...run, visible }, now));
    const cur = { handle: NO_RUN, sessionId: run.sessionId, step: run.step, startedAt: now, stage: run.stage, wroteAt: now };
    cur.handle = {
      stage: (stage) => this.onStage(cur, stage),
      end: (outcome) => this.onEnd(cur, outcome),
    };
    this.current = cur;
    return cur.handle;
  }

  /** The stored record when it is still the open run `cur` (not ended, not replaced by another tab). */
  private openRecord(cur: NonNullable<RunJournal['current']>): RunRecord | null {
    if (this.current !== cur) return null;
    const rec = this.read();
    return rec && rec.step === cur.step && rec.sessionId === cur.sessionId && rec.startedAt === cur.startedAt ? rec : null;
  }

  private onStage(cur: NonNullable<RunJournal['current']>, stage: string): void {
    if (this.current !== cur) return;
    const now = this.now();
    if (stage === cur.stage && now - cur.wroteAt < BEAT_MS) return;
    const rec = this.openRecord(cur);
    if (!rec) return;
    cur.stage = stage;
    cur.wroteAt = now;
    this.write({ ...rec, stage, lastBeat: now });
  }

  private onEnd(cur: NonNullable<RunJournal['current']>, outcome: 'done' | 'stopped'): void {
    const rec = this.openRecord(cur);
    if (this.current === cur) this.current = null;
    if (rec) this.write(endRecord(rec, outcome));
  }

  /** visibilitychange: recorded while this page's run is open (not during the page's own unload). */
  setVisible(visible: boolean): void {
    // visible again: the page is alive, whatever an earlier pagehide said
    if (visible) this.unloading = false;
    if (this.unloading || !this.current) return;
    const rec = this.openRecord(this.current);
    if (!rec || rec.visible === visible) return;
    const now = this.now();
    this.current.wroteAt = now;
    this.write({ ...rec, visible, lastBeat: now });
  }

  /** pagehide: an unload (not persisted) is not a trip to the background. */
  pageHide(persisted: boolean): void {
    if (!persisted) this.unloading = true;
  }

  /** pageshow: the page is live again (restored from the back-forward cache). */
  pageShow(): void {
    this.unloading = false;
  }
}
