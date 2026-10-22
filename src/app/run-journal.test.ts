/**
 * Run journal (run-journal.ts): kill / eviction classification at launch, kill counts per step and
 * scan, light mode, closing on success / cancel / reported error, and storage that is missing,
 * throwing or corrupt (no card); the card copy, the hand-over to the screen a scan opens in and the
 * viewer's Build hint. A "launch" is a new RunJournal over the same storage.
 */
import { describe, expect, it } from 'vitest';
import {
  BEAT_MS,
  RUN_JOURNAL_KEY,
  RunJournal,
  beginRecord,
  endRecord,
  interruptionCopy,
  lightModeFor,
  parseRunRecord,
  recoverRecord,
  resumeHint,
  type JournalStorage,
  type RunRecord,
  type RunStart,
} from './run-journal';
import type { Results } from './app';

class MemStorage implements JournalStorage {
  readonly map = new Map<string, string>();
  sets = 0;
  getItem(k: string): string | null {
    return this.map.get(k) ?? null;
  }
  setItem(k: string, v: string): void {
    this.sets++;
    this.map.set(k, v);
  }
  removeItem(k: string): void {
    this.map.delete(k);
  }
}

const start = (o: Partial<RunStart> = {}): Omit<RunStart, 'visible'> & { visible?: boolean } => ({
  sessionId: 's1',
  name: 'Garden wall',
  step: 'densify',
  stage: 'depth_maps',
  quality: 'balanced',
  visible: true,
  ...o,
});

/** A clock the test moves. */
function clock(t0 = 1_700_000_000_000): { now: () => number; advance: (ms: number) => void } {
  let t = t0;
  return { now: () => t, advance: (ms) => (t += ms) };
}

/** The page is killed while the run is open: the next launch recovers it. */
function kill(storage: JournalStorage, run = start()): ReturnType<RunJournal['recover']> {
  new RunJournal(storage).begin(run);
  return new RunJournal(storage).recover();
}

describe('run journal: classification at launch', () => {
  it('a run killed while the page was visible is reported as killed, with the scan, step and stage', () => {
    const st = new MemStorage();
    const c = clock();
    const j = new RunJournal(st, c.now);
    const run = j.begin(start());
    c.advance(BEAT_MS + 1);
    run.stage('fusion');
    const it = new RunJournal(st).recover();
    expect(it).toEqual({ sessionId: 's1', name: 'Garden wall', step: 'densify', stage: 'fusion', quality: 'balanced', kind: 'killed', startedAt: c.now() - BEAT_MS - 1, lastBeat: c.now(), kills: 1 });
  });

  it('a run stopped while the page was hidden is reported as evicted; visible again → killed', () => {
    const st = new MemStorage();
    const j = new RunJournal(st);
    j.begin(start({ step: 'refine', stage: 'matching' }));
    j.setVisible(false);
    expect(new RunJournal(st).interruption()).toBeNull(); // still open: nothing reported yet
    const a = new MemStorage();
    a.map.set(RUN_JOURNAL_KEY, st.getItem(RUN_JOURNAL_KEY)!);
    expect(new RunJournal(a).recover()).toMatchObject({ step: 'refine', kind: 'evicted' });
    j.setVisible(true);
    expect(new RunJournal(st).recover()).toMatchObject({ step: 'refine', kind: 'killed' });
  });

  it('the page’s own unload (pagehide, not persisted) does not record its visibilitychange; a bfcache pagehide does', () => {
    const st = new MemStorage();
    const j = new RunJournal(st);
    j.begin(start());
    j.pageHide(false);
    j.setVisible(false);
    expect(new RunJournal(st).recover()?.kind).toBe('killed');

    const st2 = new MemStorage();
    const j2 = new RunJournal(st2);
    j2.begin(start());
    j2.pageHide(true);
    j2.setVisible(false);
    expect(new RunJournal(st2).recover()?.kind).toBe('evicted');
    // pageshow: the page is live again, its visibility is recorded
    const st3 = new MemStorage();
    const j3 = new RunJournal(st3);
    j3.begin(start());
    j3.pageHide(false);
    j3.pageShow();
    j3.setVisible(false);
    expect(new RunJournal(st3).recover()?.kind).toBe('evicted');
    // … or once it is visible again (a pagehide that did not unload it)
    const st4 = new MemStorage();
    const j4 = new RunJournal(st4);
    j4.begin(start());
    j4.pageHide(false);
    j4.setVisible(true);
    j4.setVisible(false);
    expect(new RunJournal(st4).recover()?.kind).toBe('evicted');
  });

  it('no card for missing, throwing or corrupt storage', () => {
    expect(new RunJournal(null).recover()).toBeNull();
    const nullRun = new RunJournal(null).begin(start());
    expect(() => nullRun.end('done')).not.toThrow();

    const throwing: JournalStorage = {
      getItem: () => {
        throw new DOMException('denied', 'SecurityError');
      },
      setItem: () => {
        throw new DOMException('full', 'QuotaExceededError');
      },
      removeItem: () => {
        throw new DOMException('denied', 'SecurityError');
      },
    };
    const tj = new RunJournal(throwing);
    expect(() => tj.begin(start()).stage('fusion')).not.toThrow();
    expect(() => tj.setVisible(false)).not.toThrow();
    expect(tj.recover()).toBeNull();
    expect(tj.lightMode('s1', 'densify')).toBe(false);
    expect(() => tj.acknowledge()).not.toThrow();

    // writes fail (quota) but reads work: the run is not recorded
    const ro = new MemStorage();
    ro.setItem = () => {
      throw new DOMException('full', 'QuotaExceededError');
    };
    new RunJournal(ro).begin(start());
    expect(new RunJournal(ro).recover()).toBeNull();

    const valid = beginRecord(null, { ...start(), visible: true }, 1000);
    const corrupt = [
      'not json',
      'null',
      '42',
      JSON.stringify({ ...valid, v: 2 }),
      JSON.stringify({ ...valid, sessionId: '' }),
      JSON.stringify({ ...valid, step: 'export' }),
      JSON.stringify({ ...valid, quality: 'ultra' }),
      JSON.stringify({ ...valid, visible: 'yes' }),
      JSON.stringify({ ...valid, startedAt: 'today' }),
      JSON.stringify({ ...valid, kills: { densify: -1 } }),
      JSON.stringify({ ...valid, kills: { densify: 1.5 } }),
      JSON.stringify({ ...valid, kills: 3 }),
      JSON.stringify({ ...valid, interrupted: 'export' }),
    ];
    for (const raw of corrupt) {
      const st = new MemStorage();
      st.map.set(RUN_JOURNAL_KEY, raw);
      const j = new RunJournal(st);
      expect(parseRunRecord(raw)).toBeNull();
      expect(j.recover()).toBeNull();
      expect(j.lightMode('s1', 'densify')).toBe(false);
    }
    // unknown extra fields are fine
    expect(parseRunRecord(JSON.stringify({ ...valid, appVersion: 'x', kills: { densify: 2, other: 5 } }))?.kills).toEqual({ densify: 2 });
  });

  it('kill counts accumulate per step of the scan; light mode follows them; another scan starts with none', () => {
    const st = new MemStorage();
    expect(kill(st)?.kills).toBe(1);
    expect(kill(st)?.kills).toBe(2);
    expect(kill(st, start({ step: 'splat', stage: 'splat_training' }))).toMatchObject({ step: 'splat', kills: 1 });
    expect(parseRunRecord(st.getItem(RUN_JOURNAL_KEY))?.kills).toEqual({ densify: 2, splat: 1 });
    const j = new RunJournal(st);
    expect(j.lightMode('s1', 'densify')).toBe(true);
    expect(j.lightMode('s1', 'splat')).toBe(true);
    expect(j.lightMode('s1', 'mesh')).toBe(false);
    expect(j.lightMode('s2', 'densify')).toBe(false);
    // a run of another scan: no counts carried over
    j.begin(start({ sessionId: 's2', name: 'Fence' }));
    expect(j.lightMode('s1', 'densify')).toBe(false);
    expect(new RunJournal(st).recover()).toMatchObject({ sessionId: 's2', name: 'Fence', kills: 1 });
  });
});

describe('run journal: closing a run', () => {
  it('success closes the run and resets the step’s kill count', () => {
    const st = new MemStorage();
    kill(st);
    const j = new RunJournal(st);
    j.acknowledge();
    expect(j.lightMode('s1', 'densify')).toBe(true);
    const run = j.begin(start());
    run.end('done');
    expect(new RunJournal(st).recover()).toBeNull();
    expect(j.lightMode('s1', 'densify')).toBe(false);
  });

  it('a cancel or a reported error closes the run (no card) and keeps the kill count', () => {
    for (const label of ['cancel', 'error']) {
      const st = new MemStorage();
      kill(st, start({ step: 'splat', stage: 'splat_prepare' }));
      const j = new RunJournal(st);
      j.acknowledge();
      const run = j.begin(start({ step: 'splat', stage: 'splat_prepare' }));
      run.stage('splat_training');
      run.end('stopped');
      expect(new RunJournal(st).recover(), label).toBeNull();
      expect(j.lightMode('s1', 'splat'), label).toBe(true);
    }
  });

  it('a handle of an earlier run does not close or write the run that replaced it', () => {
    const st = new MemStorage();
    const j = new RunJournal(st);
    const splat = j.begin(start({ step: 'splat', stage: 'splat_training' }));
    const dense = j.begin(start({ step: 'densify', stage: 'depth_maps' })); // e.g. the aborted build settles later
    splat.stage('splat_finishing');
    splat.end('stopped');
    expect(parseRunRecord(st.getItem(RUN_JOURNAL_KEY))).toMatchObject({ step: 'densify', stage: 'depth_maps' });
    dense.end('done');
    dense.end('done');
    expect(parseRunRecord(st.getItem(RUN_JOURNAL_KEY))?.step).toBeNull();
  });

  it('another tab’s run in the storage is not closed by this page’s handle', () => {
    const st = new MemStorage();
    const mine = new RunJournal(st).begin(start());
    new RunJournal(st).begin(start({ sessionId: 's9', name: 'Other tab' }));
    mine.end('stopped');
    expect(parseRunRecord(st.getItem(RUN_JOURNAL_KEY))).toMatchObject({ sessionId: 's9', step: 'densify' });
  });

  it('stage changes are written at once, the same stage at most every BEAT_MS', () => {
    const st = new MemStorage();
    const c = clock();
    const j = new RunJournal(st, c.now);
    const run = j.begin(start());
    const sets = st.sets;
    run.stage('depth_maps');
    c.advance(BEAT_MS - 1);
    run.stage('depth_maps');
    expect(st.sets).toBe(sets);
    c.advance(1);
    run.stage('depth_maps');
    expect(st.sets).toBe(sets + 1);
    expect(parseRunRecord(st.getItem(RUN_JOURNAL_KEY))?.lastBeat).toBe(c.now());
    run.stage('fusion');
    expect(st.sets).toBe(sets + 2);
    expect(parseRunRecord(st.getItem(RUN_JOURNAL_KEY))?.stage).toBe('fusion');
    // visibility: written only when it changes
    j.setVisible(true);
    expect(st.sets).toBe(sets + 2);
    j.setVisible(false);
    expect(parseRunRecord(st.getItem(RUN_JOURNAL_KEY))?.visible).toBe(false);
    run.end('done');
    j.setVisible(true); // no run open: nothing written
    expect(parseRunRecord(st.getItem(RUN_JOURNAL_KEY))?.visible).toBe(false);
  });
});

describe('run journal: the Home card', () => {
  it('is reported on every launch until acknowledged, counted once; the next run drops it', () => {
    const st = new MemStorage();
    kill(st);
    expect(new RunJournal(st).recover()).toMatchObject({ step: 'densify', kills: 1 });
    expect(new RunJournal(st).interruption()).toMatchObject({ step: 'densify', kills: 1 });
    const j = new RunJournal(st);
    j.acknowledge();
    expect(j.interruption()).toBeNull();
    expect(new RunJournal(st).recover()).toBeNull();
    expect(j.lightMode('s1', 'densify')).toBe(true);
    // not acknowledged, but another run started
    kill(st, start({ step: 'mesh', stage: 'meshing' }));
    j.begin(start({ step: 'splat', stage: 'splat_prepare' }));
    expect(j.interruption()).toBeNull();
    expect(j.lightMode('s1', 'mesh')).toBe(true);
  });

  it('a deleted scan’s record is dropped', () => {
    const st = new MemStorage();
    kill(st);
    const j = new RunJournal(st);
    j.forget('other');
    expect(j.interruption()).not.toBeNull();
    j.forget('s1');
    expect(j.interruption()).toBeNull();
    expect(st.getItem(RUN_JOURNAL_KEY)).toBeNull();
  });

  it('copy: names the scan and the step, says where it stopped, never claims the cause', () => {
    const killed = interruptionCopy({ name: 'Garden wall', step: 'densify', kind: 'killed', hasDense: false });
    expect(killed.title).toContain('Garden wall');
    expect(killed.body).toMatch(/dense cloud stopped while the app was open/);
    const evicted = interruptionCopy({ name: 'Yard', step: 'refine', kind: 'evicted', hasDense: true });
    expect(evicted.body).toMatch(/^Reconstructing the scan stopped while the app was in the background/);
    expect(evicted.body).not.toMatch(/lighter/); // refine has no light mode
    // a closed tab looks like a kill: memory is only a possibility, and never in the title
    expect(killed.title).toBe('Processing of “Garden wall” stopped');
    expect(killed.body).toMatch(/may have run out of memory, or the app was closed/);
  });

  it('copy: Resume never promises stored progress Home cannot see; the lighter view only with a dense cloud (review)', () => {
    // a kill before the first depth map / checkpoint stored nothing: the card must not promise one
    const dense = interruptionCopy({ name: 'Garden wall', step: 'densify', kind: 'killed', hasDense: false }).body;
    expect(dense).toMatch(/continues from the depth maps saved so far, or starts again if none were saved\.$/);
    const splat = interruptionCopy({ name: 'Bench', step: 'splat', kind: 'killed', hasDense: true }).body;
    expect(splat).toMatch(/photoreal model.*continues from the training saved so far, or starts again if none was saved, with a lighter 3D view/);
    // a sparse-only scan's preview is already small: no "lighter" claim
    expect(dense).not.toMatch(/lighter/);
    expect(interruptionCopy({ name: 'Bench', step: 'splat', kind: 'evicted', hasDense: false }).body).not.toMatch(/lighter/);
    expect(interruptionCopy({ name: 'Wall', step: 'densify', kind: 'killed', hasDense: true }).body).toMatch(/lighter 3D view meanwhile to save memory\.$/);
    expect(interruptionCopy({ name: 'Wall', step: 'mesh', kind: 'killed', hasDense: true }).body).toMatch(/build the mesh again, with a lighter/);
  });

  it('opening the interrupted scan answers the card and hands the interruption to the screen it opens, once', () => {
    const st = new MemStorage();
    kill(st, start({ step: 'splat', stage: 'splat_prepare', quality: 'fast' }));
    const j = new RunJournal(st);
    // another scan opened: nothing answered, nothing handed over
    j.open('other');
    expect(j.interruption()).not.toBeNull();
    expect(j.takeOpened('other')).toBeNull();
    j.open('s1');
    expect(j.interruption()).toBeNull();
    expect(j.lightMode('s1', 'splat')).toBe(true); // the kill count stays
    expect(j.takeOpened('s1')).toMatchObject({ sessionId: 's1', step: 'splat', stage: 'splat_prepare', quality: 'fast', kind: 'killed' });
    expect(j.takeOpened('s1')).toBeNull();
    // handed to another scan's screen: dropped
    kill(st);
    const j2 = new RunJournal(st);
    j2.open('s1');
    expect(j2.takeOpened('s2')).toBeNull();
    expect(j2.takeOpened('s1')).toBeNull();
    // a run that starts drops a hand-over nobody took
    kill(st);
    const j3 = new RunJournal(st);
    j3.open('s1');
    j3.begin(start({ step: 'refine', stage: 'matching' }));
    expect(j3.takeOpened('s1')).toBeNull();
    // no storage: nothing to open
    const none = new RunJournal(null);
    none.open('s1');
    expect(none.takeOpened('s1')).toBeNull();
  });

  it('resumeHint: the viewer points to Build only when no resume card covers the interrupted run (review)', () => {
    const sparse = {} as NonNullable<Results['sparse']>;
    const dense = { count: 10 } as NonNullable<Results['dense']>;
    // nothing stored: the build starts over
    expect(resumeHint('splat', { sparse, dense: null, stats: {} })).toMatch(/before any training was saved/);
    expect(resumeHint('densify', { sparse, dense: null, stats: {} })).toMatch(/before any depth maps were saved/);
    // a checkpoint / depth maps were stored (loadSession) or the run was stopped here: the resume card offers it
    expect(resumeHint('splat', { sparse, dense, stats: { splatResumableIteration: 500 } })).toBeNull();
    expect(resumeHint('splat', { sparse, dense, stats: { splatInterrupted: 1 } })).toBeNull();
    expect(resumeHint('densify', { sparse, dense: null, stats: { denseResumablePass: 1 } })).toBeNull();
    // a dense rebuild: the previous cloud is shown and no dense-resume card exists
    expect(resumeHint('densify', { sparse, dense, stats: { denseResumablePass: 1 } })).toMatch(/previous one is shown/);
    // the viewer has no mesh resume card; refine runs in Processing
    expect(resumeHint('mesh', { sparse, dense, stats: {} })).toMatch(/mesh was not finished — build it again/);
    expect(resumeHint('refine', { sparse: null, dense: null, stats: {} })).toBeNull();
  });
});

describe('run journal: pure helpers', () => {
  it('beginRecord keeps the same scan’s counts and drops an unreported interruption; endRecord / recoverRecord', () => {
    const r0: RunRecord = { ...beginRecord(null, { ...start(), visible: true }, 5), kills: { densify: 2 }, step: null, interrupted: 'densify' };
    const b = beginRecord(r0, { ...start({ step: 'mesh', stage: 'meshing' }), visible: false }, 9);
    expect(b).toEqual({ v: 1, sessionId: 's1', name: 'Garden wall', step: 'mesh', stage: 'meshing', quality: 'balanced', visible: false, startedAt: 9, lastBeat: 9, kills: { densify: 2 } });
    expect(endRecord(b, 'stopped')).toMatchObject({ step: null, kills: { densify: 2 } });
    expect(endRecord({ ...b, step: 'densify' }, 'done').kills).toEqual({});
    expect(recoverRecord(null)).toEqual({ record: null, interruption: null });
    const rec = recoverRecord(b);
    expect(rec.record).toMatchObject({ step: null, interrupted: 'mesh', kills: { densify: 2, mesh: 1 } });
    expect(rec.interruption).toMatchObject({ step: 'mesh', kind: 'evicted', kills: 1 });
    expect(lightModeFor(rec.record, 's1', 'mesh')).toBe(true);
    expect(lightModeFor(null, 's1', 'mesh')).toBe(false);
  });
});
