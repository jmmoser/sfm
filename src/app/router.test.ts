import { describe, expect, it } from 'vitest';
import { MIN_KEYFRAMES, nextRoute, normalizeSteps, routeForSession, type NavEvent, type Route } from './router';

const home: Route = { screen: 'home' };

function run(start: Route, events: NavEvent[]): Route {
  let r = start;
  for (const e of events) {
    const n = nextRoute(r, e);
    if (!n) throw new Error(`${e.type} not allowed in ${r.screen}`);
    r = n;
  }
  return r;
}

describe('router state machine', () => {
  it('runs the main flow home → setup → capture → processing → viewer → home', () => {
    let r = run(home, [{ type: 'NEW_SCAN' }]);
    expect(r).toEqual({ screen: 'setup' });
    r = run(r, [{ type: 'START_CAPTURE', sessionId: 's1' }]);
    expect(r).toEqual({ screen: 'capture', sessionId: 's1' });
    r = run(r, [{ type: 'FINISH_CAPTURE', steps: ['densify'] }]);
    expect(r).toEqual({ screen: 'processing', sessionId: 's1', steps: ['refine', 'densify'] });
    r = run(r, [{ type: 'PROCESSING_DONE' }]);
    expect(r).toEqual({ screen: 'viewer', sessionId: 's1' });
    expect(run(r, [{ type: 'BACK' }])).toEqual(home);
  });

  it('FINISH_CAPTURE always refines first; mesh implies densify', () => {
    const cap: Route = { screen: 'capture', sessionId: null };
    expect(nextRoute(cap, { type: 'FINISH_CAPTURE', steps: [] })).toEqual({ screen: 'processing', sessionId: null, steps: ['refine'] });
    expect(nextRoute(cap, { type: 'FINISH_CAPTURE', steps: ['mesh'] })).toEqual({ screen: 'processing', sessionId: null, steps: ['refine', 'densify', 'mesh'] });
  });

  it('rejects events that are not allowed in a screen', () => {
    expect(nextRoute(home, { type: 'FINISH_CAPTURE', steps: [] })).toBeNull();
    expect(nextRoute(home, { type: 'PROCESSING_DONE' })).toBeNull();
    expect(nextRoute({ screen: 'capture', sessionId: 'a' }, { type: 'BACK' })).toBeNull(); // must CANCEL (confirm)
    expect(nextRoute({ screen: 'capture', sessionId: 'a' }, { type: 'OPEN_DIAGNOSTICS' })).toBeNull();
    expect(nextRoute({ screen: 'processing', sessionId: 'a', steps: ['refine'] }, { type: 'HOME' })).toBeNull();
    expect(nextRoute({ screen: 'setup' }, { type: 'NEW_SCAN' })).toBeNull();
  });

  it('cancel paths', () => {
    expect(nextRoute({ screen: 'setup' }, { type: 'CANCEL' })).toEqual(home);
    expect(nextRoute({ screen: 'capture', sessionId: 'a' }, { type: 'CANCEL' })).toEqual(home);
    const proc: Route = { screen: 'processing', sessionId: 'a', steps: ['refine', 'densify'] };
    expect(nextRoute(proc, { type: 'PROCESSING_CANCELLED', hasModel: false })).toEqual(home);
    expect(nextRoute(proc, { type: 'PROCESSING_CANCELLED', hasModel: true })).toEqual({ screen: 'viewer', sessionId: 'a' });
    expect(nextRoute({ screen: 'processing', sessionId: 'a', steps: ['densify'] }, { type: 'CANCEL' })).toEqual({ screen: 'viewer', sessionId: 'a' });
    expect(nextRoute(proc, { type: 'CANCEL' })).toEqual(home);
  });

  it('BUILD from the viewer never re-runs refine', () => {
    const v: Route = { screen: 'viewer', sessionId: 'x' };
    expect(nextRoute(v, { type: 'BUILD', steps: ['mesh'] })).toEqual({ screen: 'processing', sessionId: 'x', steps: ['densify', 'mesh'] });
    expect(nextRoute(v, { type: 'BUILD', steps: ['refine'] })).toBeNull();
  });

  it('EXTEND_CAPTURE continues a stored scan from the viewer or Home; BACK returns there', () => {
    const v: Route = { screen: 'viewer', sessionId: 'x' };
    const fromViewer = nextRoute(v, { type: 'EXTEND_CAPTURE', sessionId: 'x' });
    expect(fromViewer).toEqual({ screen: 'capture', sessionId: 'x', extendFrom: 'viewer' });
    expect(nextRoute(v, { type: 'EXTEND_CAPTURE', sessionId: 'other' })).toBeNull();
    // stopped without new views: back to the stored model; finished: processing re-runs refine
    expect(nextRoute(fromViewer!, { type: 'BACK' })).toEqual(v);
    expect(nextRoute(fromViewer!, { type: 'FINISH_CAPTURE', steps: ['densify'] })).toEqual({ screen: 'processing', sessionId: 'x', steps: ['refine', 'densify'] });
    expect(nextRoute(fromViewer!, { type: 'CANCEL' })).toEqual(home);
    const fromHome = nextRoute(home, { type: 'EXTEND_CAPTURE', sessionId: 'y' });
    expect(fromHome).toEqual({ screen: 'capture', sessionId: 'y', extendFrom: 'home' });
    expect(nextRoute(fromHome!, { type: 'BACK' })).toEqual(home);
    expect(nextRoute({ screen: 'setup' }, { type: 'EXTEND_CAPTURE', sessionId: 'y' })).toBeNull();
  });

  it('diagnostics returns where it came from', () => {
    for (const from of [home, { screen: 'setup' } as Route, { screen: 'viewer', sessionId: 'q' } as Route]) {
      const d = nextRoute(from, { type: 'OPEN_DIAGNOSTICS' });
      expect(d?.screen).toBe('diagnostics');
      expect(nextRoute(d!, { type: 'BACK' })).toEqual(from);
      expect(nextRoute(d!, { type: 'HOME' })).toEqual(home);
    }
  });

  it('opens stored sessions by status', () => {
    expect(routeForSession('a', 'meshed', 30)).toEqual({ screen: 'viewer', sessionId: 'a' });
    expect(routeForSession('a', 'sparse', 30)).toEqual({ screen: 'viewer', sessionId: 'a' });
    expect(routeForSession('a', 'captured', 30)).toEqual({ screen: 'processing', sessionId: 'a', steps: ['refine'] });
    expect(routeForSession('a', 'capturing', MIN_KEYFRAMES)).toEqual({ screen: 'processing', sessionId: 'a', steps: ['refine'] });
    expect(routeForSession('a', 'capturing', MIN_KEYFRAMES - 1)).toBeNull();
    expect(nextRoute(home, { type: 'OPEN_SESSION', sessionId: 'a', status: 'dense', numFrames: 9 })).toEqual({ screen: 'viewer', sessionId: 'a' });
    expect(nextRoute(home, { type: 'OPEN_SESSION', sessionId: 'a', status: 'capturing', numFrames: 1 })).toBeNull();
  });

  it('normalizeSteps orders and de-duplicates', () => {
    expect(normalizeSteps(['mesh', 'refine', 'mesh'])).toEqual(['refine', 'densify', 'mesh']);
    expect(normalizeSteps([])).toEqual([]);
  });
});
