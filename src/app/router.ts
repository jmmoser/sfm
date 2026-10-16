/**
 * App navigation as a pure state machine (no History API: the app stays a single document
 * without navigations — iOS muted the camera on pushState (WebKit 272704) and reset permission
 * state on hash navigations in the past).
 *
 *   home ──NEW_SCAN──▶ setup ──START_CAPTURE──▶ capture ──FINISH_CAPTURE──▶ processing ──DONE──▶ viewer
 *     ▲  ◀──BACK/CANCEL──┘          │ CANCEL                                  │ CANCELLED           │ BUILD
 *     └─────────────────────────────┴──────────────────(refine cancelled)─────┘◀────────────────────┘
 *   home ──OPEN_SESSION──▶ viewer (sparse/dense/meshed) | processing (captured)
 *   home/setup/viewer ──OPEN_DIAGNOSTICS──▶ diagnostics ──BACK──▶ (where it came from)
 */
import type { SessionStatus } from '../io/storage';

export type ScreenName = 'home' | 'setup' | 'capture' | 'processing' | 'viewer' | 'diagnostics';
export type ProcessingStep = 'refine' | 'densify' | 'mesh';

export type Route =
  | { screen: 'home' }
  | { screen: 'setup' }
  | { screen: 'capture'; sessionId: string | null }
  | { screen: 'processing'; sessionId: string | null; steps: ProcessingStep[] }
  | { screen: 'viewer'; sessionId: string | null }
  | { screen: 'diagnostics'; back: 'home' | 'setup' | 'viewer'; sessionId: string | null };

export type NavEvent =
  | { type: 'NEW_SCAN' }
  | { type: 'START_CAPTURE'; sessionId: string | null }
  | { type: 'FINISH_CAPTURE'; steps: ProcessingStep[] }
  | { type: 'CANCEL' }
  | { type: 'PROCESSING_DONE' }
  | { type: 'PROCESSING_CANCELLED'; hasModel: boolean }
  | { type: 'OPEN_SESSION'; sessionId: string; status: SessionStatus; numFrames: number }
  | { type: 'BUILD'; steps: ProcessingStep[] }
  | { type: 'OPEN_DIAGNOSTICS' }
  | { type: 'BACK' }
  | { type: 'HOME' };

/** Frames needed before a capture can be finished / an interrupted capture can be processed. */
export const MIN_KEYFRAMES = 5;

const STEP_ORDER: ProcessingStep[] = ['refine', 'densify', 'mesh'];

/** Steps in pipeline order without duplicates; densify is implied by mesh. */
export function normalizeSteps(steps: readonly ProcessingStep[]): ProcessingStep[] {
  const set = new Set(steps);
  if (set.has('mesh')) set.add('densify');
  return STEP_ORDER.filter((s) => set.has(s));
}

/** Which screen a stored session opens in (null = cannot be opened). */
export function routeForSession(sessionId: string, status: SessionStatus, numFrames: number): Route | null {
  switch (status) {
    case 'sparse':
    case 'dense':
    case 'meshed':
      return { screen: 'viewer', sessionId };
    case 'captured':
      return { screen: 'processing', sessionId, steps: ['refine'] };
    case 'capturing':
      // Interrupted capture (app killed / closed): process what was stored if it is enough.
      return numFrames >= MIN_KEYFRAMES ? { screen: 'processing', sessionId, steps: ['refine'] } : null;
    default:
      return null;
  }
}

/** The route after `ev`, or null when the event is not allowed in `cur`. */
export function nextRoute(cur: Route, ev: NavEvent): Route | null {
  if (ev.type === 'OPEN_DIAGNOSTICS') {
    if (cur.screen === 'home' || cur.screen === 'setup' || cur.screen === 'viewer') {
      return { screen: 'diagnostics', back: cur.screen, sessionId: cur.screen === 'viewer' ? cur.sessionId : null };
    }
    return null;
  }
  switch (cur.screen) {
    case 'home':
      if (ev.type === 'NEW_SCAN') return { screen: 'setup' };
      if (ev.type === 'OPEN_SESSION') return routeForSession(ev.sessionId, ev.status, ev.numFrames);
      return null;
    case 'setup':
      if (ev.type === 'START_CAPTURE') return { screen: 'capture', sessionId: ev.sessionId };
      if (ev.type === 'BACK' || ev.type === 'CANCEL' || ev.type === 'HOME') return { screen: 'home' };
      return null;
    case 'capture':
      if (ev.type === 'FINISH_CAPTURE') {
        const steps = normalizeSteps(['refine', ...ev.steps]);
        return { screen: 'processing', sessionId: cur.sessionId, steps };
      }
      if (ev.type === 'CANCEL') return { screen: 'home' };
      return null;
    case 'processing':
      if (ev.type === 'PROCESSING_DONE') return { screen: 'viewer', sessionId: cur.sessionId };
      if (ev.type === 'PROCESSING_CANCELLED') {
        // Without a sparse model there is nothing to show: back home (the capture is kept).
        return ev.hasModel ? { screen: 'viewer', sessionId: cur.sessionId } : { screen: 'home' };
      }
      if (ev.type === 'CANCEL') return cur.steps.includes('refine') ? { screen: 'home' } : { screen: 'viewer', sessionId: cur.sessionId };
      return null;
    case 'viewer':
      if (ev.type === 'BACK' || ev.type === 'HOME') return { screen: 'home' };
      if (ev.type === 'BUILD') {
        const steps = normalizeSteps(ev.steps.filter((s) => s !== 'refine'));
        return steps.length ? { screen: 'processing', sessionId: cur.sessionId, steps } : null;
      }
      return null;
    case 'diagnostics':
      if (ev.type === 'BACK' || ev.type === 'HOME') {
        if (ev.type === 'HOME' || cur.back === 'home') return { screen: 'home' };
        if (cur.back === 'setup') return { screen: 'setup' };
        return { screen: 'viewer', sessionId: cur.sessionId };
      }
      return null;
  }
}
