/**
 * "Add more views" (a processed scan) / "Continue scanning" (an interrupted capture): continue the
 * capture of a stored scan — router EXTEND_CAPTURE → the capture screen with `extendFrom`, whose
 * engine init has `resume` + `extend` (the stored session's mode, quality and orientation; the new
 * photos join the same keyframes, docs/ENGINE.md "Extend").
 *
 * The sheet says what happens before the camera opens: start where the scan already has views
 * (the new photos are placed by matching them to the existing ones), and when it is finished the
 * scan is processed again — Finish re-runs refine and the stages the scan had (`rebuildSteps`:
 * dense cloud, mesh; densify also with auto-dense), which takes minutes; a photoreal model is
 * removed by the refine and trained again from Build only (it takes much longer). Its button asks
 * for the capture permissions synchronously inside the tap (iOS grants motion access only from a
 * gesture), like Setup's Start.
 */
import type { SessionStatus } from '../io/storage';
import type { App } from './app';
import type { ProcessingStep } from './router';
import { button } from './components/controls';
import { openSheet, type SheetHandle } from './components/sheet';
import { h } from './dom';
import { icon, type IconName } from './icons';

export interface ExtendTarget {
  id: string;
  /** The scan's status: which later stages exist (dense / meshed) and are rebuilt. */
  status: SessionStatus;
  /** Later stages that exist (null: unknown); only `splat` adds to `status` (it is not rebuilt). */
  built?: { dense: boolean; mesh: boolean; splat: boolean } | null;
}

/** A processed scan (it has a model to add views to) — otherwise an interrupted capture. */
export function isProcessed(status: SessionStatus): boolean {
  return status === 'sparse' || status === 'dense' || status === 'meshed';
}

/**
 * Steps Finish of a continued capture runs after the refine (FINISH_CAPTURE): the later stages the
 * scan had when the capture was continued — the refine removes them — and densify with auto-dense.
 */
export function rebuildSteps(status: SessionStatus | null | undefined, autoDense: boolean): ProcessingStep[] {
  const steps: ProcessingStep[] = [];
  if (autoDense || status === 'dense' || status === 'meshed') steps.push('densify');
  if (status === 'meshed') steps.push('mesh');
  return steps;
}

/** What the next processing rebuilds, as a phrase ("the 3D model, the dense cloud and the mesh"). */
export function rebuiltPhrase(steps: readonly ProcessingStep[]): string {
  const parts = ['the 3D model'];
  if (steps.includes('densify')) parts.push('the dense cloud');
  if (steps.includes('mesh')) parts.push('the mesh');
  return parts.length === 1 ? parts[0] : `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}`;
}

/** The confirm sheet; its button opens the capture on the stored scan. */
export function openExtendSheet(app: App, t: ExtendTarget): SheetHandle {
  const processed = isProcessed(t.status);
  const point = (name: IconName, text: string): HTMLElement => h('li', { class: 'extend-point' }, icon(name, { size: 20 }), h('span', { text }));
  const steps = rebuildSteps(t.status, app.config.autoDense);
  const points = processed
    ? [
        point('target', 'Start where you already scanned, then move on to the parts that are missing — the top, the back, the gaps.'),
        point('refresh', `When you finish, the scan is processed again with every photo: ${rebuiltPhrase(steps)} ${steps.length ? 'are' : 'is'} rebuilt. This takes a few minutes.`),
        ...(t.built?.splat ? [point('splats', 'The photoreal model is removed — build it again from Build when processing is done.')] : []),
        point('sun', 'Keep the light and the scene as they were for the best fit. Stop without new views and the scan stays as it is.'),
      ]
    : [
        point('target', 'Start where the capture stopped, then go on scanning.'),
        point('refresh', 'The photos captured so far are kept; Finish processes the whole scan.'),
      ];
  const go = button({
    label: processed ? 'Add more views' : 'Continue scanning',
    icon: 'camera',
    variant: 'primary',
    size: 'lg',
    block: true,
    testId: 'extend-start',
    onClick: () => {
      app.requestCapturePermissions(); // synchronous inside the tap
      sheet.close();
      app.startExtendScan(t.id);
    },
  });
  const sheet = openSheet({
    title: processed ? 'Add more views' : 'Continue scanning',
    subtitle: processed ? 'Fill the gaps in this scan.' : 'Add to this interrupted capture.',
    content: [h('ul', { class: 'extend-points' }, ...points), go],
    testId: 'sheet-extend',
  });
  return sheet;
}
