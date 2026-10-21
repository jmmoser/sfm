/**
 * "Add more views" (a processed scan) / "Continue scanning" (an interrupted capture): continue the
 * capture of a stored scan — router EXTEND_CAPTURE → the capture screen with `extendFrom`, whose
 * engine init has `resume` + `extend` (the stored session's mode, quality and orientation; the new
 * photos join the same keyframes, docs/ENGINE.md "Extend").
 *
 * The sheet says what happens before the camera opens: start where the scan already has views
 * (the new photos are placed by matching them to the existing ones), and when it is finished the
 * scan is processed again — the 3D model and every later stage that exists (dense cloud, mesh,
 * photoreal model) are rebuilt, which takes minutes. Its button asks for the capture permissions
 * synchronously inside the tap (iOS grants motion access only from a gesture), like Setup's Start.
 */
import type { SessionStatus } from '../io/storage';
import type { App } from './app';
import { button } from './components/controls';
import { openSheet, type SheetHandle } from './components/sheet';
import { h } from './dom';
import { icon, type IconName } from './icons';

export interface ExtendTarget {
  id: string;
  status: SessionStatus;
  /** Later stages that exist and are rebuilt (null: unknown). */
  built?: { dense: boolean; mesh: boolean; splat: boolean } | null;
}

/** A processed scan (it has a model to add views to) — otherwise an interrupted capture. */
export function isProcessed(status: SessionStatus): boolean {
  return status === 'sparse' || status === 'dense' || status === 'meshed';
}

/** What the next processing rebuilds, as a phrase ("the 3D model, the dense cloud and the mesh"). */
export function rebuiltPhrase(built: ExtendTarget['built']): string {
  const parts = ['the 3D model'];
  if (built?.dense) parts.push('the dense cloud');
  if (built?.mesh) parts.push('the mesh');
  if (built?.splat) parts.push('the photoreal model');
  return parts.length === 1 ? parts[0] : `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}`;
}

/** The confirm sheet; its button opens the capture on the stored scan. */
export function openExtendSheet(app: App, t: ExtendTarget): SheetHandle {
  const processed = isProcessed(t.status);
  const point = (name: IconName, text: string): HTMLElement => h('li', { class: 'extend-point' }, icon(name, { size: 20 }), h('span', { text }));
  const points = processed
    ? [
        point('target', 'Start where you already scanned, then move on to the parts that are missing — the top, the back, the gaps.'),
        point('refresh', `When you finish, the scan is processed again with every photo: ${rebuiltPhrase(t.built)} ${t.built?.dense || t.built?.mesh || t.built?.splat ? 'are' : 'is'} rebuilt. This takes a few minutes.`),
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
