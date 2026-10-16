/**
 * Navigation: three's OrbitControls configured for touch (1 finger rotate,
 * 2 fingers pinch-zoom + pan, damping, zoom towards the pinch centre) plus a
 * tap / double-tap detector that coexists with it, and a small eased
 * orbit-target animation used by double-tap-to-focus.
 */
import { TOUCH, MOUSE, Vector3, type PerspectiveCamera } from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export { OrbitControls };

export function createOrbitControls(camera: PerspectiveCamera, element: HTMLElement): OrbitControls {
  const c = new OrbitControls(camera, element);
  c.enableDamping = true;
  c.dampingFactor = 0.12;
  c.rotateSpeed = 0.8;
  c.zoomSpeed = 1.2;
  c.panSpeed = 1.0;
  c.screenSpacePanning = true;
  c.zoomToCursor = true;
  c.touches = { ONE: TOUCH.ROTATE, TWO: TOUCH.DOLLY_PAN };
  c.mouseButtons = { LEFT: MOUSE.ROTATE, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.PAN };
  element.style.touchAction = 'none';
  return c;
}

export interface TapInfo {
  /** CSS px relative to the element's top-left. */
  x: number;
  y: number;
  pointerType: string;
}

export interface TapHandlers {
  onTap?: (t: TapInfo) => void;
  onDoubleTap?: (t: TapInfo) => void;
}

export interface TapOptions {
  /** Max movement (CSS px) for a pointer down/up to count as a tap. */
  moveTolerancePx?: number;
  /** Max press duration (ms). */
  maxDurationMs?: number;
  /** Max delay between two taps of a double tap (ms). */
  doubleTapMs?: number;
  /** Max distance between two taps of a double tap (CSS px). */
  doubleTapDistancePx?: number;
}

interface ActivePointer {
  x0: number;
  y0: number;
  t0: number;
  moved: boolean;
}

/**
 * Detects single taps and double taps on an element (pointer events). A gesture
 * that involved two or more simultaneous pointers (pinch) never produces a tap.
 * Single taps are reported immediately (no double-tap delay); a double tap is
 * reported on the second tap (after its own single-tap callback).
 */
export class TapDetector {
  private readonly active = new Map<number, ActivePointer>();
  private multi = false;
  private last: { x: number; y: number; t: number } | null = null;
  private readonly opts: Required<TapOptions>;
  private readonly onDown: (e: PointerEvent) => void;
  private readonly onMove: (e: PointerEvent) => void;
  private readonly onUp: (e: PointerEvent) => void;
  private readonly onCancel: (e: PointerEvent) => void;

  constructor(
    private readonly element: HTMLElement,
    private readonly handlers: TapHandlers,
    opts: TapOptions = {},
  ) {
    this.opts = {
      moveTolerancePx: opts.moveTolerancePx ?? 10,
      maxDurationMs: opts.maxDurationMs ?? 450,
      doubleTapMs: opts.doubleTapMs ?? 350,
      doubleTapDistancePx: opts.doubleTapDistancePx ?? 40,
    };
    this.onDown = (e) => {
      if (e.pointerType === 'mouse' && e.button !== 0) return;
      const p = this.local(e);
      this.active.set(e.pointerId, { x0: p.x, y0: p.y, t0: e.timeStamp, moved: false });
      if (this.active.size > 1) this.multi = true;
    };
    this.onMove = (e) => {
      const a = this.active.get(e.pointerId);
      if (!a || a.moved) return;
      const p = this.local(e);
      if (Math.hypot(p.x - a.x0, p.y - a.y0) > this.opts.moveTolerancePx) a.moved = true;
    };
    this.onUp = (e) => {
      const a = this.active.get(e.pointerId);
      if (!a) return;
      this.active.delete(e.pointerId);
      const wasMulti = this.multi;
      if (this.active.size === 0) this.multi = false;
      const p = this.local(e);
      if (wasMulti || a.moved || e.timeStamp - a.t0 > this.opts.maxDurationMs) return;
      if (Math.hypot(p.x - a.x0, p.y - a.y0) > this.opts.moveTolerancePx) return;
      const tap: TapInfo = { x: a.x0, y: a.y0, pointerType: e.pointerType };
      this.handlers.onTap?.(tap);
      const last = this.last;
      if (
        last &&
        e.timeStamp - last.t <= this.opts.doubleTapMs &&
        Math.hypot(tap.x - last.x, tap.y - last.y) <= this.opts.doubleTapDistancePx
      ) {
        this.last = null;
        this.handlers.onDoubleTap?.(tap);
      } else {
        this.last = { x: tap.x, y: tap.y, t: e.timeStamp };
      }
    };
    this.onCancel = (e) => {
      this.active.delete(e.pointerId);
      this.multi = this.active.size > 0;
      this.last = null;
    };
    element.addEventListener('pointerdown', this.onDown);
    element.addEventListener('pointermove', this.onMove);
    element.addEventListener('pointerup', this.onUp);
    element.addEventListener('pointercancel', this.onCancel);
  }

  private local(e: PointerEvent): { x: number; y: number } {
    const r = this.element.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  dispose(): void {
    this.element.removeEventListener('pointerdown', this.onDown);
    this.element.removeEventListener('pointermove', this.onMove);
    this.element.removeEventListener('pointerup', this.onUp);
    this.element.removeEventListener('pointercancel', this.onCancel);
    this.active.clear();
  }
}

/** Eased animation of the orbit target (camera position kept, view turns to the target). */
export class TargetAnimator {
  private from = new Vector3();
  private to = new Vector3();
  private t0 = 0;
  private active = false;

  constructor(
    private readonly controls: OrbitControls,
    private readonly durationMs = 320,
  ) {}

  get running(): boolean {
    return this.active;
  }

  start(target: Vector3, now: number): void {
    this.from.copy(this.controls.target);
    this.to.copy(target);
    this.t0 = now;
    this.active = true;
  }

  cancel(): void {
    this.active = false;
  }

  /** Advance; returns true while still animating. */
  step(now: number): boolean {
    if (!this.active) return false;
    const u = Math.min(1, Math.max(0, (now - this.t0) / this.durationMs));
    const e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2; // easeInOutCubic
    this.controls.target.lerpVectors(this.from, this.to, e);
    if (u >= 1) this.active = false;
    return this.active;
  }
}
