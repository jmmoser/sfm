/**
 * Measure tool UI: two tapped points → markers, a line and a distance label.
 * Picking itself is done by the Viewer (GPU depth readback or CPU fallback);
 * this module only keeps the state and the three.js / DOM decorations.
 * Points are in the WORLD frame (the object lives under the world root).
 */
import {
  BufferAttribute,
  BufferGeometry,
  Group,
  Line,
  LineBasicMaterial,
  Mesh,
  MeshBasicMaterial,
  SphereGeometry,
  Vector3,
  type Object3D,
  type PerspectiveCamera,
} from 'three';
import type { Vec3 } from '../core/types';
import { formatDistance } from './helpers';

export type MeasureCallback = (distance: number, a: Vec3, b: Vec3) => void;

/** Euclidean distance of two world points. */
export function measureDistance(a: ArrayLike<number>, b: ArrayLike<number>): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

const _v = new Vector3();
const _w = new Vector3();

export class MeasureTool {
  /** Add under the world root of the overlay scene. */
  readonly object = new Group();
  private readonly markerGeom = new SphereGeometry(1, 16, 12);
  private readonly markerMatA = new MeshBasicMaterial({ color: '#ffd400', depthTest: false, transparent: true });
  private readonly markerMatB = new MeshBasicMaterial({ color: '#ff5a36', depthTest: false, transparent: true });
  private readonly markerA: Mesh;
  private readonly markerB: Mesh;
  private readonly lineGeom = new BufferGeometry();
  private readonly lineMat = new LineBasicMaterial({ color: '#ffd400', depthTest: false, transparent: true });
  private readonly line: Line;
  private readonly label: HTMLDivElement;
  private a: Vec3 | null = null;
  private b: Vec3 | null = null;
  private callback: MeasureCallback | null = null;
  private metric = false;
  /** Marker radius in CSS px. */
  markerRadiusPx = 6;
  private labelX = NaN;
  private labelY = NaN;

  constructor(container: HTMLElement) {
    this.markerA = new Mesh(this.markerGeom, this.markerMatA);
    this.markerB = new Mesh(this.markerGeom, this.markerMatB);
    this.lineGeom.setAttribute('position', new BufferAttribute(new Float32Array(6), 3));
    this.line = new Line(this.lineGeom, this.lineMat);
    for (const o of [this.markerA, this.markerB, this.line]) {
      o.renderOrder = 1000;
      o.frustumCulled = false;
      o.visible = false;
      this.object.add(o);
    }
    this.line.renderOrder = 999;
    this.object.name = 'measure';
    const el = document.createElement('div');
    el.className = 'viewer-measure-label';
    Object.assign(el.style, {
      position: 'absolute',
      left: '0px',
      top: '0px',
      display: 'none',
      pointerEvents: 'none',
      padding: '3px 8px',
      borderRadius: '6px',
      background: 'rgba(12,14,18,0.78)',
      color: '#fff',
      font: '600 13px/1.3 -apple-system, system-ui, sans-serif',
      whiteSpace: 'nowrap',
      transform: 'translate(-9999px,-9999px)',
      zIndex: '2',
      userSelect: 'none',
    } satisfies Partial<CSSStyleDeclaration>);
    container.appendChild(el);
    this.label = el;
  }

  get active(): boolean {
    return this.callback !== null;
  }

  get points(): { a: Vec3 | null; b: Vec3 | null } {
    return { a: this.a, b: this.b };
  }

  /** Current label text ('' when no complete measurement). */
  get text(): string {
    return this.a && this.b ? formatDistance(measureDistance(this.a, this.b), this.metric) : '';
  }

  start(cb: MeasureCallback): void {
    this.callback = cb;
    this.clear();
  }

  stop(): void {
    this.callback = null;
    this.clear();
  }

  clear(): void {
    this.labelX = NaN;
    this.a = null;
    this.b = null;
    this.markerA.visible = false;
    this.markerB.visible = false;
    this.line.visible = false;
    this.label.style.display = 'none';
  }

  setUnits(metric: boolean): void {
    this.metric = metric;
    if (this.a && this.b) this.label.textContent = this.text;
  }

  /**
   * Feed a picked world point. First point → marker A; second → marker B, line,
   * label and callback; a third starts a new measurement. Returns true if the
   * display changed.
   */
  addPoint(p: Vec3 | null): boolean {
    if (!this.callback || !p) return false;
    if (!this.a || this.b) {
      this.clear();
      this.a = Float64Array.from(p);
      this.markerA.position.set(p[0], p[1], p[2]);
      this.markerA.visible = true;
      return true;
    }
    this.b = Float64Array.from(p);
    this.markerB.position.set(p[0], p[1], p[2]);
    this.markerB.visible = true;
    const pos = this.lineGeom.getAttribute('position') as BufferAttribute;
    pos.setXYZ(0, this.a[0], this.a[1], this.a[2]);
    pos.setXYZ(1, p[0], p[1], p[2]);
    pos.needsUpdate = true;
    this.lineGeom.computeBoundingSphere();
    this.line.visible = true;
    this.label.textContent = this.text;
    this.label.style.display = 'block';
    const d = measureDistance(this.a, this.b);
    this.callback(d, this.a.slice() as Vec3, this.b.slice() as Vec3);
    return true;
  }

  /**
   * Per-frame update: constant on-screen marker size and label placement.
   * `root` is the world root (world → three transform), sizes in CSS px.
   */
  update(camera: PerspectiveCamera, root: Object3D, cssWidth: number, cssHeight: number): void {
    const k = (2 * Math.tan((camera.fov * Math.PI) / 360) * this.markerRadiusPx) / Math.max(1, cssHeight);
    for (const m of [this.markerA, this.markerB]) {
      if (!m.visible) continue;
      _v.copy(m.position);
      root.localToWorld(_v);
      const dist = _v.distanceTo(camera.position);
      m.scale.setScalar(Math.max(1e-12, dist * k));
    }
    if (!(this.a && this.b)) return;
    _v.set((this.a[0] + this.b[0]) / 2, (this.a[1] + this.b[1]) / 2, (this.a[2] + this.b[2]) / 2);
    root.localToWorld(_v);
    _w.copy(_v).project(camera);
    const visible = _w.z > -1 && _w.z < 1 && Math.abs(_w.x) <= 1.2 && Math.abs(_w.y) <= 1.2;
    if (!visible) {
      this.label.style.display = 'none';
      this.labelX = NaN;
      return;
    }
    const x = (_w.x * 0.5 + 0.5) * cssWidth;
    const y = (0.5 - _w.y * 0.5) * cssHeight;
    this.labelX = x;
    this.labelY = y;
    this.label.style.display = 'block';
    this.label.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) translate(-50%, -140%)`;
  }

  /** Label geometry for screenshots: text and CSS-px anchor, or null. */
  labelForScreenshot(): { text: string; x: number; y: number } | null {
    if (!(this.a && this.b) || this.label.style.display === 'none' || !Number.isFinite(this.labelX)) return null;
    return { text: this.text, x: this.labelX, y: this.labelY };
  }

  dispose(): void {
    this.markerGeom.dispose();
    this.markerMatA.dispose();
    this.markerMatB.dispose();
    this.lineGeom.dispose();
    this.lineMat.dispose();
    this.label.remove();
  }
}
