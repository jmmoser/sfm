/**
 * Coverage ring of the capture screen (object mode): the coverage map of src/app/coverage.ts
 * seen from above, as an SVG polar dial, plus its one-line prompt.
 *
 * Angle = azimuth around the object (sector 0, the first keyframe, at the bottom; the dial runs
 * counter-clockwise like an orbit seen from above, so the marker moves the way the user walks),
 * radius = elevation: the high band (looking down) is the inner ring, then mid, then low. Without
 * elevation (no motion sensor) one azimuth ring. A cell is empty, partly covered (one keyframe) or
 * covered; the empty cells of the band the prompt asks for pulse, and a dot marks where the phone
 * is (LiveUpdate.framePose). The cells are built once per layout and only re-classed per update
 * (keyframe rate); the prompt text changes only when the prompt does, so its live region speaks
 * rarely. No inline styles (CSP): colours come from classes in styles.css.
 */
import { COVERAGE_SECTORS, type Coverage, type CoverageCell } from '../coverage';
import { h } from '../dom';
import { coveragePromptText } from '../format';

const SVG_NS = 'http://www.w3.org/2000/svg';
const SECTOR = (2 * Math.PI) / COVERAGE_SECTORS;
/** Gap between neighbouring cells (rad, each side). */
const GAP = 1.5 * (Math.PI / 180);
/** [inner, outer] radius per band (viewBox −50…50): low outermost, high innermost. */
const BAND_RADII: readonly (readonly [number, number])[] = [
  [37, 45],
  [23, 35],
  [9, 21],
];
/** The single azimuth-only ring. */
const FLAT_RADII: readonly [number, number] = [25, 45];

/** Point at radius r and azimuth a (rad) in SVG coordinates: azimuth 0 at the bottom, counter-clockwise on screen. */
export function ringPoint(r: number, a: number): [number, number] {
  return [r * Math.sin(a), r * Math.cos(a)];
}

/** SVG path of the annular sector r0…r1 between azimuths a0 < a1 (rad, < π apart). */
export function sectorPath(r0: number, r1: number, a0: number, a1: number): string {
  const f = (v: number): string => {
    const r = Math.round(v * 1000) / 1000;
    return String(r === 0 ? 0 : r); // no "-0"
  };
  const p = (r: number, a: number): string => ringPoint(r, a).map(f).join(' ');
  // increasing azimuth runs counter-clockwise on screen: sweep-flag 0 outwards, 1 back
  return `M${p(r1, a0)}A${r1} ${r1} 0 0 0 ${p(r1, a1)}L${p(r0, a1)}A${r0} ${r0} 0 0 1 ${p(r0, a0)}Z`;
}

function svg<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>): SVGElementTagNameMap[K] {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

export interface CoverageRing {
  /** `.coverage`: the dial and the prompt line (hidden until the screen shows it). */
  readonly el: HTMLDivElement;
  /** Draw a coverage map; null / not stable: an empty, dimmed dial without a prompt. */
  update(cov: Coverage | null): void;
  /** Mark where the phone is (null: not known). */
  setCurrent(cell: CoverageCell | null): void;
}

export function createCoverageRing(): CoverageRing {
  const dial = svg('svg', { class: 'coverage-ring', viewBox: '-50 -50 100 100', 'aria-hidden': 'true', focusable: 'false' });
  const cellsGroup = svg('g', {});
  const marker = svg('circle', { class: 'coverage-marker', r: 4.2, cx: 0, cy: 0 });
  dial.append(cellsGroup, svg('circle', { class: 'coverage-centre', r: 3, cx: 0, cy: 0 }), marker);
  const prompt = h('div', { class: 'coverage-prompt', role: 'status', 'aria-live': 'polite', hidden: true, 'data-testid': 'coverage-prompt' });
  const el = h('div', { class: 'coverage locating', hidden: true, 'data-testid': 'coverage', dataset: { prompt: 'locating' } });
  el.append(dial, prompt);

  /** cells[band][sector] (azimuth-only: one band). */
  let cells: SVGPathElement[][] = [];
  let layout: 'bands' | 'flat' | null = null;
  let elevation = true;
  let promptKey = '';

  function build(kind: 'bands' | 'flat'): void {
    if (layout === kind) return;
    layout = kind;
    cellsGroup.replaceChildren();
    const radii = kind === 'bands' ? BAND_RADII : [FLAT_RADII];
    cells = radii.map(([r0, r1], b) =>
      Array.from({ length: COVERAGE_SECTORS }, (_, s) => {
        const a = s * SECTOR;
        const path = svg('path', { d: sectorPath(r0, r1, a - SECTOR / 2 + GAP, a + SECTOR / 2 - GAP), class: `coverage-cell b${kind === 'bands' ? b : 'f'} c0` });
        cellsGroup.append(path);
        return path;
      }),
    );
    el.classList.toggle('flat', kind === 'flat');
  }

  function setCurrent(cell: CoverageCell | null): void {
    const show = !!cell && !el.classList.contains('locating') && (layout === 'flat' || cell.band >= 0);
    marker.classList.toggle('shown', show);
    if (!show || !cell) return;
    const [r0, r1] = layout === 'flat' ? FLAT_RADII : BAND_RADII[cell.band];
    const [x, y] = ringPoint((r0 + r1) / 2, cell.sector * SECTOR);
    marker.setAttribute('cx', x.toFixed(2));
    marker.setAttribute('cy', y.toFixed(2));
  }

  function update(cov: Coverage | null): void {
    if (cov) elevation = cov.elevation;
    build(elevation ? 'bands' : 'flat');
    const stable = !!cov && cov.stable;
    el.classList.toggle('locating', !stable);
    // the band whose empty cells the prompt asks to fill (−1: none)
    const target = !stable ? -1 : cov.prompt === 'look_down' ? 2 : cov.prompt === 'go_lower' ? 1 : cov.prompt === 'other_side' || cov.prompt === 'fill_gaps' ? (elevation ? cov.primaryBand : 0) : -1;
    for (let b = 0; b < cells.length; b++) {
      for (let s = 0; s < COVERAGE_SECTORS; s++) {
        const k = !stable ? 0 : layout === 'flat' ? cov.sectors[s] : cov.counts[b][s];
        const state = k >= (cov?.minPerCell ?? 2) ? 'c2' : k > 0 ? 'c1' : 'c0';
        const pulse = b === target && state !== 'c2';
        const cls = `coverage-cell b${layout === 'flat' ? 'f' : b} ${state}${pulse ? ' target' : ''}`;
        if (cells[b][s].getAttribute('class') !== cls) cells[b][s].setAttribute('class', cls);
      }
    }
    el.dataset.prompt = stable ? cov.prompt : 'locating';
    // the live region speaks only when the prompt changes
    const text = stable ? coveragePromptText(cov.prompt, elevation) : '';
    if (text !== promptKey) {
      promptKey = text;
      prompt.textContent = text;
      prompt.hidden = !text;
    }
    if (!stable) setCurrent(null);
  }

  update(null);
  return { el, update, setCurrent };
}
