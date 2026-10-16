/**
 * Procedural, deterministic (seeded) textures drawn with Canvas 2D + per-pixel noise.
 * Browser-only (OffscreenCanvas or <canvas>). Every texture has structure at many
 * scales (shapes, edges, corners, blobs, fine grain) so that feature detectors find
 * keypoints at all distances, and nothing is a short-period repetition unless the
 * real-world material is (bricks, shingles).
 */
import { SynthRng } from './rng';
import { fbm, worley, type Raster } from './noise';

export type Canvas2D = OffscreenCanvas | HTMLCanvasElement;
type Ctx = OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;
type RGB = [number, number, number];

export function makeCanvas(w: number, h: number): { canvas: Canvas2D; ctx: Ctx } {
  w = Math.max(1, Math.round(w));
  h = Math.max(1, Math.round(h));
  let canvas: Canvas2D;
  if (typeof OffscreenCanvas !== 'undefined') canvas = new OffscreenCanvas(w, h);
  else {
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    canvas = c;
  }
  const ctx = canvas.getContext('2d', { willReadFrequently: true }) as Ctx | null;
  if (!ctx) throw new Error('2D canvas context unavailable');
  return { canvas, ctx };
}

// ---------------------------------------------------------------------------
// Colour + pixel helpers
// ---------------------------------------------------------------------------

export function rgbStr(c: RGB, a = 1): string {
  return `rgba(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])},${a})`;
}

export function hsl(h: number, s: number, l: number): RGB {
  h = ((h % 360) + 360) % 360;
  s = Math.max(0, Math.min(1, s));
  l = Math.max(0, Math.min(1, l));
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let r = 0,
    g = 0,
    b = 0;
  if (h < 60) [r, g, b] = [c, x, 0];
  else if (h < 120) [r, g, b] = [x, c, 0];
  else if (h < 180) [r, g, b] = [0, c, x];
  else if (h < 240) [r, g, b] = [0, x, c];
  else if (h < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  return [(r + m) * 255, (g + m) * 255, (b + m) * 255];
}

function mix(a: RGB, b: RGB, t: number): RGB {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

function jitterColor(c: RGB, rng: SynthRng, amount: number): RGB {
  const k = 1 + amount * (2 * rng.next() - 1);
  return [c[0] * k, c[1] * k, c[2] * k];
}

function randomVivid(rng: SynthRng): RGB {
  return hsl(rng.next() * 360, 0.45 + 0.5 * rng.next(), 0.25 + 0.5 * rng.next());
}

/** Fill the canvas with colours computed per pixel. */
function paintPixels(ctx: Ctx, w: number, h: number, fn: (x: number, y: number, i: number, out: Float32Array) => void): void {
  const img = ctx.createImageData(w, h);
  const d = img.data;
  const c = new Float32Array(3);
  for (let y = 0, i = 0; y < h; y++) {
    for (let x = 0; x < w; x++, i++) {
      fn(x, y, i, c);
      d[4 * i] = c[0];
      d[4 * i + 1] = c[1];
      d[4 * i + 2] = c[2];
      d[4 * i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

/** Multiplicative luminance modulation by a raster in [−1, 1] plus fine white grain. */
function modulate(ctx: Ctx, w: number, h: number, r: Raster | null, amount: number, grain: number, rng: SynthRng): void {
  const img = ctx.getImageData(0, 0, w, h);
  const d = img.data;
  const n = w * h;
  for (let i = 0; i < n; i++) {
    let k = 1;
    if (r) k += amount * r.data[i];
    const g = grain > 0 ? grain * 255 * (rng.next() + rng.next() - 1) : 0;
    d[4 * i] = d[4 * i] * k + g;
    d[4 * i + 1] = d[4 * i + 1] * k + g;
    d[4 * i + 2] = d[4 * i + 2] * k + g;
  }
  ctx.putImageData(img, 0, 0);
}

/** Draw `fn` at offsets so that shapes wrap around the edges (tileable textures). */
function drawWrapped(w: number, h: number, x: number, y: number, r: number, fn: (ox: number, oy: number) => void): void {
  for (const ox of [0, -w, w]) {
    if (ox !== 0 && (x + ox + r < 0 || x + ox - r > w)) continue;
    for (const oy of [0, -h, h]) {
      if (oy !== 0 && (y + oy + r < 0 || y + oy - r > h)) continue;
      fn(ox, oy);
    }
  }
}

// ---------------------------------------------------------------------------
// Pseudo text (font-independent, deterministic)
// ---------------------------------------------------------------------------

/** One pseudo glyph of cap height h at (x, y = baseline). Returns its advance. */
function drawGlyph(ctx: Ctx, x: number, y: number, h: number, rng: SynthRng): number {
  const w = h * (0.45 + 0.3 * rng.next());
  const strokes = 1 + rng.int(3);
  ctx.beginPath();
  for (let s = 0; s < strokes; s++) {
    const kind = rng.int(6);
    const x0 = x + w * rng.next(),
      x1 = x + w * rng.next();
    const yt = y - h,
      ym = y - h / 2;
    switch (kind) {
      case 0:
        ctx.moveTo(x0, y);
        ctx.lineTo(x0, yt);
        break;
      case 1: {
        const yy = rng.pick([yt, ym, y]);
        ctx.moveTo(x, yy);
        ctx.lineTo(x + w, yy);
        break;
      }
      case 2:
        ctx.moveTo(x0, y);
        ctx.lineTo(x1, yt);
        break;
      case 3:
        ctx.moveTo(x + w / 2 + w / 2, ym);
        ctx.arc(x + w / 2, ym, Math.min(w, h) / 2, 0, Math.PI * (1 + rng.next()));
        break;
      case 4:
        ctx.moveTo(x + w, y - h * 0.25);
        ctx.arc(x + w / 2, y - h * 0.25, w / 2, 0, Math.PI * 2);
        break;
      default:
        ctx.moveTo(x0, ym);
        ctx.lineTo(x1, y);
        ctx.lineTo(x + w, ym);
    }
  }
  ctx.stroke();
  return w + h * 0.18;
}

/** Lines of pseudo text inside a box. */
function drawText(ctx: Ctx, x: number, y: number, w: number, lineH: number, lines: number, color: string, rng: SynthRng, justify = false): void {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(0.6, lineH * 0.09);
  ctx.lineCap = 'round';
  const cap = lineH * 0.62;
  for (let l = 0; l < lines; l++) {
    const base = y + (l + 1) * lineH;
    let cx = x;
    const end = x + w * (l === lines - 1 && !justify ? 0.35 + 0.6 * rng.next() : 1);
    while (cx < end - cap) {
      const n = 2 + rng.int(7);
      for (let g = 0; g < n && cx < end - cap * 0.5; g++) cx += drawGlyph(ctx, cx, base, cap, rng);
      cx += cap * 0.55;
    }
  }
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Generic materials
// ---------------------------------------------------------------------------

/** Painted plaster: subtle low-frequency mottling, speckles, a few scuffs. */
export function plasterTexture(w: number, h: number, rng: SynthRng, base: RGB): Canvas2D {
  const { canvas, ctx } = makeCanvas(w, h);
  const f = fbm(w, h, rng, { baseCells: 6, octaves: 7, gain: 0.6 });
  paintPixels(ctx, w, h, (_x, _y, i, c) => {
    const k = 1 + 0.06 * f.data[i];
    c[0] = base[0] * k;
    c[1] = base[1] * k;
    c[2] = base[2] * k;
  });
  // scuffs and stains
  for (let i = 0; i < 40; i++) {
    ctx.strokeStyle = rgbStr(mix(base, [60, 55, 50], 0.5), 0.08 + 0.12 * rng.next());
    ctx.lineWidth = 1 + 3 * rng.next();
    ctx.beginPath();
    const x = rng.next() * w,
      y = rng.next() * h;
    ctx.moveTo(x, y);
    ctx.quadraticCurveTo(x + rng.uniform(-60, 60), y + rng.uniform(-30, 30), x + rng.uniform(-90, 90), y + rng.uniform(-40, 40));
    ctx.stroke();
  }
  for (let i = 0; i < 2500; i++) {
    ctx.fillStyle = rgbStr(mix(base, rng.next() < 0.5 ? [0, 0, 0] : [255, 255, 255], 0.25), 0.5);
    ctx.fillRect(rng.next() * w, rng.next() * h, 1 + rng.next() * 2, 1 + rng.next() * 2);
  }
  modulate(ctx, w, h, null, 0, 0.02, rng);
  return canvas;
}

/** Wood boards (horizontal grain) with seams, knots and fine grain. */
export function woodTexture(w: number, h: number, rng: SynthRng, opts: { light?: RGB; dark?: RGB; boards?: number; stagger?: boolean } = {}): Canvas2D {
  const light = opts.light ?? [196, 150, 100];
  const dark = opts.dark ?? [120, 78, 44];
  const boards = opts.boards ?? 6;
  const { canvas, ctx } = makeCanvas(w, h);
  const warp = fbm(w, h, rng, { baseCells: 3, octaves: 5, gain: 0.5 });
  const fine = fbm(w, h, rng, { baseCells: 64, octaves: 3, gain: 0.6 });
  const bh = h / boards;
  const boardTint: RGB[] = [];
  const boardPhase: number[] = [];
  const boardFreq: number[] = [];
  const cuts: number[][] = [];
  for (let b = 0; b < boards; b++) {
    boardTint.push(jitterColor([1, 1, 1], rng, 0.12));
    boardPhase.push(rng.next() * 10);
    boardFreq.push((3 + 4 * rng.next()) / bh);
    const c: number[] = [];
    if (opts.stagger) {
      let x = -rng.next() * w * 0.5;
      while (x < w) {
        x += w * (0.3 + 0.4 * rng.next());
        c.push(x);
      }
    }
    cuts.push(c);
  }
  paintPixels(ctx, w, h, (x, y, i, c) => {
    const b = Math.min(boards - 1, Math.floor(y / bh));
    const yy = y - b * bh;
    const ring = (yy + 18 * warp.data[i] * (bh / 40)) * boardFreq[b] + boardPhase[b] + 0.002 * x;
    let v = 0.5 + 0.5 * Math.sin(2 * Math.PI * ring);
    v = Math.pow(v, 3);
    const t = Math.min(1, Math.max(0, 0.15 + 0.55 * v + 0.25 * fine.data[i]));
    const col = mix(light, dark, t);
    const tint = boardTint[b];
    let seam = 1;
    if (yy < 1.5 || yy > bh - 1.5) seam = 0.45;
    for (const cx of cuts[b]) if (Math.abs(x - cx) < 1.2) seam = 0.45;
    c[0] = col[0] * tint[0] * seam;
    c[1] = col[1] * tint[1] * seam;
    c[2] = col[2] * tint[2] * seam;
  });
  // knots
  const knots = Math.round(boards * (1 + 2 * rng.next()));
  for (let k = 0; k < knots; k++) {
    const x = rng.next() * w,
      y = rng.next() * h;
    const r = bh * (0.06 + 0.12 * rng.next());
    for (let ring = 4; ring >= 0; ring--) {
      ctx.fillStyle = rgbStr(mix(dark, [50, 30, 15], 0.2 + 0.15 * ring), 0.35);
      ctx.beginPath();
      ctx.ellipse(x, y, r * (1 + ring * 0.6) * 1.8, r * (1 + ring * 0.4), 0, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  modulate(ctx, w, h, null, 0, 0.025, rng);
  return canvas;
}

/** Geometric / art / text posters. */
export function posterTexture(w: number, h: number, rng: SynthRng, style: 'abstract' | 'photo' | 'text' | 'pattern'): Canvas2D {
  const { canvas, ctx } = makeCanvas(w, h);
  const bg = hsl(rng.next() * 360, 0.3 + 0.4 * rng.next(), 0.6 + 0.3 * rng.next());
  const g = ctx.createLinearGradient(0, 0, w, h);
  g.addColorStop(0, rgbStr(bg));
  g.addColorStop(1, rgbStr(hsl(rng.next() * 360, 0.5, 0.5)));
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
  const S = Math.min(w, h);
  if (style === 'abstract') {
    for (let i = 0; i < 70; i++) {
      ctx.fillStyle = rgbStr(randomVivid(rng), 0.5 + 0.5 * rng.next());
      ctx.strokeStyle = rgbStr(randomVivid(rng), 1);
      ctx.lineWidth = 1 + rng.next() * S * 0.02;
      const x = rng.next() * w,
        y = rng.next() * h,
        r = S * (0.01 + 0.18 * Math.pow(rng.next(), 2));
      const kind = rng.int(5);
      ctx.beginPath();
      if (kind === 0) ctx.arc(x, y, r, 0, Math.PI * 2);
      else if (kind === 1) ctx.rect(x - r, y - r * 0.6, 2 * r, 1.2 * r);
      else if (kind === 2) {
        ctx.moveTo(x, y - r);
        ctx.lineTo(x + r, y + r);
        ctx.lineTo(x - r, y + r * 0.7);
        ctx.closePath();
      } else if (kind === 3) {
        ctx.moveTo(x, y);
        ctx.bezierCurveTo(x + rng.uniform(-3, 3) * r, y + rng.uniform(-3, 3) * r, x + rng.uniform(-3, 3) * r, y + rng.uniform(-3, 3) * r, x + rng.uniform(-3, 3) * r, y + rng.uniform(-3, 3) * r);
        ctx.stroke();
        continue;
      } else {
        for (let k = 0; k < 7; k++) ctx.arc(x, y, r * (1 - k / 7), 0, Math.PI * 2);
        ctx.stroke();
        continue;
      }
      if (rng.next() < 0.7) ctx.fill();
      else ctx.stroke();
    }
    drawText(ctx, w * 0.08, h * 0.82, w * 0.84, S * 0.05, 2, 'rgba(20,20,20,0.9)', rng);
  } else if (style === 'photo') {
    const f = fbm(w, h, rng, { baseCells: 4, octaves: 8, gain: 0.6 });
    const sky = hsl(200 + 30 * rng.next(), 0.5, 0.7);
    const land = hsl(80 + 60 * rng.next(), 0.4, 0.35);
    const rock = hsl(30, 0.15, 0.45);
    const horizon = h * (0.35 + 0.2 * rng.next());
    paintPixels(ctx, w, h, (x, y, i, c) => {
      const ridge = horizon + 0.15 * h * f.data[(Math.floor(h * 0.3) * w + x) % (w * h)];
      let col: RGB;
      if (y < ridge) col = mix(sky, [250, 250, 255], 0.5 + 0.5 * f.data[i] * 0.6);
      else col = mix(mix(rock, land, Math.min(1, (y - ridge) / (h * 0.3))), [20, 30, 10], 0.3 + 0.3 * f.data[i]);
      c[0] = col[0];
      c[1] = col[1];
      c[2] = col[2];
    });
    for (let i = 0; i < 90; i++) {
      const x = rng.next() * w,
        y = horizon + rng.next() * (h - horizon);
      const s = (S * 0.01 + S * 0.05 * ((y - horizon) / h)) * (0.5 + rng.next());
      ctx.fillStyle = rgbStr(hsl(100 + 40 * rng.next(), 0.5, 0.15 + 0.2 * rng.next()), 0.9);
      ctx.beginPath();
      ctx.ellipse(x, y - s, s * 0.6, s * 1.4, 0, 0, Math.PI * 2);
      ctx.fill();
    }
  } else if (style === 'text') {
    ctx.fillStyle = 'rgb(240,236,226)';
    ctx.fillRect(0, 0, w, h);
    drawText(ctx, w * 0.06, h * 0.03, w * 0.88, S * 0.09, 1, 'rgb(150,20,20)', rng);
    const cols = 2 + rng.int(2);
    const cw = (w * 0.88) / cols;
    for (let c = 0; c < cols; c++) {
      let y = h * 0.16;
      while (y < h * 0.92) {
        if (rng.next() < 0.25) {
          const ih = Math.min(h * 0.94 - y, cw * (0.5 + 0.4 * rng.next()));
          ctx.fillStyle = rgbStr(randomVivid(rng));
          ctx.fillRect(w * 0.06 + c * cw + 4, y, cw - 12, ih);
          for (let k = 0; k < 12; k++) {
            ctx.fillStyle = rgbStr(randomVivid(rng), 0.8);
            ctx.beginPath();
            ctx.arc(w * 0.06 + c * cw + 4 + rng.next() * (cw - 12), y + rng.next() * ih, rng.next() * cw * 0.15, 0, Math.PI * 2);
            ctx.fill();
          }
          y += ih + S * 0.02;
        } else {
          const n = 3 + rng.int(6);
          drawText(ctx, w * 0.06 + c * cw + 4, y, cw - 12, S * 0.028, n, 'rgb(30,30,30)', rng, true);
          y += (n + 1) * S * 0.028;
        }
      }
    }
  } else {
    const n = 4 + rng.int(4);
    const cs = w / n;
    const rows = Math.ceil(h / cs);
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < n; c++) {
        const x = c * cs,
          y = r * cs;
        ctx.fillStyle = rgbStr(randomVivid(rng));
        ctx.fillRect(x, y, cs, cs);
        ctx.fillStyle = rgbStr(randomVivid(rng));
        ctx.strokeStyle = rgbStr(randomVivid(rng));
        ctx.lineWidth = cs * (0.02 + 0.06 * rng.next());
        const k = rng.int(6);
        ctx.beginPath();
        if (k === 0) {
          ctx.arc(x + cs / 2, y + cs / 2, cs * (0.15 + 0.3 * rng.next()), 0, Math.PI * 2);
          ctx.fill();
        } else if (k === 1) {
          ctx.moveTo(x, y);
          ctx.lineTo(x + cs, y + cs);
          ctx.moveTo(x + cs, y);
          ctx.lineTo(x, y + cs);
          ctx.stroke();
        } else if (k === 2) {
          for (let s = 0; s < 6; s++) ctx.rect(x + (s * cs) / 6, y, cs / 12, cs);
          ctx.fill();
        } else if (k === 3) {
          ctx.moveTo(x + rng.next() * cs, y);
          ctx.lineTo(x + cs, y + rng.next() * cs);
          ctx.lineTo(x, y + cs);
          ctx.fill();
        } else if (k === 4) {
          ctx.arc(x + (rng.next() < 0.5 ? 0 : cs), y + (rng.next() < 0.5 ? 0 : cs), cs * 0.7, 0, Math.PI * 2);
          ctx.save();
          ctx.clip();
          ctx.fillRect(x, y, cs, cs);
          ctx.restore();
        } else {
          for (let s = 0; s < 9; s++) {
            ctx.moveTo(x + cs / 2 + (cs / 2) * Math.cos(s), y + cs / 2 + (cs / 2) * Math.sin(s * 1.7));
            ctx.lineTo(x + cs / 2, y + cs / 2);
          }
          ctx.stroke();
        }
      }
  }
  // frame
  ctx.strokeStyle = 'rgb(25,25,25)';
  ctx.lineWidth = S * 0.025;
  ctx.strokeRect(0, 0, w, h);
  modulate(ctx, w, h, null, 0, 0.02, rng);
  return canvas;
}

/** Oriental-style rug: borders of repeating motifs, central medallion, noise. */
export function rugTexture(w: number, h: number, rng: SynthRng): Canvas2D {
  const { canvas, ctx } = makeCanvas(w, h);
  const field = hsl(355 + 10 * rng.next(), 0.55, 0.32);
  ctx.fillStyle = rgbStr(field);
  ctx.fillRect(0, 0, w, h);
  const pal: RGB[] = [hsl(40, 0.6, 0.6), hsl(210, 0.5, 0.3), hsl(30, 0.3, 0.85), hsl(150, 0.35, 0.3), hsl(15, 0.7, 0.45)];
  for (let b = 0; b < 4; b++) {
    const m = (b + 1) * Math.min(w, h) * 0.04;
    ctx.strokeStyle = rgbStr(pal[b % pal.length]);
    ctx.lineWidth = Math.min(w, h) * 0.012;
    ctx.strokeRect(m, m, w - 2 * m, h - 2 * m);
    const step = Math.min(w, h) * 0.03;
    ctx.fillStyle = rgbStr(pal[(b + 2) % pal.length]);
    for (let x = m; x < w - m; x += step) {
      const s = step * (0.2 + 0.2 * rng.next());
      ctx.beginPath();
      ctx.moveTo(x, m - s);
      ctx.lineTo(x + s, m);
      ctx.lineTo(x, m + s);
      ctx.lineTo(x - s, m);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x + step / 2, h - m, s * 0.7, 0, Math.PI * 2);
      ctx.fill();
    }
    for (let y = m; y < h - m; y += step) {
      const s = step * (0.2 + 0.2 * rng.next());
      ctx.fillRect(m - s / 2, y, s, s * 0.6);
      ctx.beginPath();
      ctx.arc(w - m, y + step / 2, s * 0.6, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  // medallion & scattered flowers
  const cx = w / 2,
    cy = h / 2;
  for (let r = 10; r > 0; r--) {
    ctx.fillStyle = rgbStr(pal[r % pal.length]);
    ctx.beginPath();
    const R = Math.min(w, h) * 0.035 * r;
    for (let k = 0; k <= 64; k++) {
      const a = (k / 64) * Math.PI * 2;
      const rr = R * (1 + 0.12 * Math.sin(a * (6 + (r % 3) * 2)));
      if (k === 0) ctx.moveTo(cx + rr * Math.cos(a) * 1.4, cy + rr * Math.sin(a));
      else ctx.lineTo(cx + rr * Math.cos(a) * 1.4, cy + rr * Math.sin(a));
    }
    ctx.fill();
  }
  for (let i = 0; i < 260; i++) {
    const x = w * (0.2 + 0.6 * rng.next()),
      y = h * (0.2 + 0.6 * rng.next());
    const s = Math.min(w, h) * (0.004 + 0.012 * rng.next());
    ctx.fillStyle = rgbStr(rng.pick(pal));
    for (let p = 0; p < 5; p++) {
      ctx.beginPath();
      ctx.arc(x + s * Math.cos((p * 2 * Math.PI) / 5), y + s * Math.sin((p * 2 * Math.PI) / 5), s * 0.7, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  const f = fbm(w, h, rng, { baseCells: 8, octaves: 6 });
  modulate(ctx, w, h, f, 0.12, 0.06, rng); // woven grain
  return canvas;
}

/** Book spines on shelves. */
export function bookshelfTexture(w: number, h: number, rng: SynthRng, shelves: number): Canvas2D {
  const { canvas, ctx } = makeCanvas(w, h);
  ctx.fillStyle = 'rgb(90,60,38)';
  ctx.fillRect(0, 0, w, h);
  const sh = h / shelves;
  for (let s = 0; s < shelves; s++) {
    let x = w * 0.02;
    const y1 = (s + 1) * sh - sh * 0.06;
    while (x < w * 0.98) {
      const bw = sh * (0.06 + 0.12 * rng.next());
      const bh = sh * (0.55 + 0.35 * rng.next());
      if (rng.next() < 0.06) {
        x += bw * 2;
        continue;
      }
      const col = randomVivid(rng);
      ctx.fillStyle = rgbStr(col);
      ctx.fillRect(x, y1 - bh, bw - 1, bh);
      ctx.fillStyle = rgbStr(mix(col, [255, 230, 160], 0.6));
      for (let b = 0; b < 3; b++) ctx.fillRect(x + 1, y1 - bh * (0.15 + 0.3 * b) , bw - 3, sh * 0.01 + 1);
      ctx.save();
      ctx.translate(x + bw * 0.7, y1 - bh * 0.1);
      ctx.rotate(-Math.PI / 2);
      drawText(ctx, 0, -bw * 0.55, bh * 0.75, bw * 0.45, 1, rgbStr(mix(col, [255, 255, 255], 0.7)), rng);
      ctx.restore();
      x += bw;
    }
    ctx.fillStyle = 'rgb(70,45,28)';
    ctx.fillRect(0, y1, w, sh * 0.06);
  }
  modulate(ctx, w, h, null, 0, 0.02, rng);
  return canvas;
}

/** Woven fabric with a printed motif. */
export function fabricTexture(w: number, h: number, rng: SynthRng, base: RGB, motif = true): Canvas2D {
  const { canvas, ctx } = makeCanvas(w, h);
  const f = fbm(w, h, rng, { baseCells: 5, octaves: 6 });
  paintPixels(ctx, w, h, (x, y, i, c) => {
    const weave = ((x >> 1) + (y >> 1)) & 1 ? 0.92 : 1.06;
    const k = weave * (1 + 0.1 * f.data[i]);
    c[0] = base[0] * k;
    c[1] = base[1] * k;
    c[2] = base[2] * k;
  });
  if (motif) {
    for (let i = 0; i < 160; i++) {
      ctx.strokeStyle = rgbStr(mix(base, randomVivid(rng), 0.6), 0.8);
      ctx.lineWidth = 2 + 4 * rng.next();
      const x = rng.next() * w,
        y = rng.next() * h,
        r = 6 + 30 * rng.next();
      ctx.beginPath();
      if (rng.next() < 0.5) ctx.arc(x, y, r, 0, Math.PI * (0.5 + 1.5 * rng.next()));
      else {
        ctx.moveTo(x - r, y);
        ctx.lineTo(x, y - r);
        ctx.lineTo(x + r, y);
        ctx.lineTo(x, y + r);
        ctx.closePath();
      }
      ctx.stroke();
    }
  }
  modulate(ctx, w, h, null, 0, 0.03, rng);
  return canvas;
}

/** Magazine / newspaper page. */
export function magazineTexture(w: number, h: number, rng: SynthRng): Canvas2D {
  return posterTexture(w, h, rng, 'text');
}

// ---------------------------------------------------------------------------
// Object textures
// ---------------------------------------------------------------------------

/** Colourful Voronoi cells with dark borders, speckles and stripes (torus knot). */
export function cellsTexture(w: number, h: number, rng: SynthRng, cells: number): Canvas2D {
  const { canvas, ctx } = makeCanvas(w, h);
  const wo = worley(w, h, cells, rng);
  const pal: RGB[] = [];
  for (let i = 0; i < cells; i++) pal.push(randomVivid(rng));
  const f = fbm(w, h, rng, { baseCells: 32, octaves: 4 });
  paintPixels(ctx, w, h, (_x, _y, i, c) => {
    const col = pal[wo.id[i]];
    const edge = wo.dist2[i] - wo.dist[i];
    const k = edge < 2.5 ? 0.15 : 1 + 0.2 * f.data[i] - 0.004 * wo.dist[i];
    c[0] = col[0] * k;
    c[1] = col[1] * k;
    c[2] = col[2] * k;
  });
  for (let i = 0; i < cells * 3; i++) {
    ctx.fillStyle = rgbStr(rng.next() < 0.5 ? [250, 250, 240] : [15, 15, 20], 0.85);
    const x = rng.next() * w,
      y = rng.next() * h,
      r = 1.5 + 5 * rng.next();
    drawWrapped(w, h, x, y, r, (ox, oy) => {
      ctx.beginPath();
      ctx.arc(x + ox, y + oy, r, 0, Math.PI * 2);
      ctx.fill();
    });
  }
  modulate(ctx, w, h, null, 0, 0.03, rng);
  return canvas;
}

/** "Globe" sphere texture: continents, grid lines, dots and labels (equirectangular). */
export function globeTexture(w: number, h: number, rng: SynthRng): Canvas2D {
  const { canvas, ctx } = makeCanvas(w, h);
  const f = fbm(w, h, rng, { baseCells: 6, octaves: 8, gain: 0.55 });
  const sea = hsl(205, 0.6, 0.35),
    seaL = hsl(195, 0.6, 0.55);
  paintPixels(ctx, w, h, (_x, _y, i, c) => {
    const v = f.data[i];
    let col: RGB;
    if (v > 0.08) col = mix(hsl(95, 0.45, 0.4), hsl(35, 0.45, 0.5), Math.min(1, (v - 0.08) * 3));
    else if (v > 0.03) col = [225, 210, 160];
    else col = mix(sea, seaL, Math.max(0, 1 + v * 3));
    c[0] = col[0];
    c[1] = col[1];
    c[2] = col[2];
  });
  ctx.strokeStyle = 'rgba(255,255,255,0.55)';
  ctx.lineWidth = 1.2;
  for (let k = 1; k < 18; k++) {
    ctx.beginPath();
    ctx.moveTo((k * w) / 18, 0);
    ctx.lineTo((k * w) / 18, h);
    ctx.stroke();
  }
  for (let k = 1; k < 9; k++) {
    ctx.beginPath();
    ctx.moveTo(0, (k * h) / 9);
    ctx.lineTo(w, (k * h) / 9);
    ctx.stroke();
  }
  for (let i = 0; i < 220; i++) {
    const x = rng.next() * w,
      y = h * (0.12 + 0.76 * rng.next());
    ctx.fillStyle = 'rgb(200,30,30)';
    ctx.beginPath();
    ctx.arc(x, y, 2 + 2 * rng.next(), 0, Math.PI * 2);
    ctx.fill();
    if (rng.next() < 0.5) drawText(ctx, x + 5, y - 10, 40 + 50 * rng.next(), 10, 1, 'rgb(20,20,20)', rng);
  }
  modulate(ctx, w, h, null, 0, 0.025, rng);
  return canvas;
}

/** Atlas of `cols × rows` distinct "packaging" faces (labels, barcodes, logos). */
export function packagingAtlas(cols: number, rows: number, cell: number, rng: SynthRng): Canvas2D {
  const w = cols * cell,
    h = rows * cell;
  const { canvas, ctx } = makeCanvas(w, h);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      const x = c * cell,
        y = r * cell;
      ctx.save();
      ctx.beginPath();
      ctx.rect(x, y, cell, cell);
      ctx.clip();
      ctx.fillStyle = rgbStr(hsl(rng.next() * 360, 0.5 + 0.4 * rng.next(), 0.4 + 0.3 * rng.next()));
      ctx.fillRect(x, y, cell, cell);
      // panels
      for (let k = 0; k < 5; k++) {
        ctx.fillStyle = rgbStr(randomVivid(rng), 0.9);
        ctx.fillRect(x + rng.next() * cell * 0.8, y + rng.next() * cell * 0.8, cell * (0.1 + 0.4 * rng.next()), cell * (0.05 + 0.3 * rng.next()));
      }
      // logo
      ctx.fillStyle = 'rgb(250,250,245)';
      ctx.beginPath();
      ctx.arc(x + cell * 0.3, y + cell * 0.3, cell * 0.14, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = rgbStr(randomVivid(rng));
      ctx.beginPath();
      ctx.moveTo(x + cell * 0.3, y + cell * 0.18);
      ctx.lineTo(x + cell * 0.42, y + cell * 0.38);
      ctx.lineTo(x + cell * 0.18, y + cell * 0.38);
      ctx.fill();
      drawText(ctx, x + cell * 0.48, y + cell * 0.18, cell * 0.45, cell * 0.06, 3, 'rgb(15,15,15)', rng);
      drawText(ctx, x + cell * 0.08, y + cell * 0.5, cell * 0.84, cell * 0.035, 6, 'rgb(250,250,250)', rng, true);
      // barcode
      let bx = x + cell * 0.1;
      ctx.fillStyle = 'white';
      ctx.fillRect(bx - 4, y + cell * 0.78, cell * 0.5, cell * 0.17);
      ctx.fillStyle = 'black';
      while (bx < x + cell * 0.56) {
        const bw = 1 + rng.int(4);
        if (rng.next() < 0.55) ctx.fillRect(bx, y + cell * 0.8, bw, cell * 0.13);
        bx += bw + 1;
      }
      ctx.restore();
    }
  modulate(ctx, w, h, null, 0, 0.02, rng);
  return canvas;
}

/** Marble (veins from turbulent sine). */
export function marbleTexture(w: number, h: number, rng: SynthRng): Canvas2D {
  const { canvas, ctx } = makeCanvas(w, h);
  const f = fbm(w, h, rng, { baseCells: 4, octaves: 8, gain: 0.6 });
  const g = fbm(w, h, rng, { baseCells: 16, octaves: 5 });
  paintPixels(ctx, w, h, (x, y, i, c) => {
    const v = Math.abs(Math.sin((x + y) * 0.012 + 7 * f.data[i]));
    const vein = Math.pow(v, 0.25);
    const col = mix([70, 70, 80], [235, 232, 225], vein);
    const k = 1 + 0.08 * g.data[i];
    c[0] = col[0] * k;
    c[1] = col[1] * k;
    c[2] = col[2] * k;
  });
  modulate(ctx, w, h, null, 0, 0.02, rng);
  return canvas;
}

// ---------------------------------------------------------------------------
// Outdoor textures
// ---------------------------------------------------------------------------

/** Grayscale (mean ≈ 0.5) tileable grass-blade detail map. */
export function grassDetailTexture(size: number, rng: SynthRng): Canvas2D {
  const { canvas, ctx } = makeCanvas(size, size);
  ctx.fillStyle = 'rgb(105,105,105)';
  ctx.fillRect(0, 0, size, size);
  const n = Math.round(size * size * 0.045);
  ctx.lineCap = 'round';
  for (let i = 0; i < n; i++) {
    const x = rng.next() * size,
      y = rng.next() * size;
    const len = size * (0.006 + 0.016 * rng.next());
    const a = rng.next() * Math.PI * 2;
    const l = Math.round(40 + 190 * Math.pow(rng.next(), 1.3));
    ctx.strokeStyle = `rgb(${l},${l},${l})`;
    ctx.lineWidth = 0.8 + 1.6 * rng.next();
    const dx = Math.cos(a) * len,
      dy = Math.sin(a) * len;
    drawWrapped(size, size, x, y, len, (ox, oy) => {
      ctx.beginPath();
      ctx.moveTo(x + ox, y + oy);
      ctx.quadraticCurveTo(x + ox + dx * 0.5 + dy * 0.2, y + oy + dy * 0.5 - dx * 0.2, x + ox + dx, y + oy + dy);
      ctx.stroke();
    });
  }
  return canvas;
}

/** Grayscale (mean ≈ 0.5) tileable soil / pebble detail map. */
export function soilDetailTexture(size: number, rng: SynthRng): Canvas2D {
  const { canvas, ctx } = makeCanvas(size, size);
  const f = fbm(size, size, rng, { baseCells: 8, octaves: 7, gain: 0.6 });
  paintPixels(ctx, size, size, (_x, _y, i, c) => {
    const v = 128 + 50 * f.data[i];
    c[0] = c[1] = c[2] = v;
  });
  for (let i = 0; i < size * 6; i++) {
    const x = rng.next() * size,
      y = rng.next() * size,
      r = 1 + 6 * Math.pow(rng.next(), 2);
    const l = Math.round(rng.next() < 0.5 ? 50 + 50 * rng.next() : 170 + 70 * rng.next());
    ctx.fillStyle = `rgb(${l},${l},${l})`;
    const rot = rng.next() * Math.PI;
    drawWrapped(size, size, x, y, r * 2, (ox, oy) => {
      ctx.beginPath();
      ctx.ellipse(x + ox, y + oy, r, r * (0.5 + 0.5 * rng.next()), rot, 0, Math.PI * 2);
      ctx.fill();
    });
  }
  return canvas;
}

/** Roof shingles (tileable, mid-grey-ish; tint with material colour). */
export function shingleTexture(size: number, rng: SynthRng): Canvas2D {
  const { canvas, ctx } = makeCanvas(size, size);
  const rows = 32;
  const rh = size / rows;
  for (let r = 0; r < rows; r++) {
    let x = r % 2 ? -rh * 1.2 : 0;
    while (x < size) {
      const sw = rh * (1.6 + 1.6 * rng.next());
      const l = 120 + 80 * rng.next();
      ctx.fillStyle = `rgb(${l},${l * 0.97},${l * 0.94})`;
      ctx.fillRect(x, r * rh, sw - 2, rh - 2);
      ctx.fillStyle = 'rgba(0,0,0,0.35)';
      ctx.fillRect(x, (r + 1) * rh - 3, sw - 2, 2);
      x += sw;
    }
  }
  const f = fbm(size, size, rng, { baseCells: 8, octaves: 7 });
  modulate(ctx, size, size, f, 0.18, 0.06, rng);
  return canvas;
}

/** Bark: vertical fissures. */
export function barkTexture(w: number, h: number, rng: SynthRng): Canvas2D {
  const { canvas, ctx } = makeCanvas(w, h);
  const f = fbm(w, h, rng, { baseCells: 24, octaves: 6, gain: 0.6 });
  const g = fbm(w, h, rng, { baseCells: 4, octaves: 6 });
  const base = hsl(25 + 10 * rng.next(), 0.25, 0.32);
  paintPixels(ctx, w, h, (x, y, i, c) => {
    // stretch vertically: sample the fbm at a compressed y coordinate
    const yi = Math.floor((y * 0.25) % h);
    const v = f.data[yi * w + x];
    const fiss = Math.abs(v) < 0.12 ? 0.35 : 1;
    const k = fiss * (1 + 0.25 * g.data[i]);
    c[0] = base[0] * k;
    c[1] = base[1] * k;
    c[2] = base[2] * k;
  });
  modulate(ctx, w, h, null, 0, 0.05, rng);
  return canvas;
}

/** Leaves: many small rotated leaf ellipses over a dark base (tileable). */
export function leavesTexture(size: number, rng: SynthRng, hue: number): Canvas2D {
  const { canvas, ctx } = makeCanvas(size, size);
  ctx.fillStyle = rgbStr(hsl(hue, 0.4, 0.12));
  ctx.fillRect(0, 0, size, size);
  const n = Math.round(size * size * 0.012);
  for (let i = 0; i < n; i++) {
    const x = rng.next() * size,
      y = rng.next() * size;
    const r = size * (0.004 + 0.008 * rng.next());
    ctx.fillStyle = rgbStr(hsl(hue + 25 * (rng.next() - 0.5), 0.35 + 0.35 * rng.next(), 0.15 + 0.35 * rng.next()));
    const rot = rng.next() * Math.PI;
    drawWrapped(size, size, x, y, r * 2, (ox, oy) => {
      ctx.beginPath();
      ctx.ellipse(x + ox, y + oy, r * 1.8, r * 0.8, rot, 0, Math.PI * 2);
      ctx.fill();
    });
  }
  return canvas;
}

/** Board fence (vertical boards), `pxPerM` pixels per metre. */
export function fenceTexture(lengthM: number, heightM: number, pxPerM: number, rng: SynthRng): Canvas2D {
  const w = Math.round(lengthM * pxPerM),
    h = Math.round(heightM * pxPerM);
  const { canvas, ctx } = makeCanvas(w, h);
  const base = hsl(28 + 8 * rng.next(), 0.35, 0.45);
  const f = fbm(w, h, rng, { baseCells: Math.max(4, Math.round(lengthM * 2)), octaves: 7, gain: 0.6 });
  const bw = 0.14 * pxPerM;
  const nb = Math.ceil(w / bw);
  const tint: number[] = [];
  const ph: number[] = [];
  for (let b = 0; b < nb; b++) {
    tint.push(0.8 + 0.4 * rng.next());
    ph.push(rng.next() * 100);
  }
  paintPixels(ctx, w, h, (x, y, i, c) => {
    const b = Math.floor(x / bw);
    const xx = x - b * bw;
    const grain = 0.5 + 0.5 * Math.sin((xx * 0.9 + ph[b] + 6 * f.data[i]) * 1.3);
    const gap = xx < 1.5 ? 0.25 : 1;
    const k = gap * tint[b] * (0.8 + 0.25 * grain + 0.15 * f.data[i]);
    c[0] = base[0] * k;
    c[1] = base[1] * k;
    c[2] = base[2] * k;
  });
  // rails shadow + nails
  ctx.fillStyle = 'rgba(0,0,0,0.25)';
  for (const ry of [0.25, 0.75]) ctx.fillRect(0, h * (1 - ry), w, 0.02 * pxPerM);
  ctx.fillStyle = 'rgb(40,40,40)';
  for (let b = 0; b < nb; b++)
    for (const ry of [0.24, 0.76]) {
      ctx.fillRect(b * bw + bw * 0.3, h * (1 - ry) - 2, 3, 3);
      ctx.fillRect(b * bw + bw * 0.7, h * (1 - ry) - 2, 3, 3);
    }
  // weathering at the bottom
  const g = ctx.createLinearGradient(0, h, 0, h * 0.75);
  g.addColorStop(0, 'rgba(40,50,20,0.5)');
  g.addColorStop(1, 'rgba(40,50,20,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, h * 0.75, w, h * 0.25);
  modulate(ctx, w, h, null, 0, 0.03, rng);
  return canvas;
}

/** Concrete pavers in running bond, with joints and moss. */
export function paverTexture(w: number, h: number, pxPerM: number, rng: SynthRng): Canvas2D {
  const { canvas, ctx } = makeCanvas(w, h);
  ctx.fillStyle = 'rgb(90,88,80)';
  ctx.fillRect(0, 0, w, h);
  const pw = 0.4 * pxPerM,
    ph = 0.2 * pxPerM;
  for (let r = 0; r * ph < h; r++) {
    for (let x = r % 2 ? -pw / 2 : 0; x < w; x += pw) {
      const l = 150 + 60 * rng.next();
      ctx.fillStyle = rgbStr(mix([l, l * 0.95, l * 0.88], hsl(15, 0.4, 0.5), 0.3 * rng.next()));
      ctx.fillRect(x + 2, r * ph + 2, pw - 4, ph - 4);
    }
  }
  const f = fbm(w, h, rng, { baseCells: 10, octaves: 7 });
  modulate(ctx, w, h, f, 0.2, 0.06, rng);
  for (let i = 0; i < 400; i++) {
    ctx.fillStyle = rgbStr(hsl(90, 0.4, 0.25), 0.5);
    ctx.beginPath();
    ctx.arc(rng.next() * w, rng.next() * h, 1 + 4 * rng.next(), 0, Math.PI * 2);
    ctx.fill();
  }
  return canvas;
}

/** Ground macro map: RGB colour + mask (1 = grass, 0 = soil/gravel), as RGBA bytes
 * with row 0 = SOUTH edge (v = 0; DataTexture without flipY). The map covers
 * [−half, half]² metres. `paint` draws extra features in canvas coordinates
 * (canvas row 0 = north) via a world→canvas transform already applied (metres). */
export function groundMacroTexture(
  size: number,
  half: number,
  rng: SynthRng,
  paint: (ctx: Ctx, mask: Ctx) => void,
): Uint8Array {
  const { ctx } = makeCanvas(size, size);
  const { ctx: mctx } = makeCanvas(size, size);
  const f1 = fbm(size, size, rng, { baseCells: 6, octaves: 9, gain: 0.58 });
  const f2 = fbm(size, size, rng, { baseCells: 12, octaves: 6, gain: 0.5 });
  const grassA = hsl(95, 0.45, 0.33),
    grassB = hsl(70, 0.45, 0.42),
    dirt = hsl(30, 0.35, 0.38);
  paintPixels(ctx, size, size, (_x, _y, i, c) => {
    const v = f1.data[i];
    const col = mix(grassA, grassB, 0.5 + 0.5 * f2.data[i]);
    const d = Math.max(0, Math.min(1, (v - 0.35) * 4)); // dirt patches
    const cc = mix(col, dirt, d);
    c[0] = cc[0] * (1 + 0.15 * v);
    c[1] = cc[1] * (1 + 0.15 * v);
    c[2] = cc[2] * (1 + 0.15 * v);
  });
  paintPixels(mctx, size, size, (_x, _y, i, c) => {
    const d = Math.max(0, Math.min(1, (f1.data[i] - 0.35) * 4));
    c[0] = c[1] = c[2] = 255 * (1 - d);
  });
  // world (metres, x east / y north) → canvas pixels
  const s = size / (2 * half);
  for (const c of [ctx, mctx]) {
    c.save();
    c.setTransform(s, 0, 0, -s, half * s, half * s);
  }
  paint(ctx, mctx);
  ctx.restore();
  mctx.restore();
  // flowers / clover specks
  for (let i = 0; i < size * 20; i++) {
    ctx.fillStyle = rgbStr(rng.next() < 0.5 ? [235, 235, 210] : hsl(50 + 20 * rng.next(), 0.8, 0.6), 0.7);
    ctx.fillRect(rng.next() * size, rng.next() * size, 1, 1);
  }
  const rgb = ctx.getImageData(0, 0, size, size).data;
  const msk = mctx.getImageData(0, 0, size, size).data;
  const out = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    const src = y * size * 4,
      dst = (size - 1 - y) * size * 4;
    for (let x = 0; x < size; x++) {
      const i = src + 4 * x,
        o = dst + 4 * x;
      out[o] = rgb[i];
      out[o + 1] = rgb[i + 1];
      out[o + 2] = rgb[i + 2];
      out[o + 3] = msk[i];
    }
  }
  return out;
}

export type FacadeStyle = 'brick' | 'siding' | 'stucco' | 'stone';

export interface FacadeSpec {
  /** Physical width / height of the wall (m). */
  widthM: number;
  heightM: number;
  style: FacadeStyle;
  color: RGB;
  trim: RGB;
  door: boolean;
  garage?: boolean;
  floors: number;
}

/** Background wall material in metre coordinates (ctx transformed to metres). */
function drawWallMaterial(ctx: Ctx, wM: number, hM: number, style: FacadeStyle, color: RGB, rng: SynthRng): void {
  ctx.fillStyle = rgbStr(color);
  ctx.fillRect(0, 0, wM, hM);
  if (style === 'brick') {
    ctx.fillStyle = 'rgba(200,195,185,1)';
    ctx.fillRect(0, 0, wM, hM);
    const bh = 0.075,
      bl = 0.225;
    for (let r = 0; r * bh < hM; r++) {
      for (let x = r % 2 ? -bl / 2 : 0; x < wM; x += bl) {
        ctx.fillStyle = rgbStr(jitterColor(mix(color, [60, 30, 20], 0.25 * rng.next()), rng, 0.18));
        ctx.fillRect(x + 0.006, r * bh + 0.006, bl - 0.012, bh - 0.012);
      }
    }
  } else if (style === 'siding') {
    const bh = 0.2;
    for (let r = 0; r * bh < hM; r++) {
      ctx.fillStyle = rgbStr(jitterColor(color, rng, 0.04));
      ctx.fillRect(0, r * bh, wM, bh);
      ctx.fillStyle = 'rgba(0,0,0,0.3)';
      ctx.fillRect(0, (r + 1) * bh - 0.02, wM, 0.02);
      ctx.fillStyle = 'rgba(255,255,255,0.15)';
      ctx.fillRect(0, r * bh, wM, 0.01);
      if (rng.next() < 0.5) {
        ctx.fillStyle = 'rgba(0,0,0,0.25)';
        ctx.fillRect(rng.next() * wM, r * bh, 0.01, bh); // butt joint
      }
    }
  } else if (style === 'stone') {
    let y = 0;
    while (y < hM) {
      const rh = 0.15 + 0.2 * rng.next();
      let x = -rng.next() * 0.3;
      while (x < wM) {
        const sw = 0.25 + 0.4 * rng.next();
        ctx.fillStyle = rgbStr(jitterColor(mix(color, [140, 130, 120], rng.next() * 0.6), rng, 0.2));
        ctx.fillRect(x + 0.012, y + 0.012, sw - 0.024, rh - 0.024);
        x += sw;
      }
      y += rh;
    }
  } else {
    // stucco: speckles
    for (let i = 0; i < wM * hM * 300; i++) {
      ctx.fillStyle = rgbStr(jitterColor(color, rng, 0.15), 0.6);
      ctx.fillRect(rng.next() * wM, rng.next() * hM, 0.012, 0.012);
    }
  }
}

function drawWindow(ctx: Ctx, x: number, y: number, w: number, h: number, trim: RGB, rng: SynthRng): void {
  // frame
  ctx.fillStyle = rgbStr(trim);
  ctx.fillRect(x - 0.08, y - 0.08, w + 0.16, h + 0.16);
  // glass with sky reflection / interior
  const g = ctx.createLinearGradient(x, y, x + w * 0.6, y + h);
  const tone = 40 + 60 * rng.next();
  g.addColorStop(0, `rgb(${tone + 90},${tone + 110},${tone + 130})`);
  g.addColorStop(0.5, `rgb(${tone},${tone + 15},${tone + 30})`);
  g.addColorStop(1, `rgb(${tone * 0.6},${tone * 0.7},${tone * 0.8})`);
  ctx.fillStyle = g;
  ctx.fillRect(x, y, w, h);
  // reflections streaks
  for (let k = 0; k < 4; k++) {
    ctx.fillStyle = `rgba(255,255,255,${0.05 + 0.12 * rng.next()})`;
    ctx.beginPath();
    const sx = x + rng.next() * w;
    ctx.moveTo(sx, y);
    ctx.lineTo(sx + w * 0.15, y);
    ctx.lineTo(sx - w * 0.2, y + h);
    ctx.lineTo(sx - w * 0.35, y + h);
    ctx.fill();
  }
  // curtains
  if (rng.next() < 0.6) {
    ctx.fillStyle = rgbStr(randomVivid(rng), 0.85);
    const cw = w * (0.15 + 0.25 * rng.next());
    ctx.fillRect(x, y, cw, h);
    ctx.fillRect(x + w - cw * (0.6 + 0.8 * rng.next()), y, cw, h);
    ctx.fillStyle = 'rgba(0,0,0,0.2)';
    for (let k = 0; k < 6; k++) ctx.fillRect(x + (k * cw) / 6, y, 0.015, h);
  }
  // mullions
  ctx.fillStyle = rgbStr(trim);
  ctx.fillRect(x + w / 2 - 0.03, y, 0.06, h);
  if (rng.next() < 0.7) ctx.fillRect(x, y + h * 0.45, w, 0.05);
  // sill
  ctx.fillStyle = rgbStr(mix(trim, [0, 0, 0], 0.25));
  ctx.fillRect(x - 0.12, y + h + 0.06, w + 0.24, 0.06);
}

function drawDoor(ctx: Ctx, x: number, y: number, w: number, h: number, trim: RGB, rng: SynthRng): void {
  ctx.fillStyle = rgbStr(trim);
  ctx.fillRect(x - 0.1, y - 0.1, w + 0.2, h + 0.1);
  ctx.fillStyle = rgbStr(randomVivid(rng));
  ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = 'rgba(0,0,0,0.4)';
  ctx.lineWidth = 0.025;
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 2; c++) ctx.strokeRect(x + 0.08 + (c * (w - 0.16)) / 2 + 0.03, y + 0.1 + (r * (h - 0.2)) / 3 + 0.03, (w - 0.16) / 2 - 0.06, (h - 0.2) / 3 - 0.06);
  ctx.fillStyle = 'rgb(210,180,60)';
  ctx.beginPath();
  ctx.arc(x + w * 0.85, y + h * 0.52, 0.04, 0, Math.PI * 2);
  ctx.fill();
  // house number + lamp
  drawText(ctx, x + w + 0.2, y + 0.1, 0.3, 0.12, 1, 'rgb(20,20,20)', rng);
  ctx.fillStyle = 'rgb(30,30,30)';
  ctx.fillRect(x - 0.45, y + 0.3, 0.16, 0.3);
  ctx.fillStyle = 'rgb(250,230,170)';
  ctx.fillRect(x - 0.43, y + 0.34, 0.12, 0.2);
}

/**
 * Facade (wall elevation) drawn into a region of `ctx` given in PIXELS
 * (px, py, pw, ph). The facade's top-left is the top of the wall. Windows, door,
 * garage door, downspout, vents, grime at the bottom.
 */
export function drawFacade(ctx: Ctx, px: number, py: number, pw: number, ph: number, spec: FacadeSpec, rng: SynthRng): void {
  const { widthM: wM, heightM: hM } = spec;
  ctx.save();
  ctx.beginPath();
  ctx.rect(px, py, pw, ph);
  ctx.clip();
  ctx.translate(px, py);
  ctx.scale(pw / wM, ph / hM);
  drawWallMaterial(ctx, wM, hM, spec.style, spec.color, rng);
  // foundation band
  ctx.fillStyle = 'rgb(120,118,112)';
  ctx.fillRect(0, hM - 0.35, wM, 0.35);
  // windows per floor
  const floorH = hM / Math.max(1, spec.floors);
  const doorX = spec.door ? wM * (0.25 + 0.5 * rng.next()) : -10;
  const garageX = spec.garage ? (doorX < wM / 2 ? wM - 3.4 : 0.6) : -10;
  for (let f = 0; f < spec.floors; f++) {
    const wy = hM - (f + 1) * floorH + floorH * 0.28;
    const wh = Math.min(1.5, floorH * 0.45);
    let x = 0.8 + rng.next() * 0.6;
    while (x < wM - 1.6) {
      const ww = 0.8 + 0.8 * rng.next();
      const clashDoor = f === 0 && x + ww > doorX - 0.4 && x < doorX + 1.4;
      const clashGarage = f === 0 && x + ww > garageX - 0.3 && x < garageX + 3.1;
      if (!clashDoor && !clashGarage && rng.next() < 0.85) drawWindow(ctx, x, wy, ww, wh, spec.trim, rng);
      x += ww + 0.9 + 1.4 * rng.next();
    }
  }
  if (spec.door) drawDoor(ctx, doorX, hM - 0.35 - 2.1, 1.0, 2.1, spec.trim, rng);
  if (spec.garage) {
    const gy = hM - 0.35 - 2.3;
    ctx.fillStyle = rgbStr(spec.trim);
    ctx.fillRect(garageX - 0.1, gy - 0.1, 2.8, 2.4);
    ctx.fillStyle = rgbStr(mix(spec.trim, [120, 120, 120], 0.3));
    ctx.fillRect(garageX, gy, 2.6, 2.3);
    ctx.fillStyle = 'rgba(0,0,0,0.25)';
    for (let k = 1; k < 5; k++) ctx.fillRect(garageX, gy + (k * 2.3) / 5, 2.6, 0.03);
    for (let k = 0; k < 4; k++) {
      ctx.fillStyle = 'rgba(40,50,60,0.9)';
      ctx.fillRect(garageX + 0.15 + k * 0.62, gy + 0.2, 0.5, 0.3);
    }
  }
  // downspout
  const dx = rng.next() < 0.5 ? 0.15 : wM - 0.25;
  ctx.fillStyle = rgbStr(mix(spec.trim, [80, 80, 80], 0.3));
  ctx.fillRect(dx, 0, 0.08, hM - 0.1);
  // vents / meter box / hose reel
  for (let k = 0; k < 3; k++) {
    const vx = rng.next() * (wM - 0.5),
      vy = hM - 0.35 - 0.3 - rng.next() * 1.2;
    ctx.fillStyle = 'rgb(90,90,95)';
    ctx.fillRect(vx, vy, 0.3, 0.22);
    ctx.fillStyle = 'rgb(200,200,200)';
    for (let s = 0; s < 4; s++) ctx.fillRect(vx + 0.03, vy + 0.03 + s * 0.05, 0.24, 0.02);
  }
  // grime gradient at the bottom
  const g = ctx.createLinearGradient(0, hM, 0, hM - 1.2);
  g.addColorStop(0, 'rgba(40,35,25,0.35)');
  g.addColorStop(1, 'rgba(40,35,25,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, hM - 1.2, wM, 1.2);
  ctx.restore();
}

/** Triangular gable (wall material + louvered vent), drawn into a pixel region. */
export function drawGable(ctx: Ctx, px: number, py: number, pw: number, ph: number, widthM: number, heightM: number, style: FacadeStyle, color: RGB, trim: RGB, rng: SynthRng): void {
  ctx.save();
  ctx.beginPath();
  ctx.rect(px, py, pw, ph);
  ctx.clip();
  ctx.translate(px, py);
  ctx.scale(pw / widthM, ph / heightM);
  drawWallMaterial(ctx, widthM, heightM, style, color, rng);
  const vw = 0.6,
    vh = 0.7;
  const vx = widthM / 2 - vw / 2,
    vy = heightM * 0.45;
  ctx.fillStyle = rgbStr(trim);
  ctx.fillRect(vx - 0.06, vy - 0.06, vw + 0.12, vh + 0.12);
  ctx.fillStyle = 'rgb(50,50,55)';
  ctx.fillRect(vx, vy, vw, vh);
  ctx.fillStyle = rgbStr(trim);
  for (let s = 0; s < 7; s++) ctx.fillRect(vx, vy + 0.04 + s * 0.095, vw, 0.04);
  ctx.restore();
}

/** Per-pixel grain over a whole canvas (after facades are drawn). */
export function addGrain(canvas: Canvas2D, amount: number, rng: SynthRng, lowFreq = 0.08): void {
  const ctx = canvas.getContext('2d') as Ctx;
  const w = canvas.width,
    h = canvas.height;
  const f = lowFreq > 0 ? fbm(w, h, rng, { baseCells: Math.max(4, Math.round(w / 128)), octaves: 6 }) : null;
  modulate(ctx, w, h, f, lowFreq, amount, rng);
}
