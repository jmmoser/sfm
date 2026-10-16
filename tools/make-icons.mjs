// Generates the PWA / apple-touch icons procedurally (no binary assets in git
// besides the outputs). Run: node tools/make-icons.mjs
import { PNG } from 'pngjs';
import { writeFileSync } from 'node:fs';

function render(size, maskable) {
  const png = new PNG({ width: size, height: size });
  const pad = maskable ? size * 0.18 : size * 0.08;
  const cx = size / 2, cy = size / 2;
  // Simple icon: dark gradient background, a camera-orbit ring and a point-cloud cube.
  const pts = [];
  let seed = 7;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  for (let i = 0; i < 900; i++) {
    // points on the visible faces of an isometric cube
    const face = i % 3, a = rnd(), b = rnd();
    let x, y, z;
    if (face === 0) { x = a; y = b; z = 1; } else if (face === 1) { x = 1; y = a; z = b; } else { x = a; y = 1; z = b; }
    x -= 0.5; y -= 0.5; z -= 0.5;
    const sx = (x - y) * 0.866, sy = (x + y) * 0.5 - z;
    pts.push([sx, sy, face]);
  }
  const scale = (size - 2 * pad) * 0.42;
  const buf = png.data;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      const t = y / size;
      buf[i] = Math.round(14 + 20 * t);
      buf[i + 1] = Math.round(20 + 30 * t);
      buf[i + 2] = Math.round(38 + 50 * t);
      buf[i + 3] = 255;
      // orbit ring (ellipse)
      const ex = (x - cx) / (scale * 1.55), ey = (y - cy - scale * 0.15) / (scale * 0.55);
      const r = Math.sqrt(ex * ex + ey * ey);
      if (Math.abs(r - 1) < 2.2 / scale) { buf[i] = 120; buf[i + 1] = 220; buf[i + 2] = 255; }
    }
  }
  const colors = [[255, 196, 92], [92, 220, 170], [120, 170, 255]];
  for (const [sx, sy, face] of pts) {
    const px = Math.round(cx + sx * scale), py = Math.round(cy + sy * scale);
    const rad = Math.max(1, Math.round(size / 160));
    for (let dy = -rad; dy <= rad; dy++) for (let dx = -rad; dx <= rad; dx++) {
      if (dx * dx + dy * dy > rad * rad) continue;
      const X = px + dx, Y = py + dy;
      if (X < 0 || Y < 0 || X >= size || Y >= size) continue;
      const i = (Y * size + X) * 4;
      const c = colors[face];
      buf[i] = c[0]; buf[i + 1] = c[1]; buf[i + 2] = c[2];
    }
  }
  return PNG.sync.write(png);
}

writeFileSync('public/icons/icon-192.png', render(192, false));
writeFileSync('public/icons/icon-512.png', render(512, false));
writeFileSync('public/icons/icon-maskable-512.png', render(512, true));
writeFileSync('public/icons/apple-touch-icon.png', render(180, false));
console.log('icons written');
