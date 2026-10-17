import { describe, expect, it } from 'vitest';
import type { Pose } from '../core/types';
import { createColmapZip, exportColmapText, importColmapText, type ColmapModelInput } from './colmap';
import { readZip } from './zip';

function rotFromAxisAngle(ax: number, ay: number, az: number, angle: number): Float64Array {
  const n = Math.hypot(ax, ay, az);
  const x = ax / n, y = ay / n, z = az / n;
  const c = Math.cos(angle), s = Math.sin(angle), C = 1 - c;
  return Float64Array.of(
    c + x * x * C, x * y * C - z * s, x * z * C + y * s,
    y * x * C + z * s, c + y * y * C, y * z * C - x * s,
    z * x * C - y * s, z * y * C + x * s, c + z * z * C,
  );
}

function project(pose: Pose, f: number, cx: number, cy: number, X: ArrayLike<number>): [number, number] {
  const R = pose.R, t = pose.t;
  const xc = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0];
  const yc = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1];
  const zc = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
  return [f * (xc / zc) + cx, f * (yc / zc) + cy];
}

function makeModel(): ColmapModelInput {
  const cams = [
    { id: 1, cam: { width: 1920, height: 1440, f: 1500.123456789, cx: 959.5, cy: 719.5, k1: -0.0123, k2: 0.00045 } },
    { id: 2, cam: { width: 640, height: 480, f: 512.25, cx: 320.1, cy: 239.7, k1: 0, k2: 0 } },
  ];
  // Rotations covering all Shepperd branches (trace > 0, and each diagonal dominant).
  const rots = [
    rotFromAxisAngle(0.1, 0.2, 0.3, 0.4),
    rotFromAxisAngle(1, 0.01, 0.02, 3.1),
    rotFromAxisAngle(0.02, 1, 0.01, 3.0),
    rotFromAxisAngle(0.01, 0.03, 1, 3.14159),
  ];
  const points = [
    Float64Array.of(0.1, 0.2, 5),
    Float64Array.of(-0.5, 0.3, 6.25),
    Float64Array.of(0.7, -0.4, 4.123456789012345),
  ];
  const images: ColmapModelInput['images'] = rots.map((R, i) => ({
    id: i + 1,
    cameraId: i === 3 ? 2 : 1,
    name: `frame_${String(i).padStart(4, '0')}.jpg`,
    pose: { R, t: Float64Array.of(0.01 * i, -0.02 * i, 0.3 + i) },
    points2D: [],
  }));
  const model: ColmapModelInput = { cameras: cams, images, points: [] };
  // Image 1..3 observe the points; image 4 has no observations (empty POINTS2D line).
  points.forEach((X, pi) => {
    const track: { image: number; point2D: number }[] = [];
    for (let k = 0; k < 3; k++) {
      const im = images[k];
      const c1 = cams[0].cam;
      const [u, v] = project(im.pose, c1.f, c1.cx, c1.cy, X);
      im.points2D.push({ x: u, y: v, point3D: 100 + pi });
      track.push({ image: im.id, point2D: im.points2D.length - 1 });
    }
    model.points.push({ id: 100 + pi, xyz: X, color: [pi * 50, 255, 3], error: 0.25 * pi, track });
  });
  images[0].points2D.push({ x: 12.25, y: 7.75, point3D: -1 });
  return model;
}

describe('COLMAP text export', () => {
  it('writes the standard headers, counts and +0.5 pixel shift', () => {
    const files = exportColmapText(makeModel());
    expect(files.map((f) => f.name)).toEqual(['cameras.txt', 'images.txt', 'points3D.txt']);
    const [cams, imgs, pts] = files.map((f) => f.data.split('\n'));
    expect(cams[0]).toBe('# Camera list with one line of data per camera:');
    expect(cams[1]).toBe('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]');
    expect(cams[2]).toBe('# Number of cameras: 2');
    expect(cams[3]).toBe('1 RADIAL 1920 1440 1500.123456789 960 720 -0.0123 0.00045');
    expect(cams[4]).toBe('2 RADIAL 640 480 512.25 320.6 240.2 0 0');
    expect(imgs[0]).toBe('# Image list with two lines of data per image:');
    expect(imgs[1]).toBe('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME');
    expect(imgs[2]).toBe('#   POINTS2D[] as (X, Y, POINT3D_ID)');
    expect(imgs[3]).toBe('# Number of images: 4, mean observations per image: 2.25');
    // image 4: pose line followed by an empty POINTS2D line
    const idx4 = imgs.findIndex((l) => l.startsWith('4 '));
    expect(imgs[idx4]).toMatch(/ 2 frame_0003\.jpg$/);
    expect(imgs[idx4 + 1]).toBe('');
    // unmatched keypoint: shifted coords and -1
    const line1 = imgs[imgs.findIndex((l) => l.startsWith('1 ')) + 1];
    expect(line1.endsWith('12.75 8.25 -1')).toBe(true);
    expect(pts[0]).toBe('# 3D point list with one line of data per point:');
    expect(pts[1]).toBe('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)');
    expect(pts[2]).toBe('# Number of points: 3, mean track length: 3');
    expect(pts[3]).toBe('100 0.1 0.2 5 0 255 3 0 1 0 2 0 3 0');
  });

  it('writes the world-to-camera quaternion (w ≥ 0, Hamilton)', () => {
    const model = makeModel();
    const R = rotFromAxisAngle(0, 0, 1, Math.PI / 2); // 90° about z
    model.images[0].pose.R = R;
    const imgs = exportColmapText(model)[1].data.split('\n');
    const tok = imgs.find((l) => l.startsWith('1 '))!.split(' ').map(Number);
    expect(tok[1]).toBeCloseTo(Math.SQRT1_2, 12);
    expect(tok[2]).toBeCloseTo(0, 12);
    expect(tok[3]).toBeCloseTo(0, 12);
    expect(tok[4]).toBeCloseTo(Math.SQRT1_2, 12);
  });

  it('rejects image names with whitespace and non-finite numbers', () => {
    const m = makeModel();
    m.images[0].name = 'my frame.jpg';
    expect(() => exportColmapText(m)).toThrow(/whitespace/);
    const m2 = makeModel();
    m2.points[0].xyz = [NaN, 0, 0];
    expect(() => exportColmapText(m2)).toThrow(/non-finite/);
  });
});

describe('COLMAP round trip', () => {
  it('import(export(model)) reproduces the model', () => {
    const model = makeModel();
    const back = importColmapText(exportColmapText(model));
    expect(back.cameras.length).toBe(2);
    for (let i = 0; i < 2; i++) {
      const a = model.cameras[i].cam, b = back.cameras[i].cam;
      expect(back.cameras[i].id).toBe(model.cameras[i].id);
      expect(b.width).toBe(a.width);
      expect(b.height).toBe(a.height);
      expect(b.f).toBe(a.f);
      expect(b.cx).toBeCloseTo(a.cx, 10);
      expect(b.cy).toBeCloseTo(a.cy, 10);
      expect(b.k1).toBe(a.k1);
      expect(b.k2).toBe(a.k2);
    }
    expect(back.images.length).toBe(4);
    for (let i = 0; i < 4; i++) {
      const a = model.images[i], b = back.images[i];
      expect(b.id).toBe(a.id);
      expect(b.cameraId).toBe(a.cameraId);
      expect(b.name).toBe(a.name);
      for (let k = 0; k < 9; k++) expect(b.pose.R[k]).toBeCloseTo(a.pose.R[k], 12);
      for (let k = 0; k < 3; k++) expect(b.pose.t[k] === a.pose.t[k]).toBe(true);
      expect(b.points2D.length).toBe(a.points2D.length);
      for (let k = 0; k < a.points2D.length; k++) {
        expect(b.points2D[k].x).toBeCloseTo(a.points2D[k].x, 9);
        expect(b.points2D[k].y).toBeCloseTo(a.points2D[k].y, 9);
        expect(b.points2D[k].point3D).toBe(a.points2D[k].point3D);
      }
    }
    expect(back.points.length).toBe(3);
    for (let i = 0; i < 3; i++) {
      const a = model.points[i], b = back.points[i];
      expect(b.id).toBe(a.id);
      expect(Array.from(b.xyz)).toEqual(Array.from(a.xyz));
      expect(Array.from(b.color)).toEqual(Array.from(a.color));
      expect(b.error).toBe(a.error);
      expect(b.track).toEqual(a.track);
    }
    // Re-export is byte-identical up to float noise in the quaternion → compare cameras & points.
    const again = exportColmapText(back);
    expect(again[2].data).toBe(exportColmapText(model)[2].data);
  });

  it('tracks are consistent with POINTS2D (reprojection of imported data)', () => {
    const back = importColmapText(exportColmapText(makeModel()));
    const cams = new Map(back.cameras.map((c) => [c.id, c.cam]));
    for (const p of back.points) {
      for (const t of p.track) {
        const im = back.images.find((i) => i.id === t.image)!;
        const kp = im.points2D[t.point2D];
        expect(kp.point3D).toBe(p.id);
        const cam = cams.get(im.cameraId)!;
        const [u, v] = project(im.pose, cam.f, cam.cx, cam.cy, p.xyz);
        expect(Math.hypot(u - kp.x, v - kp.y)).toBeLessThan(1e-6);
      }
    }
  });

  it('imports other camera models and COLMAP-written files (CRLF, extra blank lines)', () => {
    const cameras = '# c\r\n1 PINHOLE 100 80 50 52 49.5 39.5\r\n2 SIMPLE_RADIAL 100 80 60 50.5 40.5 0.1\r\n3 OPENCV 100 80 70 72 50.5 40.5 0.1 0.2 0.001 0.002\r\n';
    const images = '# i\n1 1 0 0 0 0 0 0 1 a.jpg\n1.5 2.5 -1\n\n2 0 0 0 1 1 2 3 3 b.png\n\n';
    const points = '# p\n';
    const m = importColmapText({ cameras, images, points3D: points });
    expect(m.cameras[0].cam).toEqual({ width: 100, height: 80, f: 51, cx: 49, cy: 39, k1: 0, k2: 0 });
    expect(m.cameras[1].cam.k1).toBe(0.1);
    expect(m.cameras[2].cam).toEqual({ width: 100, height: 80, f: 71, cx: 50, cy: 40, k1: 0.1, k2: 0.2 });
    expect(m.images.length).toBe(2);
    expect(m.images[0].points2D).toEqual([{ x: 1, y: 2, point3D: -1 }]);
    expect(m.images[1].points2D).toEqual([]);
    // q = (0,0,0,1): 180° about z
    expect(Array.from(m.images[1].pose.R)).toEqual([-1, 0, 0, 0, -1, 0, 0, 0, 1]);
    expect(() => importColmapText({ cameras: '1 FISHEYE 1 1 1 1 1', images: '', points3D: '' })).toThrow(/unsupported/);
  });

  it('packs a standard project layout into a ZIP', () => {
    const jpeg = Uint8Array.of(0xff, 0xd8, 0xff, 0xd9);
    const zip = createColmapZip(makeModel(), [{ name: 'frame_0000.jpg', data: jpeg }]);
    const names = readZip(zip).map((e) => e.name);
    expect(names).toEqual(['sparse/0/cameras.txt', 'sparse/0/images.txt', 'sparse/0/points3D.txt', 'images/frame_0000.jpg']);
  });
});
