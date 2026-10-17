/**
 * Adversarial browser tests for CameraCapture lifecycle (headless Chromium, fake camera).
 */
import { afterAll, describe, expect, it } from 'vitest';
import { CameraCapture } from './camera';

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
const videos: HTMLVideoElement[] = [];
function makeVideo(): HTMLVideoElement {
  const v = document.createElement('video');
  v.style.width = '160px';
  document.body.appendChild(v);
  videos.push(v);
  return v;
}
afterAll(() => {
  for (const v of videos) v.remove();
});

/** Record every track handed out by getUserMedia (to detect leaked, still-live cameras). */
function trackGum(): { tracks: MediaStreamTrack[]; restore: () => void } {
  const md = navigator.mediaDevices;
  const orig = md.getUserMedia.bind(md);
  const tracks: MediaStreamTrack[] = [];
  md.getUserMedia = async (c?: MediaStreamConstraints): Promise<MediaStream> => {
    const s = await orig(c);
    tracks.push(...s.getTracks());
    return s;
  };
  return { tracks, restore: () => (md.getUserMedia = orig) };
}

async function framesWithin(cam: CameraCapture, n: number, ms: number): Promise<number> {
  let count = 0;
  const off = cam.onFrame(() => count++);
  const t0 = performance.now();
  while (count < n && performance.now() - t0 < ms) await sleep(20);
  off();
  return count;
}

describe('CameraCapture lifecycle (adversarial)', () => {
  it('start() while restart() is in flight leaves exactly one live camera', async () => {
    const video = makeVideo();
    const cam = new CameraCapture();
    await cam.start(video, { quality: 'fast' });
    const g = trackGum();
    try {
      const r = cam.restart('test');
      const s = cam.start(video, { quality: 'fast' });
      await Promise.allSettled([r, s]);
      await sleep(100);
      const live = g.tracks.filter((t) => t.readyState === 'live');
      expect(live.length).toBe(1);
      expect(live[0]).toBe(cam.track);
      expect(cam.state).toBe('running');
      cam.stop();
      expect(g.tracks.every((t) => t.readyState === 'ended')).toBe(true);
    } finally {
      g.restore();
      cam.stop();
    }
  });

  it('stop() while restart() attaches the new stream: stays stopped, nothing live', async () => {
    const video = makeVideo();
    const cam = new CameraCapture();
    await cam.start(video, { quality: 'fast' });
    const g = trackGum();
    const origPlay = video.play.bind(video);
    try {
      video.play = (): Promise<void> => {
        cam.stop(); // user leaves the capture screen exactly while the restart re-attaches
        return origPlay();
      };
      await cam.restart('test');
      video.play = origPlay;
      await sleep(100);
      expect(cam.state).toBe('stopped');
      expect(g.tracks.filter((t) => t.readyState === 'live').length).toBe(0);
    } finally {
      video.play = origPlay;
      g.restore();
      cam.stop();
    }
  });

  it('frames keep flowing after restart() (frame loop re-armed on the new stream)', async () => {
    const video = makeVideo();
    const cam = new CameraCapture();
    try {
      await cam.start(video, { quality: 'fast' });
      let count = 0;
      const off = cam.onFrame(() => count++);
      const t0 = performance.now();
      while (count < 3 && performance.now() - t0 < 3000) await sleep(20);
      expect(count).toBeGreaterThanOrEqual(3);
      expect(await cam.restart('test')).toBe(true);
      const c0 = count;
      const t1 = performance.now();
      while (count < c0 + 5 && performance.now() - t1 < 3000) await sleep(20);
      off();
      expect(count).toBeGreaterThanOrEqual(c0 + 5);
    } finally {
      cam.stop();
    }
  });

  it('resumes a <video> that the platform paused during an interruption (iOS background)', async () => {
    const video = makeVideo();
    const cam = new CameraCapture();
    try {
      await cam.start(video, { quality: 'fast' });
      const track = cam.track!;
      track.dispatchEvent(new Event('mute'));
      expect(cam.state).toBe('interrupted');
      video.pause(); // WebKit pauses media elements of a backgrounded page
      track.dispatchEvent(new Event('unmute'));
      expect(cam.state).toBe('running');
      expect(cam.segment).toBe(1);
      expect(await framesWithin(cam, 3, 3000)).toBeGreaterThanOrEqual(3);
    } finally {
      cam.stop();
    }
  });

  it('hidden → visible with a live (unmuted) track starts the new segment immediately, once', async () => {
    const video = makeVideo();
    const cam = new CameraCapture();
    let hidden = false;
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => hidden });
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => (hidden ? 'hidden' : 'visible') });
    try {
      await cam.start(video, { quality: 'fast' });
      const segs: number[] = [];
      const off = cam.onFrame((_t, m) => segs.push(m.segment));
      hidden = true;
      document.dispatchEvent(new Event('visibilitychange'));
      expect(cam.state).toBe('interrupted');
      await sleep(100);
      hidden = false;
      document.dispatchEvent(new Event('visibilitychange'));
      expect(cam.state).toBe('running');
      expect(cam.segment).toBe(1);
      const n0 = segs.length;
      await sleep(1500); // past the 1.2 s unmute grace timer: no second increment
      off();
      expect(cam.segment).toBe(1);
      expect(segs.slice(n0).length).toBeGreaterThan(3);
      expect(segs.slice(n0).every((x) => x === 1)).toBe(true);
    } finally {
      delete (document as unknown as Record<string, unknown>).hidden;
      delete (document as unknown as Record<string, unknown>).visibilityState;
      cam.stop();
    }
  });

  it('a manual restart() is not refused by the automatic-restart rate limit', async () => {
    const video = makeVideo();
    const cam = new CameraCapture();
    try {
      await cam.start(video, { quality: 'fast' });
      for (let i = 0; i < 4; i++) expect(await cam.restart()).toBe(true);
      expect(cam.state).toBe('running');
      expect(cam.segment).toBe(4);
    } finally {
      cam.stop();
    }
  });
});
