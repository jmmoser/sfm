import { describe, expect, it, vi } from 'vitest';
import { CameraCapture, candidateSizes, chooseCamera, classifyCamera, classifyCameraLabel, focalPriorPx, hfovPriorDeg, type CameraInfo } from './camera';

describe('classifyCameraLabel', () => {
  const cases: [string, string, string][] = [
    // English AVFoundation names
    ['Back Camera', 'wide', 'back'],
    ['Back Ultra Wide Camera', 'ultrawide', 'back'],
    ['Back Telephoto Camera', 'tele', 'back'],
    ['Back Dual Wide Camera', 'virtual', 'back'],
    ['Back Dual Camera', 'virtual', 'back'],
    ['Back Triple Camera', 'virtual', 'back'],
    ['Front Camera', 'front', 'front'],
    ['Front TrueDepth Camera', 'front', 'front'],
    // German (Scandit tables) and other localisations
    ['Rückkamera', 'wide', 'back'],
    ['Rückseitige Ultra-Weitwinkelkamera', 'ultrawide', 'back'],
    ['Rückseitige Dual-Weitwinkelkamera', 'virtual', 'back'],
    ['Rückseitige Triple-Kamera', 'virtual', 'back'],
    ['Rückseitige Telekamera', 'tele', 'back'],
    ['Vorderseitige Kamera', 'front', 'front'],
    ['Caméra arrière', 'wide', 'back'],
    ['Caméra ultra grand-angle arrière', 'ultrawide', 'back'],
    ['Caméra double grand-angle arrière', 'virtual', 'back'],
    ['Caméra avant', 'front', 'front'],
    ['Cámara trasera', 'wide', 'back'],
    ['Cámara frontal', 'front', 'front'],
    ['Câmera Traseira', 'wide', 'back'],
    ['Fotocamera posteriore', 'wide', 'back'],
    ['Fotocamera anteriore', 'front', 'front'],
    ['背面カメラ', 'wide', 'back'],
    ['背面超広角カメラ', 'ultrawide', 'back'],
    ['背面デュアル広角カメラ', 'virtual', 'back'],
    ['背面トリプルカメラ', 'virtual', 'back'],
    ['背面望遠カメラ', 'tele', 'back'],
    ['前面カメラ', 'front', 'front'],
    ['后置相机', 'wide', 'back'],
    ['后置超广角相机', 'ultrawide', 'back'],
    ['后置双广角相机', 'virtual', 'back'],
    ['前置相机', 'front', 'front'],
    ['후면 카메라', 'wide', 'back'],
    ['후면 초광각 카메라', 'ultrawide', 'back'],
    ['전면 카메라', 'front', 'front'],
    ['Задняя камера', 'wide', 'back'],
    ['Передняя камера', 'front', 'front'],
    // Android Chrome / desktop
    ['camera2 0, facing back', 'wide', 'back'],
    ['camera2 1, facing front', 'front', 'front'],
    ['fake_device_0', 'unknown', 'unknown'],
    ['FaceTime HD Camera', 'front', 'front'],
    ['HD Pro Webcam C920', 'unknown', 'unknown'],
    ['Telecamera integrata', 'unknown', 'unknown'],
    ['', 'unknown', 'unknown'],
  ];
  for (const [label, lens, facing] of cases) {
    it(`"${label}" → ${lens}/${facing}`, () => {
      expect(classifyCameraLabel(label)).toEqual({ lens, facing });
    });
  }
  it('capabilities refine the label: zoom.min 0.5 ⇒ virtual, facingMode user ⇒ front', () => {
    expect(classifyCamera('Rückkamera X', { zoom: { min: 0.5, max: 10 } }).lens).toBe('virtual');
    expect(classifyCamera('Kamera 1', { facingMode: ['environment'], zoom: { min: 1, max: 10 } })).toEqual({ lens: 'unknown', facing: 'back' });
    expect(classifyCamera('Kamera 2', { facingMode: ['user'] })).toEqual({ lens: 'front', facing: 'front' });
  });
});

describe('chooseCamera', () => {
  const mk = (label: string, id = label): CameraInfo => ({ deviceId: id, label, ...classifyCameraLabel(label) });
  it('picks the physical wide camera on a Pro iPhone (not the lens-switching virtual one)', () => {
    const cams = ['Front Camera', 'Back Triple Camera', 'Back Dual Wide Camera', 'Back Ultra Wide Camera', 'Back Dual Camera', 'Back Camera', 'Back Telephoto Camera'].map((l) =>
      mk(l),
    );
    expect(chooseCamera(cams)!.label).toBe('Back Camera');
    expect(chooseCamera(cams, 'front')!.label).toBe('Front Camera');
  });
  it('falls back sensibly', () => {
    expect(chooseCamera([mk('Front Camera'), mk('Back Dual Wide Camera'), mk('Back Ultra Wide Camera')])!.label).toBe('Back Dual Wide Camera');
    expect(chooseCamera([mk('camera2 1, facing front'), mk('camera2 0, facing back'), mk('camera2 2, facing back')])!.label).toBe('camera2 0, facing back');
    expect(chooseCamera([mk('fake_device_0')])!.label).toBe('fake_device_0');
    expect(chooseCamera([])).toBeNull();
  });
});

describe('FOV / focal priors', () => {
  it('Apple main camera 4:3 ≈ 67.3° (26 mm), Pro 69.5°, 16:9 slightly wider, cropped formats narrower', () => {
    expect(hfovPriorDeg('wide', 1920, 1440, { apple: true })).toBeCloseTo(67.3, 6);
    expect(hfovPriorDeg('wide', 1440, 1920, { apple: true })).toBeCloseTo(67.3, 6); // portrait UI: same long-side FOV
    expect(hfovPriorDeg('wide', 4032, 3024, { apple: true, pro: true })).toBeCloseTo(69.5, 6);
    expect(hfovPriorDeg('wide', 1920, 1080, { apple: true })).toBeCloseTo((67.3 * 68.16) / 67.1, 6);
    expect(hfovPriorDeg('wide', 1440, 1080, { apple: true })).toBeLessThan(60);
    expect(hfovPriorDeg('ultrawide', 1920, 1440, { apple: true })).toBeGreaterThan(100);
    expect(hfovPriorDeg('unknown', 640, 480)).toBe(63);
  });
  it('focal prior uses the long side and matches the measured 12 Pro Max value (≈1448 px @1920)', () => {
    const f = focalPriorPx(67.1, 1920, 1440);
    expect(f).toBeGreaterThan(1440);
    expect(f).toBeLessThan(1456);
    expect(focalPriorPx(67.1, 1440, 1920)).toBeCloseTo(f, 9);
  });
});

describe('candidateSizes', () => {
  it('max tries the largest 4:3 then 16:9 formats (landscape numbers)', () => {
    expect(candidateSizes('max', true)).toEqual([
      [4032, 3024],
      [3264, 2448],
      [2592, 1944],
      [1920, 1440],
      [3840, 2160],
      [1920, 1080],
    ]);
  });
  it('filters by the device maximum (either orientation)', () => {
    expect(candidateSizes('max', false, { width: 1920, height: 1080 })).toEqual([[1920, 1080]]);
    expect(candidateSizes('max', true, { width: 3024, height: 4032 })[0]).toEqual([4032, 3024]);
  });
  it('balanced ≈ 1920×1440 then 1920×1080; fast ≈ 1280×960 (native 1920×1440 on Apple: no 1280×960 preset)', () => {
    expect(candidateSizes('balanced', true)).toEqual([
      [1920, 1440],
      [1920, 1080],
    ]);
    expect(candidateSizes('fast', false)[0]).toEqual([1280, 960]);
    expect(candidateSizes('fast', true)[0]).toEqual([1920, 1440]);
  });
});

describe('CameraCapture.restart (fake media devices)', () => {
  /** Fake camera track: `ended` events, settings, focusMode capability, applyConstraints held on `gate`. */
  class FakeTrack extends EventTarget {
    kind = 'video';
    label = 'Back Camera';
    readyState: 'live' | 'ended' = 'live';
    muted = false;
    applyCalls = 0;
    constructor(private readonly env: { gate: Promise<void> | null }) {
      super();
    }
    stop(): void {
      this.readyState = 'ended';
    }
    getSettings(): Record<string, unknown> {
      return { width: 1920, height: 1440, frameRate: 30, deviceId: 'cam0' };
    }
    getCapabilities(): Record<string, unknown> {
      return { focusMode: ['continuous'] };
    }
    applyConstraints(): Promise<void> {
      this.applyCalls++;
      return this.env.gate ?? Promise.resolve();
    }
  }
  function fakeStream(track: FakeTrack): MediaStream {
    return { getVideoTracks: () => [track], getTracks: () => [track] } as unknown as MediaStream;
  }
  /** <video> whose pending rVFC callbacks survive a srcObject change (as in Chromium). */
  function fakeVideo(): {
    video: HTMLVideoElement;
    pending: Map<number, (now: number, md: Record<string, number>) => void>;
    present: () => void;
  } {
    const pending = new Map<number, (now: number, md: Record<string, number>) => void>();
    let handle = 0;
    let frames = 0;
    const v = {
      srcObject: null as MediaStream | null,
      videoWidth: 1920,
      videoHeight: 1440,
      readyState: 4,
      paused: false,
      setAttribute: () => undefined,
      play: () => Promise.resolve(),
      pause: () => undefined,
      requestVideoFrameCallback: (cb: (now: number, md: Record<string, number>) => void) => {
        pending.set(++handle, cb);
        return handle;
      },
      cancelVideoFrameCallback: (h: number) => void pending.delete(h),
    };
    const present = (): void => {
      const track = v.srcObject?.getVideoTracks()[0];
      if (!track || track.readyState !== 'live') return;
      const cbs = [...pending.values()];
      pending.clear();
      frames++;
      for (const cb of cbs) cb(performance.now(), { presentationTime: performance.now(), presentedFrames: frames, width: 1920, height: 1440 });
    };
    return { video: v as unknown as HTMLVideoElement, pending, present };
  }

  it("a restart after the track ended delivers no frames until it is done, then only the new segment's (one loop)", async () => {
    const env: { gate: Promise<void> | null } = { gate: null };
    const tracks: FakeTrack[] = [];
    vi.stubGlobal('navigator', {
      userAgent: '',
      mediaDevices: {
        enumerateDevices: async () => [{ kind: 'videoinput', deviceId: 'cam0', label: 'Back Camera', groupId: 'g0' }],
        getUserMedia: async () => {
          const t = new FakeTrack(env);
          tracks.push(t);
          return fakeStream(t);
        },
      },
    });
    const { video, pending, present } = fakeVideo();
    const cam = new CameraCapture();
    try {
      await cam.start(video, { quality: 'balanced' });
      const got: { segment: number; state: string }[] = [];
      cam.onFrame((_t, m) => got.push({ segment: m.segment, state: cam.state }));
      present();
      expect(got).toEqual([{ segment: 0, state: 'running' }]);

      let release!: () => void;
      env.gate = new Promise<void>((r) => (release = r));
      tracks[0].readyState = 'ended'; // camera taken by another app while visible
      tracks[0].dispatchEvent(new Event('ended'));
      await vi.waitFor(() => expect(tracks[1]?.applyCalls).toBe(1)); // restart re-applies the settings
      expect(cam.state).toBe('restarting');
      for (let i = 0; i < 3; i++) present(); // the new stream's first frames
      expect(got.filter((f) => f.state === 'restarting')).toEqual([]);
      expect(pending.size).toBe(0);

      env.gate = null;
      release();
      await vi.waitFor(() => expect(cam.state).toBe('running'));
      expect(cam.segment).toBe(1);
      expect(pending.size).toBe(1);
      present();
      expect(got.slice(1)).toEqual([{ segment: 1, state: 'running' }]);
    } finally {
      cam.stop();
      vi.unstubAllGlobals();
    }
  });
});
