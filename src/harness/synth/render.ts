/**
 * Frame renderer for synthetic datasets (WebGL2 via three.js).
 *
 * Colour frames: the scene is rendered UNDISTORTED (pinhole, linear HDR half-float)
 * over a window of normalized coordinates slightly larger than the undistorted image
 * of the sensor, at `supersample`× the focal length. A full-screen pass then
 * resamples it so that output pixels follow the RADIAL model exactly: for each output
 * pixel (pixel centres at integers) and each of ss×ss sub-pixel samples, the distorted
 * normalized coordinate (u − cx)/f is undistorted by Newton iteration and the
 * undistorted render is sampled bilinearly. Then exposure gain, vignetting and
 * signal-dependent sensor noise (seeded per-pixel hash) are applied in linear light and
 * the result is sRGB-encoded into the canvas (→ JPEG via canvas.toBlob).
 * Motion blur: several sub-frame renders (poses across the exposure) are averaged
 * before resampling.
 * Rolling shutter (renderFrameBands): the output image is split into bands of pixels;
 * each band is rendered with its own sub-frame poses into the part of the undistorted
 * target its pixels sample (an off-axis sub-frustum + viewport, so the rasterised samples
 * are identical to a full render) and resampled into the canvas under a scissor.
 * Focus breathing: setFocal() changes f of the resampling (the undistorted windows are
 * sized for the smallest focal given at construction).
 *
 * Depth frames: view-space z rendered to float targets, resampled with nearest
 * lookup through the same distortion map (GT depth aligned with the JPEG pixels), or
 * kept undistorted at low resolution for visibility tests.
 */
import {
  AdditiveBlending,
  Camera,
  DoubleSide,
  FloatType,
  GLSL3,
  HalfFloatType,
  LinearFilter,
  Mesh,
  NearestFilter,
  NoBlending,
  OrthographicCamera,
  PlaneGeometry,
  RGBAFormat,
  Scene,
  ShaderMaterial,
  WebGLRenderTarget,
  WebGLRenderer,
} from 'three';
import type { CameraIntrinsics, Pose } from '../../core/types';
import { type NormalizedWindow, undistortedWindow, undistortNormalized } from './conventions';
import type { PixelRect } from './realism';
import type { WindowDepth } from './sampling';
import type { SynthScene } from './scene-common';
import { DEFAULT_RENDER_EFFECTS, type RenderEffects } from './render-effects';
import { applyPoseToThreeCamera, setWindowProjection } from './three-camera';

export type { RenderEffects } from './render-effects';
export { DEFAULT_RENDER_EFFECTS } from './render-effects';

const FULLSCREEN_VS = /* glsl */ `
void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

const UNDISTORT_GLSL = /* glsl */ `
uniform vec4 win;   // xmin, xmax, ymin, ymax (undistorted normalized coords of the source)
uniform vec4 intr;  // f, cx, cy, image height
uniform vec2 kd;    // k1, k2
vec2 undistortN(vec2 d) {
  float rd = length(d);
  if (rd < 1e-9 || (kd.x == 0.0 && kd.y == 0.0)) return d;
  float r = rd;
  for (int i = 0; i < 4; i++) { // quadratic convergence: 4 steps reach float precision for |k| < 0.2
    float r2 = r * r;
    float fr = r * (1.0 + kd.x * r2 + kd.y * r2 * r2) - rd;
    float fp = 1.0 + 3.0 * kd.x * r2 + 5.0 * kd.y * r2 * r2;
    r -= fr / fp;
  }
  return d * (r / rd);
}
// pixel (OpenCV, centres at integers) -> source texture coordinate
vec2 sourceST(vec2 p) {
  vec2 xu = undistortN((p - intr.yz) / intr.x);
  return vec2((xu.x - win.x) / (win.y - win.x), (win.w - xu.y) / (win.w - win.z));
}
`;

const COLOR_FS = /* glsl */ `
precision highp float;
precision highp int;
uniform sampler2D src;
uniform vec2 imageSize;
uniform int ss;
uniform float gain;
uniform vec3 wb;
uniform float vignette;
uniform vec2 noiseVar;  // shot, read
uniform uint seed;
${UNDISTORT_GLSL}
out highp vec4 fragColor;

uvec3 pcg3d(uvec3 v) {
  v = v * 1664525u + 1013904223u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  v ^= v >> 16u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  return v;
}
float srgbEncode(float c) {
  c = clamp(c, 0.0, 1.0);
  return c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}
void main() {
  vec2 pix = vec2(gl_FragCoord.x - 0.5, intr.w - gl_FragCoord.y - 0.5);
  vec3 acc = vec3(0.0);
  float inv = 1.0 / float(ss);
  for (int j = 0; j < ss; j++) {
    for (int i = 0; i < ss; i++) {
      vec2 p = pix + (vec2(float(i), float(j)) + 0.5) * inv - 0.5;
      acc += texture(src, sourceST(p)).rgb;
    }
  }
  vec3 c = acc * (inv * inv);
  vec2 dc = (pix - intr.yz) / (0.5 * length(imageSize));
  float vig = 1.0 - vignette * dot(dc, dc);
  c *= gain * vig;
  c *= wb;
  if (noiseVar.x > 0.0 || noiseVar.y > 0.0) {
    uvec3 h1 = pcg3d(uvec3(uint(pix.x), uint(pix.y), seed));
    uvec3 h2 = pcg3d(h1 ^ uvec3(0x9e3779b9u, 0x85ebca6bu, 0xc2b2ae35u));
    vec3 u1 = (vec3(h1 >> 8u) + 0.5) / 16777216.0;
    vec3 u2 = vec3(h2 >> 8u) / 16777216.0;
    vec3 g = sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
    vec3 sigma = sqrt(max(vec3(0.0), noiseVar.x * c + noiseVar.y));
    c += sigma * g;
  }
  fragColor = vec4(srgbEncode(c.r), srgbEncode(c.g), srgbEncode(c.b), 1.0);
}
`;

const DEPTH_RESAMPLE_FS = /* glsl */ `
precision highp float;
precision highp int;
uniform highp sampler2D src;
${UNDISTORT_GLSL}
out highp vec4 fragColor;
void main() {
  vec2 pix = vec2(gl_FragCoord.x - 0.5, intr.w - gl_FragCoord.y - 0.5);
  vec2 st = sourceST(pix);
  ivec2 sz = textureSize(src, 0);
  ivec2 ij = ivec2(floor(st * vec2(sz)));
  float z = 0.0;
  if (ij.x >= 0 && ij.y >= 0 && ij.x < sz.x && ij.y < sz.y) z = texelFetch(src, ij, 0).r;
  fragColor = vec4(z, 0.0, 0.0, 1.0);
}
`;

const ACCUM_FS = /* glsl */ `
precision highp float;
uniform sampler2D src;
uniform float weight;
uniform vec2 size;
out highp vec4 fragColor;
void main() { fragColor = vec4(texture(src, gl_FragCoord.xy / size).rgb * weight, 1.0); }
`;

const DEPTH_VS = /* glsl */ `
out float vZ;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vZ = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`;

const DEPTH_FS = /* glsl */ `
precision highp float;
in float vZ;
out highp vec4 fragColor;
void main() { fragColor = vec4(vZ, 0.0, 0.0, 1.0); }
`;

interface Pass {
  scene: Scene;
  material: ShaderMaterial;
}

function makePass(fs: string, uniforms: Record<string, { value: unknown }>): Pass {
  const material = new ShaderMaterial({
    glslVersion: GLSL3,
    vertexShader: FULLSCREEN_VS,
    fragmentShader: fs,
    uniforms,
    depthTest: false,
    depthWrite: false,
  });
  const scene = new Scene();
  const quad = new Mesh(new PlaneGeometry(2, 2), material);
  quad.frustumCulled = false;
  scene.add(quad);
  return { scene, material };
}

/** Window grown to an integer pixel grid of pitch 1/(f·scale). */
function gridWindow(w: NormalizedWindow, f: number, scale: number): { win: NormalizedWindow; width: number; height: number } {
  const pitch = 1 / (f * scale);
  const width = Math.ceil((w.xmax - w.xmin) / pitch);
  const height = Math.ceil((w.ymax - w.ymin) / pitch);
  return { win: { xmin: w.xmin, xmax: w.xmin + width * pitch, ymin: w.ymin, ymax: w.ymin + height * pitch }, width, height };
}

export interface SynthRendererOptions {
  /**
   * Smallest focal length (px) setFocal() will be given (focus breathing). The undistorted
   * render windows are sized for it so that every focal ≥ minFocal is covered. Default cam.f.
   */
  minFocal?: number;
}

/** One band of a rolling-shutter frame: its output pixels and its sub-frame poses. */
export interface RenderBand {
  rect: PixelRect;
  /** Poses averaged over the band's exposure (≥ 1). */
  poses: readonly Pose[];
}

/** Texel rectangle [c0, c1) × [r0, r1) of a render target (GL rows, 0 = bottom). */
interface TexelRect {
  c0: number;
  c1: number;
  r0: number;
  r1: number;
}

export class SynthRenderer {
  readonly renderer: WebGLRenderer;
  readonly canvas: HTMLCanvasElement;
  readonly effects: RenderEffects;
  private readonly camera = new Camera();
  private readonly ortho = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly colorWin: NormalizedWindow;
  private readonly depthWin: NormalizedWindow;
  private readonly sceneRT: WebGLRenderTarget;
  private readonly accumRT: WebGLRenderTarget | null;
  private readonly depthRT: WebGLRenderTarget;
  private readonly depthOutRT: WebGLRenderTarget;
  private visRT: WebGLRenderTarget | null = null;
  private visWin: NormalizedWindow | null = null;
  private readonly colorPass: Pass;
  private readonly depthPass: Pass;
  private readonly accumPass: Pass;
  private readonly depthMaterial: ShaderMaterial;
  private depthBuf: Float32Array | null = null;
  private readonly colorPitch: number;
  private readonly baseWin: NormalizedWindow;
  private readonly intrU: { value: number[] };
  private focal: number;

  constructor(
    readonly synth: SynthScene,
    readonly cam: CameraIntrinsics,
    effects: Partial<RenderEffects> = {},
    opts: SynthRendererOptions = {},
  ) {
    this.effects = { ...DEFAULT_RENDER_EFFECTS, ...effects };
    const ss = Math.max(1, Math.round(this.effects.supersample));
    this.canvas = document.createElement('canvas');
    this.canvas.width = cam.width;
    this.canvas.height = cam.height;
    this.renderer = new WebGLRenderer({
      canvas: this.canvas,
      antialias: false,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
      alpha: false,
    });
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(cam.width, cam.height, false);
    this.renderer.autoClear = true;
    this.renderer.setClearColor(0x000000, 1);

    this.focal = cam.f;
    const minF = Math.min(cam.f, opts.minFocal ?? cam.f);
    const base = undistortedWindow({ ...cam, f: minF }, 3);
    this.baseWin = base;
    const cg = gridWindow(base, cam.f, ss);
    this.colorWin = cg.win;
    this.colorPitch = 1 / (cam.f * ss);
    this.sceneRT = new WebGLRenderTarget(cg.width, cg.height, {
      type: HalfFloatType,
      format: RGBAFormat,
      minFilter: LinearFilter,
      magFilter: LinearFilter,
      depthBuffer: true,
      generateMipmaps: false,
    });
    this.accumRT =
      this.effects.motionBlurSamples > 1
        ? new WebGLRenderTarget(cg.width, cg.height, { type: HalfFloatType, format: RGBAFormat, minFilter: LinearFilter, magFilter: LinearFilter, depthBuffer: false, generateMipmaps: false })
        : null;
    const dg = gridWindow(base, cam.f, 1);
    this.depthWin = dg.win;
    this.depthRT = new WebGLRenderTarget(dg.width, dg.height, { type: FloatType, format: RGBAFormat, minFilter: NearestFilter, magFilter: NearestFilter, depthBuffer: true, generateMipmaps: false });
    this.depthOutRT = new WebGLRenderTarget(cam.width, cam.height, { type: FloatType, format: RGBAFormat, minFilter: NearestFilter, magFilter: NearestFilter, depthBuffer: false, generateMipmaps: false });

    const intr = { value: [cam.f, cam.cx, cam.cy, cam.height] };
    this.intrU = intr;
    const kd = { value: [cam.k1, cam.k2] };
    this.colorPass = makePass(COLOR_FS, {
      src: { value: null },
      win: { value: [this.colorWin.xmin, this.colorWin.xmax, this.colorWin.ymin, this.colorWin.ymax] },
      intr,
      kd,
      imageSize: { value: [cam.width, cam.height] },
      ss: { value: ss },
      gain: { value: 1 },
      wb: { value: [1, 1, 1] },
      vignette: { value: this.effects.vignetting },
      noiseVar: { value: [0, 0] },
      seed: { value: 0 },
    });
    this.depthPass = makePass(DEPTH_RESAMPLE_FS, {
      src: { value: this.depthRT.texture },
      win: { value: [this.depthWin.xmin, this.depthWin.xmax, this.depthWin.ymin, this.depthWin.ymax] },
      intr,
      kd,
    });
    this.accumPass = makePass(ACCUM_FS, { src: { value: this.sceneRT.texture }, weight: { value: 1 }, size: { value: [cg.width, cg.height] } });
    this.accumPass.material.blending = AdditiveBlending;
    this.depthMaterial = new ShaderMaterial({ glslVersion: GLSL3, vertexShader: DEPTH_VS, fragmentShader: DEPTH_FS, side: DoubleSide, blending: NoBlending });
  }

  /** Size of the undistorted colour render target (diagnostics). */
  get renderTargetSize(): [number, number] {
    return [this.sceneRT.width, this.sceneRT.height];
  }

  /** Current focal length (px) of the colour / depth resampling. */
  get currentFocal(): number {
    return this.focal;
  }

  /**
   * Focal length (px) of the following colour and depth renders (focus breathing). The
   * principal point and distortion stay; f must be ≥ the constructor's minFocal.
   */
  setFocal(f: number): void {
    this.focal = f;
    this.intrU.value = [f, this.cam.cx, this.cam.cy, this.cam.height];
  }

  /** Linear-light white-balance gains (r, g, b) of the following colour renders. */
  setWhiteBalance(rgb: readonly [number, number, number]): void {
    this.colorPass.material.uniforms.wb.value = [rgb[0], rgb[1], rgb[2]];
  }

  /**
   * Texels of the undistorted colour target sampled by the output pixels of `rect` at the
   * current focal (pixel extents ±0.5 px, undistorted, plus a 3-texel margin for bilinear
   * sampling), clamped to the target.
   */
  private bandTexelRect(rect: PixelRect): TexelRect {
    const { cx, cy, k1, k2 } = this.cam;
    const f = this.focal;
    const ua = rect.u0 - 0.5,
      ub = rect.u1 + 0.5,
      va = rect.v0 - 0.5,
      vb = rect.v1 + 0.5;
    let xmin = Infinity,
      xmax = -Infinity,
      ymin = Infinity,
      ymax = -Infinity;
    const o = new Float64Array(2);
    const visit = (u: number, v: number) => {
      undistortNormalized(k1, k2, (u - cx) / f, (v - cy) / f, o);
      if (o[0] < xmin) xmin = o[0];
      if (o[0] > xmax) xmax = o[0];
      if (o[1] < ymin) ymin = o[1];
      if (o[1] > ymax) ymax = o[1];
    };
    const n = 48;
    for (let i = 0; i <= n; i++) {
      const u = ua + ((ub - ua) * i) / n,
        v = va + ((vb - va) * i) / n;
      visit(u, va);
      visit(u, vb);
      visit(ua, v);
      visit(ub, v);
    }
    for (let j = 1; j < 8; j++) for (let i = 1; i < 8; i++) visit(ua + ((ub - ua) * i) / 8, va + ((vb - va) * j) / 8);
    const w = this.colorWin,
      p = this.colorPitch,
      W = this.sceneRT.width,
      H = this.sceneRT.height,
      m = 3;
    const clamp = (x: number, hi: number) => Math.max(0, Math.min(hi, x));
    return {
      c0: clamp(Math.floor((xmin - w.xmin) / p) - m, W),
      c1: clamp(Math.ceil((xmax - w.xmin) / p) + m, W),
      r0: clamp(Math.floor((w.ymax - ymax) / p) - m, H),
      r1: clamp(Math.ceil((w.ymax - ymin) / p) + m, H),
    };
  }

  /** Render the scene into the texel rectangle `tr` of `target` only (sub-frustum + viewport + scissor). */
  private renderSceneRect(pose: Pose, tr: TexelRect, target: WebGLRenderTarget): void {
    const w = this.colorWin,
      p = this.colorPitch;
    applyPoseToThreeCamera(this.camera, pose);
    const sub: NormalizedWindow = { xmin: w.xmin + tr.c0 * p, xmax: w.xmin + tr.c1 * p, ymin: w.ymax - tr.r1 * p, ymax: w.ymax - tr.r0 * p };
    setWindowProjection(this.camera, sub, this.synth.near, this.synth.far);
    target.viewport.set(tr.c0, tr.r0, tr.c1 - tr.c0, tr.r1 - tr.r0);
    target.scissor.set(tr.c0, tr.r0, tr.c1 - tr.c0, tr.r1 - tr.r0);
    target.scissorTest = true;
    this.renderer.setRenderTarget(target);
    this.renderer.setClearColor(0x000000, 1);
    this.renderer.clear(true, true, true);
    this.renderer.render(this.synth.scene, this.camera);
  }

  /**
   * Render one colour frame band by band (rolling shutter and/or adaptive motion blur):
   * band b's output pixels come from the average of renders at `b.poses`. The bands must
   * partition the image. Returns the number of scene renders.
   */
  renderFrameBands(bands: readonly RenderBand[], gain: number, seed: number): number {
    const r = this.renderer;
    const u = this.colorPass.material.uniforms;
    u.gain.value = gain;
    u.seed.value = seed >>> 0;
    u.noiseVar.value = this.effects.noise ? [this.effects.shotNoise, this.effects.readNoise] : [0, 0];
    const H = this.cam.height;
    let renders = 0;
    const prevAutoClear = r.autoClear;
    try {
      for (const band of bands) {
        if (band.poses.length === 0) throw new Error('renderFrameBands: band without poses');
        const tr = this.bandTexelRect(band.rect);
        if (tr.c1 <= tr.c0 || tr.r1 <= tr.r0) throw new Error('renderFrameBands: band outside the render window');
        let src = this.sceneRT;
        if (band.poses.length > 1) {
          if (!this.accumRT) throw new Error('renderFrameBands: several sub-frames need effects.motionBlurSamples > 1');
          const acc = this.accumRT;
          acc.viewport.set(0, 0, acc.width, acc.height);
          acc.scissor.set(tr.c0, tr.r0, tr.c1 - tr.c0, tr.r1 - tr.r0);
          acc.scissorTest = true;
          r.autoClear = true;
          r.setRenderTarget(acc);
          r.setClearColor(0x000000, 0);
          r.clear(true, false, false);
          this.accumPass.material.uniforms.weight.value = 1 / band.poses.length;
          for (const p of band.poses) {
            this.renderSceneRect(p, tr, this.sceneRT);
            renders++;
            r.setRenderTarget(acc);
            r.autoClear = false;
            r.render(this.accumPass.scene, this.ortho);
            r.autoClear = true;
          }
          src = acc;
        } else {
          r.autoClear = true;
          this.renderSceneRect(band.poses[0], tr, this.sceneRT);
          renders++;
        }
        u.src.value = src.texture;
        r.setRenderTarget(null);
        r.setScissor(band.rect.u0, H - 1 - band.rect.v1, band.rect.u1 - band.rect.u0 + 1, band.rect.v1 - band.rect.v0 + 1);
        r.setScissorTest(true);
        r.autoClear = false;
        r.render(this.colorPass.scene, this.ortho);
        r.setScissorTest(false);
      }
    } finally {
      for (const t of [this.sceneRT, this.accumRT]) {
        if (!t) continue;
        t.viewport.set(0, 0, t.width, t.height);
        t.scissor.set(0, 0, t.width, t.height);
        t.scissorTest = false;
      }
      r.setScissorTest(false);
      r.autoClear = prevAutoClear;
      r.setRenderTarget(null);
    }
    return renders;
  }

  private renderScene(pose: Pose, win: NormalizedWindow, target: WebGLRenderTarget): void {
    applyPoseToThreeCamera(this.camera, pose);
    setWindowProjection(this.camera, win, this.synth.near, this.synth.far);
    this.renderer.setRenderTarget(target);
    this.renderer.setClearColor(0x000000, 1);
    this.renderer.clear(true, true, true);
    this.renderer.render(this.synth.scene, this.camera);
  }

  /**
   * Render one colour frame into the canvas. `poses` are the sub-frame poses across
   * the exposure (1 = no motion blur). `gain` = exposure multiplier, `seed` = noise seed.
   */
  renderFrame(poses: readonly Pose[], gain: number, seed: number): void {
    const r = this.renderer;
    let src = this.sceneRT;
    if (poses.length > 1 && this.accumRT) {
      r.setRenderTarget(this.accumRT);
      r.setClearColor(0x000000, 0);
      r.clear(true, false, false);
      this.accumPass.material.uniforms.weight.value = 1 / poses.length;
      for (const p of poses) {
        this.renderScene(p, this.colorWin, this.sceneRT);
        r.setRenderTarget(this.accumRT);
        r.autoClear = false;
        r.render(this.accumPass.scene, this.ortho);
        r.autoClear = true;
      }
      src = this.accumRT;
    } else {
      this.renderScene(poses[0], this.colorWin, this.sceneRT);
    }
    const u = this.colorPass.material.uniforms;
    u.src.value = src.texture;
    u.gain.value = gain;
    u.seed.value = seed >>> 0;
    u.noiseVar.value = this.effects.noise ? [this.effects.shotNoise, this.effects.readNoise] : [0, 0];
    r.setRenderTarget(null);
    r.render(this.colorPass.scene, this.ortho);
  }

  /** JPEG of the current canvas content. */
  toJpeg(quality = 0.92): Promise<Blob> {
    return new Promise((resolve, reject) =>
      this.canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), 'image/jpeg', quality),
    );
  }

  /** RGBA8 of the current canvas content (row 0 = top), for tests. */
  readPixels(): Uint8Array {
    const gl = this.renderer.getContext();
    const W = this.cam.width,
      H = this.cam.height;
    const buf = new Uint8Array(W * H * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, buf);
    const out = new Uint8Array(W * H * 4);
    for (let y = 0; y < H; y++) out.set(buf.subarray((H - 1 - y) * W * 4, (H - y) * W * 4), y * W * 4);
    return out;
  }

  private withDepthMaterial(fn: () => void): void {
    const s = this.synth.scene;
    const prevOverride = s.overrideMaterial;
    const hidden = this.synth.depthHidden.map((o) => [o, o.visible] as const);
    for (const [o] of hidden) o.visible = false;
    s.overrideMaterial = this.depthMaterial;
    try {
      fn();
    } finally {
      s.overrideMaterial = prevOverride;
      for (const [o, v] of hidden) o.visible = v;
    }
  }

  private readFloatR(rt: WebGLRenderTarget): Float32Array {
    const W = rt.width,
      H = rt.height;
    if (!this.depthBuf || this.depthBuf.length < W * H * 4) this.depthBuf = new Float32Array(W * H * 4);
    const buf = this.depthBuf;
    this.renderer.readRenderTargetPixels(rt, 0, 0, W, H, buf);
    const out = new Float32Array(W * H);
    for (let y = 0; y < H; y++) {
      const src = (H - 1 - y) * W * 4,
        dst = y * W;
      for (let x = 0; x < W; x++) out[dst + x] = buf[src + 4 * x];
    }
    return out;
  }

  /**
   * Ground-truth z-depth (m) for every pixel of the DISTORTED output image (row 0 =
   * top), 0 where no surface (sky).
   */
  renderDepth(pose: Pose): Float32Array {
    this.withDepthMaterial(() => this.renderScene(pose, this.depthWin, this.depthRT));
    this.renderer.setRenderTarget(this.depthOutRT);
    this.renderer.render(this.depthPass.scene, this.ortho);
    this.renderer.setRenderTarget(null);
    return this.readFloatR(this.depthOutRT);
  }

  /** Low-resolution undistorted depth over the colour window (for visibility tests). */
  renderWindowDepth(pose: Pose, scale = 0.25): WindowDepth {
    if (!this.visRT || !this.visWin) {
      const g = gridWindow(this.baseWin, this.cam.f, scale);
      this.visWin = g.win;
      this.visRT = new WebGLRenderTarget(g.width, g.height, { type: FloatType, format: RGBAFormat, minFilter: NearestFilter, magFilter: NearestFilter, depthBuffer: true, generateMipmaps: false });
    }
    const rt = this.visRT,
      win = this.visWin;
    this.withDepthMaterial(() => this.renderScene(pose, win, rt));
    this.renderer.setRenderTarget(null);
    return { data: this.readFloatR(rt), width: rt.width, height: rt.height, window: win };
  }

  dispose(): void {
    this.sceneRT.dispose();
    this.accumRT?.dispose();
    this.depthRT.dispose();
    this.depthOutRT.dispose();
    this.visRT?.dispose();
    for (const p of [this.colorPass, this.depthPass, this.accumPass]) {
      p.material.dispose();
      (p.scene.children[0] as Mesh).geometry.dispose();
    }
    this.depthMaterial.dispose();
    this.renderer.dispose();
    this.renderer.forceContextLoss();
  }
}
