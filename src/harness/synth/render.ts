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
import { type NormalizedWindow, undistortedWindow } from './conventions';
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

  constructor(
    readonly synth: SynthScene,
    readonly cam: CameraIntrinsics,
    effects: Partial<RenderEffects> = {},
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

    const base = undistortedWindow(cam, 3);
    const cg = gridWindow(base, cam.f, ss);
    this.colorWin = cg.win;
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
    const kd = { value: [cam.k1, cam.k2] };
    this.colorPass = makePass(COLOR_FS, {
      src: { value: null },
      win: { value: [this.colorWin.xmin, this.colorWin.xmax, this.colorWin.ymin, this.colorWin.ymax] },
      intr,
      kd,
      imageSize: { value: [cam.width, cam.height] },
      ss: { value: ss },
      gain: { value: 1 },
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
      const g = gridWindow(undistortedWindow(this.cam, 3), this.cam.f, scale);
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
