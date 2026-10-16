/**
 * Eye-Dome Lighting (Potree-style) post-process + depth readback for picking.
 *
 * Design:
 *  - The "content" layer (points, mesh) is rendered into an offscreen target with
 *    an sRGB8_ALPHA8 colour texture and a DEPTH TEXTURE (DEPTH_COMPONENT32F,
 *    falling back to DEPTH_COMPONENT24). No float colour target is needed, so it
 *    works on every WebGL2 implementation; if even that framebuffer is
 *    incomplete, `supported` is false and the viewer renders directly.
 *  - Composite pass (fullscreen triangle → canvas): linearises the hardware depth
 *    with the frame's near/far, takes log2, and computes the obscurance from 8
 *    neighbours at `radius` px: shade = exp(−300·strength·mean(max(0, l₀ − lᵢ))).
 *    Only fragments whose colour alpha is 1 (points) are shaded; the mesh writes
 *    alpha 0 into the target (it has its own lighting). The pass
 *    writes gl_FragDepth = sampled depth so overlays (cameras, grid, measure
 *    markers) drawn afterwards depth-test against the content. Background
 *    fragments are discarded (the canvas clear colour shows through).
 *  - Picking: a tiny pass texelFetches the depth texture in a window around the
 *    tap and packs each depth into RGBA8 (three's packDepthToRGBA), which is read
 *    back synchronously (≤ 64×64 px, ~1 ms). Cost is independent of the number of
 *    points and it is occlusion-correct for points and meshes. Because Safari's
 *    fingerprinting protection may add noise to readbacks (Private Browsing), a
 *    known-pattern self-test runs once; if it fails the viewer uses its CPU picker.
 */
import {
  BufferAttribute,
  BufferGeometry,
  DepthTexture,
  FloatType,
  Mesh,
  NearestFilter,
  NoBlending,
  OrthographicCamera,
  RGBAFormat,
  SRGBColorSpace,
  Scene,
  ShaderMaterial,
  UnsignedByteType,
  UnsignedIntType,
  Vector2,
  WebGLRenderTarget,
  AlwaysDepth,
  type Camera,
  type Object3D,
  type WebGLRenderer,
} from 'three';
import { unpackRGBAToDepth } from './helpers';

const fullscreenVertex = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

const compositeFragment = /* glsl */ `
uniform sampler2D tColor;
uniform sampler2D tDepth;
uniform vec2 uTexel;
uniform float uNear;
uniform float uFar;
uniform float uStrength;
uniform float uRadius;
varying vec2 vUv;

float viewLog2(float d) {
  float z = d * 2.0 - 1.0;
  return log2(2.0 * uNear * uFar / (uFar + uNear - z * (uFar - uNear)));
}

void main() {
  float d0 = texture2D(tDepth, vUv).r;
  if (d0 >= 1.0) discard;
  vec4 c0 = texture2D(tColor, vUv);
  vec3 col = c0.rgb;
#ifdef EDL
  const vec2 N[8] = vec2[8](
    vec2(1.0, 0.0), vec2(0.70710678, 0.70710678), vec2(0.0, 1.0), vec2(-0.70710678, 0.70710678),
    vec2(-1.0, 0.0), vec2(-0.70710678, -0.70710678), vec2(0.0, -1.0), vec2(0.70710678, -0.70710678));
  float l0 = viewLog2(d0);
  float sum = 0.0;
  for (int i = 0; i < 8; i++) {
    float dn = texture2D(tDepth, vUv + N[i] * uRadius * uTexel).r;
    if (dn < 1.0) sum += max(0.0, l0 - viewLog2(dn));
  }
  // Potree: exp(-response * 300 * strength) with response = sum / 8.
  // Alpha marks EDL-eligible fragments (points = 1, lit mesh = 0).
  col *= mix(1.0, exp(-sum * 37.5 * uStrength), c0.a);
#endif
  gl_FragColor = vec4(col, 1.0);
  gl_FragDepth = d0;
  #include <colorspace_fragment>
}
`;

const pickFragment = /* glsl */ `
#include <packing>
uniform sampler2D tDepth;
uniform vec2 uOrigin;
uniform vec2 uSize;
void main() {
  vec2 pix = uOrigin + floor(gl_FragCoord.xy);
#ifdef SELF_TEST
  vec2 q = floor(gl_FragCoord.xy);
  float d = (q.x + q.y * 16.0 + 0.5) / 256.0;
#else
  float d = 1.0;
  if (pix.x >= 0.0 && pix.y >= 0.0 && pix.x < uSize.x && pix.y < uSize.y)
    d = texelFetch(tDepth, ivec2(pix), 0).r;
#endif
  gl_FragColor = packDepthToRGBA(d);
}
`;

/** Max side of the pick window in drawing-buffer pixels. */
export const PICK_WINDOW_MAX = 63;

export interface DepthWindow {
  /** Window depths (row-major, row 0 = BOTTOM row in GL convention), 1 = background. */
  depths: Float32Array;
  /** Side length of the window. */
  size: number;
  /** Drawing-buffer pixel (GL convention, origin bottom-left) of depths[0]. */
  x0: number;
  y0: number;
}

export interface EDLSettings {
  enabled: boolean;
  strength: number;
  /** Neighbour radius in drawing-buffer pixels. */
  radiusPx: number;
}

function fullscreenTriangle(): BufferGeometry {
  const g = new BufferGeometry();
  g.setAttribute('position', new BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
  return g;
}

export class EDLPass {
  /** Offscreen colour + depth targets can be created on this context. */
  readonly supported: boolean;
  readonly depthTextureType: 'float' | 'uint' | null;

  private target: WebGLRenderTarget | null = null;
  private width = 1;
  private height = 1;
  private readonly quadScene = new Scene();
  private readonly quadCamera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly quad: Mesh;
  private readonly geometry = fullscreenTriangle();
  private readonly compositeMaterial: ShaderMaterial;
  private readonly pickMaterial: ShaderMaterial;
  private pickTarget: WebGLRenderTarget | null = null;
  private readonly pickBytes = new Uint8Array(4 * (PICK_WINDOW_MAX + 1) * (PICK_WINDOW_MAX + 1));
  private readbackOk: boolean | null = null;
  private edlDefine = true;

  constructor(renderer: WebGLRenderer) {
    this.depthTextureType = EDLPass.probe(renderer);
    this.supported = this.depthTextureType !== null;
    this.compositeMaterial = new ShaderMaterial({
      vertexShader: fullscreenVertex,
      fragmentShader: compositeFragment,
      defines: { EDL: '' },
      uniforms: {
        tColor: { value: null },
        tDepth: { value: null },
        uTexel: { value: new Vector2(1, 1) },
        uNear: { value: 0.1 },
        uFar: { value: 100 },
        uStrength: { value: 1 },
        uRadius: { value: 1.4 },
      },
      depthTest: true,
      depthWrite: true,
      depthFunc: AlwaysDepth,
      blending: NoBlending,
    });
    this.pickMaterial = new ShaderMaterial({
      vertexShader: fullscreenVertex,
      fragmentShader: pickFragment,
      uniforms: {
        tDepth: { value: null },
        uOrigin: { value: new Vector2() },
        uSize: { value: new Vector2(1, 1) },
      },
      depthTest: false,
      depthWrite: false,
      blending: NoBlending,
    });
    this.quad = new Mesh(this.geometry, this.compositeMaterial);
    this.quad.frustumCulled = false;
    this.quadScene.add(this.quad);
  }

  /** Try DEPTH_COMPONENT32F then DEPTH_COMPONENT24 depth textures; null if neither works. */
  private static probe(renderer: WebGLRenderer): 'float' | 'uint' | null {
    const gl = renderer.getContext();
    for (const type of [FloatType, UnsignedIntType] as const) {
      const rt = EDLPass.createTarget(4, 4, type);
      let ok = false;
      try {
        renderer.setRenderTarget(rt);
        ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
      } catch {
        ok = false;
      } finally {
        renderer.setRenderTarget(null);
        rt.depthTexture?.dispose();
        rt.dispose();
      }
      if (ok) return type === FloatType ? 'float' : 'uint';
    }
    return null;
  }

  private static createTarget(w: number, h: number, depthType: typeof FloatType | typeof UnsignedIntType) {
    const depthTexture = new DepthTexture(w, h, depthType);
    depthTexture.minFilter = NearestFilter;
    depthTexture.magFilter = NearestFilter;
    return new WebGLRenderTarget(w, h, {
      type: UnsignedByteType,
      format: RGBAFormat,
      colorSpace: SRGBColorSpace,
      minFilter: NearestFilter,
      magFilter: NearestFilter,
      generateMipmaps: false,
      depthBuffer: true,
      depthTexture,
    });
  }

  /** Set the drawing-buffer size (allocation is lazy). */
  setSize(width: number, height: number): void {
    this.width = Math.max(1, Math.floor(width));
    this.height = Math.max(1, Math.floor(height));
    if (this.target && (this.target.width !== this.width || this.target.height !== this.height)) {
      this.target.setSize(this.width, this.height);
    }
  }

  private ensureTarget(): WebGLRenderTarget {
    if (!this.target) {
      this.target = EDLPass.createTarget(
        this.width,
        this.height,
        this.depthTextureType === 'uint' ? UnsignedIntType : FloatType,
      );
    }
    return this.target;
  }

  /** True once renderContent() has produced a depth texture for the current size. */
  get hasTarget(): boolean {
    return this.target !== null;
  }

  /**
   * Render `scene` into the offscreen target (cleared to transparent black, depth 1).
   * Leaves the renderer's render target set to null afterwards.
   */
  renderContent(renderer: WebGLRenderer, scene: Object3D, camera: Camera): void {
    const rt = this.ensureTarget();
    renderer.setRenderTarget(rt);
    renderer.setClearColor(0x000000, 0);
    renderer.clear(true, true, false);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
  }

  /**
   * Composite the offscreen content to the current render target (the canvas)
   * with optional EDL shading. Writes depth. Call renderContent() first.
   */
  composite(renderer: WebGLRenderer, near: number, far: number, s: EDLSettings): void {
    const rt = this.target;
    if (!rt) return;
    const m = this.compositeMaterial;
    if (s.enabled !== this.edlDefine) {
      this.edlDefine = s.enabled;
      m.defines = s.enabled ? { EDL: '' } : {};
      m.needsUpdate = true;
    }
    const u = m.uniforms;
    u.tColor.value = rt.texture;
    u.tDepth.value = rt.depthTexture;
    (u.uTexel.value as Vector2).set(1 / rt.width, 1 / rt.height);
    u.uNear.value = near;
    u.uFar.value = far;
    u.uStrength.value = s.strength;
    u.uRadius.value = s.radiusPx;
    this.quad.material = m;
    renderer.render(this.quadScene, this.quadCamera);
  }

  private ensurePickTarget(): WebGLRenderTarget {
    if (!this.pickTarget) {
      this.pickTarget = new WebGLRenderTarget(PICK_WINDOW_MAX + 1, PICK_WINDOW_MAX + 1, {
        type: UnsignedByteType,
        format: RGBAFormat,
        minFilter: NearestFilter,
        magFilter: NearestFilter,
        generateMipmaps: false,
        depthBuffer: false,
      });
    }
    return this.pickTarget;
  }

  private runPickPass(renderer: WebGLRenderer, size: number): Uint8Array {
    const prt = this.ensurePickTarget();
    const prev = renderer.getRenderTarget();
    this.quad.material = this.pickMaterial;
    renderer.setRenderTarget(prt);
    renderer.render(this.quadScene, this.quadCamera);
    const out = this.pickBytes.subarray(0, 4 * size * size);
    renderer.readRenderTargetPixels(prt, 0, 0, size, size, out);
    renderer.setRenderTarget(prev);
    this.quad.material = this.compositeMaterial;
    return out;
  }

  /**
   * One-time known-pattern readback test (detects readback noise injection /
   * broken readPixels). Cached.
   */
  readbackReliable(renderer: WebGLRenderer): boolean {
    if (this.readbackOk !== null) return this.readbackOk;
    try {
      this.pickMaterial.defines = { SELF_TEST: '' };
      this.pickMaterial.needsUpdate = true;
      const bytes = this.runPickPass(renderer, 16);
      let ok = true;
      for (let y = 0; y < 16 && ok; y++) {
        for (let x = 0; x < 16; x++) {
          const i = 4 * (y * 16 + x);
          const d = unpackRGBAToDepth(bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]);
          const expected = (x + y * 16 + 0.5) / 256;
          if (!(Math.abs(d - expected) < 2e-6)) {
            ok = false;
            break;
          }
        }
      }
      this.readbackOk = ok;
    } catch {
      this.readbackOk = false;
    } finally {
      this.pickMaterial.defines = {};
      this.pickMaterial.needsUpdate = true;
    }
    return this.readbackOk ?? false;
  }

  /**
   * Read the content depth (window depth in [0,1], 1 = background) in a square
   * window of side 2r+1 centred on drawing-buffer pixel (cx, cy), GL convention
   * (origin bottom-left). renderContent() must have been called for the current
   * camera. Returns null when no target exists.
   */
  readDepthWindow(renderer: WebGLRenderer, cx: number, cy: number, r: number): DepthWindow | null {
    const rt = this.target;
    if (!rt || !rt.depthTexture) return null;
    const rr = Math.max(0, Math.min(Math.floor(PICK_WINDOW_MAX / 2), Math.floor(r)));
    const size = 2 * rr + 1;
    const x0 = Math.floor(cx) - rr;
    const y0 = Math.floor(cy) - rr;
    const u = this.pickMaterial.uniforms;
    u.tDepth.value = rt.depthTexture;
    (u.uOrigin.value as Vector2).set(x0, y0);
    (u.uSize.value as Vector2).set(rt.width, rt.height);
    const bytes = this.runPickPass(renderer, size);
    const depths = new Float32Array(size * size);
    for (let i = 0; i < size * size; i++) {
      depths[i] = unpackRGBAToDepth(bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]);
    }
    return { depths, size, x0, y0 };
  }

  /** Forget cached GPU-side state after a context restore (targets are re-created lazily by three). */
  onContextRestored(): void {
    this.readbackOk = null;
  }

  dispose(): void {
    this.target?.depthTexture?.dispose();
    this.target?.dispose();
    this.target = null;
    this.pickTarget?.dispose();
    this.pickTarget = null;
    this.geometry.dispose();
    this.compositeMaterial.dispose();
    this.pickMaterial.dispose();
  }
}
