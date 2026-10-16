/**
 * Point-cloud material: per-point sRGB colour (Uint8 normalised attribute),
 * optional per-point world-space radius (screen size = 2·radius·projScale/depth,
 * clamped), round splats (fragments outside the unit disc are discarded) with an
 * optional soft "sphere" shading, optional confidence threshold.
 *
 * Output is linear and goes through three's colour-space conversion, so it
 * composes correctly both when drawn to the canvas and into the sRGB EDL target.
 */
import { ShaderMaterial, type Material } from 'three';

export interface PointsMaterialOptions {
  /** Size in drawing-buffer pixels when no radius attribute is used. */
  sizePx?: number;
  /** Use the per-point `radius` attribute (world units) for the size. */
  useRadius?: boolean;
  /** Round splats (discard outside the disc). Default true. */
  round?: boolean;
  /** Fake sphere shading of round splats. Default false. */
  soft?: boolean;
  /** Use the per-point `confidence` attribute and discard below minConfidence. */
  useConfidence?: boolean;
  minConfidence?: number;
  /** Clamp range of the computed size in drawing-buffer pixels. */
  minSizePx?: number;
  maxSizePx?: number;
}

const vertexShader = /* glsl */ `
uniform float uSize;
uniform float uProjScale;
uniform float uMinSize;
uniform float uMaxSize;
uniform float uSizeScale;
#ifdef USE_RADIUS
attribute float radius;
#endif
#ifdef USE_CONFIDENCE
attribute float confidence;
uniform float uMinConfidence;
#endif
varying vec3 vCol;

vec3 srgbToLinear3(vec3 c) {
  return mix(c * 0.0773993808, pow(c * 0.9478672986 + 0.0521327014, vec3(2.4)), step(vec3(0.04045), c));
}

void main() {
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPosition;
#ifdef USE_RADIUS
  float size = 2.0 * radius * uProjScale / max(-mvPosition.z, 1e-9);
#else
  float size = uSize;
#endif
  gl_PointSize = clamp(size * uSizeScale, uMinSize, uMaxSize);
#ifdef USE_COLOR
  vCol = srgbToLinear3(color);
#else
  vCol = vec3(0.8);
#endif
#ifdef USE_CONFIDENCE
  if (confidence < uMinConfidence) gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // clipped
#endif
}
`;

const fragmentShader = /* glsl */ `
varying vec3 vCol;
void main() {
  vec3 col = vCol;
#ifdef ROUND
  vec2 c = 2.0 * gl_PointCoord - 1.0;
  float r2 = dot(c, c);
  if (r2 > 1.0) discard;
#ifdef SOFT
  col *= 0.55 + 0.45 * sqrt(1.0 - r2);
#endif
#endif
  gl_FragColor = vec4(col, 1.0);
  #include <colorspace_fragment>
}
`;

/** ShaderMaterial specialised for point clouds; see module doc. */
export class PointCloudMaterial extends ShaderMaterial {
  readonly isPointCloudMaterial = true;

  constructor(opts: PointsMaterialOptions = {}) {
    super({
      vertexShader,
      fragmentShader,
      vertexColors: true,
      uniforms: {
        uSize: { value: opts.sizePx ?? 2 },
        uProjScale: { value: 1000 },
        uMinSize: { value: opts.minSizePx ?? 1 },
        uMaxSize: { value: opts.maxSizePx ?? 64 },
        uSizeScale: { value: 1 },
        uMinConfidence: { value: opts.minConfidence ?? 0 },
      },
    });
    this.setOptions(opts);
  }

  private readonly flags = { round: true, soft: false, useRadius: false, useConfidence: false };

  /** True when sizes come from the per-point radius attribute. */
  get usesRadius(): boolean {
    return this.flags.useRadius;
  }

  /** Update defines / uniforms (unspecified options keep their value). Recompiles only when a define changes. */
  setOptions(opts: PointsMaterialOptions): void {
    const f = this.flags;
    if (opts.round !== undefined) f.round = opts.round;
    if (opts.soft !== undefined) f.soft = opts.soft;
    if (opts.useRadius !== undefined) f.useRadius = opts.useRadius;
    if (opts.useConfidence !== undefined) f.useConfidence = opts.useConfidence;
    const defines: Record<string, string> = {};
    if (f.round) defines.ROUND = '';
    if (f.round && f.soft) defines.SOFT = '';
    if (f.useRadius) defines.USE_RADIUS = '';
    if (f.useConfidence) defines.USE_CONFIDENCE = '';
    const prev = this.defines ?? {};
    const changed =
      Object.keys(prev).length !== Object.keys(defines).length || Object.keys(defines).some((k) => !(k in prev));
    if (changed) {
      this.defines = defines;
      this.needsUpdate = true;
    }
    if (opts.sizePx !== undefined) this.uniforms.uSize.value = opts.sizePx;
    if (opts.minSizePx !== undefined) this.uniforms.uMinSize.value = opts.minSizePx;
    if (opts.maxSizePx !== undefined) this.uniforms.uMaxSize.value = opts.maxSizePx;
    if (opts.minConfidence !== undefined) this.uniforms.uMinConfidence.value = opts.minConfidence;
  }

  /**
   * Per-frame camera-dependent uniforms. `projScale` = drawing-buffer height /
   * (2 tan(fovY/2)); `pixelRatio` scales the clamp range given in CSS px.
   */
  updateView(projScale: number, sizeScale: number): void {
    this.uniforms.uProjScale.value = projScale;
    this.uniforms.uSizeScale.value = sizeScale;
  }
}

/**
 * Patch a built-in mesh material (MeshStandardMaterial / MeshBasicMaterial with
 * vertexColors) for the viewer:
 *  - converts the sRGB vertex colour attribute to linear in the vertex shader
 *    (three treats vertex colours as linear);
 *  - multiplies the output alpha by the shared `edlWeight` uniform. The viewer
 *    sets it to 0 while rendering into the EDL target so that the composite pass
 *    applies Eye-Dome Lighting to points only (alpha = 1) and leaves the lit
 *    mesh untouched (alpha = 0); it is 1 for direct rendering.
 */
export function patchMeshMaterial(material: Material, edlWeight: { value: number }): void {
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uEdlWeight = edlWeight;
    shader.vertexShader = shader.vertexShader
      .replace(
        'void main() {',
        `vec3 viewerSrgbToLinear(vec3 c) {
  return mix(c * 0.0773993808, pow(c * 0.9478672986 + 0.0521327014, vec3(2.4)), step(vec3(0.04045), c));
}
void main() {`,
      )
      .replace(
        '#include <color_vertex>',
        `#include <color_vertex>
#if defined( USE_COLOR ) && !defined( USE_COLOR_ALPHA )
  vColor.rgb = viewerSrgbToLinear(vColor.rgb);
#endif`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace('void main() {', 'uniform float uEdlWeight;\nvoid main() {')
      .replace('#include <dithering_fragment>', '#include <dithering_fragment>\n  gl_FragColor.a *= uEdlWeight;');
  };
  material.customProgramCacheKey = () => 'viewer-mesh-v1';
}
