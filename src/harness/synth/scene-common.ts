/**
 * Shared scene-building utilities: texture/material creation, baked (view-independent)
 * Lambertian shading in vertex colours, simple non-indexed geometry builder, and the
 * SynthScene bundle returned by the scene factories.
 *
 * All materials are UNLIT (MeshBasicMaterial; the ground/floor variant only injects
 * extra detail-map sampling), so pixel colours are a pure function of the surface
 * point: perfect multi-view photo-consistency.
 */
import {
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  ClampToEdgeWrapping,
  Color,
  DataTexture,
  DoubleSide,
  FrontSide,
  LinearFilter,
  LinearMipmapLinearFilter,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  NoColorSpace,
  Object3D,
  RepeatWrapping,
  RGBAFormat,
  SRGBColorSpace,
  Scene,
  Texture,
  UnsignedByteType,
  Vector3,
  type Material,
  type Side,
} from 'three';
import type { Canvas2D } from './textures';

export interface Box3Like {
  min: [number, number, number];
  max: [number, number, number];
}

export interface SynthScene {
  name: string;
  scene: Scene;
  /** Meshes contributing to ground-truth surface sampling (world matrices up to date). */
  surfaceMeshes: Mesh[];
  /** Objects hidden while rendering depth / visibility (sky dome). */
  depthHidden: Object3D[];
  near: number;
  far: number;
  /** Region of interest for dense evaluation (world, metres). */
  roi: Box3Like;
  /** Bounding box of the object of interest (object scene) or the yard. */
  focus: Box3Like;
  dispose(): void;
}

export interface MaterialOptions {
  /** Texture repeat (uv multiplier). */
  repeat?: [number, number];
  offset?: [number, number];
  side?: Side;
  color?: [number, number, number];
  /** Linear data texture (no sRGB decode), e.g. detail maps. */
  linear?: boolean;
}

/** Owns GPU-side resources of one synthetic scene. */
export class SceneBuilder {
  readonly scene = new Scene();
  readonly surfaceMeshes: Mesh[] = [];
  readonly depthHidden: Object3D[] = [];
  private readonly textures: Texture[] = [];
  private readonly geometries: BufferGeometry[] = [];
  private readonly materials: Material[] = [];

  constructor(
    readonly anisotropy: number,
    readonly light: Vector3,
    readonly ambient: number,
  ) {
    this.scene.matrixWorldAutoUpdate = true;
  }

  texture(canvas: Canvas2D, o: MaterialOptions = {}): Texture {
    const t = new CanvasTexture(canvas as unknown as HTMLCanvasElement);
    this.setupTexture(t, o);
    return t;
  }

  dataTexture(data: Uint8Array, w: number, h: number, o: MaterialOptions = {}): Texture {
    const t = new DataTexture(data, w, h, RGBAFormat, UnsignedByteType);
    t.flipY = false;
    this.setupTexture(t, o);
    t.needsUpdate = true;
    return t;
  }

  private setupTexture(t: Texture, o: MaterialOptions): void {
    t.colorSpace = o.linear ? NoColorSpace : SRGBColorSpace;
    t.anisotropy = this.anisotropy;
    t.generateMipmaps = true;
    t.minFilter = LinearMipmapLinearFilter;
    t.magFilter = LinearFilter;
    const rep = o.repeat && (o.repeat[0] !== 1 || o.repeat[1] !== 1);
    t.wrapS = t.wrapT = rep || o.offset ? RepeatWrapping : ClampToEdgeWrapping;
    if (o.repeat) t.repeat.set(o.repeat[0], o.repeat[1]);
    if (o.offset) t.offset.set(o.offset[0], o.offset[1]);
    this.textures.push(t);
  }

  material(map: Texture | null, o: MaterialOptions = {}): MeshBasicMaterial {
    const m = new MeshBasicMaterial({ map, vertexColors: true, side: o.side ?? FrontSide });
    if (o.color) m.color = new Color().setRGB(o.color[0], o.color[1], o.color[2], SRGBColorSpace);
    this.materials.push(m);
    return m;
  }

  /** Unlit material modulating a macro map by up to two tiled detail maps. */
  detailMaterial(
    map: Texture,
    detailA: { tex: Texture; scale: number; rot: number },
    detailB: { tex: Texture; scale: number; rot: number },
    opts: { maskFromAlpha: boolean; strength: number },
  ): MeshBasicMaterial {
    const m = new MeshBasicMaterial({ map, vertexColors: true });
    const rot = (a: number) => [Math.cos(a), Math.sin(a), -Math.sin(a), Math.cos(a)];
    m.onBeforeCompile = (shader) => {
      shader.uniforms.detailA = { value: detailA.tex };
      shader.uniforms.detailB = { value: detailB.tex };
      shader.uniforms.detailRotA = { value: rot(detailA.rot) };
      shader.uniforms.detailRotB = { value: rot(detailB.rot) };
      shader.uniforms.detailScale = { value: [detailA.scale, detailB.scale] };
      shader.fragmentShader = shader.fragmentShader
        .replace(
          '#include <map_pars_fragment>',
          `#include <map_pars_fragment>
uniform sampler2D detailA;
uniform sampler2D detailB;
uniform mat2 detailRotA;
uniform mat2 detailRotB;
uniform vec2 detailScale;`,
        )
        .replace(
          '#include <map_fragment>',
          `#ifdef USE_MAP
  vec4 macroC = texture2D( map, vMapUv );
  float dA = texture2D( detailA, detailRotA * ( vMapUv * detailScale.x ) ).r;
  float dB = texture2D( detailB, detailRotB * ( vMapUv * detailScale.y ) ).r;
  float maskG = ${opts.maskFromAlpha ? 'macroC.a' : '0.5'};
  float detail = ${opts.maskFromAlpha ? 'mix( dB, dA, maskG )' : '0.5 * ( dA + dB )'};
  diffuseColor.rgb *= macroC.rgb * ( 1.0 + ${opts.strength.toFixed(4)} * ( 2.0 * detail - 1.0 ) );
#endif`,
        );
    };
    m.customProgramCacheKey = () => `detail-${opts.maskFromAlpha}-${opts.strength}`;
    this.materials.push(m);
    return m;
  }

  /**
   * Add a mesh (shading baked from its world-space normals). `gt=false` excludes it
   * from ground-truth surface sampling; `depth=false` hides it in depth passes.
   */
  add(mesh: Mesh, o: { gt?: boolean; depth?: boolean; shade?: boolean; ao?: (z: number) => number; parent?: Object3D } = {}): Mesh {
    (o.parent ?? this.scene).add(mesh);
    mesh.updateMatrixWorld(true);
    if (o.shade !== false) bakeShading(mesh.geometry, mesh.matrixWorld, this.light, this.ambient, o.ao);
    if (!this.geometries.includes(mesh.geometry)) this.geometries.push(mesh.geometry);
    if (o.gt !== false) this.surfaceMeshes.push(mesh);
    if (o.depth === false) this.depthHidden.push(mesh);
    mesh.matrixAutoUpdate = false;
    return mesh;
  }

  trackGeometry(g: BufferGeometry): void {
    this.geometries.push(g);
  }

  dispose(): void {
    for (const t of this.textures) t.dispose();
    for (const g of this.geometries) g.dispose();
    for (const m of this.materials) m.dispose();
    this.textures.length = 0;
    this.geometries.length = 0;
    this.materials.length = 0;
  }
}

/**
 * Bake view-independent shading into a `color` attribute:
 * c = ambient + (1 − ambient)·max(0, n·L) (+ optional height-based occlusion).
 */
export function bakeShading(g: BufferGeometry, world: Matrix4, light: Vector3, ambient: number, ao?: (z: number) => number): void {
  if (!g.getAttribute('normal')) g.computeVertexNormals();
  const nAttr = g.getAttribute('normal');
  const pAttr = g.getAttribute('position');
  const n = pAttr.count;
  const colors = new Float32Array(3 * n);
  const nm = new Matrix4().copy(world).invert().transpose();
  const v = new Vector3(),
    p = new Vector3();
  for (let i = 0; i < n; i++) {
    v.set(nAttr.getX(i), nAttr.getY(i), nAttr.getZ(i)).applyMatrix4(nm).normalize();
    // n·L with a soft wrap so faces turned away are not flat black.
    const ndl = v.dot(light);
    let s = ambient + (1 - ambient) * Math.max(0, 0.15 + 0.85 * ndl);
    if (ao) {
      p.set(pAttr.getX(i), pAttr.getY(i), pAttr.getZ(i)).applyMatrix4(world);
      s *= ao(p.z);
    }
    colors[3 * i] = colors[3 * i + 1] = colors[3 * i + 2] = Math.min(1, s);
  }
  g.setAttribute('color', new BufferAttribute(colors, 3));
}

/** Non-indexed triangle soup builder (positions, normals, uvs). */
export class GeoBuilder {
  private pos: number[] = [];
  private nor: number[] = [];
  private uv: number[] = [];

  /** Triangle a, b, c (counter-clockwise seen from the front). */
  tri(a: number[], b: number[], c: number[], ua: number[], ub: number[], uc: number[]): void {
    const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const l = Math.hypot(n[0], n[1], n[2]) || 1;
    for (const [p, u] of [
      [a, ua],
      [b, ub],
      [c, uc],
    ]) {
      this.pos.push(p[0], p[1], p[2]);
      this.nor.push(n[0] / l, n[1] / l, n[2] / l);
      this.uv.push(u[0], u[1]);
    }
  }

  /** Quad a-b-c-d counter-clockwise from the front. */
  quad(a: number[], b: number[], c: number[], d: number[], ua: number[], ub: number[], uc: number[], ud: number[]): void {
    this.tri(a, b, c, ua, ub, uc);
    this.tri(a, c, d, ua, uc, ud);
  }

  build(): BufferGeometry {
    const g = new BufferGeometry();
    g.setAttribute('position', new BufferAttribute(new Float32Array(this.pos), 3));
    g.setAttribute('normal', new BufferAttribute(new Float32Array(this.nor), 3));
    g.setAttribute('uv', new BufferAttribute(new Float32Array(this.uv), 2));
    return g;
  }
}

/** Orient a plane mesh (local +Z normal, +Y up) to face `normal` with `up`. */
export function orientPlane(obj: Object3D, normal: [number, number, number], up: [number, number, number] = [0, 0, 1]): void {
  const z = new Vector3(...normal).normalize();
  const y = new Vector3(...up);
  y.addScaledVector(z, -y.dot(z)).normalize();
  const x = new Vector3().crossVectors(y, z);
  obj.quaternion.setFromRotationMatrix(new Matrix4().makeBasis(x, y, z));
}

/** Remap the UVs of a three BoxGeometry so each face uses its own cell of a cols×rows atlas. */
export function atlasBoxUVs(g: BufferGeometry, cols: number, rows: number, cells: number[] = [0, 1, 2, 3, 4, 5]): BufferGeometry {
  const uv = g.getAttribute('uv') as BufferAttribute;
  // BoxGeometry: 6 faces (+x, −x, +y, −y, +z, −z), 4 vertices each (no segments).
  for (let f = 0; f < 6; f++) {
    const cell = cells[f];
    const c = cell % cols,
      r = Math.floor(cell / cols);
    for (let k = 0; k < 4; k++) {
      const i = 4 * f + k;
      uv.setXY(i, (c + uv.getX(i)) / cols, 1 - (r + 1 - uv.getY(i)) / rows);
    }
  }
  uv.needsUpdate = true;
  return g;
}

/** Linear-light vertex colours for a sky dome gradient (sRGB inputs). */
export function skyColors(g: BufferGeometry, horizon: [number, number, number], zenith: [number, number, number], radius: number): void {
  const p = g.getAttribute('position');
  const c = new Float32Array(3 * p.count);
  const lin = (v: number) => Math.pow(v / 255, 2.2);
  for (let i = 0; i < p.count; i++) {
    const t = Math.max(0, Math.min(1, p.getZ(i) / radius));
    const k = Math.pow(t, 0.6);
    for (let j = 0; j < 3; j++) c[3 * i + j] = lin(horizon[j] + (zenith[j] - horizon[j]) * k);
  }
  g.setAttribute('color', new BufferAttribute(c, 3));
}

export { DoubleSide, FrontSide };
