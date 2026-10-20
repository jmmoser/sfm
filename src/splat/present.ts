/**
 * Presents a splat frame — the f32 colour buffer of `SplatRenderer.encodeFrame` (colour with
 * T·background added, transmittance T in w; rows top to bottom) — in a render pass: a full-screen
 * triangle whose fragment shader reads the buffer, bilinear when the frame is smaller than the
 * target (the viewer renders at a lower resolution while the view moves). Resampling follows
 * ARCHITECTURE §2: source x = (target x + 0.5)·s − 0.5, so equal sizes copy pixels exactly.
 *
 * Colours are written as they are: sRGB values into a non-sRGB target such as the canvas's
 * preferred format (the model is fitted to sRGB images, src/splat/types.ts). `transparent`:
 * premultiplied output with alpha 1 − T (render the frame over a black background), for a canvas
 * configured with `alphaMode: 'premultiplied'` over page content; otherwise alpha 1.
 */

const PRESENT_WGSL = /* wgsl */ `
struct Params {
  src: vec2u,
  dst: vec2u,
  transparent: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> img: array<vec4f>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  // full-screen triangle (-1,-1), (3,-1), (-1,3)
  let x = f32((i << 1u) & 2u);
  let y = f32(i & 2u);
  return vec4f(2.0 * x - 1.0, 2.0 * y - 1.0, 0.0, 1.0);
}

fn texel(x: i32, y: i32) -> vec4f {
  let cx = clamp(x, 0, i32(P.src.x) - 1);
  let cy = clamp(y, 0, i32(P.src.y) - 1);
  return img[u32(cy) * P.src.x + u32(cx)];
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  // pos.xy = target pixel centre (x + 0.5, y + 0.5), +y down like the frame rows
  let q = pos.xy * (vec2f(P.src) / vec2f(P.dst)) - 0.5;
  let q0 = floor(q);
  let f = q - q0;
  let x0 = i32(q0.x);
  let y0 = i32(q0.y);
  let c = mix(mix(texel(x0, y0), texel(x0 + 1, y0), f.x), mix(texel(x0, y0 + 1), texel(x0 + 1, y0 + 1), f.x), f.y);
  if (P.transparent != 0u) {
    let a = clamp(1.0 - c.w, 0.0, 1.0);
    return vec4f(clamp(c.rgb, vec3f(0.0), vec3f(a)), a);
  }
  return vec4f(clamp(c.rgb, vec3f(0.0), vec3f(1.0)), 1.0);
}
`;

export interface PresentSource {
  /** `SplatRenderer.encodeFrame`'s colour buffer. */
  buffer: GPUBuffer;
  width: number;
  height: number;
}

export class SplatPresenter {
  private readonly uniform: GPUBuffer;
  private bindGroup: GPUBindGroup | null = null;
  private boundBuffer: GPUBuffer | null = null;

  private constructor(
    readonly device: GPUDevice,
    private readonly pipeline: GPURenderPipeline,
    readonly format: GPUTextureFormat,
  ) {
    this.uniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'splat-present-params' });
  }

  /** Compiles the pipeline (at stage start: compilation blocks the queue on Safari). */
  static async create(device: GPUDevice, format: GPUTextureFormat): Promise<SplatPresenter> {
    const module = device.createShaderModule({ code: PRESENT_WGSL, label: 'splat-present' });
    const pipeline = await device.createRenderPipelineAsync({
      label: 'splat-present',
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });
    return new SplatPresenter(device, pipeline, format);
  }

  /** Records the pass drawing `src` over the whole of `target` (`width` × `height` pixels). */
  encode(enc: GPUCommandEncoder, target: GPUTextureView, width: number, height: number, src: PresentSource, transparent = false): void {
    if (this.boundBuffer !== src.buffer || !this.bindGroup) {
      this.bindGroup = this.device.createBindGroup({
        label: 'splat-present',
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: { buffer: src.buffer } },
        ],
      });
      this.boundBuffer = src.buffer;
    }
    this.device.queue.writeBuffer(this.uniform, 0, new Uint32Array([src.width, src.height, width, height, transparent ? 1 : 0, 0, 0, 0]));
    const pass = enc.beginRenderPass({
      label: 'splat-present',
      colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: transparent ? 0 : 1 } }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
    pass.end();
  }

  destroy(): void {
    this.uniform.destroy();
    this.bindGroup = null;
    this.boundBuffer = null;
  }
}
