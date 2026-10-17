// Point-cloud readers: PLY (ascii, binary_little_endian, binary_big_endian; vertex
// x/y/z of any scalar type, other properties/elements skipped) and raw Float32 xyz.

const TYPES = {
  char: ['getInt8', 1], int8: ['getInt8', 1],
  uchar: ['getUint8', 1], uint8: ['getUint8', 1],
  short: ['getInt16', 2], int16: ['getInt16', 2],
  ushort: ['getUint16', 2], uint16: ['getUint16', 2],
  int: ['getInt32', 4], int32: ['getInt32', 4],
  uint: ['getUint32', 4], uint32: ['getUint32', 4],
  float: ['getFloat32', 4], float32: ['getFloat32', 4],
  double: ['getFloat64', 8], float64: ['getFloat64', 8],
};

/** Parse a PLY buffer (Uint8Array) and return vertex positions as Float32Array xyz. */
export function parsePLYPoints(u8) {
  const headEnd = findHeaderEnd(u8);
  if (headEnd < 0) throw new Error('PLY: no end_header');
  const header = new TextDecoder().decode(u8.subarray(0, headEnd));
  const lines = header.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  if (lines[0] !== 'ply') throw new Error('not a PLY file');
  let format = null;
  const elements = [];
  for (const l of lines.slice(1)) {
    const p = l.split(/\s+/);
    if (p[0] === 'format') format = p[1];
    else if (p[0] === 'element') elements.push({ name: p[1], count: Number(p[2]), props: [] });
    else if (p[0] === 'property') {
      const el = elements[elements.length - 1];
      if (p[1] === 'list') el.props.push({ list: true, countType: p[2], type: p[3], name: p[4] });
      else el.props.push({ list: false, type: p[1], name: p[2] });
    }
  }
  let bodyStart = headEnd;
  // skip the newline after end_header
  while (bodyStart < u8.length && u8[bodyStart] !== 0x0a) bodyStart++;
  bodyStart++;
  const vi = elements.findIndex((e) => e.name === 'vertex');
  if (vi < 0) throw new Error('PLY: no vertex element');
  const vtx = elements[vi];
  const n = vtx.count;
  const out = new Float32Array(3 * n);
  const ix = vtx.props.findIndex((p) => p.name === 'x'),
    iy = vtx.props.findIndex((p) => p.name === 'y'),
    iz = vtx.props.findIndex((p) => p.name === 'z');
  if (ix < 0 || iy < 0 || iz < 0) throw new Error('PLY: vertex has no x/y/z');
  if (format === 'ascii') {
    const text = new TextDecoder().decode(u8.subarray(bodyStart));
    const rows = text.split(/\r?\n/);
    let r = 0;
    for (let e = 0; e < vi; e++) r += elements[e].count;
    for (let i = 0; i < n; i++) {
      const f = rows[r + i].trim().split(/\s+/).map(Number);
      out[3 * i] = f[ix];
      out[3 * i + 1] = f[iy];
      out[3 * i + 2] = f[iz];
    }
    return out;
  }
  const le = format === 'binary_little_endian';
  if (!le && format !== 'binary_big_endian') throw new Error(`PLY: unsupported format ${format}`);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  let off = bodyStart;
  // skip preceding elements (only fixed-size properties supported there)
  for (let e = 0; e < vi; e++) {
    for (let k = 0; k < elements[e].count; k++)
      for (const p of elements[e].props) {
        if (p.list) {
          const [g, s] = TYPES[p.countType];
          const c = dv[g](off, le);
          off += s + c * TYPES[p.type][1];
        } else off += TYPES[p.type][1];
      }
  }
  const fixed = vtx.props.every((p) => !p.list);
  if (fixed) {
    const offs = [];
    let stride = 0;
    for (const p of vtx.props) {
      offs.push(stride);
      stride += TYPES[p.type][1];
    }
    const gx = TYPES[vtx.props[ix].type][0],
      gy = TYPES[vtx.props[iy].type][0],
      gz = TYPES[vtx.props[iz].type][0];
    for (let i = 0; i < n; i++) {
      const b = off + i * stride;
      out[3 * i] = dv[gx](b + offs[ix], le);
      out[3 * i + 1] = dv[gy](b + offs[iy], le);
      out[3 * i + 2] = dv[gz](b + offs[iz], le);
    }
    return out;
  }
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < vtx.props.length; k++) {
      const p = vtx.props[k];
      if (p.list) {
        const [g, s] = TYPES[p.countType];
        const c = dv[g](off, le);
        off += s + c * TYPES[p.type][1];
        continue;
      }
      const [g, s] = TYPES[p.type];
      const v = dv[g](off, le);
      if (k === ix) out[3 * i] = v;
      else if (k === iy) out[3 * i + 1] = v;
      else if (k === iz) out[3 * i + 2] = v;
      off += s;
    }
  }
  return out;
}

function findHeaderEnd(u8) {
  const pat = 'end_header';
  const lim = Math.min(u8.length, 1 << 16);
  outer: for (let i = 0; i + pat.length <= lim; i++) {
    for (let k = 0; k < pat.length; k++) if (u8[i + k] !== pat.charCodeAt(k)) continue outer;
    return i + pat.length;
  }
  return -1;
}

/** Points from a .ply or raw Float32 xyz buffer (decided by the PLY magic). */
export function parsePoints(u8) {
  if (u8.length >= 3 && u8[0] === 0x70 && u8[1] === 0x6c && u8[2] === 0x79) return parsePLYPoints(u8);
  if (u8.byteLength % 12 !== 0) throw new Error('raw point file size is not a multiple of 12 bytes (Float32 xyz)');
  const copy = new Uint8Array(u8); // aligned copy
  return new Float32Array(copy.buffer, 0, copy.byteLength / 4);
}
