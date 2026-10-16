// Binary PLY writer for the dense cloud a harness result embeds as base64 (`densePoints`, used
// when the page could not stream dense.ply). Pure; unit-tested in src/harness/engine/e2e-lib.test.ts.

/** Binary little-endian PLY (float x y z + uchar red green blue) from base64 Float32 xyz / Uint8 rgb. */
export function plyFromBase64(count, xyzB64, rgbB64) {
  const xyz = Buffer.from(xyzB64, 'base64');
  const rgb = Buffer.from(rgbB64, 'base64');
  if (xyz.length < 12 * count || rgb.length < 3 * count) throw new Error(`densePoints: ${count} points need ${12 * count} + ${3 * count} bytes, got ${xyz.length} + ${rgb.length}`);
  const header = Buffer.from(
    `ply\nformat binary_little_endian 1.0\ncomment sfm e2e harness: dense (embedded)\nelement vertex ${count}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n`,
  );
  const body = Buffer.alloc(count * 15);
  for (let i = 0; i < count; i++) {
    xyz.copy(body, 15 * i, 12 * i, 12 * i + 12);
    rgb.copy(body, 15 * i + 12, 3 * i, 3 * i + 3);
  }
  return Buffer.concat([header, body]);
}
