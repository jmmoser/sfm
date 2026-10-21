/**
 * Feature probe worker (src/sensors/diagnostics.ts): reports which of the requested global names
 * exist in a dedicated worker. Some APIs are worker-only in Safari (MediaStreamTrackProcessor).
 */
const scope = self as unknown as {
  postMessage(msg: Record<string, boolean>): void;
  onmessage: ((ev: MessageEvent<string[]>) => void) | null;
};

scope.onmessage = (ev) => {
  const out: Record<string, boolean> = {};
  for (const k of ev.data) out[k] = k in globalThis;
  scope.postMessage(out);
};
