/**
 * PLACEHOLDER engine worker (module worker). The real implementation (PhotogrammetryEngine
 * behind the protocol of ./protocol.ts) replaces this file. Until then every request is
 * answered with a NotImplementedError "engine not implemented yet", so the UI can show a clear
 * message (use `?mock=1` for the in-page MockEngine).
 */
import { ENGINE_ERRORS, isEngineRequest, type EngineFailure, type WorkerToMainMessage } from './protocol';

const scope = self as unknown as {
  onmessage: ((ev: MessageEvent<unknown>) => void) | null;
  postMessage(msg: WorkerToMainMessage, transfer?: Transferable[]): void;
};

scope.onmessage = (ev: MessageEvent<unknown>): void => {
  const req = ev.data;
  if (!isEngineRequest(req)) return;
  // Release transferred frames at once (decoded bitmaps count against the memory budget).
  if (req.method === 'frame') (req.params as { bitmap?: { close?: () => void } } | undefined)?.bitmap?.close?.();
  const res: EngineFailure = {
    id: req.id,
    ok: false,
    error: { name: ENGINE_ERRORS.notImplemented, message: 'engine not implemented yet' },
  };
  scope.postMessage(res);
};
