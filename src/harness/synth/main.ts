/**
 * Entry of synth.html. Exposes `window.generateSynth(config)`, which renders a
 * synthetic dataset frame by frame and streams every file to Node through
 * `window.__synthWrite` (installed with Playwright's page.exposeFunction by
 * e2e/generate-synth.mjs), so the browser never holds the whole dataset in memory.
 * Without a sink (manual use) files are only counted and the last frame is shown.
 */
import type { SynthConfigInput } from './config';
import { generateDataset, type SynthSink, type SynthSummary } from './generate';

type WriteFn = (path: string, data: string, encoding: 'base64' | 'utf8', append: boolean) => Promise<void>;

declare global {
  interface Window {
    generateSynth: (config: SynthConfigInput) => Promise<SynthSummary>;
    __synthWrite?: WriteFn;
    __synthLog?: (msg: string) => void;
    synthReady?: boolean;
  }
}

function blobToBase64(b: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => {
      const s = r.result as string;
      resolve(s.slice(s.indexOf(',') + 1));
    };
    r.onerror = () => reject(r.error);
    r.readAsDataURL(b);
  });
}

const statusEl = document.getElementById('status');

function makeSink(): SynthSink {
  const write = window.__synthWrite;
  return {
    async write(path, data, append = false) {
      if (!write) return;
      if (typeof data === 'string') return write(path, data, 'utf8', append);
      const blob = data instanceof Blob ? data : new Blob([data as unknown as BlobPart]);
      return write(path, await blobToBase64(blob), 'base64', append);
    },
    log(msg) {
      if (statusEl) statusEl.textContent = msg;
      if (window.__synthLog) window.__synthLog(msg);
      else console.log(msg);
    },
  };
}

window.generateSynth = (config: SynthConfigInput) => generateDataset(config, makeSink());
window.synthReady = true;
if (statusEl) statusEl.textContent = 'ready: call generateSynth({ scene: "object" | "yard", ... })';
