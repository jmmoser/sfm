import { defineConfig } from 'vite';
import { resolve } from 'node:path';

// Static, relative-path build so the app can be hosted on any HTTPS static host
// (GitHub Pages, Netlify, S3...). Camera + motion sensors require a secure context.
export default defineConfig({
  base: './',
  build: {
    target: 'es2022',
    // No source maps in the deployed app (they would publish the full source next to it).
    // For on-device debugging build with `npx vite build --sourcemap`.
    sourcemap: false,
    rollupOptions: {
      // Only the app is deployed. The test pages (harness.html, synth.html, dense-gt.html, nvs.html) are
      // served by the Vite dev server for the e2e scripts and are NOT part of the production build:
      // harness.html exposes `window.runPipeline` (fetches `/e2e/data/<?dataset=…>/` paths, creates
      // sessions in the app's IndexedDB, runs the GPU engine on load).
      input: {
        main: resolve(__dirname, 'index.html'),
      },
    },
  },
  worker: {
    format: 'es',
    // One file per worker. With code splitting, the chunks a worker loads lazily (the feature
    // extractor, GPU kernels) import shared code back from the worker's entry chunk. WebKit does
    // not reuse the running worker script for that import: it evaluates a second copy, whose
    // `onmessage` (fresh state, no engine) then receives every later request ("Engine not
    // initialised" on the first frame of the first scan).
    rolldownOptions: {
      output: { codeSplitting: false },
    },
  },
  server: {
    host: true,
  },
});
