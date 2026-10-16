import { defineConfig } from 'vite';
import { resolve } from 'node:path';

// Static, relative-path build so the app can be hosted on any HTTPS static host
// (GitHub Pages, Netlify, S3...). Camera + motion sensors require a secure context.
export default defineConfig({
  base: './',
  build: {
    target: 'es2022',
    sourcemap: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        harness: resolve(__dirname, 'harness.html'),
      },
    },
  },
  worker: {
    format: 'es',
  },
  server: {
    host: true,
  },
});
