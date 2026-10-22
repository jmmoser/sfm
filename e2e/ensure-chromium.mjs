#!/usr/bin/env node
/**
 * CI helper: make sure Playwright's Chromium is installed before the GPU tests or an e2e tool run.
 *
 *   node e2e/ensure-chromium.mjs
 *
 * vitest.config.ts (gpu project) and every e2e tool launch Playwright's own Chromium without a
 * pinned path, so Playwright looks in PLAYWRIGHT_BROWSERS_PATH (images that ship the browsers,
 * e.g. /opt/pw-browsers with PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1) or in its default cache
 * (~/.cache/ms-playwright on GitHub-hosted runners, empty on a fresh runner). Only when the
 * executable is missing does this run `npx playwright install --with-deps chromium` (the browser,
 * its headless shell and the system libraries; sudo on GitHub runners).
 */
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { chromium } from 'playwright';

const exe = chromium.executablePath();
if (existsSync(exe)) {
  console.log(`[chromium] found ${exe}`);
  process.exit(0);
}
console.log(`[chromium] missing (${exe}): npx playwright install --with-deps chromium`);
const r = spawnSync('npx', ['playwright', 'install', '--with-deps', 'chromium'], { stdio: 'inherit', env: { ...process.env, PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD: '' } });
process.exit(r.status ?? 1);
