// Chromium launch options for the e2e tools. Default: Playwright's Chromium on SwiftShader
// (deterministic, runs on any CI host). E2E_GPU=host: the installed Google Chrome on the host's
// real GPU (Metal on macOS), for realistic GPU timings.
export function launchOptions(swiftshaderArgs, { headless = true } = {}) {
  if (process.env.E2E_GPU === 'host') {
    const keep = swiftshaderArgs.filter((a) => !/swiftshader|vulkan|Vulkan/.test(a));
    return { channel: 'chrome', headless, args: [...new Set(['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', ...keep])] };
  }
  return { headless, args: swiftshaderArgs };
}
