import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

// Two test projects:
//  - unit: pure TypeScript logic, runs in Node (fast). Files: src/**/*.test.ts
//  - gpu:  WebGPU / DOM code, runs in headless Chromium with SwiftShader WebGPU
//          (E2E_GPU=host: installed Chrome on the host GPU).
//          Files: src/**/*.gpu.test.ts
export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: 'unit',
          environment: 'node',
          include: ['src/**/*.test.ts'],
          exclude: ['src/**/*.gpu.test.ts', 'node_modules/**'],
          testTimeout: 120_000,
        },
      },
      {
        test: {
          name: 'gpu',
          include: ['src/**/*.gpu.test.ts'],
          testTimeout: 600_000,
          browser: {
            enabled: true,
            headless: true,
            provider: playwright({
              // E2E_GPU=host: the installed Google Chrome on the host's real GPU (Metal on macOS).
              launchOptions:
                process.env.E2E_GPU === 'host'
                  ? {
                      channel: 'chrome',
                      args: [
                        '--enable-unsafe-webgpu',
                        '--ignore-gpu-blocklist',
                        // gc() for the leak tests (splat-layer.gpu.test.ts)
                        '--js-flags=--expose-gc',
                        '--use-fake-ui-for-media-stream',
                        '--use-fake-device-for-media-stream',
                        '--autoplay-policy=no-user-gesture-required',
                      ],
                    }
                  : {
                      args: [
                        '--enable-unsafe-webgpu',
                        '--js-flags=--expose-gc',
                        '--enable-features=Vulkan',
                        '--use-vulkan=swiftshader',
                        '--use-webgpu-adapter=swiftshader',
                        '--disable-vulkan-surface',
                        '--use-angle=swiftshader',
                        '--use-fake-ui-for-media-stream',
                        '--use-fake-device-for-media-stream',
                        '--autoplay-policy=no-user-gesture-required',
                      ],
                    },
            }),
            instances: [{ browser: 'chromium' }],
          },
        },
      },
    ],
  },
});
