import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

// Three test projects:
//  - unit: pure TypeScript logic, runs in Node (fast). Files: src/**/*.test.ts
//  - slow: Node tests that run the whole pipeline (SfM / BA / PatchMatch on synthetic
//          captures) and take seconds to tens of seconds each. Files: src/**/*.slow.test.ts
//  - gpu:  WebGPU / DOM code, runs in headless Chromium with SwiftShader WebGPU.
//          Files: src/**/*.gpu.test.ts
export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: 'unit',
          environment: 'node',
          include: ['src/**/*.test.ts'],
          exclude: ['src/**/*.gpu.test.ts', 'src/**/*.slow.test.ts', 'node_modules/**'],
          testTimeout: 120_000,
        },
      },
      {
        test: {
          name: 'slow',
          environment: 'node',
          include: ['src/**/*.slow.test.ts'],
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
              launchOptions: {
                args: [
                  '--enable-unsafe-webgpu',
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
