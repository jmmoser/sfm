/**
 * Static security checks of the shipped app: Content-Security-Policy of index.html, no
 * third-party URLs, HTML-injection sinks only where the markup is static, dev-only pages kept out
 * of the production build, icon markup never taken from outside.
 */
import { describe, expect, it } from 'vitest';
import indexHtml from '../../index.html?raw';
import manifest from '../../public/manifest.webmanifest?raw';
import viteConfig from '../../vite.config.ts?raw';
import { applyProps } from './dom';
import { iconSvg, type IconName } from './icons';

// keys: '/src/app/dom.ts', …
const sources = import.meta.glob<string>(['/src/**/*.ts', '!/src/**/*.test.ts', '!/src/harness/**'], { query: '?raw', import: 'default', eager: true });
const rel = (file: string): string => file.replace(/^\/src\//, '');

function csp(): Map<string, string[]> {
  const m = /<meta\s+http-equiv="Content-Security-Policy"\s+content="([^"]+)"/.exec(indexHtml);
  expect(m, 'index.html has a CSP meta').not.toBeNull();
  const out = new Map<string, string[]>();
  for (const d of m![1].split(';')) {
    const [name, ...values] = d.trim().split(/\s+/);
    if (name) out.set(name, values);
  }
  return out;
}

describe('index.html Content-Security-Policy', () => {
  it('comes before any script or stylesheet', () => {
    const at = indexHtml.indexOf('Content-Security-Policy');
    expect(at).toBeGreaterThan(0);
    for (const tag of ['<script', '<link']) expect(indexHtml.indexOf(tag)).toBeGreaterThan(at);
  });

  it('allows only same-origin code, WebAssembly but no eval, no plugins, no form posts', () => {
    const p = csp();
    expect(p.get('default-src')).toEqual(["'self'"]);
    expect(p.get('script-src')).toEqual(["'self'", "'wasm-unsafe-eval'"]);
    expect(p.get('object-src')).toEqual(["'none'"]);
    expect(p.get('base-uri')).toEqual(["'self'"]);
    expect(p.get('form-action')).toEqual(["'none'"]);
    expect(p.get('worker-src')).toEqual(["'self'"]);
    expect(p.get('connect-src')).toEqual(["'self'"]);
    for (const [name, values] of p) {
      expect(values, name).not.toContain("'unsafe-eval'");
      if (name !== 'style-src') expect(values, name).not.toContain("'unsafe-inline'");
      // no network origin other than our own (no CDN, no analytics)
      for (const v of values) expect(v, `${name} ${v}`).toMatch(/^('[a-z-]+'|blob:|data:|mediastream:)$/);
    }
  });

  it('has no inline scripts or third-party resources', () => {
    expect(indexHtml).not.toMatch(/<script(?![^>]*\bsrc=)[^>]*>/);
    expect(indexHtml).not.toMatch(/\son[a-z]+\s*=/i);
    expect(indexHtml).not.toMatch(/(src|href)\s*=\s*["'](https?:)?\/\//);
    expect(manifest).not.toMatch(/https?:\/\//);
  });
});

describe('source code', () => {
  const files = Object.entries(sources);

  it('was collected', () => {
    expect(files.length).toBeGreaterThan(100);
  });

  it('requests no third-party URL at runtime', () => {
    const offenders: string[] = [];
    for (const [file, text] of files) {
      for (const m of text.matchAll(/["'`](https?:)?\/\/[^"'`\s]+/g)) {
        if (m[0].slice(1) === 'http://www.w3.org/2000/svg') continue; // SVG namespace, not a request
        offenders.push(`${file}: ${m[0]}`);
      }
    }
    expect(offenders).toEqual([]);
  });

  it('uses HTML sinks only for static markup (icons, the home illustration)', () => {
    const sinks: string[] = [];
    for (const [file, text] of files) {
      for (const m of text.matchAll(/\.(innerHTML|outerHTML)\s*=|insertAdjacentHTML\(|document\.write\(|createContextualFragment\(|\bsrcdoc\b|\bnew Function\(|\beval\(/g)) {
        sinks.push(`${rel(file)}: ${m[0]}`);
      }
    }
    expect(sinks.sort()).toEqual(['app/dom.ts: .innerHTML =', 'app/icons.ts: .innerHTML =']);
    // the `html` prop of h() is used for the static hero SVG only
    const htmlProps = files.flatMap(([file, text]) => [...text.matchAll(/\bhtml:\s*([^,}\s]+)/g)].map((m) => `${rel(file)}: ${m[1]}`));
    expect(htmlProps).toEqual(['app/screens/home.ts: HERO_SVG']);
  });

  it('h() / applyProps never create inline event-handler attributes from strings', () => {
    const attrs: string[] = [];
    const el = { className: '', setAttribute: (k: string) => attrs.push(k), addEventListener: () => undefined } as unknown as HTMLElement;
    for (const k of ['onclick', 'onerror', 'ONLOAD', 'onFocus']) expect(() => applyProps(el, { [k]: 'alert(1)' })).toThrow(TypeError);
    applyProps(el, { onclick: () => undefined, 'aria-label': 'x', class: 'btn' });
    expect(attrs).toEqual(['aria-label']);
  });

  it('iconSvg never looks up prototype keys or non-numeric sizes', () => {
    for (const name of ['constructor', '__proto__', 'toString', 'hasOwnProperty']) {
      expect(iconSvg(name as IconName)).not.toMatch(/function|\[object|native code/);
    }
    expect(iconSvg('plus', '"><script>' as unknown as number)).toContain('width="24"');
    expect(iconSvg('plus', 18)).toContain('<path d="M12 5v14M5 12h14"/>');
  });
});

describe('production build', () => {
  it('ships only the app page (the e2e harness pages are dev-server only)', () => {
    const input = /input:\s*\{([^}]*)\}/.exec(viteConfig);
    expect(input).not.toBeNull();
    expect(input![1]).toMatch(/index\.html/);
    expect(input![1]).not.toMatch(/harness|synth|dense-gt/);
  });
});
