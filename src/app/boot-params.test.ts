import { describe, expect, it } from 'vitest';
import { parseBootParams, SIMULATED_FAILURE } from './boot-params';

describe('parseBootParams', () => {
  it('defaults: real engine, no test hooks', () => {
    expect(parseBootParams('', false)).toEqual({ mock: false, mockOptions: {}, screen: null });
    expect(parseBootParams('?screen=diagnostics', false).screen).toBe('diagnostics');
    expect(parseBootParams('?screen=setup', true).screen).toBe('setup');
    expect(parseBootParams('?screen=javascript:alert(1)', true).screen).toBeNull();
  });

  it('mock* parameters only apply with mock', () => {
    expect(parseBootParams('?mockFail=refine:x&mockStall=refine&mockMemoryLimit=3', true).mockOptions).toEqual({});
    expect(parseBootParams('?mock=0&mockFail=refine', true).mock).toBe(false);
    expect(parseBootParams('?mock=true', true).mock).toBe(true);
  });

  it('dev builds keep the custom failure message (e2e error UI), control characters stripped and length capped', () => {
    const p = parseBootParams(`?mock=1&mockSpeed=3&mockFail=${encodeURIComponent('refine:Not enough overlap: between photos')}`, true);
    expect(p.mockOptions).toEqual({ speed: 3, failOn: { refine: 'Not enough overlap: between photos' } });
    const long = parseBootParams(`?mock=1&mockFail=refine:${'x'.repeat(500)}%0A%0Dy`, true).mockOptions.failOn!.refine!;
    expect(long.length).toBeLessThanOrEqual(160);
    expect(/[\r\n]/.test(long)).toBe(false);
    expect(parseBootParams('?mock=1&mockFail=refine', true).mockOptions.failOn).toEqual({ refine: SIMULATED_FAILURE });
  });

  it('production builds never show URL-controlled text', () => {
    const p = parseBootParams(`?mock=1&mockFail=${encodeURIComponent('export:Your device is infected — call +1 555 0100')}`, false);
    expect(p.mockOptions.failOn).toEqual({ export: SIMULATED_FAILURE });
  });

  it('unknown methods, prototype keys and bad numbers are ignored', () => {
    const p = parseBootParams('?mock=1&mockFail=__proto__:x&mockStall=constructor,refine,toString,mesh&mockMemoryLimit=-2&mockSpeed=NaN', true);
    expect(p.mockOptions.failOn).toBeUndefined();
    expect(p.mockOptions.stallOn).toEqual({ refine: true, mesh: true });
    expect(Object.getPrototypeOf(p.mockOptions.stallOn)).toBe(Object.prototype);
    expect(p.mockOptions.memoryLimitKeyframes).toBeUndefined();
    expect(p.mockOptions.speed).toBe(1);
    expect(parseBootParams('?mock=1&mockMemoryLimit=6&mockSpeed=0.3', false).mockOptions).toEqual({ speed: 0.3, memoryLimitKeyframes: 6 });
    expect(parseBootParams('?mock=1&mockSpeed=1e9', false).mockOptions.speed).toBe(50);
    expect(parseBootParams('?mock=1&mockMemoryLimit=2.5', false).mockOptions.memoryLimitKeyframes).toBeUndefined();
  });
});
