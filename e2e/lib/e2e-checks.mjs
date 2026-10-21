// Result → metrics input, threshold checks and tables for e2e/run-e2e.mjs (pure; unit-tested in
// src/harness/engine/e2e-lib.test.ts).

/**
 * The e2e/metrics.mjs result format from a harness result (window.runPipeline):
 * { intrinsics, images: [{index, pose}], metric, numInputImages, pointsFile?, objectRoi? }.
 * `numInputImages` = images the engine tried to register: `stats.images` when the engine
 * reports it, else the live keyframe count, else the number of cameras. `objectRoi`: the
 * mesh's object box (null = none proposed), absent without a mesh.
 */
export function metricsInput(r, { pointsFile = null } = {}) {
  const sp = r.sparse;
  const statImages = sp && typeof sp.stats?.images === 'number' ? sp.stats.images : null;
  const numInput = statImages ?? (r.live?.keyframes || null) ?? (sp ? sp.numCameras : null);
  const out = {
    intrinsics: sp?.intrinsics ?? null,
    images: (sp?.images ?? []).map((im) => ({ index: im.index, pose: im.pose })),
    metric: sp?.metric
      ? { metric: !!sp.metric.metric, gravityAligned: !!sp.metric.gravityAligned, northAligned: !!sp.metric.northAligned, source: sp.metric.source, relativeUncertainty: sp.metric.relativeUncertainty }
      : { metric: false, gravityAligned: false, northAligned: false },
    numInputImages: numInput || null,
  };
  if (pointsFile) out.pointsFile = pointsFile;
  if (r.mesh) out.objectRoi = r.mesh.objectRoi ?? null;
  return out;
}

/** Thresholds for a dataset: default ← scene (meta.scene) ← datasets[name]. */
export function resolveThresholds(spec, scene, name) {
  const pick = (o) => (o && typeof o === 'object' ? o : {});
  const merged = { ...pick(spec?.default), ...pick(spec?.[scene]), ...pick(spec?.datasets?.[name]) };
  const claims = { ...pick(spec?.default?.claims), ...pick(spec?.[scene]?.claims), ...pick(spec?.datasets?.[name]?.claims) };
  delete merged.claims;
  return { limits: merged, claims };
}

const abs = (v) => (typeof v === 'number' ? Math.abs(v) : v);

/**
 * Named check values from computeMetrics output. `when` decides whether a check applies:
 *   'always', 'dense' (dense cloud requested), 'mesh' (mesh requested), 'metric' / 'gravity' /
 *   'north' (claimed by the result). The mesh / object-box checks score object isolation against
 *   meta.focus (metrics `focus` / `objectRoi`); objectRoiContainsFocus is 1 when the box contains
 *   meta.focus shrunk by 1 cm, else 0.
 */
export const CHECKS = {
  registrationRate: { label: 'registration rate (of input)', when: 'always', value: (m) => m.registration?.rateOfInput ?? m.registration?.rate ?? null },
  ateRelative: { label: 'ATE / scene extent', when: 'always', value: (m) => m.trajectory?.ateRelative ?? null },
  rotErrDegMedian: { label: 'rotation error median (deg)', when: 'always', value: (m) => m.trajectory?.rotErrDegMedian ?? null },
  focalErrorPct: { label: '|focal error| (%)', when: 'always', value: (m) => abs(m.intrinsics?.focalErrorPct ?? null) },
  scaleError: { label: '|scale error|', when: 'metric', value: (m) => abs(m.metric?.scaleError ?? null) },
  gravityErrorDeg: { label: 'gravity error (deg)', when: 'gravity', value: (m) => m.frame?.gravityErrorDeg ?? null },
  headingErrorDeg: { label: '|heading error| (deg)', when: 'north', value: (m) => abs(m.frame?.headingErrorDeg ?? null) },
  denseAccuracyMedian: { label: 'dense accuracy p50 (m)', when: 'dense', value: (m) => m.dense?.accuracyMedian ?? null },
  denseAccuracyMedianRel: { label: 'dense accuracy p50 / extent', when: 'dense', value: (m) => (m.dense && m.trajectory ? m.dense.accuracyMedian / m.trajectory.sceneExtent : null) },
  densePrecision: { label: 'dense precision', when: 'dense', value: (m) => m.dense?.precision ?? null },
  denseCompleteness: { label: 'dense completeness', when: 'dense', value: (m) => m.dense?.completeness ?? null },
  denseFscore: { label: 'dense F-score', when: 'dense', value: (m) => m.dense?.fscore ?? null },
  meshFocusOutside: { label: 'mesh vertices outside focus ± 2 cm', when: 'mesh', value: (m) => m.focus?.outsideFraction ?? null },
  meshFocusCompleteness: { label: 'mesh completeness in focus box', when: 'mesh', value: (m) => m.focus?.completeness ?? null },
  objectRoiIoU: { label: 'object box IoU with focus', when: 'mesh', value: (m) => m.objectRoi?.iou ?? null },
  objectRoiContainsFocus: { label: 'object box ⊇ focus − 1 cm', when: 'mesh', value: (m) => (m.objectRoi ? (m.objectRoi.containsFocusShrunk ? 1 : 0) : null) },
};

/**
 * Evaluate thresholds. ctx = { dense: bool (dense requested), mesh: bool (mesh requested), claims:
 * {metric, gravityAligned, northAligned} (from the result), sensors: {imu, gps, heading} (streams
 * that were passed) }.
 * Returns [{ name, label, value, limit, status: 'pass' | 'fail' | 'skip', reason? }].
 * A check that applies but has no value (e.g. no trajectory) FAILS. `claims` thresholds
 * (e.g. { metric: true }) require the result to claim that property when the sensors needed
 * for it were passed (metric/gravity: IMU or GPS; north: heading or GPS).
 */
export function runChecks(metrics, thresholds, ctx) {
  const out = [];
  const claimed = { metric: !!ctx.claims?.metric, gravity: !!ctx.claims?.gravityAligned, north: !!ctx.claims?.northAligned };
  const order = Object.keys(CHECKS);
  const rank = (k) => (order.includes(k) ? order.indexOf(k) : order.length);
  const limits = Object.entries(thresholds.limits ?? {}).sort((a, b) => rank(a[0]) - rank(b[0]));
  for (const [name, limit] of limits) {
    const def = CHECKS[name];
    if (!def) {
      out.push({ name, label: name, value: null, limit, status: 'fail', reason: 'unknown check name' });
      continue;
    }
    const applies =
      def.when === 'always' ||
      (def.when === 'dense' && !!ctx.dense) ||
      (def.when === 'mesh' && !!ctx.mesh) ||
      (def.when === 'metric' && claimed.metric) ||
      (def.when === 'gravity' && claimed.gravity) ||
      (def.when === 'north' && claimed.north);
    if (!applies) {
      const reason = def.when === 'dense' ? 'no dense run' : def.when === 'mesh' ? 'no mesh run' : `not claimed (${def.when})`;
      out.push({ name, label: def.label, value: null, limit, status: 'skip', reason });
      continue;
    }
    const v = def.value(metrics);
    if (v === null || v === undefined || Number.isNaN(v)) {
      out.push({ name, label: def.label, value: null, limit, status: 'fail', reason: 'missing' });
      continue;
    }
    let ok = true;
    if (typeof limit.min === 'number' && !(v >= limit.min)) ok = false;
    if (typeof limit.max === 'number' && !(v <= limit.max)) ok = false;
    out.push({ name, label: def.label, value: v, limit, status: ok ? 'pass' : 'fail' });
  }
  const needs = {
    metric: !!(ctx.sensors?.imu || ctx.sensors?.gps),
    gravityAligned: !!(ctx.sensors?.imu || ctx.sensors?.gps),
    northAligned: !!(ctx.sensors?.heading || ctx.sensors?.gps),
  };
  for (const [k, want] of Object.entries(thresholds.claims ?? {})) {
    if (!want) continue;
    const label = `claims ${k}`;
    if (!(k in needs)) {
      out.push({ name: `claims.${k}`, label, value: null, limit: { equals: true }, status: 'fail', reason: 'unknown claim' });
      continue;
    }
    if (!needs[k]) {
      out.push({ name: `claims.${k}`, label, value: null, limit: { equals: true }, status: 'skip', reason: 'sensor withheld' });
      continue;
    }
    const v = !!ctx.claims?.[k];
    out.push({ name: `claims.${k}`, label, value: v, limit: { equals: true }, status: v ? 'pass' : 'fail' });
  }
  return out;
}

const fmtNum = (v, d = 4) =>
  v === null || v === undefined
    ? '—'
    : typeof v === 'boolean'
      ? String(v)
      : typeof v !== 'number'
        ? String(v)
        : !Number.isFinite(v)
          ? String(v)
          : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0)
            ? v.toExponential(2)
            : v.toFixed(d);

export function formatChecks(checks) {
  const rows = [['check', 'value', 'limit', 'status']];
  for (const c of checks) {
    const lim = c.limit?.equals !== undefined ? `= ${c.limit.equals}` : [c.limit?.min !== undefined ? `≥ ${c.limit.min}` : null, c.limit?.max !== undefined ? `≤ ${c.limit.max}` : null].filter(Boolean).join(', ');
    rows.push([c.label, fmtNum(c.value), lim, c.status.toUpperCase() + (c.reason ? ` (${c.reason})` : '')]);
  }
  return table(rows);
}

export function table(rows, alignLeft = [0]) {
  const w = rows[0].map((_, c) => Math.max(...rows.map((r) => String(r[c]).length)));
  return rows.map((r) => r.map((s, c) => (alignLeft.includes(c) ? String(s).padEnd(w[c]) : String(s).padStart(w[c]))).join('  ')).join('\n');
}

/** One summary row per dataset run. */
export function summaryRow(name, r, m, checks) {
  const fails = checks.filter((c) => c.status === 'fail').length;
  return [
    name,
    r?.ok ? 'ok' : r ? `errors ${r.errors?.length ?? 0}` : 'failed',
    m ? `${m.registration.numRegistered}/${m.registration.numInput ?? '?'}` : '—',
    fmtNum(m?.trajectory?.ateRelative ?? null),
    fmtNum(m?.trajectory?.rotErrDegMedian ?? null, 3),
    fmtNum(m?.intrinsics?.focalErrorPct ?? null, 2),
    fmtNum(m?.metric?.scaleError ?? null, 3),
    fmtNum(m?.frame?.gravityErrorDeg ?? null, 2),
    fmtNum(m?.frame?.headingErrorDeg ?? null, 1),
    fmtNum(m?.dense?.accuracyMedian ?? null),
    fmtNum(m?.dense?.completeness ?? null, 3),
    fmtNum(m?.dense?.fscore ?? null, 3),
    r ? fmtNum((r.timings?.total ?? 0) / 1000, 0) : '—',
    checks.length ? (fails ? `FAIL ${fails}` : 'PASS') : '—',
  ];
}

export const SUMMARY_HEADER = ['dataset', 'pipeline', 'reg/input', 'ATE/ext', 'rot° p50', 'f err %', 'scale err', 'grav°', 'head°', 'dense p50', 'compl', 'F', 'wall s', 'checks'];

export { fmtNum };
