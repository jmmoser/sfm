// GT-free scoring of a run on a dataset without ground-truth poses — a raw capture recording
// (src/io/recording.ts; e2e/README.md "Raw capture recordings") — for e2e/run-e2e.mjs and
// e2e/metrics.mjs (pure; unit-tested in src/harness/engine/e2e-lib.test.ts).
//
// Self-consistency only, from what the engine reports (refine's SparseResult.stats, the live
// summary) and, for recordings, the device's own answers (result.recording): registered fraction,
// reprojection error, track length, gyro breaks / outliers (refine's gyro consistency of the live
// and offline models), explained correspondences, which model won, the scale / gravity / north
// sources, the time offset, the focal against the hfov prior, live init / tracking, and the
// agreement of the replay's keyframe decisions with the device's. These are proxies: real
// accuracy needs ground truth (a gyro-consistent model can still be off in scale or shape).

const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/** True when the dataset cannot be scored against ground truth (a recording, or frames without poses). */
export function isGtFree(meta, frames) {
  if (meta?.source === 'sfm-raw-capture' || meta?.scene === 'recording' || meta?.gt === null) return true;
  return !Array.isArray(frames) || frames.length === 0 || frames.some((f) => !f?.pose);
}

/** "3 (+1 outliers, median 0.42°)" (refine stats gyroBreaksLive / Offline) → {breaks, outliers, medianDeg}. */
export function parseGyroBreaks(s) {
  if (typeof s !== 'string') return null;
  const m = /^\s*(\d+)\s*\(\+(\d+) outliers?, median ([\d.eE+-]+)°\)/.exec(s);
  return m ? { breaks: Number(m[1]), outliers: Number(m[2]), medianDeg: Number(m[3]) } : null;
}

/** Focal (px, long side `longSide`) of a field-of-view prior across the long side. */
function focalFromHfov(hfovDeg, longSide) {
  return hfovDeg > 0 && longSide > 0 ? longSide / 2 / Math.tan((hfovDeg * Math.PI) / 360) : null;
}

/**
 * Self-consistency metrics of a harness result (result.json of run-e2e: the harness fields) on a
 * GT-free dataset. `meta` = the dataset's meta.json. Never throws on missing parts (a run that
 * failed early still gets the fields it has).
 */
export function computeSelfMetrics(r, meta = {}) {
  const sp = r?.sparse ?? null;
  const st = sp?.stats ?? {};
  const lv = r?.live ?? {};
  const numInput = num(st.images) ?? (num(lv.keyframes) || null) ?? (sp ? num(sp.numCameras) : null);
  const numRegistered = num(st.registeredImages) ?? num(sp?.images?.length) ?? num(sp?.numCameras) ?? 0;
  const intr = sp?.intrinsics ?? null;
  const prior = num(r?.config?.hfovPriorDegUsed);
  const longSide = intr ? Math.max(intr.width ?? 0, intr.height ?? 0) : 0;
  const fPrior = prior !== null && longSide ? focalFromHfov(prior, longSide) : null;
  const metric = sp?.metric ?? null;
  const gl = parseGyroBreaks(st.gyroBreaksLive);
  const go = parseGyroBreaks(st.gyroBreaksOffline);
  const model = typeof st.model === 'string' ? st.model : null;
  const chosen = model === 'offline' ? go : model === 'live' ? gl : (gl ?? go);
  const outliers = typeof st.gyroOutliers === 'string' && st.gyroOutliers ? st.gyroOutliers.split(',').filter(Boolean).length : 0;
  const rec = r?.recording ?? null;
  return {
    dataset: meta?.name ?? r?.dataset?.name ?? '',
    scene: meta?.scene ?? r?.dataset?.scene ?? '',
    groundTruth: false,
    numFrames: num(r?.dataset?.numFrames) ?? 0,
    registration: {
      numFrames: num(r?.dataset?.numFrames) ?? 0,
      delivered: num(lv.delivered) ?? 0,
      keyframes: num(lv.keyframes) ?? 0,
      liveRegistered: num(st.liveRegistered) ?? num(lv.registered),
      numInput,
      numRegistered,
      rateOfInput: numInput ? numRegistered / numInput : null,
    },
    reprojection: { meanPx: num(st.meanReprojErrorPx) },
    tracks: { mean: num(st.meanTrackLength), median: num(st.medianTrackLength), points: num(sp?.pointsCount) ?? num(st.points) },
    gyro: {
      model,
      breaks: chosen?.breaks ?? null,
      breakOutliers: chosen?.outliers ?? null,
      medianErrDeg: chosen?.medianDeg ?? null,
      live: gl,
      offline: go,
      deregisteredOutliers: outliers,
      rejectedPairs: num(st.gyroRejectedPairs),
      rejectedPoses: num(st.gyroRejectedPoses),
    },
    explained: { live: num(st.liveExplained), offline: num(st.offlineExplained) },
    offlineSfM: typeof st.offlineSfM === 'string' ? st.offlineSfM : null,
    scale: metric
      ? {
          metric: !!metric.metric,
          source: metric.source ?? null,
          relativeUncertainty: num(metric.relativeUncertainty),
          gravityAligned: !!metric.gravityAligned,
          northAligned: !!metric.northAligned,
          yawSource: typeof st.yawSource === 'string' ? st.yawSource : null,
          gravityStdDeg: num(st.gravityStdDeg),
          timeOffsetMs: num(st.timeOffsetMs),
          warnings: typeof st.alignmentWarnings === 'string' ? st.alignmentWarnings : null,
        }
      : null,
    intrinsics: intr
      ? { f: intr.f, k1: num(intr.k1), k2: num(intr.k2), fPrior, focalVsPriorPct: fPrior ? (100 * (intr.f - fPrior)) / fPrior : null }
      : null,
    live: {
      timeToInit: lv.timeToInit ?? null,
      trackingLostFrames: num(lv.trackingLostFrames) ?? 0,
      frameMsMedian: num(lv.frameMs?.median),
      frameMsP90: num(lv.frameMs?.p90),
      decisions: r?.decisions ?? {},
    },
    dense: r?.dense ? { points: r.dense.count } : null,
    mesh: r?.mesh ? { triangles: r.mesh.triangleCount } : null,
    device: rec?.device ?? null,
    agreement: rec?.agreement ?? null,
    // the GT blocks of computeMetrics, absent here (summary rows read them)
    trajectory: null,
    metric: null,
    frame: null,
  };
}

const fmt = (v, d = 4) =>
  v === null || v === undefined
    ? '—'
    : typeof v !== 'number'
      ? String(v)
      : !Number.isFinite(v) || Number.isInteger(v)
        ? String(v)
        : Math.abs(v) >= 1e4 || Math.abs(v) < 1e-3
          ? v.toExponential(2)
          : v.toFixed(d);

/** Human-readable table of computeSelfMetrics. */
export function formatSelfTable(m) {
  const rows = [];
  const add = (k, v, unit = '') => rows.push([k, fmt(v) + (unit && v !== null && v !== undefined ? ` ${unit}` : '')]);
  add('dataset', `${m.dataset} (no ground truth: self-consistency only)`);
  add('frames delivered / keyframes', `${m.registration.delivered} / ${m.registration.keyframes}`);
  add('registered / input', `${m.registration.numRegistered} / ${m.registration.numInput ?? '?'}`);
  add('registered fraction', m.registration.rateOfInput);
  add('mean reprojection error', m.reprojection.meanPx, 'px');
  add('track length mean / median', `${fmt(m.tracks.mean, 2)} / ${fmt(m.tracks.median, 1)}`);
  add('points', m.tracks.points);
  add(`gyro breaks (${m.gyro.model ?? '?'} model)`, m.gyro.breaks === null ? null : `${m.gyro.breaks} (+${m.gyro.breakOutliers} outliers, median ${fmt(m.gyro.medianErrDeg, 2)}°)`);
  if (m.gyro.deregisteredOutliers) add('gyro outliers deregistered', m.gyro.deregisteredOutliers);
  add('explained correspondences (live / offline)', `${fmt(m.explained.live, 3)} / ${fmt(m.explained.offline, 3)}`);
  if (m.offlineSfM) add('offline SfM', m.offlineSfM);
  if (m.scale) {
    add('scale', m.scale.metric ? `metric (${m.scale.source}${m.scale.relativeUncertainty !== null ? `, ±${fmt(100 * m.scale.relativeUncertainty, 1)} %` : ''})` : 'not metric');
    add('gravity / north aligned', `${m.scale.gravityAligned} / ${m.scale.northAligned}${m.scale.yawSource ? ` (yaw: ${m.scale.yawSource})` : ''}`);
    if (m.scale.timeOffsetMs !== null) add('camera–IMU time offset', m.scale.timeOffsetMs, 'ms');
    if (m.scale.warnings) add('alignment warnings', m.scale.warnings);
  }
  if (m.intrinsics) add('focal (vs hfov prior)', `${fmt(m.intrinsics.f, 1)} px (${m.intrinsics.focalVsPriorPct === null ? 'no prior' : `${fmt(m.intrinsics.focalVsPriorPct, 2)} %`})`);
  add('live init after frames', m.live.timeToInit ? m.live.timeToInit.delivered : 'never');
  add('tracking lost frames', m.live.trackingLostFrames);
  add('frame ms p50 / p90', `${fmt(m.live.frameMsMedian, 0)} / ${fmt(m.live.frameMsP90, 0)}`);
  if (m.dense) add('dense points', m.dense.points);
  if (m.device) {
    add('device: frames / keyframes / registered', `${m.device.frames} / ${m.device.keyframes ?? '?'} / ${m.device.registered ?? '?'}`);
    add('device: frame ms p50 / p90 (recorder wait p90)', `${fmt(m.device.frameMs?.median, 0)} / ${fmt(m.device.frameMs?.p90, 0)} (${fmt(m.device.waitMs?.p90, 0)})`);
  }
  if (m.agreement) add('decisions = device (keyframes device / replay)', `${fmt(m.agreement.rate, 3)} of ${m.agreement.compared} (${m.agreement.keyframesDevice} / ${m.agreement.keyframesReplay})`);
  const w = Math.max(...rows.map((x) => x[0].length));
  return rows.map(([k, v]) => `${k.padEnd(w)}  ${v}`).join('\n');
}
