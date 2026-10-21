# True-north heading: WMM declination and compass ⊕ GPS yaw

Measurements from 2026-10; the code may have changed since — check the source before relying on a number.

Code: `src/sensors/wmm.ts` (WMM2025), `CompassSensor.setLocationFromFix` (`src/sensors/orientation.ts`),
`yawFromCompass` (`src/sfm/geo.ts`), `fuseYaw` / `computeWorldAlignment` (`src/sfm/world.ts`),
`HeadingNorthEmulator` (`src/harness/engine/feed.ts`).

## WMM2025 validation

* Against all 100 official `WMM2025_TestValues.txt` rows: |ΔD|, |ΔI| ≤ 0.005° (the table's print precision),
  |ΔX..F| < 1e-3 nT, rates < 1e-5 °/yr and 1e-4 nT/yr (`wmm.test.ts`).
* The embedded `WMM.COF` and the test table are byte-identical to the official `WMM2025COF.zip` (sha256 of WMM.COF
  dfa85978…b075582f).
* `wmm-independent.test.ts` re-synthesises the field a different way (potential from explicit Legendre polynomials,
  −∇V by central differences in ECEF, geodetic NED projection) and agrees with the official rows and with
  `magneticField` at 300 random places / heights / dates and at the poles (≤ 5e-3 nT, ≤ 1e-4°).

## Yaw accuracy

Unit (`world-heading.test.ts`, 30–40 m walks, Seattle, GPS σ 3 m, 5° compass noise):

* Magnetic headings converted with the first-fix declination: |ΔD| < 1e-3° vs the GPS origin.
* 6 seeds: fused yaw RMS 0.62° vs GPS-only 3.21°, z-score RMS 0.49 (σ conservative: the synthetic compass has no
  systematic error).
* A 15° (mislabelled magnetic) or 40° compass bias is rejected against the GPS yaw; a 22 m track (σ_g 5.2°) gives the
  compass weight > 0.5.

e2e yard (worker, CPU features, `--no-dense`; magnetic headings by default):

| Run | Heading error | Notes |
|---|---|---|
| default (magnetic headings) | −0.64° | yaw `gps+compass`; declination 14.869° at the first fix, 80/80 headings converted; σ_gps 2.97°, σ_compass 1.47° ⇒ compass weight 0.80, fused σ 1.32°; compass − GPS 2.5° = 0.76σ |
| `--no-heading` (GPS yaw only) | −2.66° | the behaviour before compass fusion |
| `headingNorth: 'app'` | −0.64° | headings already true north |
| object (no location ⇒ true-north replay) | −0.36° | compass yaw |
| `--no-gps`, magnetic headings | yaw 14.7° off true north | = magnetic north, `northAligned: false`; the yard north claim fails by design |

## Limits

* The WMM ignores crustal anomalies and disturbance fields (declination RMS ≈ 0.3–0.5°, worse near the magnetic
  poles and at high magnetic activity).
* The systematic compass σ (25 % of the reported accuracy) is a prior, not calibrated on devices.
* The compass reference axis (device top vs optical axis when upright) and the iOS 26 landscape heading behaviour
  are unverified on hardware (ios-sensors.md); the confident-GPS rule in `fuseYaw` limits the damage.
* The app has no coarse-location source: with GPS denied the yaw is magnetic and `northAligned` false.
* Persisted keyframes without `headingNorth` are read as true north. Pre-release builds sent unflagged magnetic
  headings; a released app would need a session-version check to read such sessions as magnetic.
