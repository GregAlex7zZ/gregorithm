# Changelog

All notable public changes to gregorithm are documented here. The project follows
[Semantic Versioning](https://semver.org/); during the 0.x series, minor releases may change the
API and result schemas when documented here.

## [0.1.0] — unreleased

First public version.

### Included

- Inspection and canonical import of CSV, TXT and Parquet sources with explicit column and unit
  mapping, elapsed-duration parsing, documented row cleaning and disk-space admission.
- Streaming elaboration of half-cycles, cycles, rest/CC/CV/other steps, cumulative capacity and
  energy, compact indexes, plot overviews and windowed reads.
- Per-cycle Capacity, energy, efficiency, normalization and C-rate with explicit diagnostics.
- DCIR pulse detection, explicit review, automatic and manual measurement, capacity-based SOC,
  immutable publication and optional cycle correction.
- dQ/dV selectable with `DqdvParameters.method`: `interpolated_gradient` (default;
  interpolate-then-`numpy.gradient` with point count or percentage and edge trim) and the
  derivative-free `histogram` and `histogram_weighted` methods of E. Flores and S. Clark,
  J. Electrochem. Soc. 173 (2026) 120520, with optional Gaussian filtering, optional NumPy-only
  Savitzky–Golay smoothing and an optional voltage window.
- Voltage reconstruction of every dQ/dV curve with APE and RMSE quality metrics, enabled by default;
  cycle correction re-indexes the reconstruction table.
- Pure dQ/dV functions (`dqdv_histogram`, `interpolated_gradient`, `interpolation_point_count`,
  `gaussian_smooth`, `reconstruct_voltage_curve`, `reconstruction_metrics`,
  `dqdv_method_references`).
- **GITT/ICI diffusion analysis** (`gregorithm.gitt_ici`, compact API re-exported at top level):
  - typed contracts, canonical event association (GITT rest–CC–rest, ICI CC–interruption) on a
    verified elaboration (`load_canonical_scope`), SI physical inputs and exact sample selection;
  - sampled OCV (last rest sample, an exact row, or the last sample within a global
    `ocv_sample_time_s` or a per-rest `OcvSampleTime`) and on-demand `LOG_TIME_2P` relaxation fit
    and extrapolation, with per-rest targets and fit start fractions (`OcvTargetPlan`);
  - shared square-root-time OLS fit and dimensionally explicit GITT_VA, GITT_DELTA and ICI
    kernels with conditional covariance propagation;
  - analytical spherical Delacourt fit with exact short/long-time response, fixed/free offset,
    log-dimensionless search, variable projection and identifiability diagnostics;
  - event-level analysis plans (global and local recalculation), manual/fixed/statistical/
    physically guided windows, advisory quality indicators, accept/accept-all/revoke/reject,
    targeted invalidation, late-result protection, a versioned window cache and impulse-level
    progress;
  - `measure_ir_drops` (pre-pulse R_on and post-pulse R_off with the real samples used) and
    `measure_rest_ocv` (sampled and extrapolated OCV of every rest), with targeted dependencies
    for manual choices;
  - immutable, checksummed snapshot publication, a typed round-trip loader and CSV/Parquet export
    of accepted results.
- `ParameterDescriptor.conditions` and `AnalysisDescriptor.references` so interfaces can show
  only relevant settings and cite methods.
- Streaming CSV/Parquet export, immutable checksummed result generations, fingerprints, progress
  and cooperative cancellation; storage locks checked on Windows, Linux and macOS.
- User and scientific documentation, six executable example notebooks, synthetic data and a
  public test suite.
