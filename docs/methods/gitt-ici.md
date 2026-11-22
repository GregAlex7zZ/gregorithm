# GITT/ICI — diffusion coefficients from pulse techniques

The native GITT/ICI API starts from an existing gregorithm elaboration. It does not import raw
data, classify a protocol from durations, or replace the authoritative `rest`, `step`, `cycle`,
`half_cycle` and `source_row` identities. The operator chooses the protocol (GITT or ICI); it is
never inferred from pulse durations.

## Events and OCV

The API provides explicit protocol selection, immutable canonical sample/segment/event contracts,
GITT rest–CC–rest and ICI CC–interruption associations, stable event identities, exact sample
selection, SI physical inputs, distinct OCV sources, sampled OCV and on-demand `LOG_TIME_2P`
fitting on strictly positive times. A finite target may be beyond the observed rest; the returned
extrapolation ratio lets callers warn without blocking a mathematically valid evaluation.

## Square-root-time fit and diffusion kernels

A shared unweighted OLS fit `a + s√t` with free intercept, inclusive time bounds,
canonical source-row provenance, residuals, RMSE, R² and covariance only when residual degrees of
freedom permit it. Constant voltage correction changes the intercept but not the slope when rows
and weights are unchanged.

The public kernels implement three methods:

```text
GITT_VA    D = 4/π (V_act/A_eff)² [ΔU/(τ s_on)]²
GITT_DELTA D = 4/(πτ) (V_act/A_eff)² (ΔU/ΔE_τ)²
ICI        D = 4/π (V_act/A_eff)² [ΔU/(Δt_I s_off)]²
```

They require the caller to report whether the authoritative segment is CC; a false value leaves
D unavailable and never substitutes an average current. OCV differences reject mixed sources.
Zero or binary64-indistinguishable slopes keep a valid fit but cannot produce D. Optional standard
uncertainties use local covariance propagation when the necessary inputs are estimable.

Resistance helpers preserve signed ΔV and ΔI while reporting `abs(ΔV/ΔI)`. IR correction is one
explicit constant selected from none, a measured sample or the GITT intercept. GITT_DELTA always
uses the complete observed pulse duration; selecting a 1 s correction does not shorten τ.

## Analytical spherical fit (Delacourt)

`fit_delacourt` fits the full observed CC pulse. For `θ ≥ 0.02` the response uses the
eigenfunction series (20 terms by default, never fewer than 14); below it uses the exact short-time
solution `f(θ) = e^θ(1 + erf√θ) − 1` of the Laplace-domain response, whose neglected image terms
are of order `exp(−1/θ)`. Both are accurate to about 1e-13 against independent references, give
`f(0)=0`, the short-time limit `2√(θ/π)` and the long-time limit `3θ+1/5`.

The optimized variable is `log10(θτ)`, with default bounds −6…3. A grid exploration locates
candidate minima before bounded scalar refinement. Fixed offset is the default; free offset uses
variable projection. Results include D, `r_p²/D`, offset, predictions, residuals, RMSE, R²,
boundary status, sensitivity, identifiability and covariance-derived uncertainty only when
estimable. `r_p` is always a direct required input.

## Analysis plans, windows and quality

`plan_analysis` turns canonical events into one task per event and selected series
(`MethodVariant`: method, OCV source and variant). A global plan covers every complete event of
the scope; a local plan covers one event and all its checked series. Series that are not listed
are never recalculated or deleted. Incomplete events produce a `GITT_ICI_INCOMPLETE_EVENT`
diagnostic and no task. Automatic analysis on recognition is off by default; GITT preselects
GITT_VA and GITT_DELTA with sampled OCV, ICI preselects ICI with the intercept pseudo-OCV.

Each task computes the science in the library: sampled or on-demand extrapolated OCV (with an
extrapolation warning when the target exceeds the observed rest), the single constant IR
correction, the sqrt-time window and the kernel. D is placed at the canonical capacity at the
start of the CC segment. Classic ICI pairs the pseudo-OCV of the interruption directly before the
pulse in the same branch, with `Δt_I` equal to that pulse's current time; the first interruption
keeps its intermediates (`k`, signed resistance, pseudo-OCV) and reports D unavailable without a
fabricated ΔU. Missing `V_act`/`A_eff` leaves D unavailable with intermediates; Delacourt requires
`r_p` and fixes its offset at `(V0 − U0) + c_IR`.

Windows can be manual, fixed, a statistical search inside the suggested (editable) range, or
physically guided for GITT. The statistical color is the worse of
`q_L = RMSE/(|s|(√t1 − √t0))` and `q_S`, the median relative slope change when each endpoint is
moved by 10 % of the window's distinct times. Both are N/D when the slope is zero or within two
standard errors of zero, or when there are too few samples or neighbours. Among windows with at
least five distinct times the search prefers the best color, then the longest and best-sampled
window. The physical indicator compares the window end with `k r_p²/D` (green, +50 % orange,
red, otherwise N/D); `k` is explicit because no coefficient has been validated. ICI has no
validated useful-time function, so its physical indicator is always N/D. Guided searches use one
chosen D source per run, resolved per event without fallback, and are refined deterministically
until the window stabilizes. All thresholds are provisional presets and never block acceptance.

## Review, snapshots and batches

`ResultSnapshot` keeps one current record per identity. Review states are provisional, accepted,
rejected and error; freshness (stale after a targeted dependency change) is separate. A successful
recalculation replaces the identity and returns it to provisional; a recoverable error replaces it
with an error record; cancellation raises before anything is returned or cached. "Accept all"
acts on one series and skips rejected, stale, error and D-less records. Only accepted current
records are exportable. `commit_batch` refuses an outcome computed against an older snapshot, so
late results are discarded. The intermediate cache stores only automatic window bounds, keyed by
generation, dependencies, thresholds and algorithm version.

## Data access, defaults and limits

`load_canonical_scope` reads a verified elaboration generation (segments index and elaborated
points) restricted to selected cycles/half-cycles, plus two neighbouring segments on each side as
context; `CanonicalScope.events` associates only the pulses inside the scope. It never re-imports,
deduplicates or re-segments. Active segments labelled CV or OTHER become events with
`constant_current=False`: their intermediates are computed, D is unavailable
(`constant_current_required`) and Delacourt is not run.

`PhysicalInputs` values are independent and optional: Delacourt needs only `r_p`; GITT_VA,
GITT_DELTA and ICI report `physical_input_missing` without `V_act`/`A_eff`. The Delacourt fixed
offset takes its IR constant from `delacourt_ir_correction`, by default the voltage of the last
pulse sample within `ir_sample_time_s` (1 s) of the last rest sample; the row, time, correction and
total offset are recorded, and a missing sample is an error, never a silent fallback. The IR
default of the other methods stays "none". Because that sample also contains the diffusive
response of its first second, the fixed-offset D is biased high when the sampling is not much
faster than the diffusion transient; the free offset does not have this bias.

Automatic statistical windows use the longest adequate window by default and are
marked `early_data_excluded`, and therefore "to review", when they start after the first available
time. `WindowRanking.EARLY_ANCHORED` and `InitialWindowRule.LEADING_FRACTION` are experimental,
non-default alternatives. On spherical synthetic pulses no statistical rule keeps the sqrt-time
error small, while the physically guided search does. The suggested initial window is the whole
real segment.

The top-level `gregorithm` package re-exports only the compact analysis, review and batch API;
kernels, fits and window helpers are available from `gregorithm.gitt_ici`.

## Pre-pulse and post-pulse IR drops

`measure_ir_drops(event_data, settings, samples=...)` is the single source of the IR drops of one
impulse and returns `ImpulseIrDrops` with one `IrDrop` per `IrDropSide`:

- `PRE_PULSE` (R_on): reference = last sample of the rest before the pulse; measured = last pulse
  sample with `t <= ir_sample_time_s` after it (reference row excluded, ties by source row);
- `POST_PULSE` (R_off): reference = last sample under current; measured = last following-rest
  sample with `t <= post_pulse_sample_time_s` (None: same as `ir_sample_time_s`).

Each `IrDrop` exposes both `IrDropSample`s (source row, absolute and relative time, voltage,
current, branch capacity), signed `delta_voltage_v` and `delta_current_a`, `resistance_ohm =
|ΔV/ΔI|` and, when unavailable, a stable `reason` (`rest_before_missing`,
`pulse_samples_missing`, `rest_after_missing`, `no_sample_within_target`,
`manual_sample_invalid`, `zero_current_step`). Nothing is interpolated or replaced by a fallback.

`samples` maps `(event_id, side)` to the exact `source_row` of a real sample of that side (a click
keeps the row identity). The selected-sample IR correction of GITT_DELTA and of the Delacourt
fixed offset uses the same pre-pulse sample, including a manual one passed as `ir_samples` to
`plan_analysis`; an invalid manual row is an error record (`<label>_sample_invalid`). Those series
carry the dependency `ir_sample:<event_id>` (`ir_sample_dependencies`), so a manual change
invalidates exactly them. The post-pulse time and sample feed no D and have no dependency. Records
do not embed a `resistances` copy that could become stale.

## Rest-level OCV

`measure_rest_ocv(event_data, settings, samples=..., extrapolate=True)` measures every distinct rest
of the events once, in order of appearance, and returns one `RestOcv` per rest: the rest rows, the
relaxation origin (last sample under current), the sampled point and, when requested, the
LOG_TIME_2P fit and the extrapolated point. Both products are separate objects; the extrapolation
never replaces the sampled point. The extrapolation uses the same helper, fit interval
(`relaxation_fit_start_fraction`) and per-rest target (`OcvTargetPlan`) as the D results, so the
values always agree. Missing products carry a stable reason (`rest_empty`, `ocv_sample_invalid`,
`relaxation_reference_missing`, `relaxation_fit_unavailable: ...`, `not_requested`).

`sampled_ocv(..., source_row=)` and `plan_analysis(..., ocv_samples={rest_id: row})` select an
exact real sample of a rest as its sampled OCV; a row outside the rest gives no point (records:
`<label>_ocv_sample_invalid`), never a nearby sample. Sampled-OCV series carry the dependency
`ocv_sample:<rest_id>` (`ocv_sample_dependencies`), so a manual change invalidates exactly them.

## Persistence and export

`publish_snapshot` writes the complete current `ResultSnapshot` through the existing
`ResultStore`. Each publication is a new immutable generation with operation `gitt_ici`, input
dependencies, parameter/dependency hashes, snapshot content fingerprint, row counts and logical
schema `gregorithm.gitt_ici.results.long.v1`. `load_snapshot` first verifies the manifest and every
file checksum, reconstructs the typed records, and verifies the snapshot fingerprint. Publication
uses the existing atomic `current.json` replacement; cancellation never publishes a partial
snapshot or changes the preceding current generation.

The generation contains a complete technical table and a separate immutable export table built
only from accepted, non-stale current records having a finite D. Provisional, rejected, stale,
error and D-less records remain in the technical table but never enter the export table. Negative
D values remain diagnostic/export data while `AnalysisRecord.log_plottable` excludes them from a
logarithmic plot without adding an epsilon.

`AnalysisRecord` rejects NaN and infinity in its result, quality, coordinates and details with
`NonFiniteRecordError` (a `ValueError`): kernels report unavailable quantities as None, and a
non-finite number could not be persisted losslessly. Inside `run_batch` such an output becomes an
error record for that identity only, so one faulty event never blocks the publication of the
snapshot. `details` is normalized to plain JSON at construction (tuples become lists), so a
reopened snapshot compares equal to the published one.

In the export table, and therefore in both the CSV and the Parquet export, missing optional text
fields are empty strings because the canonical CSV path cannot preserve a null string; numeric
nulls stay null. The technical table keeps every original null.

`export_accepted_results` reuses the canonical verified exporter and produces the same long schema
as CSV or Parquet. If no accepted current D exists, it raises `NoAcceptedResultsError` before
creating the destination or its parent. Existing destinations are never overwritten.
