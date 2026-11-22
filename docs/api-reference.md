# API reference

Everything listed here is importable from the top-level package (`from gregorithm import ...`).
Each object has a detailed docstring describing its parameters, results and limits; use
`help(gregorithm.<name>)` or your editor's tooltips. Names not listed here (for example those
starting with `_`) are internal and may change without notice.

All request classes are frozen dataclasses; analysis functions accept optional keyword arguments
`progress=` (a callable receiving `ProgressEvent`) and `cancel=` (a `CancelToken`). The
`fault_hook=` argument exists only for testing failure handling and should be left unset.

## Import

Inspect sources and import them into canonical form.

| Name | Kind | Summary |
|---|---|---|
| `ColumnMapping` | class | Explicit assignment of source columns and units to canonical roles. |
| `ImportAdmission` | class | Disk-space estimate for an import, returned by `estimate_import`. |
| `ImportReport` | class | Outcome of an import. |
| `ImportRequest` | class | Parameters of `import_source` and `estimate_import`. |
| `ImportResult` | class | Published canonical import: result reference `ref` and `report`. |
| `SourceImportError` | exception | Structured import or inspection failure; `diagnostic` holds the reason. |
| `SourceInspection` | class | Bounded preview of a source file, produced by `inspect_source`. |
| `SourceKind` | enum | Recognized source containers, derived from the file extension. |
| `TextOptions` | class | Parsing options for CSV and TXT sources. |
| `TimeFormat` | enum | How the mapped time column is written. |
| `estimate_import` | function | Estimate the disk space an import needs, before writing anything. |
| `import_source` | function | Import a source file into an immutable canonical generation. |
| `inspect_source` | function | Preview a source without writing anything or changing it. |
| `CANONICAL_SCHEMA` | constant | Arrow schema of the canonical measurement table. |

## Elaboration

Half-cycles, cycles, steps, cumulative quantities and windowed reading.

| Name | Kind | Summary |
|---|---|---|
| `CycleOrder` | enum | Order in which half-cycles are paired into a cycle. |
| `ElaboratedArrays` | class | In-memory result of `elaborate_arrays`. |
| `ElaborationParameters` | class | Settings of the data elaboration. |
| `ElaborationRequest` | class | Parameters of `elaborate`. |
| `ElaborationResult` | class | Published elaboration result. |
| `Step` | enum | Step type stored in the `step` column: 0 rest, 1 CC, 2 CV, 3 other. |
| `WindowRequest` | class | Parameters of `read_window`. |
| `classify_steps` | function | Classify each point as rest, CC, CV or other (pure function). |
| `elaborate` | function | Elaborate a canonical import and publish the result. |
| `elaborate_arrays` | function | Apply segmentation, step classification and integration in memory. |
| `integrate_cumulatives` | function | Integrate capacity and energy per half-cycle (pure function). |
| `overview_indices` | function | Return indices of the first, last, minimum and maximum rows per bucket. |
| `read_window` | function | Read elaborated points, optionally filtered by time and cycle. |
| `segment_half_cycles` | function | Label half-cycles and cycles from a current array (pure function). |

## Capacity

Per-cycle capacity, energy, efficiency, normalization and C-rate.

| Name | Kind | Summary |
|---|---|---|
| `CapacityParameters` | class | Settings of the Capacity analysis. |
| `CapacityRequest` | class | Parameters of `analyze_capacity`. |
| `CapacityResult` | class | Published Capacity result. |
| `analyze_capacity` | function | Compute per-cycle Capacity from an elaboration result and publish it. |
| `capacity_descriptor` | function | Describe the Capacity analysis for user interfaces. |
| `capacity_rows_from_halves` | function | Aggregate half-cycle totals into per-cycle Capacity rows (pure). |
| `CAPACITY_SCHEMA` | constant | Arrow schema of the Capacity table. |

## DCIR

Pulse detection, review, measurement and publication.

| Name | Kind | Summary |
|---|---|---|
| `DcirDetectionParameters` | class | Pulse-detection criteria. |
| `DcirDetectionRequest` | class | Parameters of `detect_dcir`. |
| `DcirEvent` | class | A DCIR candidate or reviewed event. |
| `DcirMeasurement` | class | Measured DCIR of one event. |
| `DcirPublicationRequest` | class | Parameters of `publish_dcir_events`. |
| `DcirResult` | class | Published DCIR result. |
| `EventSource` | enum | Origin of a DCIR event: automatic detection or manual entry. |
| `ManualValue` | class | A manually entered DCIR value, typed or taken from a real sample. |
| `PulsePolarity` | enum | Pulse current sign: charge (positive) or discharge (negative). |
| `ReviewState` | enum | Review state of a DCIR event: pending, accepted or rejected. |
| `SampleRef` | class | A real measured sample identified by its canonical source row. |
| `SegmentRecord` | class | One contiguous step segment, as stored in ``indexes/segments.parquet``. |
| `SocContext` | class | Capacities used for the optional SOC estimate. |
| `ValueProvenance` | enum | Whether a manual DCIR value was typed or taken from a real sample. |
| `accepted_event_fingerprint` | function | Return the fingerprint of the Accepted events and their spans. |
| `create_manual_event` | function | Create an Accepted manual DCIR event and its measurement. |
| `dcir_descriptor` | function | Describe DCIR detection (`OperationId.DETECT_DCIR`) or measurement (`OperationId.MEASURE_DCIR`) for user interfaces. |
| `detect_candidates` | function | Find DCIR candidates in an ordered sequence of segments (pure). |
| `detect_dcir` | function | Detect DCIR pulse candidates in an elaboration result and publish them. |
| `event_labels` | function | Return chronological display labels ``DCIR1``, ``DCIR2``, ... for Accepted events. |
| `measure_automatic` | function | Measure DCIR of an Accepted automatic event from supplied samples (pure). |
| `measure_automatic_from_generation` | function | Measure DCIR of an Accepted automatic event. |
| `publish_dcir_events` | function | Publish reviewed DCIR events and measurements immutably. |
| `review_event` | function | Return a copy of `event` with a new review state. |

## Cycle correction

Recalculate cycles around Accepted DCIR pulses.

| Name | Kind | Summary |
|---|---|---|
| `CorrectedArrays` | class | In-memory result of `recalculate_cycle_arrays`. |
| `CycleCorrectionParameters` | class | Settings of the cycle correction. |
| `CycleCorrectionRequest` | class | Parameters of `recalculate_cycles`. |
| `CycleCorrectionResult` | class | Published corrected generation. |
| `accepted_correction_spans` | function | Return the merged inclusive source-row spans used for correction. |
| `cycle_correction_descriptor` | function | Describe the cycle-correction operation for user interfaces. |
| `recalculate_cycle_arrays` | function | Recompute half-cycle and cycle labels with spans excluded (pure). |
| `recalculate_cycles` | function | Apply (or remove) cycle correction and publish a corrected generation. |

## dQ/dV

Differential capacity curves.

| Name | Kind | Summary |
|---|---|---|
| `DQDV_HISTOGRAM_REFERENCE` | constant | Flores and Clark (2026): source of the histogram methods and the reconstruction. |
| `DqdvCurve` | class | dQ/dV curve of one half-cycle. |
| `DqdvMethod` | enum | `interpolated_gradient` (default), `histogram`, `histogram_weighted`. |
| `DqdvParameters` | class | Settings of the dQ/dV analysis. |
| `DqdvPhase` | enum | Phase of a dQ/dV curve: charge or discharge. |
| `DqdvReconstruction` | class | Voltage curve rebuilt by integrating a dQ/dV curve, with APE and RMSE. |
| `DqdvRequest` | class | Parameters of `analyze_dqdv`. |
| `DqdvResult` | class | Published dQ/dV result: reference, number of curves and points, fingerprint. |
| `DqdvSpan` | class | One piece of a dQ/dV curve. |
| `InterpolationPoints` | enum | Interpolation grid size given as `percent` or `count`. |
| `analyze_dqdv` | function | Compute dQ/dV curves for selected cycles and publish them. |
| `compute_dqdv` | function | Compute dQ/dV curves from elaborated point arrays (pure function). |
| `dqdv_descriptor` | function | Describe the dQ/dV analysis for user interfaces. |
| `dqdv_histogram` | function | Derivative-free dQ/dV as a voltage histogram (pure function). |
| `dqdv_method_references` | function | Publications to cite for results computed with given parameters. |
| `gaussian_smooth` | function | Gaussian filter implemented with NumPy only. |
| `interpolated_gradient` | function | Classic Gregorithm interpolate-then-gradient dQ/dV (pure function). |
| `interpolation_point_count` | function | Grid size of the `interpolated_gradient` method. |
| `reconstruct_voltage_curve` | function | Integrate a dQ/dV curve back into a voltage curve (pure function). |
| `reconstruction_metrics` | function | APE and RMSE between measured and reconstructed curves. |
| `savgol_interp` | function | Savitzky-Golay smoothing implemented with NumPy only. |

## Export

Export result tables to CSV or Parquet.

| Name | Kind | Summary |
|---|---|---|
| `ExportError` | exception | Structured export failure; `diagnostic` holds the reason code. |
| `estimate_export` | function | Estimate the bytes an export will write. |
| `export_table` | function | Export one table of a published result to a new CSV or Parquet file. |

## Results and storage

Open, verify and clean up result generations.

| Name | Kind | Summary |
|---|---|---|
| `GenerationManifest` | class | Contents of `manifest.json`, the identity card of a result generation. |
| `ResultStore` | class | Low-level access to a caller-owned result root. |
| `StagingGeneration` | class | A staging directory owned by one running job. |
| `StorageError` | exception | Structured storage failure; `diagnostic` holds the reason code. |
| `cleanup_job` | function | Remove an abandoned staging job, for example after a cancelled operation. |
| `open_current` | function | Open the generation referenced by `<sample_root>/current.json`. |
| `open_result` | function | Open a published generation directory and verify every file. |
| `sweep_stale` | function | Remove abandoned staging jobs and pointer temporaries older than `older_than_s`. |

## Contracts

Shared request, reference and descriptor types.

| Name | Kind | Summary |
|---|---|---|
| `AnalysisDescriptor` | class | Self-description of an analysis: parameters, outputs and warnings. |
| `CleanupReport` | class | Paths removed and paths deliberately skipped by a cleanup operation. |
| `DependencyRef` | class | Identifier and fingerprint of one input a result depends on. |
| `ExportFormat` | enum | File formats supported by `export_table`: CSV and Parquet. |
| `ExportReceipt` | class | Description of a completed export. |
| `ExportRequest` | class | Request to export one table of a published result to an external file. |
| `LibraryInfo` | class | Library identification; `version` is the installed gregorithm version. |
| `LiteratureReference` | class | Bibliographic reference of a published method; `citation()` and `url`. |
| `OperationId` | enum | Stable identifiers of the operations recorded in result manifests. |
| `OperationRequest` | class | Generic operation description used only by the generic `estimate` call. |
| `ParameterCondition` | class | Rule stating when a descriptor parameter is relevant (e.g. only for one dQ/dV method). |
| `ResourceEstimate` | class | Byte estimate for an operation that writes files. |
| `ResultInfo` | class | Validated information about a reopened result generation. |
| `ResultRef` | class | Reference to one immutable published result generation. |

## Diagnostics

Structured diagnostic records and codes.

| Name | Kind | Summary |
|---|---|---|
| `Diagnostic` | class | One structured diagnostic record. |
| `DiagnosticCode` | enum | Stable diagnostic codes. |
| `DiagnosticKind` | enum | Category of a diagnostic. |
| `first_cause` | function | Return the first non-None diagnostic, or None when all candidates are None. |

## Progress and cancellation

Progress events and cooperative cancellation.

| Name | Kind | Summary |
|---|---|---|
| `CancelToken` | class | Protocol for objects that tell an operation whether it should stop. |
| `CancellationToken` | class | Thread-safe, caller-owned cancellation token. |
| `OperationCancelled` | exception | Raised when an operation stops because its cancel token was set. |
| `ProgressEvent` | class | Immutable progress notification emitted by streaming operations. |
| `ProgressSink` | class | Protocol for callables that receive `ProgressEvent` objects. |
| `check_cancelled` | function | Raise `OperationCancelled` if `token` is set; do nothing when it is None. |
| `report_progress` | function | Deliver `event` to `sink` when a sink was supplied. |

## Resources

Disk-space admission.

| Name | Kind | Summary |
|---|---|---|
| `ResourceEstimationError` | exception | Raised when a resource estimate is impossible or space is insufficient. |
| `estimate` | function | Reject generic resource estimates, which cannot be computed reliably. |
| `require_free_space` | function | Check that `root` has room for `resource_estimate` and return free bytes. |

## Identity

Deterministic parameter and dependency hashing.

| Name | Kind | Summary |
|---|---|---|
| `canonical_parameter_bytes` | function | Encode a JSON-like value as unambiguous, type-tagged UTF-8 bytes. |
| `hash_dependencies` | function | Return the SHA-256 fingerprint of a set of dependency references. |
| `hash_parameters` | function | Return the SHA-256 fingerprint of a parameter mapping. |

## Version

`gregorithm.__version__` is the installed version; it is recorded in every result manifest.
