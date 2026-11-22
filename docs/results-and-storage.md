# Results and storage

## Result store layout

```text
<caller_root>/
  sample-<sample_id>/
    current.json                  pointer to the most recently published generation
    .publish.lock                 per-sample publication lock (OS file lock)
    generations/<generation_id>/  one immutable result
      manifest.json
      ...result files...
    jobs/<job_id>.staging/        work in progress; renamed into generations/ on success
```

Publication is atomic: a generation appears only after every file has been written, fsynced and
checked against its manifest, and `current.json` is then replaced atomically. A failure or
cancellation leaves the previous state untouched. Symbolic links and Windows junctions inside the
store are rejected.

Limitation: on Windows the standard library cannot flush directory metadata, so renames are
atomic but their durability after a power loss is not guaranteed.

## Files per operation

| Operation | Files |
|---|---|
| `import_source` | `points/canonical.parquet`, `import_report.json` |
| `elaborate` | `points/elaborated.parquet`, `indexes/half_cycles.parquet`, `indexes/segments.parquet`, `overview/level-64.parquet`, `overview/level-4096.parquet`, `overview/level-262144.parquet`, `reports/elaboration.json` |
| `analyze_capacity` | `tables/capacity.parquet`, `tables/capacity_diagnostics.parquet`, `reports/capacity.json` |
| `detect_dcir` | `tables/dcir_events.jsonl`, `reports/dcir.json` |
| `publish_dcir_events` | `tables/dcir_events.json`, `tables/dcir_measurements.json` (if any), `reports/dcir.json` |
| `recalculate_cycles` | `points/elaborated.parquet`, `indexes/*.parquet`, `tables/capacity.parquet`, `tables/capacity_diagnostics.parquet`, `tables/dcir_events.json`, `tables/dcir_measurements.json` (if measurements were given), `tables/dqdv.parquet` and `tables/dqdv_reconstruction.parquet` (if an existing dQ/dV was given), `reports/cycle_correction.json` |
| `analyze_dqdv` | `tables/dqdv.parquet`, `tables/dqdv_reconstruction.parquet` (if reconstruction is enabled), `reports/dqdv.json` |

Every generation also contains `manifest.json`.

## Table columns

### `points/canonical.parquet`

| Column | Type | Unit |
|---|---|---|
| `time_s` | float64, not null | s |
| `current_a` | float64, not null | A (positive = charge) |
| `voltage_v` | float64, not null | V |

Rows are the retained source rows in order; their position is the `source_row` used downstream.

### `points/elaborated.parquet`

| Column | Type | Meaning |
|---|---|---|
| `time_s`, `current_a`, `voltage_v` | float64 | canonical measurements |
| `charge_capacity_ah`, `discharge_capacity_ah` | float64 | cumulative directional capacity, reset per half-cycle |
| `charge_energy_wh`, `discharge_energy_wh` | float64, nullable | cumulative directional energy; null if the half-cycle has negative voltage |
| `cycle` | int32 | cycle number (0 = leading rest) |
| `half_cycle` | int32 | half-cycle number (0 = leading rest) |
| `rest` | bool | `|I| <= rest threshold` |
| `step` | int8 | 0 rest, 1 CC, 2 CV, 3 other |
| `source_row` | uint64 | row of the canonical table |

### `indexes/half_cycles.parquet`

`half_cycle`, `cycle`, `first_row`, `last_row`, `first_direction` (+1 if half-cycle 1 is a
charge, -1 if a discharge) and the final totals `charge_capacity_ah`, `discharge_capacity_ah`,
`charge_energy_wh`, `discharge_energy_wh` (including zero-crossing contributions).

### `indexes/segments.parquet`

Contiguous runs of equal step type: `segment_id`, `first_row`, `last_row`, `first_time_s`,
`last_time_s`, `cycle`, `half_cycle`, `step`, `s4_portion` (CC portion identifier, -1 otherwise)
and `polarity` (current sign).

### `overview/level-<width>.parquet`

Same columns as the point table; contains only real rows (first, last and extrema of time,
current, voltage and capacity per bucket of `width` points). Intended for plotting.

### `tables/capacity.parquet`

| Column | Unit |
|---|---|
| `cycle` | – |
| `charge_capacity_ah`, `discharge_capacity_ah` | Ah |
| `coulombic_efficiency_pct` | % |
| `charge_energy_wh`, `discharge_energy_wh` | Wh |
| `energy_efficiency_pct` | % |
| `charge_c_rate`, `discharge_c_rate` | C |
| `charge_capacity_mah_g`, `discharge_capacity_mah_g` | mAh/g |
| `charge_capacity_mah_cm2`, `discharge_capacity_mah_cm2` | mAh/cm² |
| `charge_energy_wh_kg`, `discharge_energy_wh_kg` | Wh/kg |
| `charge_energy_mwh_cm2`, `discharge_energy_mwh_cm2` | mWh/cm² |

All value columns are nullable. `tables/capacity_diagnostics.parquet` has one row
(`cycle`, `column`, `code`) for each null cell.

### `tables/dqdv.parquet`

`cycle`, `half_cycle`, `phase` (`charge`/`discharge`), `anchor_source_row`, `segment_index`
(always 0; kept for compatibility), `voltage_mid_v` (V; interpolation grid nodes or histogram bin
centres depending on the method), `dqdv_raw_ah_per_v` and
`dqdv_smoothed_ah_per_v` (Ah/V; null when smoothing is disabled). The method and all settings are
recorded in the manifest parameters and in `reports/dqdv.json`.

### `tables/dqdv_reconstruction.parquet`

`cycle`, `half_cycle`, `phase`, `anchor_source_row`, `curve` (`measured`: the points the
dQ/dV curve was computed from and compared with; `reconstructed`: the integrated curve),
`point_index`, `capacity_ah` (Ah) and `voltage_v` (V), in measurement order.
`reconstruction_point_records` in the report counts both kinds. `reports/dqdv.json` (schema
version 2) lists the method, the publications to cite (`references`) and, under
`reconstruction`, one entry per curve (`cycle`, `half_cycle`, `phase`, `anchor_source_row`) with
`ape_percent`, `rmse_v`, `points` and a `code` that is
`DQDV_RECONSTRUCTION_UNAVAILABLE` when a metric is null.

### DCIR files

`dcir_events.jsonl` / `dcir_events.json` contain `DcirEvent` fields; `dcir_measurements.json`
contains `DcirMeasurement` fields (reference and measurement values, ΔI, ΔV, resistance, SOC,
samples used and diagnostics). See the docstrings of `DcirEvent` and `DcirMeasurement`.

## Manifest

`manifest.json` records: `schema_version`, `library_version`, `generation_id`, `operation`,
`created_utc`, `parameter_hash`, `dependency_hash`, `dependencies` (identifier and fingerprint of
each input actually used), `content_hash` (the result fingerprint), `files` (relative path, size
and SHA-256 of every file), `diagnostics`, `row_counts` and `logical_schemas` (schema identifiers
such as `gregorithm.capacity.v1`). `open_result` verifies all of it.

## Fingerprints

- The **canonical content hash** is a SHA-256 over the retained rows as little-endian float64
  triples. It depends on content only, not on file name, path or batch size.
- Every later result fingerprint combines the library version, the parameters that actually
  affect it and the fingerprints of its inputs. Display labels, paths, timestamps, Pending or
  Rejected DCIR events do not change scientific fingerprints.
- Capacity additionally reports separate fingerprints for base values, mass and area
  normalization and C-rate, so you can tell which outputs a parameter change affects.

Because the library version is part of each fingerprint, a new gregorithm version produces new
fingerprints even for identical numbers. Result schemas may change during the 0.x series.

## Diagnostic codes

| Code | Meaning |
|---|---|
| `IMPORT_MAPPING_REQUIRED` | header not recognized, or a mapped column is missing |
| `IMPORT_INVALID_TIME` | a surviving row has invalid time (`source_row` given) |
| `IMPORT_TIME_DECREASE` | elapsed time decreased |
| `IMPORT_INSUFFICIENT_ROWS` | fewer than two rows remain |
| `IMPORT_REMOVED_INVALID_MEASUREMENT` | warning: rows with invalid current/voltage were removed |
| `IMPORT_REMOVED_DUPLICATE_TIME` | warning: later duplicates of a timestamp were removed |
| `IMPORT_UNSUPPORTED_FORMAT` | extension is not .csv, .txt, .parquet or .pq |
| `IMPORT_PARSE_ERROR`, `IMPORT_IO_ERROR` | the source could not be parsed or read |
| `INSUFFICIENT_DISK_SPACE` | not enough free space for the operation |
| `INPUT_INVALID_PARAMETER` | an input parameter is invalid |
| `INTEGRATION_ENERGY_UNAVAILABLE` | energy unavailable (negative voltage in the half-cycle) |
| `CAPACITY_MISSING_PHASE` | the required charge or discharge phase is absent |
| `CAPACITY_ZERO_DENOMINATOR` | efficiency denominator is exactly zero |
| `CAPACITY_NO_ACTIVE_PAIR_CYCLE_ZERO` | cycle 0 has no efficiency |
| `NORMALIZATION_BASIS_MISSING` | no mass or area was supplied |
| `NUMERIC_RESULT_UNREPRESENTABLE` | the exact result does not fit in a float64 |
| `C_RATE_REFERENCE_UNAVAILABLE` | reference cycle missing or its capacity is not positive |
| `C_RATE_CC_UNAVAILABLE` | the phase has no CC portion |
| `DCIR_NO_CANDIDATES` | no pulse candidate was found |
| `DCIR_MISSING_PRE_PULSE_REFERENCE` | no real sample precedes the pulse; it cannot be accepted |
| `DCIR_ZERO_CURRENT_DIFFERENCE` | ΔI is exactly zero; resistance unavailable |
| `DCIR_TIME_AFTER_LAST_SAMPLE` | warning: requested time beyond the pulse, last sample used |
| `DCIR_SOC_UNAVAILABLE` | SOC cannot be estimated unambiguously |
| `DCIR_MANUAL_CORRECTION_BOUNDARY_INVALID` | manual correction boundaries are not two distinct samples |
| `DQDV_NO_VALID_SPAN` | a dQ/dV curve has no valid CC data for the selected method |
| `DQDV_RECONSTRUCTION_UNAVAILABLE` | the voltage curve cannot be reconstructed or its quality metrics are undefined |
| `OPERATION_CANCELLED` | the caller cancelled the operation |
| `PATH_INVALID`, `PATH_OUTSIDE_ROOT` | unsafe identifier or path, or a link escaping the root |
| `RESULT_JOB_EXISTS`, `RESULT_JOB_ACTIVE` | job identifier in use, or cleanup of a live job |
| `RESULT_GENERATION_EXISTS` | generation identifier already used for this sample |
| `RESULT_MANIFEST_INVALID`, `RESULT_FILE_MISSING`, `RESULT_FILE_UNEXPECTED`, `RESULT_FILE_CHECKSUM_MISMATCH`, `RESULT_POINTER_INVALID` | a stored result is corrupt or was modified |
| `RESULT_PUBLICATION_CONFLICT` | another process is publishing for the same sample |
| `STORAGE_IO_ERROR` | filesystem error; previous results remain current |
| `EXPORT_INVALID_SOURCE`, `EXPORT_DESTINATION_EXISTS`, `EXPORT_IO_ERROR` | export failures |
| `ESTIMATE_INVALID_REQUEST` | the generic `estimate` cannot be used; use a typed estimator |
