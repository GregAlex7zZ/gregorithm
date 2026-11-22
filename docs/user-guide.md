# User guide

This guide walks through a complete analysis. Every code block uses only the public API
(`from gregorithm import ...`). For the scientific definitions behind each step, see
[Scientific methods](methods/README.md); for every output file and column, see
[Results and storage](results-and-storage.md).

## Key ideas

**Result store.** You choose a directory (the *caller root*). Each operation writes a new
*generation* below `<root>/sample-<sample_id>/generations/<generation_id>/` and returns a
`ResultRef` whose `path` points to it. Generations are immutable: nothing is ever overwritten.

**Identifiers.** Every request takes three identifiers (1–128 ASCII letters, digits, `.`, `_`, `-`):

| Identifier | Meaning | Rule |
|---|---|---|
| `sample_id` | the cell or sample being analysed | use the same value for all its operations |
| `job_id` | one run of an operation | unique among unfinished jobs of the sample |
| `generation_id` | the published result | unique within the sample, never reused |

Using the same string for `job_id` and `generation_id` (for example `"capacity-2026-10-25"`) is
the simplest convention.

**Fingerprints.** Each step returns a fingerprint that identifies its content. The next step
receives the previous result's path *and* fingerprint, and refuses to run if they do not match.
This chain makes it impossible to combine results of different inputs by mistake.

```text
import_source ──content_hash──► elaborate ──fingerprint──► analyze_capacity
                                          ├──fingerprint──► detect_dcir → review → measure → publish
                                          │                                  └──► recalculate_cycles
                                          └──fingerprint──► analyze_dqdv
```

## 1. Inspect and import a source

```python
from pathlib import Path

from gregorithm import ImportRequest, estimate_import, import_source, inspect_source

store = Path("results")
cell = "cell-01"
source = Path("cell-01.csv")

inspection = inspect_source(source)  # reads only a bounded preview
print(inspection.columns, inspection.mapping_confirmed)

request = ImportRequest(source, store, cell, "import-1", "import-1")
print(estimate_import(request).admitted)  # optional disk-space check
imported = import_source(request)
report = imported.report
print(report.retained_rows, report.removed_invalid_measurement, report.removed_duplicate_time)
```

If the header is not one of the two recognized forms (`time_s,current_a,voltage_v` and
`time_s,current_mA,voltage_V`), pass an explicit mapping, and text options when needed:

```python
from gregorithm import ColumnMapping, TextOptions, TimeFormat

mapping = ColumnMapping(
    time_column="Total Time",
    current_column="Current(mA)",
    voltage_column="Voltage(V)",
    time_unit="s",
    current_unit="mA",
    voltage_unit="V",
    time_format=TimeFormat.DURATION,
)
options = TextOptions(delimiter=";", decimal_separator=",", encoding="utf-8", skip_rows=0)
imported = import_source(
    ImportRequest(
        source, store, cell, "import-2", "import-2", mapping=mapping, text_options=options
    )
)
```

Supported units: time `s`, `ms`, `min`, `h`; current `A`, `mA`, `uA`; voltage `V`, `mV`.
`TimeFormat.DURATION` reads elapsed durations such as `179:22:08.25` (its unit must be `s`).

What import does, in this order: drop rows with missing/non-finite current or voltage; validate
time on the remaining rows; keep the first row of each repeated timestamp; abort if time decreases.
The canonical table is `imported.ref.path / "points" / "canonical.parquet"`. Import warnings are
in `report.diagnostics`; they describe the import only and are not carried into later results.

## 2. Elaborate: cycles, steps and cumulative quantities

```python
from gregorithm import CycleOrder, ElaborationParameters, ElaborationRequest, elaborate

elaborated = elaborate(
    ElaborationRequest(
        imported.ref.path / "points" / "canonical.parquet",
        imported.report.content_hash,
        store,
        cell,
        "elaboration-1",
        "elaboration-1",
        parameters=ElaborationParameters(
            rest_current_threshold_a=0.0,  # |I| <= threshold is rest
            cycle_order=CycleOrder.CHARGE_DISCHARGE,
            cc_relative_tolerance=0.01,  # 1 % current range for CC
            cv_absolute_tolerance_v=0.001,  # 1 mV voltage range for CV
        ),
    )
)
print(elaborated.rows, elaborated.half_cycles, elaborated.cycles)
```

The point table `points/elaborated.parquet` adds `cycle`, `half_cycle`, `rest`, `step`
(0 rest, 1 CC, 2 CV, 3 other), four cumulative columns and `source_row`. To read a slice for a
plot without loading the whole file:

```python
from gregorithm import WindowRequest, read_window

window = read_window(
    WindowRequest(
        elaborated.ref.path,
        start_time_s=0,
        end_time_s=3600,
        cycles=(1, 2),
        columns=("time_s", "voltage_v", "current_a"),
        point_budget=5000,
    )
)
```

With `point_budget`, at most that many *real* rows are returned, always including the first,
last, minimum and maximum of each requested numeric column. Precomputed real-point overviews are
also stored under `overview/` for very large files.

## 3. Capacity, efficiency, normalization and C-rate

```python
import pyarrow.parquet as pq

from gregorithm import CapacityParameters, CapacityRequest, analyze_capacity

capacity = analyze_capacity(
    CapacityRequest(
        elaborated.ref.path,
        elaborated.fingerprint,
        store,
        cell,
        "capacity-1",
        "capacity-1",
        CapacityParameters(
            cycle_order=CycleOrder.CHARGE_DISCHARGE,  # must match the elaboration
            reference_cycle=2,  # C-rate reference cycle
            active_material_mass_mg=5.0,  # optional -> mAh/g, Wh/kg
            electrode_area_cm2=1.13,  # optional -> mAh/cm², mWh/cm²
        ),
    )
)
table = pq.read_table(capacity.ref.path / "tables" / "capacity.parquet")
why = pq.read_table(capacity.ref.path / "tables" / "capacity_diagnostics.parquet")
```

`capacity_diagnostics.parquet` explains every `null` cell with its first causal code (for example
`CAPACITY_MISSING_PHASE` for an incomplete last cycle, or `NORMALIZATION_BASIS_MISSING` when no
mass was given).

## 4. DCIR: detect, review, measure, publish

### Detect candidates

```python
from gregorithm import DcirDetectionParameters, DcirDetectionRequest, PulsePolarity, detect_dcir

detection = DcirDetectionParameters(
    rest_before_s=300,
    pulse_s=30,
    rest_after_s=300,  # nominal durations
    rest_before_tolerance_pct=20,
    pulse_tolerance_pct=20,
    rest_after_tolerance_pct=20,
    polarity=PulsePolarity.BOTH,
)
detected = detect_dcir(
    DcirDetectionRequest(
        elaborated.ref.path,
        elaborated.fingerprint,
        store,
        cell,
        "dcir-detect-1",
        "dcir-detect-1",
        detection,
    )
)
```

Choose the durations to match your pulse protocol; a nominal rest of `0` disables that check.
Candidates are written as JSON lines. Load them back into `DcirEvent` objects:

```python
import json

from gregorithm import DcirEvent, EventSource, ReviewState


def load_events(path):
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row["source"] = EventSource(row["source"])
        row["state"] = ReviewState(row["state"])
        row["polarity"] = PulsePolarity(row["polarity"]) if row["polarity"] else None
        events.append(DcirEvent(**row))
    return tuple(events)


candidates = load_events(detected.ref.path / "tables" / "dcir_events.jsonl")
```

### Review

Detection never accepts anything. Inspect each candidate (for example by plotting it with
`read_window`) and decide:

```python
from gregorithm import review_event

accepted = review_event(candidates[0], ReviewState.ACCEPTED, acceptance_order=0)
# review_event(other_candidate, ReviewState.REJECTED) marks a candidate as rejected
```

To keep earlier decisions when detecting again with other parameters, pass them as
`existing_events=`; Accepted events are preserved and never duplicated.

### Measure

```python
from gregorithm import event_labels, measure_automatic_from_generation

labels = event_labels((accepted,))  # {"<event_id>": "DCIR1"}
measurement = measure_automatic_from_generation(
    elaborated.ref.path,
    elaborated.fingerprint,
    accepted,
    requested_time_s=10.0,
    label=labels[accepted.event_id],
)
print(measurement.resistance_ohm, measurement.delta_v_v, measurement.delta_i_a)
print(measurement.actual_time_s, measurement.soc_pct, measurement.diagnostics)
```

Manual measurements use four values (typed, or taken from real samples with
`ValueProvenance.SAMPLE`):

```python
from gregorithm import ManualValue, create_manual_event

manual = create_manual_event(
    ManualValue(0.0),
    ManualValue(3.70),  # I0 (A), V0 (V)
    ManualValue(-1.0),
    ManualValue(3.65),  # I1 (A), V1 (V)
    acceptance_order=1,
)
```

### Publish

```python
from gregorithm import DcirPublicationRequest, OperationId, publish_dcir_events

published = publish_dcir_events(
    DcirPublicationRequest(
        store,
        cell,
        "dcir-measure-1",
        "dcir-measure-1",
        elaborated.fingerprint,
        events=(accepted, manual.event),
        operation=OperationId.MEASURE_DCIR,
        parameters={"requested_time_s": 10.0},
        measurements=(measurement, manual),
        elaboration_path=elaborated.ref.path,
    )
)
```

Every Accepted event needs exactly one measurement. Sample-backed values are re-verified against
the stored data before publication.

## 5. Optional: recalculate cycles around DCIR pulses

A pulse of opposite polarity can split a half-cycle in two. Cycle correction excludes the rows of
Accepted pulses from half-cycle segmentation and publishes a corrected generation:

```python
from gregorithm import CycleCorrectionParameters, CycleCorrectionRequest, recalculate_cycles

corrected = recalculate_cycles(
    CycleCorrectionRequest(
        elaborated.ref.path,
        elaborated.fingerprint,
        store,
        cell,
        "correct-1",
        "correct-1",
        events=(accepted,),
        measurements=(measurement,),
        parameters=CycleCorrectionParameters(
            enabled=True, capacity=CapacityParameters(reference_cycle=2)
        ),
        existing_dqdv_path=None,  # or a dQ/dV result of the same elaboration to re-index
    )
)
```

The corrected generation contains corrected points and indexes plus recomputed Capacity, the DCIR
records with updated cycle context and, if given, the re-indexed dQ/dV table. Measured values and
resistances are unchanged. Run again with `enabled=False` to obtain the uncorrected view.

## 6. dQ/dV

```python
from gregorithm import DqdvParameters, DqdvRequest, accepted_correction_spans, analyze_dqdv

dqdv = analyze_dqdv(
    DqdvRequest(
        elaborated.ref.path,
        elaborated.fingerprint,
        store,
        cell,
        "dqdv-1",
        "dqdv-1",
        cycles=(1, 2, 10),  # empty = all cycles
        accepted_spans=accepted_correction_spans((accepted,)),  # exclude pulse rows
        parameters=DqdvParameters(
            interpolation_value=10.0, smoothing_enabled=True, sg_window_points=5, sg_poly_order=2
        ),
    )
)
curves = pq.read_table(dqdv.ref.path / "tables" / "dqdv.parquet")
rebuilt = pq.read_table(dqdv.ref.path / "tables" / "dqdv_reconstruction.parquet")
```

`DqdvParameters.method` selects how dQ/dV is computed (see
[the method page](methods/dqdv.md)):

```python
from gregorithm import DqdvMethod, InterpolationPoints

# Classic Gregorithm: interpolate onto 10 % of the points, then numpy.gradient, 3.4-4.1 V only.
classic = DqdvParameters(
    method=DqdvMethod.INTERPOLATED_GRADIENT,
    interpolation_mode=InterpolationPoints.PERCENT,
    interpolation_value=10.0,
    voltage_min_v=3.4,
    voltage_max_v=4.1,
)
# Derivative-free voltage histogram (Flores and Clark, 2026), 1 mV bins, Gaussian filter of 1 bin.
histogram = DqdvParameters(
    method=DqdvMethod.HISTOGRAM_WEIGHTED, histogram_bin_mv=1.0, smoothing_enabled=True
)
```

Every curve is also integrated back into a voltage curve. `reports/dqdv.json` lists the APE and
RMSE of each reconstruction, so you can compare methods and bin sizes objectively. Set
`reconstruction_enabled=False` to skip it. The histogram methods and the reconstruction come from
E. Flores and S. Clark, *J. Electrochem. Soc.* 173 (2026) 120520,
doi:10.1149/1945-7111/ae7e5c. `gregorithm.DQDV_HISTOGRAM_REFERENCE.citation()` returns the
citation for reports and figures.

The same functions also work on plain arrays, without a result store: `dqdv_histogram`,
`interpolated_gradient`, `gaussian_smooth`, `reconstruct_voltage_curve` and
`reconstruction_metrics`.

Use the corrected generation (`corrected.ref.path`, `corrected.fingerprint`) as input if you
applied cycle correction.

## 7. Export and reproducibility

```python
from gregorithm import ExportFormat, ExportRequest, export_table

receipt = export_table(
    ExportRequest(
        capacity.ref.path,
        "tables/capacity.parquet",
        Path("exports/cell-01-capacity.csv"),
        ExportFormat.CSV,
    )
)
print(receipt.rows, receipt.sha256)
```

Any Parquet table under `points/`, `indexes/`, `overview/` or `tables/` can be exported to CSV or
Parquet, optionally with a column subset (`columns=(...)`). Existing files are never overwritten.
Exports contain data only; keep a record of the provenance next to them:

```python
import gregorithm

record = {
    "gregorithm_version": gregorithm.__version__,
    "canonical_content_hash": imported.report.content_hash,
    "elaboration_fingerprint": elaborated.fingerprint,
    "capacity_fingerprint": capacity.fingerprint,
    "capacity_parameters": CapacityParameters(reference_cycle=2).as_dict(),
    "export_sha256": receipt.sha256,
}
```

Each generation's `manifest.json` also records the version, operation, parameter and dependency
hashes, file checksums and diagnostics.

## 8. Reopening results later

```python
from gregorithm import open_current, open_result

info = open_result(capacity.ref.path)  # verifies every checksum
latest = open_current(store / "sample-cell-01")  # the most recently published generation
print(info.operation, info.library_version, latest.ref.generation_id)
```

`current.json` points to the latest generation of the sample, whatever its operation. Keep the
`ResultRef` paths (or generation identifiers) of the results you need.

## 9. Progress, cancellation and errors

```python
from gregorithm import CancellationToken, OperationCancelled

token = CancellationToken()  # call token.cancel() from another thread to stop
request = ElaborationRequest(
    imported.ref.path / "points" / "canonical.parquet",
    imported.report.content_hash,
    store,
    cell,
    "elaboration-2",
    "elaboration-2",
)
try:
    elaborate(request, progress=print, cancel=token)
except OperationCancelled as error:
    print(error.diagnostic.code)
```

Operations raise structured exceptions with a `diagnostic` attribute: `SourceImportError`,
`StorageError`, `ExportError`, `ResourceEstimationError` and `OperationCancelled`. Invalid
parameters or inconsistent inputs raise `ValueError` with an explanatory message. A failed or
cancelled operation never changes the previously published results. Abandoned staging folders
(for example after a crash) can be removed with:

```python
from gregorithm import cleanup_job, sweep_stale

sweep_stale(store, older_than_s=24 * 3600)
```

## 10. Large files

All operations stream data in bounded batches (`batch_size`, default 65 536 rows) and write
intermediate results to disk inside the job folder, so memory use does not grow with file size.
Make sure the result store has enough free space (import refuses to start otherwise). Avoid
cloud-synchronised or network folders for the result store: atomic renames and file locks are
most reliable on a local disk.

## 11. Building applications on gregorithm

User interfaces should call the public API rather than re-implementing calculations.
`capacity_descriptor`, `dcir_descriptor`, `dqdv_descriptor` and `cycle_correction_descriptor`
return machine-readable parameter and output descriptions (labels, units, defaults, limits and
availability rules) that can drive settings panels and tables. Run operations in a worker thread
or process, forward `ProgressEvent`s to the UI and use a `CancellationToken` for a Cancel button.
