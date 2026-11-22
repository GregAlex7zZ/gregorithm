# Provenance and diagnostics

## Provenance

Each result generation records the gregorithm version, the operation, a fingerprint of the
parameters that affect it and the fingerprints of the inputs it actually used. The chain starts
from the content hash of the retained canonical rows: renaming or moving a file changes no
fingerprint, while changing one measured value changes them all.

Only scientific dependencies count. Changing the DCIR requested time changes the DCIR measurement
but not Capacity or dQ/dV; renumbering `DCIRn` labels or adding Rejected events changes nothing.
Results are never recalculated automatically: a parameter change requires a new explicit run,
which publishes a new generation.

Exports (CSV/Parquet) contain data only. Store the version, fingerprints, parameters and export
checksum alongside them (see the [user guide](../user-guide.md#7-export-and-reproducibility)).

## Diagnostics

A diagnostic has a stable code, a kind (`error`, `warning`, `unavailable`, `empty`), an English
message and optional context (cycle, source row, parameter, ...). When a value is null, the
diagnostic names the **first** failing check in a fixed order, while independent values of the
same row are still computed. Outputs never contain `NaN` or infinity; values outside the float64
range are null with `NUMERIC_RESULT_UNREPRESENTABLE`.

All codes are listed in [Results and storage](../results-and-storage.md#diagnostic-codes).
