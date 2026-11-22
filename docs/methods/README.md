# Scientific methods

These pages define how gregorithm computes every quantity. They are the reference for reporting
methods in publications. Each page lists the rules, parameters and edge cases, with small
hand-checkable examples.

| Page | Topic |
|---|---|
| [Import and data quality](import-and-quality.md) | units, sign convention, row cleaning |
| [Cycles, half-cycles and steps](cycles-and-steps.md) | segmentation and rest/CC/CV classification |
| [Cumulative quantities](cumulative-quantities.md) | capacity and energy integration |
| [Capacity, normalization and C-rate](capacity-and-c-rate.md) | per-cycle results |
| [DCIR](dcir.md) | pulse detection, review and resistance |
| [Cycle correction](cycle-correction.md) | recalculating cycles around DCIR pulses |
| [dQ/dV](dqdv.md) | differential capacity |
| [GITT/ICI](gitt-ici.md) | canonical events, physical inputs and OCV contracts |
| [Provenance and diagnostics](provenance-and-diagnostics.md) | fingerprints and unavailable values |

General principles:

- Only real retained samples are used. No sample is interpolated, imputed or invented; the only
  exception is that integration splits an interval at the mathematical zero crossing of current.
- All values are IEEE 754 binary64 (float64) in s, A, V, Ah and Wh.
- An unavailable value is `null` with a diagnostic code, never `0`, `NaN` or infinity.
- Nothing is recomputed implicitly; every analysis is an explicit call with explicit parameters.
