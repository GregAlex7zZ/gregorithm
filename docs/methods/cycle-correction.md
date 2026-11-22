# Cycle correction

A DCIR pulse of opposite polarity that lasts three or more samples is recognized as a separate
half-cycle. Cycle correction removes this effect, and only when explicitly requested.

- Spans: rows of valid **Accepted** events — the pulse rows of automatic events and the rows
  between the two boundary samples of manual events. Pending, Rejected and unbounded events are
  ignored. Overlapping or adjacent spans are merged.
- Half-cycle segmentation is recomputed on the rows outside the spans. Rows inside a span take
  the label of the neighbouring half-cycle; when the neighbours differ, the span is split at the
  first sample whose sign matches the following half-cycle.
- All samples are kept. Cumulative quantities and Capacity are recomputed for the new
  half-cycles, so the charge transported by the pulse is still counted.
- Step classification is unchanged.
- DCIR resistances are unchanged; event cycle context and optional SOC are refreshed.
- An existing dQ/dV table keeps its numerical values; only its cycle and half-cycle indices are
  re-mapped through each curve's anchor row.
- Running with `enabled=False` produces the uncorrected interpretation again.
