# DC internal resistance (DCIR)

## Detection

A candidate pulse is a CC segment (see [steps](cycles-and-steps.md)) whose polarity matches the
filter (charge, discharge or both) and whose duration lies in the inclusive band
`T × (1 − p/100) ≤ d ≤ T × (1 + p/100)` around the nominal duration (default 30 s ± 20 %). If the
nominal rest before (or after) is positive, a rest segment of matching duration must be directly
before (or after) the pulse (default 300 s ± 20 %). A nominal rest of 0 s disables that check.

A segment's duration is the time from its first sample to the first sample of the next segment.
A final rest reaching the end of the data uses its last sample; a CC segment at the very end of
the data is not eligible. One rest may serve two neighbouring pulses. There are no additional
current-step or separation thresholds.

Candidates are **Pending**: a user must accept or reject them. Repeated detection keeps Accepted
events and never duplicates them. A candidate without a preceding real sample can be detected but
not accepted.

## Automatic measurement

- Reference: current and voltage of the real sample immediately before the pulse
  (`I_ref`, `V_ref`), whatever its step type.
- Current: `I_3`, the current of the **third** pulse sample (default). With the optional
  `current_time_s` the current is that of the real pulse sample nearest to that time after V0
  instead, with the same tie and beyond-the-pulse rules as the voltage; it is stored as
  `current_requested_time_s` and re-verified at publication. Current is never interpolated.
- Voltage: `V_meas` of the real pulse sample nearest to `requested_time_s` after the reference
  sample V0 (default 10 s); ties choose the later sample. A request beyond the pulse uses the
  last pulse sample and adds `DCIR_TIME_AFTER_LAST_SAMPLE`. Voltage is never interpolated.
- Time origin: V0 is the last recorded instant before the current step, so it is time zero. With a
  sampling interval `dt` the first pulse sample sits at about `+dt`; for a 30 s pulse logged at
  1 s, `requested_time_s = 30` selects the last pulse sample (30 s after V0) without warning,
  and `requested_time_s = 10` selects the sample 10 s after V0. `actual_time_s` is measured from
  V0 as well.
- `ΔV = V_meas − V_ref`, `ΔI = I_3 − I_ref`, `R = |ΔV / ΔI|` in Ω. ΔV and ΔI are kept signed.
- Exactly zero `ΔI` makes R null (`DCIR_ZERO_CURRENT_DIFFERENCE`); the event stays Accepted.

## Manual measurement

Four values I0, V0, I1, V1 (typed or taken from real samples) give
`R = |(V1 − V0) / (I1 − I0)|`. Two optional boundary samples define a span for cycle correction.

## Optional state of charge

When the pulse forms its own half-cycle between two half-cycles of the same direction,
`f = Q1 / (Q1 + Q2 − Q_pulse)`, where Q1 and Q2 are the capacities of the flanking half-cycles and
Q_pulse the pulse capacity. SOC is `100 f` for charge flanks and `100 (1 − f)` for discharge
flanks. It is a relative, capacity-based estimate, not a calibrated SOC. Whenever the mapping is
ambiguous or a value is missing, SOC is null (`DCIR_SOC_UNAVAILABLE`); this never affects the
resistance.

## Labels

`event_labels` numbers Accepted events chronologically as `DCIR1`, `DCIR2`, ... Labels are for
display only and do not affect any fingerprint.
