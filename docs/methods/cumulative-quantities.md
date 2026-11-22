# Cumulative quantities

Four non-negative cumulative quantities are stored per point and reset at each half-cycle:
`charge_capacity_ah`, `discharge_capacity_ah`, `charge_energy_wh` and `discharge_energy_wh`. Net
values are charge minus discharge.

For two adjacent retained points with time step `dt` (s) and no sign change (trapezoidal rule):

```text
ΔQ = dt × (I0 + I1) / 7200            [Ah]
ΔE = dt × (V0·I0 + V1·I1) / 7200      [Wh]
```

A positive increment adds to the charge quantity and a negative one to the discharge quantity.
If current changes sign within an interval, the interval is split at the zero crossing
`α = |I0| / (|I0| + |I1|)`: the left part is attributed to the sign of `I0`, the right part to the
sign of `I1`. No point is added to the table. When the crossing coincides with a half-cycle
boundary, the left part is included in the final totals of the previous half-cycle.

The measured current is integrated everywhere, including rest points and brief sign excursions
that did not start a new half-cycle.

If any point of a half-cycle has negative voltage, the energy of that whole half-cycle is
unavailable (null); capacity remains available.

Example: `dt = 3600 s` and `I0 = I1 = 1 A` give 1 Ah of charge capacity; at `V = 3.7 V` this is
3.7 Wh of charge energy.
