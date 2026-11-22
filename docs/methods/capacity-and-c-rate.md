# Capacity, normalization and C-rate

## Phase capacity and energy

Within a cycle, the charge-labelled half-cycle gives `Q_charge = Q⁺ − Q⁻` and the
discharge-labelled half-cycle gives `Q_discharge = Q⁻ − Q⁺`, from the half-cycle's final
directional totals (energy likewise). A missing phase is null; a present phase with zero net value
is 0. No step filtering is applied.

## Efficiencies

With charge → discharge order, `CE = 100 × Q_discharge / Q_charge` and
`EE = 100 × E_discharge / E_charge`. With discharge → charge order the ratio is inverted (charge
over discharge). An efficiency is null when a phase is missing, when energy is unavailable or when
the denominator is exactly zero. Values below 0 % or above 100 % are reported as computed.

**Cycle 0** (leading rest, if present) reports the directional rest contributions (normally 0 at
a 0 A rest threshold) and always has null efficiencies and C-rates.

## Normalization

Only with positive, finite values supplied by the user:

| From | Mass `m` (mg) | Area `A` (cm²) |
|---|---|---|
| capacity Q (Ah) | `Q × 10⁶ / m` → mAh/g | `Q × 10³ / A` → mAh/cm² |
| energy E (Wh) | `E × 10⁶ / m` → Wh/kg | `E × 10³ / A` → mWh/cm² |

Intermediate arithmetic is exact, so extreme but representable results do not overflow. A result
outside the float64 range is null (`NUMERIC_RESULT_UNREPRESENTABLE`).

## C-rate

- Reference `Q_ref`: the net capacity of the **second phase** of `reference_cycle` (default 2),
  i.e. the discharge capacity for charge → discharge order and the charge capacity otherwise. If
  that cycle or phase is missing, or `Q_ref ≤ 0`, all C-rates are null; no other cycle is used.
- For each CC segment `k` of a phase: `I_k` is the arithmetic mean of its sample currents and
  `D_k = t_last − t_first` its duration.
- Phase current: `I_CC = Σ D_k I_k / Σ D_k` (duration-weighted over all CC segments).
- `C_rate = |I_CC| / Q_ref`, reported separately for charge and discharge in each cycle.
- A phase without CC segments has a null C-rate.

Example: 1 A for 10 min and 0.5 A for 20 min give `I_CC = 2/3 A`; with `Q_ref = 2 Ah` the C-rate
is 0.333 C.
