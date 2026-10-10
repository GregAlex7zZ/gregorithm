# Scientific methods

This document defines the calculations performed by the public prepared-source API.
Operational thresholds are configurable. Software tests establish mathematical and
implementation consistency; they do not replace experimental validation.

## Measurements and time

Each row is a point measurement at its recorded elapsed time. Current is expressed in mA
and voltage in V. Acquisition cadence may vary within a file. Row position is retained by
the table itself; normal execution creates no row, trace or result identifiers.

Equal timestamps trigger a warning. The first complete row is retained and every later
row with that timestamp is removed before analysis. Distinct timestamps that move
backwards raise an error. Data are never globally reordered.

An isolated missing current or voltage may be estimated by linear interpolation only
when immediate finite neighbours exist and span no more than 10 s. The result is marked
as estimated and warns. Missing values are never replaced by zero, and unresolved gaps
are never connected by an integration interval.

## Capacity

Current is linear between two real or explicitly estimated endpoint measurements. For
an interval of duration `dt`:

```text
Q [mAh] = dt / 3600 * (I0 + I1) / 2
```

If current changes sign inside an interval, linear interpolation locates the zero
crossing and splits the contribution there. Positive contributions are charge; the
absolute value of negative contributions is discharge. Reported net capacity is signed
charge minus discharge. No samples are inserted into stored measurement tables.

Half-cycles follow consecutive current direction. Rest is associated with the preceding
half-cycle. A charge-then-discharge cycle closes on discharge; a discharge-then-charge
cycle closes on charge. A leading closing-polarity half-cycle is retained for capacity
but excluded from complete-cycle pairing.

## Energy

Current and voltage are interpolated separately as linear point signals. Energy uses the
exact integral of their product, which is quadratic within an interval:

```text
I(u) = I0 + (I1 - I0)u
V(u) = V0 + (V1 - V0)u
E [Wh] = dt / 3,600,000 * integral from 0 to 1 of I(u)V(u) du
```

Charge, discharge and net energy use the same sign split as capacity. Missing support
marks affected results partial rather than silently treating it as zero.

## Cycle summaries and normalization

Complete cycles pair their opening and closing half-cycles. Coulombic and energy
efficiencies divide discharge by charge and are reported as percentages. Missing
counterparts or invalid denominators remain unavailable with a reason.

Mass normalization divides additive results by user-supplied active mass in g. Area
normalization divides by user-supplied area in cm². Both can be requested together and
add columns while preserving all existing columns. No mass, area or nominal capacity is
inferred from filenames or measurements.

## C-rate

Signed C-rate is measured current divided by a fixed capacity reference. By default the
reference is the closing half-cycle of complete cycle 2, for either supported cycle
order. The user may supply another positive reference explicitly. Means are weighted by
covered time, not by row count.

## Step classification

Step classification is opt-in. Rest requires a finite current within the configurable
rest threshold. Missing current is unknown, never rest. Active regimes use bounded,
nonoverlapping signal windows:

- stable current with changing voltage: CC;
- changing current with stable voltage and a current signal: CV;
- stable current and stable voltage: CC;
- otherwise: variable or unknown according to available support.

Rest periods are steps. A `step_type` measurement column is produced only through the
explicit labelled-measurement view. Capacity and energy calculations do not invoke step
classification automatically.

## Pulse discovery and DCIR

Pulse discovery looks for measured rest-active-rest opportunities in both current
directions. Discovery does not accept an event, exclude it from cycling, identify a
specific experimental protocol or calculate resistance. Known expected rest and pulse
durations can be supplied with independent tolerances; a nominal rest duration of zero
may be absent. Pulse stability uses the complete real-current range and a configurable
relative tolerance, without nominal current.

Manual DCIR selection accepts one or more explicit pulse boundaries or consecutive cycle
scopes. Automatic and manual selection both measure every eligible rest-to-current
transition in the selected support. Time zero is the last real sample of the preceding
rest. One requested delay is evaluated per call.

The target sample is the real sample exactly at the requested time when present,
otherwise the first real sample after it, or the last real sample before it if no later
sample exists. No interpolation is used for resistance. With current converted from mA
to A:

```text
R [ohm] = (V_target - V_rest) / (I_target_A - I_rest_A)
```

Signed resistance and timing error are retained. Detection, user acceptance/exclusion
and resistance measurement remain separate result layers.

## dQ/dV

The corrected historical method is the default. A fixed voltage-step grid is an explicit
alternative. Only continuous CC portions matching the requested direction are eligible;
CV portions are always excluded. Separate CC steps and portions separated by missing or
unsupported measurements are processed independently.

Capacity is interpolated as a function of voltage on a uniform voltage grid. When
multiple capacity values map to the same voltage, their mean is used. Optional
Savitzky-Golay smoothing applies to capacity before differentiation. The reported
derivative retains its sign. Peak selection uses the maximum for charge and minimum for
discharge; equal extrema are marked ambiguous. Curve comparison reports the selected
curve minus the reference curve on common supported voltage.

## Plotting and multi-file execution

Point reduction applies only to displayed figures. Scientific tables and exports retain
their calculated support. A point budget applies to the complete figure, including all
series. Progressive figures expose updates while calculation is running.

Multi-file execution keeps one prepared handle per complete filename, rejects duplicate
names, divides the configured managed-memory budget across active workers and isolates
per-file failures. Serial execution is the default. Results do not depend on source batch
size or worker count.

## Scope

The pulse API can describe generic rest-current-rest measurements, but this release does
not calculate GITT or ICI diffusion parameters and does not assign those protocol labels.
