# Differential capacity (dQ/dV)

## Common rules

1. For each selected cycle and half-cycle, keep only CC points outside excluded spans. Charge
   phases use `charge_capacity_ah`, discharge phases `discharge_capacity_ah`. Points stay in
   measurement order.
2. The optional voltage window `voltage_min_v` / `voltage_max_v` (E min / E max) restricts every
   method as described below. An empty bound means the measured extreme.
3. Values are in Ah/V and keep their mathematical sign: negative when capacity grows while voltage
   falls (normally discharge). Charge and discharge curves are separate.
4. `DqdvParameters.method` selects one of three methods. `interpolated_gradient` is the default.

A half-cycle without enough data for the method is reported with `DQDV_NO_VALID_SPAN`. No mass- or
area-normalized dQ/dV is computed.

## `interpolated_gradient` (default): classic Gregorithm workflow

This method reproduces the dQ/dV of the original Gregorithm application.

1. Drop points within `edge_trim_mv` (default 5 mV) of the maximum or minimum voltage of the
   half-cycle. This removes constant-voltage tails and cut-off edges. 0 disables the trim.
2. Sort the remaining points by voltage and keep the first point of equal voltages.
3. Choose the number of grid points `N`. It is either `interpolation_value` percent of the points
   left after step 1 (default 10 %, rounded) or a fixed count (`interpolation_mode = "count"`).
   `N` is clamped to `[3, interpolation_max_points]` (default 5000).
4. Interpolate Q(V) linearly onto `N` evenly spaced voltages between the measured extremes,
   clipped to the window.
5. Differentiate with `numpy.gradient`: second-order central differences inside the grid and
   one-sided differences at the two ends. Values are located at the grid nodes.
6. Optional Savitzky–Golay smoothing (window in grid points, polynomial order) is applied to the
   derivative and stored next to the raw values. Curves shorter than the window are left
   unsmoothed. The implementation uses NumPy only (Legendre basis for numerical stability,
   nearest complete window at the edges).

Limit: sorting folds any non-monotonic voltage segment onto a single voltage axis. Use this method
for essentially monotonic half-cycles.

## `histogram` and `histogram_weighted`: derivative-free voltage histograms

These methods implement E. Flores and S. Clark, "Simple and Comparable Computation of Incremental
Capacity Curves", *J. Electrochem. Soc.* **173**, 120520 (2026),
[doi:10.1149/1945-7111/ae7e5c](https://doi.org/10.1149/1945-7111/ae7e5c) (CC BY 4.0). The
underlying probability-density idea comes from X. Feng et al., *J. Power Sources* **232**, 209
(2013). Results computed with these methods should cite Flores and Clark.

The idea: when every measurement carries the same capacity, the number of points in a voltage bin
is proportional to the capacity passed in that bin. No differentiation takes place, so voltage
noise is not amplified, and flat plateaus give finite values.

Within the window, with `n` points, bin size `ΔV = histogram_bin_mv` and capacity range
`Q = max(q) − min(q)`:

1. `N_b = floor((max V − min V) / ΔV)` bins span the measured voltages.
2. `histogram`: count the points `c_j` in each bin (the last edge is inclusive) and compute
   `ρ_j = c_j / (n · w)`, where `w` is the actual bin width, then `I_j = ρ_j · Q`
   (Algorithm 1 of the paper).
3. `histogram_weighted`: each measurement interval contributes its capacity `q_i − q_{i−1}` to
   the bin containing its midpoint voltage, and `I_j = Σ Δq / w`. This is the capacity weighting
   proposed in the paper for non-uniform sampling or a changing C-rate. Intervals that are not
   between consecutive measurements contribute nothing; for example, rows removed as excluded
   spans or non-CC steps break the interval.
4. Values are located at the bin centres, in ascending voltage order. Empty bins hold 0.
5. Optional smoothing applies a Gaussian filter with standard deviation `gaussian_sigma_bins`
   (default 1 bin, the kernel reported by the authors). It is truncated at 4σ, with mirrored
   edges, and implemented in NumPy only.

Choosing the bin size. The bin size is the only parameter, and it sets the voltage resolution.
Small bins resolve subtle plateaus but leave sparse, noisy histograms. The bin should not be much
smaller than the typical voltage spacing between measurements. The paper uses 1 mV (the default)
for low-rate curves. Use the voltage reconstruction below to compare bin sizes objectively.

Limits:

- `histogram` assumes uniform capacity increments. When the sampling rate or the current changes
  within a half-cycle, or when excluded spans remove capacity from the middle, use
  `histogram_weighted`.
- Sparse data (for example GITT) produces histograms with many empty bins.

## Voltage reconstruction and quality metrics

When `reconstruction_enabled` is true (the default), each curve is integrated back into a voltage
curve and compared with the measured points used to compute it, following Flores and Clark:

1. Integration (Eq. 5): `q_j = q_{j−1} + (v_j − v_{j−1}) (I_j + I_{j−1}) / 2` along the ascending
   voltages of the curve. The smoothed values are used when smoothing is enabled. The curve starts
   at the first measured capacity of the half-cycle. It starts at the low-voltage end when the
   integral is positive and at the high-voltage end otherwise, because cumulative capacity only
   grows during a measurement.
2. APE (Eq. 6): `|Q − Q_rec| / Q × 100`, where `Q` is the measured and `Q_rec` the reconstructed
   capacity range.
3. RMSE (Eq. 7): the root mean square of the voltage differences at the reconstructed capacities
   inside the overlapping capacity interval. The measured curve is linearly interpolated there.

A perfect reconstruction gives APE = 0 % and RMSE = 0 V. A metric is null, with
`DQDV_RECONSTRUCTION_UNAVAILABLE`, when the measured capacity range is zero, when the curves do not
overlap, or when the curve has fewer than two values.

Example: a linear curve `V = 3 + q/2` (true dQ/dV = 2 Ah/V) from 0 to 2 Ah, with weighted
histogram bins of 1/64 V, gives `I = 2` Ah/V in every bin. Integrating from the first to the last
bin centre misses half a bin at each end. The result is `APE = 1.5625 %`, and `RMSE = 1/128 V`,
which is the half-bin shift.

A cumulative integral cannot reproduce non-monotonic voltage features such as local maxima before
a phase transition. Such features are smoothed out and show up as a larger RMSE.
