"""Bounded vectorized annotation for the common, unexcluded full-trace path."""

from dataclasses import replace

import numpy as np
import pyarrow as pa

from .execution import CancellationToken
from .segmentation import annotated_batches


def _local(values, labels, previous_label, previous_value):
    """Accumulate contiguous label groups while carrying the previous batch group total; reset only at original assignment boundaries."""
    if not len(labels):
        return values.copy()
    sums = np.cumsum(values, axis=0)
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    ends = np.r_[starts[1:], len(labels)]
    origins = np.zeros((len(starts), values.shape[1]))
    origins[1:] = sums[starts[1:] - 1]
    result = sums - np.repeat(origins, ends - starts, axis=0)
    if labels[0] == previous_label:
        result[: ends[0]] += previous_value
    return result


def full_batches(trace, index, config, cancellation=None, store=None, diagnostic_callback=None):
    """Vectorize the unexcluded full-trace integration path with bounded arrays and the same ownership rules as the reference contribution path.

    Parameters
    ----------
    trace
        BatteryTrace source handle; raw signals are never rewritten.
    index
        Disk-backed SegmentIndex captured for the scientific assignment revision.
    config
        Validated immutable AnalysisConfig or DQDVConfig as appropriate; None uses operational defaults.
    cancellation
        CancellationToken checked at cooperative boundaries; None creates a local token.
    store
        Bounded operation-specific value described by the function contract and API reference.
    diagnostic_callback
        Bounded operation-specific value described by the function contract and API reference."""
    from .integration import calculate_increments

    token = cancellation or CancellationToken()
    previous = None
    previous_info = np.zeros(4, dtype=np.int64)
    total = np.zeros(4)
    half_total = np.zeros(4)
    cycle_total = np.zeros(4)
    step_total = np.zeros(4)
    partial = False
    previous_step = 0
    cfg = replace(
        config,
        batch_rows=min(config.batch_rows, 65536, max(1, config.managed_memory_bytes // 2048)),
    )
    for b in annotated_batches(trace, index, cfg):
        token.check()
        n = b.num_rows
        if not n:
            continue

        def col(k):
            """Internal helper for full_batches.col; receives only the bounded state owned by its containing operation."""
            return b.column(k).to_numpy(zero_copy_only=False)

        t, i, v = [col(k).astype(float) for k in ("time_s", "current_mA", "voltage_V")]
        h, cy, main, sections, step, ids = [
            col(k)
            for k in (
                "half_cycle_id",
                "cycle_id",
                "main_sign",
                "time_section_id",
                "step_id",
                "sample_id",
            )
        ]
        inc, _ = calculate_increments(t, i, v, config, previous)
        lh = np.r_[previous_info[0], h[:-1]]
        lc = np.r_[previous_info[1], cy[:-1]]
        lm = np.r_[previous_info[2], main[:-1]]
        ls = np.r_[previous_info[3], sections[:-1]]
        reset = sections != ls
        if previous is None:
            reset[0] = False
        inc[reset] = np.nan
        qvalid = np.isfinite(inc[:, 0])
        evalid = np.isfinite(inc[:, 2])
        safe = np.where(np.isfinite(inc), inc, 0.0)
        cumulative = np.cumsum(safe, axis=0) + total
        total = cumulative[-1].copy()
        owned = safe.copy()
        changed = (h != lh) & (main != lm)
        owned[changed & (main == 1), 1] = 0.0
        owned[changed & (main == 1), 3] = 0.0
        owned[changed & (main == -1), 0] = 0.0
        owned[changed & (main == -1), 2] = 0.0
        local = _local(owned, h, previous_info[0], half_total)
        half_total = local[-1].copy()
        cycle_values = safe.copy()
        cycle_values[cy != lc] = owned[cy != lc]
        cycle_values = _local(cycle_values, cy, previous_info[1], cycle_total)
        cycle_total = cycle_values[-1].copy()
        step_values = _local(safe, step, previous_step, step_total)
        step_total = step_values[-1].copy()
        capacity = np.where(main == 1, local[:, 0], np.where(main == -1, local[:, 1], 0.0))
        owners = []
        left_current = np.r_[previous[1] if previous else i[0], i[:-1]]
        for sign in (1, -1):
            right = main == sign
            left = (~right) & (lm == sign) & (~reset)
            fallback_left = (
                (~right) & (~left) & (~reset) & (left_current * sign > 0) & (i * sign <= 0)
            )
            owners.append(
                (
                    np.where(right, h, np.where(left | fallback_left, lh, h)),
                    np.where(right, cy, np.where(left | fallback_left, lc, cy)),
                )
            )
        dt = np.diff(np.r_[previous[0] if previous else t[0], t])
        if store is not None:
            cc_mask = (step == np.r_[previous_step, step[:-1]]) & (
                np.asarray(b.column("mode").to_pylist()) == "CC"
            )
            store.add_arrays(inc, dt, owners, step, cc_mask)
        invalid = ~(qvalid & evalid)
        partial_values = np.maximum.accumulate(invalid | partial)
        partial = bool(partial_values[-1])
        if diagnostic_callback:
            starts = np.flatnonzero(invalid & np.r_[True, ~invalid[:-1]])
            ends = np.flatnonzero(invalid & np.r_[~invalid[1:], True])
            for a, z in zip(starts, ends):
                diagnostic_callback(
                    "invalid_interval",
                    int(ids[a]),
                    int(ids[z]),
                    "Q or E adjacent interval not integrated; no zero replacement",
                )
        for k, name in enumerate(("charge_mAh", "discharge_mAh", "charge_Wh", "discharge_Wh")):
            for prefix, values in (
                ("d_", inc[:, k]),
                ("global_", cumulative[:, k]),
                ("cycling_", cumulative[:, k]),
                ("excluded_d_", np.zeros(n)),
                ("cycle_", cycle_values[:, k]),
                ("step_", step_values[:, k]),
            ):
                b = b.append_column(prefix + name, pa.array(values))
        for name, values in (
            ("global_net_mAh", cumulative[:, 0] - cumulative[:, 1]),
            ("global_throughput_mAh", cumulative[:, 0] + cumulative[:, 1]),
            ("global_net_Wh", cumulative[:, 2] - cumulative[:, 3]),
            ("global_throughput_Wh", cumulative[:, 2] + cumulative[:, 3]),
            ("capacity_mAh", capacity),
            ("q_interval_valid", qvalid),
            ("e_interval_valid", evalid),
            ("coverage_partial", partial_values),
        ):
            b = b.append_column(name, pa.array(values))
        previous = (float(t[-1]), float(i[-1]), float(v[-1]))
        previous_step = int(step[-1])
        previous_info = np.array([h[-1], cy[-1], main[-1], sections[-1]], dtype=np.int64)
        yield b
