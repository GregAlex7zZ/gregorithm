"""Analytic integrals of pointwise linear I,V with original interval ownership."""

import math
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .execution import CancellationToken
from .segmentation import annotated_batches


def interval_parts(t0, i0, v0, t1, i1, v1, method="linear_signals", lo=0.0, hi=1.0):
    """Integrate one original interval, optionally clipped to fractional bounds lo/hi. Return positive-current Q, negative-current Q, positive-current E, negative-current E in mAh/mAh/Wh/Wh. Energy uses the analytic product of pointwise linear I and V. Invalid Q/E support returns NaN independently; dt must be positive.

    Parameters
    ----------
    t0
        Original left endpoint timestamp in seconds.
    i0
        Original left endpoint current in mA.
    v0
        Original left endpoint voltage in V.
    t1
        Original right endpoint timestamp in seconds.
    i1
        Original right endpoint current in mA.
    v1
        Original right endpoint voltage in V.
    method
        Signal integration or DCIR sampling policy appropriate to this function; see its contract.
    lo
        Lower interval fraction or query boundary, as specified by this operation.
    hi
        Upper interval fraction or query boundary, as specified by this operation."""
    dt = t1 - t0
    if not (
        math.isfinite(t0)
        and math.isfinite(t1)
        and dt > 0
        and math.isfinite(i0)
        and math.isfinite(i1)
    ):
        return np.nan, np.nan, np.nan, np.nan
    if method == "left":
        ia = ib = i0
        va = vb = v0
    elif method == "right":
        ia = ib = i1
        va = vb = v1
    else:
        ia = i0
        ib = i1
        va = v0
        vb = v1
    di = ib - ia
    dv = vb - va
    zero = -ia / di if di != 0 else -1.0
    cuts = [lo, hi]
    if lo < zero < hi:
        cuts = [lo, zero, hi]
    qc = qd = ec = ed = 0.0
    ev = math.isfinite(va) and math.isfinite(vb)
    for a, b in zip(cuts[:-1], cuts[1:]):
        q = dt * (ia * (b - a) + di * (b * b - a * a) / 2) / 3600
        e = (
            dt
            * (
                va * ia * (b - a)
                + (va * di + ia * dv) * (b * b - a * a) / 2
                + dv * di * (b**3 - a**3) / 3
            )
            / 3600000
            if ev
            else np.nan
        )
        if ia + di * (a + b) / 2 >= 0:
            qc += q
            ec += e
        else:
            qd -= q
            ed -= e
    if not ev:
        ec = ed = np.nan
    return qc, qd, ec, ed


def _kernel(t, i, v, method, max_gap):
    """Reference per-interval integration scan using float64 and configured gap invalidation."""
    n = len(t)
    out = np.zeros((n, 4), dtype=np.float64)
    for j in range(1, n):
        if max_gap > 0 and t[j] - t[j - 1] > max_gap:
            out[j, :] = np.nan
        else:
            out[j, :] = interval_parts(t[j - 1], i[j - 1], v[j - 1], t[j], i[j], v[j], method)
    return out


_FAST = None


def integration_backend(name):
    """Select the reference scalar scan or cached Numba scan without fastmath. auto falls back only when Numba is unavailable; explicit numba requires that dependency.

    Parameters
    ----------
    name
        Registry/table/preset name as documented for the containing API."""
    global _FAST
    if name == "numpy":
        return _kernel, "numpy"
    try:
        import numba

        if _FAST is None:
            # Compile scalar helper and state scan without fastmath.
            scalar = numba.njit(cache=True, nogil=True, fastmath=False)(interval_parts)

            def scan(t, i, v, method, max_gap):
                """Scan bounded original data while carrying the containing operation state across batches."""
                n = len(t)
                out = np.zeros((n, 4), dtype=np.float64)
                for j in range(1, n):
                    if max_gap > 0 and t[j] - t[j - 1] > max_gap:
                        out[j, :] = np.nan
                    else:
                        out[j, :] = scalar(t[j - 1], i[j - 1], v[j - 1], t[j], i[j], v[j], method)
                return out

            _FAST = numba.njit(cache=True, nogil=True, fastmath=False)(scan)
        return _FAST, "numba"
    except ImportError:
        if name == "numba":
            raise
        return _kernel, "numpy"


def calculate_increments(t, i, v, config, previous=None):
    """Calculate bounded per-row Q/E increments attributed to right endpoints. Optionally prepend the previous batch endpoint to preserve the boundary interval. Return (n-by-4 float64 increments, backend name); the first source row has zero increment.

    Parameters
    ----------
    t
        One bounded float64 timestamp batch in seconds.
    i
        Matching point current batch in mA.
    v
        Matching point voltage batch in V.
    config
        Validated immutable AnalysisConfig or DQDVConfig as appropriate; None uses operational defaults.
    previous
        Previous original endpoint (time_s,current_mA,voltage_V) for a cross-batch interval."""
    arrays = [np.asarray(x, dtype=np.float64) for x in (t, i, v)]
    if previous is not None:
        arrays = [np.r_[previous[k], a] for k, a in enumerate(arrays)]
    fun, backend = integration_backend(config.backend)
    out = fun(*arrays, config.integration, config.max_gap_s or -1.0)
    return (out[1:] if previous is not None else out), backend


def normalize(view, *, mass_g=None, area_cm2=None, columns=("capacity_mAh",)):
    """Add per-mass and/or per-area columns lazily without replacing original columns or null masks. Example: normalize(view, mass_g=0.005, area_cm2=1.13). Output-name collisions and nonpositive denominators raise ValueError.

    Parameters
    ----------
    view
        Lazy TableView with the standardized/derived columns required by this operation.
    mass_g
        Strictly positive active mass in grams, or None to omit this normalization.
    area_cm2
        Strictly positive electrode area in cm^2, or None to omit this normalization.
    columns
        Explicit column projection or normalization input names, according to this operation."""
    for x in (mass_g, area_cm2):
        if x is not None and (not math.isfinite(x) or x <= 0):
            raise ValueError("Positive finite mass/area required")
    from .io import TableView

    schema = view.schema
    if schema is not None:
        for name in columns:
            if name not in schema.names:
                raise ValueError(f"Missing normalization column {name}")
            for suffix, denom in (("_g", mass_g), ("_cm2", area_cm2)):
                if denom is None:
                    continue
                destination = name + suffix
                if destination in schema.names:
                    raise ValueError(f"Normalization output collision: {destination}")
                output_type = pc.divide(
                    pa.array([], type=schema.field(name).type), pa.scalar(float(denom))
                ).type
                schema = schema.append(pa.field(destination, output_type))

    def factory():
        """Produce this captured lazy view in bounded batches; retain its source and scientific revision."""
        for b in view.iter_batches():
            for name in columns:
                if name not in b.schema.names:
                    raise ValueError(f"Missing normalization column {name}")
                for suffix, denom in (("_g", mass_g), ("_cm2", area_cm2)):
                    if denom is None:
                        continue
                    dest = name + suffix
                    if dest in b.schema.names:
                        raise ValueError(f"Normalization output collision: {dest}")
                    a = b.column(b.schema.get_field_index(name))
                    b = b.append_column(dest, pc.divide(a, pa.scalar(float(denom))))
            yield b

    return TableView(
        factory,
        {
            **view.manifest,
            "normalization": {"mass_g": mass_g, "area_cm2": area_cm2, "columns": columns},
        },
        memory_bytes=view.memory_bytes,
        declared_schema=schema,
    )


def integrated_batches(
    trace,
    index,
    config,
    events=None,
    cancellation=None,
    contribution_callback=None,
    diagnostic_callback=None,
):
    """Yield source-preserving annotated batches with global, cycling, cycle, step, and half-cycle accumulations. Internal contribution callbacks receive bounded blocks with original interval support; rows removed by selection are never joined.

    Parameters
    ----------
    trace
        BatteryTrace source handle; raw signals are never rewritten.
    index
        Disk-backed SegmentIndex captured for the scientific assignment revision.
    config
        Validated immutable AnalysisConfig or DQDVConfig as appropriate; None uses operational defaults.
    events
        Explicit EventSelection with source identity and decision/boundary revision.
    cancellation
        CancellationToken checked at cooperative boundaries; None creates a local token.
    contribution_callback
        Bounded operation-specific value described by the function contract and API reference.
    diagnostic_callback
        Bounded operation-specific value described by the function contract and API reference."""
    token = cancellation or CancellationToken()
    previous = None
    prev_info = None
    prev_observed = None
    totals = np.zeros(4)
    cycling = np.zeros(4)
    local = np.zeros(4)
    last_half = 0
    local_cycle = np.zeros(4)
    last_cycle = 0
    local_step = np.zeros(4)
    last_step = 0
    excluded = events.accepted_time_ranges() if events else ()
    alternate_rest = config.rest_assignment == "unassigned"
    local_ledger = {"half_cycle": {0: np.zeros(4)}, "cycle": {0: np.zeros(4)}}
    store = getattr(contribution_callback, "__self__", None)
    if (
        not alternate_rest
        and not excluded
        and (
            contribution_callback is None
            or (
                store is not None
                and not store.selection.clauses
                and store.selection.row_limit is None
            )
        )
    ):
        from ._stream_numeric import full_batches

        yield from full_batches(trace, index, config, cancellation, store, diagnostic_callback)
        return
    observed_duration = missing_duration = 0.0
    partial = False
    effective = replace(
        config,
        batch_rows=min(config.batch_rows, 32768, max(1, config.managed_memory_bytes // 8192)),
    )
    for b in annotated_batches(trace, index, effective):
        token.check()
        n = b.num_rows

        def col(name):
            """Internal helper for integrated_batches.col; receives only the bounded state owned by its containing operation."""
            return b.column(b.schema.get_field_index(name)).to_numpy(zero_copy_only=False)

        t, i, v = [col(x).astype(float) for x in ("time_s", "current_mA", "voltage_V")]
        increments, backend = calculate_increments(t, i, v, config, previous)
        hc, cy, ms, section, steps, ids = [
            col(x)
            for x in (
                "half_cycle_id",
                "cycle_id",
                "main_sign",
                "time_section_id",
                "step_id",
                "sample_id",
            )
        ]
        observed = col("observed_sign")
        cumulative = np.zeros((n, 4))
        cum_cycling = np.zeros((n, 4))
        half_values = np.zeros(n)
        cycle_values = np.zeros((n, 4))
        step_values = np.zeros((n, 4))
        excluded_values = np.zeros((n, 4))
        valid_q = np.isfinite(increments[:, 0])
        valid_e = np.isfinite(increments[:, 2])
        contributions = []
        for j in range(n):
            token.check() if j % 8192 == 0 else None
            info = (int(hc[j]), int(cy[j]), int(ms[j]), int(section[j]), int(steps[j]), int(ids[j]))
            left = (t[j - 1], i[j - 1], v[j - 1]) if j else previous
            left_info = (
                (
                    int(hc[j - 1]),
                    int(cy[j - 1]),
                    int(ms[j - 1]),
                    int(section[j - 1]),
                    int(steps[j - 1]),
                    int(ids[j - 1]),
                )
                if j
                else prev_info
            )
            inc = increments[j].copy()
            cyc_inc = inc.copy()
            exc = np.zeros(4)
            if left is not None:
                dt = t[j] - left[0]
                if left_info[3] != info[3]:
                    inc[:] = np.nan
                    cyc_inc[:] = np.nan
                if not np.isfinite(inc[0]):
                    partial = True
                    if np.isfinite(dt) and dt > 0:
                        missing_duration += dt
                    if diagnostic_callback:
                        diagnostic_callback(
                            "invalid_interval",
                            info[5],
                            info[5],
                            "Q/E adjacent interval not integrated",
                        )
                else:
                    observed_duration += dt
                    # Split exclusions on the original I,V linear model; never join across removed rows.
                    ranges = []
                    for sec, a, z in excluded:
                        if sec == info[3] and min(t[j], z) > max(left[0], a):
                            ranges.append(
                                (max(0.0, (a - left[0]) / dt), min(1.0, (z - left[0]) / dt))
                            )
                    ranges.sort()
                    merged = []
                    for a, z in ranges:
                        if merged and a <= merged[-1][1]:
                            merged[-1] = (merged[-1][0], max(z, merged[-1][1]))
                        else:
                            merged.append((a, z))
                    for a, z in merged:
                        exc += np.asarray(
                            interval_parts(*left, t[j], i[j], v[j], config.integration, a, z)
                        )
                    cyc_inc = inc - exc
                    # Direction-coherent branch ownership, including left portion at sign reversals.
                    for sign, k in ((1, 0), (-1, 1)):
                        owner = (
                            info
                            if info[2] == sign
                            else left_info
                            if left_info and left_info[2] == sign and left_info[3] == info[3]
                            else left_info
                            if left_info
                            and left_info[3] == info[3]
                            and left[1] * sign > 0
                            and i[j] * sign <= 0
                            else info
                        )
                        if alternate_rest:
                            for scope, group_id in (("half_cycle", owner[0]), ("cycle", owner[1])):
                                amount = local_ledger[scope].setdefault(group_id, np.zeros(4))
                                for component in (k, k + 2):
                                    amount[component] += (
                                        cyc_inc[component]
                                        if np.isfinite(cyc_inc[component])
                                        else 0.0
                                    )
                        for view, vals in (
                            ("measured", inc),
                            ("cycling", cyc_inc),
                            ("excluded", exc),
                        ):
                            contributions.append(
                                (
                                    owner[0],
                                    owner[1],
                                    info[4],
                                    info[3],
                                    sign,
                                    view,
                                    float(vals[k]),
                                    float(vals[k + 2]),
                                    float(dt),
                                    bool(np.isfinite(vals[k])),
                                    bool(np.isfinite(vals[k + 2])),
                                    int(ids[j]),
                                    *left,
                                    float(t[j]),
                                    float(i[j]),
                                    float(v[j]),
                                    tuple(merged),
                                    int(left_info[5]),
                                    int(left_info[4]),
                                    int(observed[j]),
                                    int(observed[j - 1]) if j else int(prev_observed),
                                )
                            )
            else:
                inc[:] = 0.0
                cyc_inc[:] = 0.0
            increments[j] = inc
            valid_q[j] = np.isfinite(inc[0])
            valid_e[j] = np.isfinite(inc[2])
            totals += np.where(np.isfinite(inc), inc, 0.0)
            cycling += np.where(np.isfinite(cyc_inc), cyc_inc, 0.0)
            if info[0] != last_half:
                local[:] = 0.0
                last_half = info[0]
            if info[1] != last_cycle:
                local_cycle[:] = 0.0
                last_cycle = info[1]
            if info[4] != last_step:
                local_step[:] = 0.0
                last_step = info[4]
            # At a reversal, the old sign's portion belongs to the preceding branch.
            owned = np.where(np.isfinite(cyc_inc), cyc_inc, 0.0).copy()
            if left_info and info[0] != left_info[0] and info[2] != left_info[2]:
                if info[2] == 1:
                    owned[[1, 3]] = 0.0
                elif info[2] == -1:
                    owned[[0, 2]] = 0.0
            local += owned
            if left_info and info[1] != left_info[1]:
                local_cycle += owned
            else:
                local_cycle += np.where(np.isfinite(cyc_inc), cyc_inc, 0.0)
            cumulative[j] = totals
            cum_cycling[j] = cycling
            excluded_values[j] = exc
            half_values[j] = local[0] if info[2] == 1 else local[1] if info[2] == -1 else 0.0
            cycle_values[j] = local_cycle
            if alternate_rest:
                for scope, group_id in (("half_cycle", info[0]), ("cycle", info[1])):
                    local_ledger[scope].setdefault(group_id, np.zeros(4))
                    if group_id:
                        # Positive IDs never return after a confirmed transition;
                        # only group zero and the current positive group can recur.
                        for old_group in tuple(local_ledger[scope]):
                            if old_group not in (0, group_id):
                                del local_ledger[scope][old_group]
                amount = local_ledger["half_cycle"][info[0]]
                half_values[j] = amount[0] if info[2] == 1 else amount[1] if info[2] == -1 else 0.0
                cycle_values[j] = local_ledger["cycle"][info[1]]
            local_step += np.where(np.isfinite(cyc_inc), cyc_inc, 0.0)
            step_values[j] = local_step
        for k, name in enumerate(("charge_mAh", "discharge_mAh", "charge_Wh", "discharge_Wh")):
            b = (
                b.append_column("d_" + name, pa.array(increments[:, k]))
                .append_column("global_" + name, pa.array(cumulative[:, k]))
                .append_column("cycling_" + name, pa.array(cum_cycling[:, k]))
                .append_column("excluded_d_" + name, pa.array(excluded_values[:, k]))
                .append_column("cycle_" + name, pa.array(cycle_values[:, k]))
                .append_column("step_" + name, pa.array(step_values[:, k]))
            )
        b = (
            b.append_column("global_net_mAh", pa.array(cumulative[:, 0] - cumulative[:, 1]))
            .append_column("global_throughput_mAh", pa.array(cumulative[:, 0] + cumulative[:, 1]))
            .append_column("global_net_Wh", pa.array(cumulative[:, 2] - cumulative[:, 3]))
            .append_column("global_throughput_Wh", pa.array(cumulative[:, 2] + cumulative[:, 3]))
            .append_column("capacity_mAh", pa.array(half_values))
            .append_column("q_interval_valid", pa.array(valid_q))
            .append_column("e_interval_valid", pa.array(valid_e))
            .append_column("coverage_partial", pa.array([partial] * n))
        )
        if contribution_callback:
            contribution_callback(contributions)
        previous = (float(t[-1]), float(i[-1]), float(v[-1]))
        prev_observed = int(observed[-1])
        prev_info = (
            int(hc[-1]),
            int(cy[-1]),
            int(ms[-1]),
            int(section[-1]),
            int(steps[-1]),
            int(ids[-1]),
        )
        yield b
