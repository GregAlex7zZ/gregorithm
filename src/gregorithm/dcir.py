"""R(delay) on explicit events; measuring never accepts or excludes events."""

import math
import uuid

import numpy as np
import pyarrow as pa

from .config import AnalysisConfig
from .io import TableView, filter_batch
from .segmentation import annotated_batches


def range_batches(trace, start, end, config=None):
    """Yield original annotated support for an inclusive sample-ID interval without materializing the complete source.

    Parameters
    ----------
    trace
        BatteryTrace source handle; raw signals are never rewritten.
    start
        Bounded operation-specific value described by the function contract and API reference.
    end
        Bounded operation-specific value described by the function contract and API reference.
    config
        Validated immutable AnalysisConfig or DQDVConfig as appropriate; None uses operational defaults."""
    config = config or AnalysisConfig()
    idx = trace.ensure_index(config)
    clause = ({"sample_id": {"min": start, "max": end}},)
    for b in annotated_batches(
        trace, idx, config, columns=("time_s", "current_mA", "voltage_V"), clauses=clause
    ):
        b = filter_batch(b, clause)
        if b.num_rows:
            yield b


def fetch_samples(trace, ids, config=None):
    """Retrieve explicitly requested original sample IDs from bounded support batches; missing IDs are errors rather than fabricated samples.

    Parameters
    ----------
    trace
        BatteryTrace source handle; raw signals are never rewritten.
    ids
        Bounded operation-specific value described by the function contract and API reference.
    config
        Validated immutable AnalysisConfig or DQDVConfig as appropriate; None uses operational defaults."""
    found = {}
    wanted = set(ids)
    for b in range_batches(trace, min(ids), max(ids), config):
        for r in b.to_pylist():
            if r["sample_id"] in wanted:
                found[r["sample_id"]] = r
    if set(found) != wanted:
        raise ValueError("Sample references absent")
    return found


def pulse_statistics(trace, event, config=None):
    """Compute time-weighted current moments over observed pulse support, reporting incomplete coverage instead of silently estimating missing intervals.

    Parameters
    ----------
    trace
        BatteryTrace source handle; raw signals are never rewritten.
    event
        One event record with original pulse support and source identity.
    config
        Validated immutable AnalysisConfig or DQDVConfig as appropriate; None uses operational defaults."""
    previous = None
    w = s = s2 = 0.0
    complete = True
    for b in range_batches(trace, event["sample_t1"], event["sample_t2"], config):
        t = b.column("time_s").to_numpy()
        i = b.column("current_mA").to_numpy()
        for tt, ii in zip(t, i):
            if previous is not None:
                dt = tt - previous[0]
                a = previous[1]
                if not np.isfinite(dt) or dt <= 0 or not np.isfinite(ii) or not np.isfinite(a):
                    complete = False
                else:
                    w += dt
                    s += dt * (a + ii) / 2
                    s2 += dt * (a * a + a * ii + ii * ii) / 3
            previous = (tt, ii)
    mean = s / w if w > 0 else np.nan
    rms = np.sqrt(max(0.0, s2 / w - mean * mean)) if w > 0 else np.nan
    return {
        "current_mean_mA": float(mean) if np.isfinite(mean) else None,
        "current_rms_mA": float(rms) if np.isfinite(rms) else None,
        "coverage_complete": bool(complete and w > 0),
    }


def measure_dcir(
    trace,
    events,
    delay_s,
    *,
    method="linear",
    nearest_tolerance_s=None,
    max_interp_gap_s=None,
    delta_i_min_mA=0.001,
    resolution_fraction=0.1,
    rest_min_duration_s=None,
    config=None,
):
    """Measure signed finite-delay resistance for each supplied event independently of acceptance/exclusion. Return a TableView including invalid outcomes and reasons. Example: measure_dcir(trace, events, delay_s=11). t0 is the exact last rest sample and R=1000*(V_tau-V0)/(I_tau_mA-I0_mA); no abs, smoothing, baseline averaging, or extrapolation.

    Parameters
    ----------
    trace
        BatteryTrace source handle; raw signals are never rewritten.
    events
        Explicit EventSelection with source identity and decision/boundary revision.
    delay_s
        One strictly positive finite delay from the last rest sample, in seconds; lists are rejected.
    method
        Signal integration or DCIR sampling policy appropriate to this function; see its contract.
    nearest_tolerance_s
        Explicit allowable nearest-sample timing error in seconds; mandatory for nearest DCIR.
    max_interp_gap_s
        Maximum DCIR support gap in seconds, independent of integration max_gap_s.
    delta_i_min_mA
        Minimum absolute pulse-minus-baseline current difference in mA.
    resolution_fraction
        Operational warning threshold for support/front timing relative to requested delay.
    rest_min_duration_s
        Optional minimum preceding rest duration in seconds; otherwise use analytical support policy.
    config
        Validated immutable AnalysisConfig or DQDVConfig as appropriate; None uses operational defaults."""
    if isinstance(delay_s, (list, tuple, np.ndarray)) or not math.isfinite(delay_s) or delay_s <= 0:
        raise ValueError("One positive finite delay_s per call")
    if method not in {"linear", "nearest"}:
        raise ValueError("method")
    if method == "nearest" and (nearest_tolerance_s is None or nearest_tolerance_s < 0):
        raise ValueError("nearest requires an explicit nonnegative tolerance")
    for name, value in (
        ("nearest_tolerance_s", nearest_tolerance_s),
        ("max_interp_gap_s", max_interp_gap_s),
        ("rest_min_duration_s", rest_min_duration_s),
    ):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(name)
    if not math.isfinite(delta_i_min_mA) or delta_i_min_mA < 0:
        raise ValueError("delta_i_min_mA")
    if not math.isfinite(resolution_fraction) or resolution_fraction < 0:
        raise ValueError("resolution_fraction")
    config = config or AnalysisConfig()
    call_id = uuid.uuid4().hex
    records = []
    for event in events.events:
        if event["trace_id"] != trace.trace_id or event["source_revision"] != trace.source_revision:
            raise ValueError("Event source revision mismatch")
        e = dict(event)
        target = e["t0"] + delay_s if e.get("t0") is not None else None
        r = {
            **e,
            "measurement_id": call_id,
            "delay_s": float(delay_s),
            "target_s": target,
            "method": method,
            "valid": False,
            "reason": "",
            "warnings": [],
            "V0_V": None,
            "I0_mA": None,
            "V_tau_V": None,
            "I_tau_mA": None,
            "deltaI_A": None,
            "deltaV_V": None,
            "R_ohm": None,
            "support_left_sample_id": None,
            "support_right_sample_id": None,
            "weight_right": None,
            "interp_gap_s": None,
            "timing_error_s": None,
            "elapsed_from_on_lower_s": target - e["t1"]
            if target is not None and e.get("t1") is not None
            else None,
            "elapsed_from_on_upper_s": delay_s,
        }
        if e.get("t0") is None:
            r["reason"] = "missing_baseline"
        elif e.get("t1") is None or e.get("t2") is None:
            r["reason"] = "invalid_active_times"
        elif e.get("duration_s") is None:
            r["reason"] = "incomplete_duration"
        elif not delay_s < e["duration_s"]:
            r["reason"] = "delay_outside_duration"
        elif not e["t1"] <= target <= e["t2"]:
            r["reason"] = "target_outside_active_support"
        if r["reason"]:
            records.append(r)
            continue
        baseline = fetch_samples(trace, [e["sample_t0"]], config)[e["sample_t0"]]
        r["V0_V"] = baseline["voltage_V"]
        r["I0_mA"] = baseline["current_mA"]
        if baseline["time_s"] != e["t0"]:
            r["reason"] = "baseline_time_mismatch"
            records.append(r)
            continue
        if not all(np.isfinite(baseline[k]) for k in ("time_s", "current_mA", "voltage_V")):
            r["reason"] = "invalid_baseline"
            records.append(r)
            continue
        if baseline["direction"] != "rest":
            r["reason"] = "baseline_not_rest"
            records.append(r)
            continue
        left = right = None
        prev = None
        for b in range_batches(trace, e["sample_t1"], e["sample_t2"], config):
            for row in b.to_pylist():
                tt = row["time_s"]
                if tt == target:
                    left = right = row
                    break
                if prev is not None and prev["time_s"] < target < tt:
                    left = prev
                    right = row
                    break
                prev = row
            if right is not None:
                break
        if left is None or right is None:
            r["reason"] = "unsupported_target"
            records.append(r)
            continue
        if (
            left["time_section_id"] != right["time_section_id"]
            or left["time_section_id"] != e["time_section_id"]
        ):
            r["reason"] = "time_reset"
            records.append(r)
            continue
        if method == "nearest":
            chosen = min((left, right), key=lambda x: abs(x["time_s"] - target))
            if abs(chosen["time_s"] - target) > nearest_tolerance_s:
                r["reason"] = "nearest_tolerance_exceeded"
                records.append(r)
                continue
            left = right = chosen
            r["timing_error_s"] = chosen["time_s"] - target
        gap = right["time_s"] - left["time_s"]
        if max_interp_gap_s is not None and gap > max_interp_gap_s:
            r["reason"] = "interpolation_gap_exceeded"
            records.append(r)
            continue
        if not all(np.isfinite(x[k]) for x in (left, right) for k in ("current_mA", "voltage_V")):
            r["reason"] = "invalid_adjacent_support"
            records.append(r)
            continue
        if any(x["direction"] != e["pulse_direction"] for x in (left, right)):
            r["reason"] = "support_not_same_pulse_direction"
            records.append(r)
            continue
        w = (target - left["time_s"]) / gap if gap > 0 else 0.0
        it = left["current_mA"] * (1 - w) + right["current_mA"] * w
        vt = left["voltage_V"] * (1 - w) + right["voltage_V"] * w
        di = it - baseline["current_mA"]
        dv = vt - baseline["voltage_V"]
        r.update(
            {
                "V_tau_V": vt,
                "I_tau_mA": it,
                "deltaI_A": di / 1000,
                "deltaV_V": dv,
                "support_left_sample_id": left["sample_id"],
                "support_right_sample_id": right["sample_id"],
                "weight_right": w,
                "interp_gap_s": gap,
            }
        )
        if abs(di) <= delta_i_min_mA:
            r["reason"] = "delta_current_too_small"
        else:
            r["R_ohm"] = 1000 * dv / di
            r["valid"] = True
            if r["R_ohm"] < 0:
                r["warnings"].append("negative_resistance")
        if max(e["t1"] - e["t0"], gap) > resolution_fraction * delay_s:
            r["warnings"].append("sampling_resolution_limited")
        if (
            rest_min_duration_s is not None
            and (e.get("rest_duration_s") or 0) < rest_min_duration_s
        ):
            r["warnings"].append("short_rest")
        if (e.get("current_rms_mA") or 0) > max(0.0001, 0.02 * abs(e.get("current_mean_mA") or 0)):
            r["warnings"].append("variable_current")
        records.append(r)
    manifest = {
        **trace.manifest(),
        "analysis": "dcir",
        "analysis_id": call_id,
        "event_revision": events.revision,
        "delay_s": delay_s,
        "config": config.to_dict(),
        "parent_result_ids": [],
    }
    # Nullable output is intentionally missing, never zero or absolute-valued.
    return TableView(
        lambda: iter([pa.RecordBatch.from_pylist(records)]) if records else iter([]),
        manifest,
        memory_bytes=config.managed_memory_bytes,
    )
