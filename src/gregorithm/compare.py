"""Explicit alignment and grid-resolution-aware peak comparisons."""

import numpy as np


def align_time(views, origin=None):
    """Return separate views with an added relative_time_s coordinate. Example: align_time({"a": result_a.samples, "b": result_b.samples}). Original timestamps and integration origins remain intact.

    Parameters
    ----------
    views
        Bounded operation-specific value described by the function contract and API reference.
    origin
        Relative-time origin in seconds; None uses the first selected sample independently per trace."""
    return {k: v.relative_time(origin) for k, v in views.items()}


def peak(curve, window_V, *, extremum=None, max_candidates=1000):
    """Find the maximum (charge) or minimum (discharge) dQ/dV on existing grid nodes inside an explicit voltage window. Select one curve_id first. Ties are reported as ambiguity; candidate storage is capped without hiding the total count. No sub-grid fit or chemical assignment is performed.

    Parameters
    ----------
    curve
        Bounded operation-specific value described by the function contract and API reference.
    window_V
        Explicit (lower, upper) voltage window for grid-node peak comparison.
    extremum
        Explicit max/min override; default max for charge and min for discharge.
    max_candidates
        Cap on retained tied peak candidates; total count and ambiguity are still reported."""
    lo, hi = window_V
    if hi < lo:
        raise ValueError("window")
    best = None
    candidates = []
    candidate_count = 0
    curve_ids = set()
    if not isinstance(max_candidates, int) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    direction = None
    for b in curve.iter_batches():
        if b.num_rows and "direction" in b.schema.names:
            direction = b.column("direction")[0].as_py()
        method = extremum or ("min" if direction == "discharge" else "max")
        if method not in {"min", "max"}:
            raise ValueError("extremum")
        for r in b.to_pylist():
            curve_ids.add(r["curve_id"])
            if len(curve_ids) > 1:
                raise ValueError("Select one curve_id before finding a peak")
            v = r["voltage_V"]
            y = r["dqdv_mAh_V"]
            if not lo <= v <= hi or not np.isfinite(y):
                continue
            if best is None or (y > best if method == "max" else y < best):
                best = y
                candidates = [r]
                candidate_count = 1
            elif y == best:
                candidate_count += 1
                if len(candidates) < max_candidates:
                    candidates.append(r)
    return {
        "valid": bool(candidates),
        "candidates": [
            {
                "voltage_V": r["voltage_V"],
                "height_mAh_V": r["dqdv_mAh_V"],
                "curve_point_id": r["curve_point_id"],
                "curve_id": r["curve_id"],
            }
            for r in candidates
        ],
        "ambiguous": candidate_count > 1,
        "candidate_count": candidate_count,
        "candidates_truncated": candidate_count > len(candidates),
        "window_V": window_V,
        "grid_step_V": curve.manifest.get("effective_step_V"),
        "smoothing": curve.manifest.get("dqdv_config", {}).get("smoothing", False),
        "parent_result_id": curve.manifest.get("result_id"),
    }


def compare_peaks(reference, comparison, window_V, *, extremum=None):
    """Compare two explicitly selected curves in the same voltage window. Return their peak outcomes and comparison-minus-reference deltaV only when both peaks are unambiguous. Different grid resolution is flagged; no resolution harmonization is implicit.

    Parameters
    ----------
    reference
        Bounded operation-specific value described by the function contract and API reference.
    comparison
        Bounded operation-specific value described by the function contract and API reference.
    window_V
        Explicit (lower, upper) voltage window for grid-node peak comparison.
    extremum
        Explicit max/min override; default max for charge and min for discharge."""
    a = peak(reference, window_V, extremum=extremum)
    b = peak(comparison, window_V, extremum=extremum)
    return {
        "reference": a,
        "comparison": b,
        "deltaV_V": b["candidates"][0]["voltage_V"] - a["candidates"][0]["voltage_V"]
        if a["valid"] and b["valid"] and not a["ambiguous"] and not b["ambiguous"]
        else None,
        "different_resolution": a["grid_step_V"] != b["grid_step_V"],
        "parent_result_ids": [a["parent_result_id"], b["parent_result_id"]],
        "precision": "grid_nodes_only",
    }
