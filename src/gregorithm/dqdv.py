"""Corrected historical interpolation, with exact bounded grid-neighbor reduction."""

import uuid

import numpy as np
import pyarrow as pa

from .config import DQDVConfig
from .diagnostics import BudgetExceeded
from .io import TableView


class GridNeighbors:
    """Exact nearest support per grid node, not a mean voltage binning."""

    def __init__(self, grid, duplicates="first"):
        """Initialize the state required by this object. Use the public factory or dataclass configuration unless constructing this internal component deliberately."""
        self.grid = grid
        self.policy = duplicates
        n = len(grid)
        self.lv = np.full(n, -np.inf)
        self.rv = np.full(n, np.inf)
        self.lq = np.zeros(n)
        self.rq = np.zeros(n)
        self.lid = np.full(n, -1, dtype=np.int64)
        self.rid = self.lid.copy()
        self.lcount = np.zeros(n, dtype=np.int64)
        self.rcount = self.lcount.copy()

    def update(self, v, q, ids):
        """Update exact left/right voltage supports for every grid node from one batch, resolving duplicate policy across batch boundaries."""
        if len(v) == 0:
            return
        order = np.argsort(v, kind="stable")
        v = v[order]
        q = q[order]
        ids = ids[order]
        unique, starts, counts = np.unique(v, return_index=True, return_counts=True)
        if self.policy == "mean":
            uq = np.add.reduceat(q, starts) / counts
            ui = ids[starts]
        elif self.policy == "first":
            uq = q[starts]
            ui = ids[starts]
        else:
            uq = q[starts + counts - 1]
            ui = ids[starts + counts - 1]
        for side in ("l", "r"):
            ix = (
                np.searchsorted(unique, self.grid, side="right") - 1
                if side == "l"
                else np.searchsorted(unique, self.grid, side="left")
            )
            ok = (ix >= 0) & (ix < len(unique))
            dest = np.flatnonzero(ok)
            src = ix[ok]
            vv = unique[src]
            qq = uq[src]
            ii = ui[src]
            cc = counts[src]
            stored = getattr(self, side + "v")
            sq = getattr(self, side + "q")
            si = getattr(self, side + "id")
            sc = getattr(self, side + "count")
            closer = vv > stored[dest] if side == "l" else vv < stored[dest]
            same = vv == stored[dest]
            take = closer | (
                same
                & (
                    (ii < si[dest])
                    if self.policy == "first"
                    else (ii > si[dest])
                    if self.policy == "last"
                    else False
                )
            )
            if self.policy == "mean":
                d = dest[same]
                c = cc[same]
                sq[d] = (sq[d] * sc[d] + qq[same] * c) / (sc[d] + c)
                sc[d] += c
            d = dest[take]
            stored[d] = vv[take]
            sq[d] = qq[take]
            si[d] = ii[take]
            sc[d] = cc[take]

    def interpolate(self):
        """Return linear Q(V) and right-support weights on supported grid nodes; unsupported nodes raise instead of extrapolating."""
        if not np.all(np.isfinite(self.lv) & np.isfinite(self.rv)):
            raise ValueError("Unsupported grid node; no extrapolation")
        gap = self.rv - self.lv
        w = np.divide(self.grid - self.lv, gap, out=np.zeros_like(gap), where=gap != 0)
        return self.lq * (1 - w) + self.rq * w, w


def compute_dqdv(view, *, config=None, rest_threshold_mA=0.0001, manifest=None):
    """Return a curve TableView for exactly one selected half-cycle, using corrected historical interpolation by default. Example: compute_dqdv(result.samples.select(half_cycle_id=1), config=DQDVConfig()). Rest, excluded diagnostic samples, nonfinite values, and invalid-Q right endpoints are omitted. Grid-neighbor support is bounded; no whole-branch sort is required.

    Parameters
    ----------
    view
        Lazy TableView with the standardized/derived columns required by this operation.
    config
        Validated immutable AnalysisConfig or DQDVConfig as appropriate; None uses operational defaults.
    rest_threshold_mA
        Inclusive rest-current threshold in mA; does not zero raw current.
    manifest
        Captured provenance dictionary or saved trace-manifest path, as specified by the operation."""
    cfg = config or DQDVConfig()
    if view.parquet_path is not None and view.columns is None and view.declared_schema is not None:
        needed = (
            "voltage_V",
            "capacity_mAh",
            "current_mA",
            "sample_id",
            "half_cycle_id",
            "cycle_id",
            "trace_id",
            "direction",
            "excluded",
            "q_interval_valid",
        )
        view = view.select(columns=[name for name in needed if name in view.declared_schema.names])
    n_input = 0
    lo = np.inf
    hi = -np.inf
    branch = None
    previous_v = None
    nonmonotonic = False

    def values():
        """Yield valid branch voltage/capacity/sample support without whole-branch materialization."""
        for b in view.iter_batches():
            v = b.column("voltage_V").to_numpy(zero_copy_only=False)
            q = b.column("capacity_mAh").to_numpy(zero_copy_only=False)
            i = b.column("current_mA").to_numpy(zero_copy_only=False)
            ids = b.column("sample_id").to_numpy(zero_copy_only=False)
            mask = (
                np.isfinite(v) & np.isfinite(q) & np.isfinite(i) & (np.abs(i) > rest_threshold_mA)
            )
            if "excluded" in b.schema.names:
                mask &= ~b.column("excluded").to_numpy(zero_copy_only=False).astype(bool)
            if "q_interval_valid" in b.schema.names:
                mask &= b.column("q_interval_valid").to_numpy(zero_copy_only=False)
            yield b, v[mask], q[mask], ids[mask]

    branches = set()
    for b, v, q, ids in values():
        n_input += b.num_rows
        if len(v):
            lo = min(lo, float(v.min()))
            hi = max(hi, float(v.max()))
        if b.num_rows:
            for h in b.column("half_cycle_id").to_pylist():
                if h is None or h <= 0:
                    raise ValueError("Select a positive confirmed half_cycle_id for dQ/dV")
                branches.add(h)
                if len(branches) > 1:
                    raise ValueError(
                        "compute_dqdv requires one half-cycle; pipeline splits branches"
                    )
            if branch is None and len(v):
                branch = {
                    k: b.column(k)[0].as_py()
                    for k in ("trace_id", "half_cycle_id", "cycle_id", "direction")
                    if k in b.schema.names
                }
                currents = b.column("current_mA").to_numpy(zero_copy_only=False)
                active = currents[np.isfinite(currents) & (np.abs(currents) > rest_threshold_mA)]
                if len(active):
                    branch["direction"] = "charge" if active[0] > 0 else "discharge"
    if not hi > lo:
        raise ValueError("Insufficient voltage domain")
    trim_lo = lo + cfg.trim_extremes_V
    trim_hi = hi - cfg.trim_extremes_V
    e_lo = max(trim_lo, cfg.e_lo if cfg.e_lo is not None else -np.inf)
    e_hi = min(trim_hi, cfg.e_hi if cfg.e_hi is not None else np.inf)
    if e_hi <= e_lo:
        raise ValueError("Empty voltage window")

    def trim_mask(v):
        """Apply the explicit endpoint trim consistently to interpolation support."""
        return (
            (v > trim_lo) & (v < trim_hi)
            if cfg.trim_extremes_V > 0
            else (v >= trim_lo) & (v <= trim_hi)
        )

    filtered = 0
    distinct = []
    available_lo = np.inf
    available_hi = -np.inf
    for b, v, q, ids in values():
        support = v[trim_mask(v)]
        if len(support):
            available_lo = min(available_lo, float(support.min()))
            available_hi = max(available_hi, float(support.max()))
        v = support[(support >= e_lo) & (support <= e_hi)]
        filtered += len(v)
        for x in v:
            if len(distinct) < 3 and x not in distinct:
                distinct.append(float(x))
            if (
                previous_v is not None
                and branch
                and (
                    (branch.get("direction") == "charge" and x < previous_v)
                    or (branch.get("direction") == "discharge" and x > previous_v)
                )
            ):
                nonmonotonic = True
            previous_v = x
    if len(distinct) < 3 or not available_hi > available_lo:
        raise ValueError("Fewer than 3 distinct valid voltages")
    e_lo = max(available_lo, cfg.e_lo) if cfg.e_lo is not None else available_lo
    e_hi = min(available_hi, cfg.e_hi) if cfg.e_hi is not None else available_hi
    requested = (
        cfg.n_points if cfg.n_points is not None else round((cfg.fraction or 0.1) * filtered)
    )
    if cfg.profile == "voltage_grid":
        step = cfg.voltage_step_V or 0.002
        n = int(np.floor((e_hi - e_lo) / step)) + 1
        if n < 3:
            raise ValueError("Voltage grid needs at least 3 nodes")
    else:
        n = (
            cfg.n_points
            if cfg.n_points is not None
            else max(
                3, requested if cfg.max_auto_points is None else min(cfg.max_auto_points, requested)
            )
        )
    if n * 192 > min(view.memory_bytes // 4, 512 * 1024**2):
        raise BudgetExceeded("Requested dQ/dV grid exceeds buffer budget")
    grid = (
        e_lo + np.arange(n) * step if cfg.profile == "voltage_grid" else np.linspace(e_lo, e_hi, n)
    )
    neighbors = GridNeighbors(grid, cfg.duplicates)
    for b, v, q, ids in values():
        m = trim_mask(v)
        neighbors.update(v[m], q[m], ids[m])
    qi, w = neighbors.interpolate()
    if cfg.signed_capacity and branch.get("direction") == "discharge":
        qi = -qi
    unsmoothed = qi.copy()
    if cfg.smoothing:
        from scipy.signal import savgol_filter

        if cfg.window_points > n:
            raise ValueError("Smoothing window longer than grid; no silent adaptation")
        qi = savgol_filter(qi, cfg.window_points, cfg.polyorder, mode="interp")
    derivative = np.gradient(qi, grid, edge_order=1)
    curve_id = uuid.uuid4().hex
    step_effective = float(grid[1] - grid[0])
    m = {
        **(manifest or view.manifest),
        "analysis": "dqdv",
        "curve_id": curve_id,
        "profile": cfg.profile,
        "dqdv_config": cfg.__dict__,
        "n_input": n_input,
        "n_filtered": filtered,
        "n_unique_V": None,
        "n_requested": requested,
        "n_effective": n,
        "effective_step_V": step_effective,
        "auto_grid_limited": cfg.n_points is None
        and cfg.profile == "legacy_corrected"
        and cfg.max_auto_points is not None
        and requested > cfg.max_auto_points,
        "nonmonotonic_voltage": nonmonotonic,
        "smoothing_width_V": (cfg.window_points - 1) * step_effective if cfg.smoothing else 0.0,
        "unit": "mAh/V",
        "voltage_domain_V": [e_lo, e_hi],
        "boundary_support_outside_window": cfg.e_lo is not None or cfg.e_hi is not None,
    }
    data = {
        **{k: [v] * n for k, v in (branch or {}).items()},
        "curve_id": [curve_id] * n,
        "curve_point_id": np.arange(n, dtype=np.int64),
        "voltage_V": grid,
        "capacity_mAh": qi,
        "capacity_unsmoothed_mAh": unsmoothed,
        "dqdv_mAh_V": derivative,
        "support_left_sample_id": neighbors.lid,
        "support_right_sample_id": neighbors.rid,
        "weight_right": w,
    }
    if cfg.smoothing:
        data["dqdv_unsmoothed_mAh_V"] = np.gradient(unsmoothed, grid, edge_order=1)
    if m.get("result_id"):
        data["result_id"] = [m["result_id"]] * n
    batch = pa.RecordBatch.from_pydict(data)
    return TableView(lambda: iter([batch]), m, memory_bytes=view.memory_bytes)
