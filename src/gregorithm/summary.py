"""Disk-backed long summaries and fixed per-trace C-rate references."""

import math
import sqlite3

import numpy as np

from .integration import interval_parts
from .segmentation import DIR


class SummaryStore:
    """Internal SQLite accumulator for long quantities, coverage, time-weighted CC current, and fixed references. Call through analyze()/summarize(); close() releases file handles."""

    def __init__(self, path, selection, config, head_end=None, events=None, index=None):
        """Initialize the state required by this object. Use the public factory or dataclass configuration unless constructing this internal component deliberately."""
        self.db = sqlite3.connect(path)
        self.db.execute("pragma cache_size=-8192")
        self.db.execute(
            "create table amounts (scope text,gid integer,view text,sign integer,selected integer,q real,e real,duration real,q_invalid integer,e_invalid integer,primary key(scope,gid,view,sign,selected))"
        )
        self.selection = selection
        self.db.execute(
            "create table cc_moments(step_id integer,view text,selected integer,integral_mA_s real,duration_s real,primary key(step_id,view,selected))"
        )
        self.config = config
        self.head_end = head_end
        from functools import lru_cache

        from .events import EventSelection

        self.events = events or EventSelection()
        self._index_db = index.connect() if index is not None else None

        @lru_cache(maxsize=4096)
        def step_info(step):
            """Retrieve one step annotation through a bounded LRU-backed SQLite lookup."""
            if self._index_db is None:
                return {}
            record = self._index_db.execute(
                "select * from steps where step_id=?", (step,)
            ).fetchone()
            return dict(record) if record else {}

        self._step_info = step_info

    def _bounds(self, c, clauses):
        """Intersect original fractional interval domains with one conjunction of scientific selection predicates."""
        (
            h,
            cy,
            step,
            section,
            sign,
            view,
            q,
            e,
            dt,
            qok,
            eok,
            sid,
            t0,
            i0,
            v0,
            t1,
            i1,
            v1,
            excluded,
            left_sid,
        ) = c[:20]
        domains = [(0.0, 1.0)]
        for clause in clauses:
            for k, wanted in clause.items():
                choices = wanted if isinstance(wanted, (list, tuple)) else [wanted]
                allowed = [(0.0, 1.0)]
                if k == "time_s" and isinstance(wanted, dict):
                    allowed = [
                        (
                            (wanted["min"] - t0) / dt if wanted.get("min") is not None else 0.0,
                            (wanted["max"] - t0) / dt if wanted.get("max") is not None else 1.0,
                        )
                    ]
                elif k == "event_id":
                    allowed = []
                    for event in self.events.events:
                        if event["event_id"] in choices and event["time_section_id"] == section:
                            a = event.get("t0")
                            z = event.get("t3")
                            a = event.get("t1") if a is None else a
                            z = event.get("t2") if z is None else z
                            if a is not None and z is not None:
                                allowed.append(((a - t0) / dt, (z - t0) / dt))
                elif k == "sample_id":
                    if isinstance(wanted, dict):
                        if (wanted.get("min") is not None and left_sid < wanted["min"]) or (
                            wanted.get("max") is not None and sid > wanted["max"]
                        ):
                            return []
                    elif sid not in choices or left_sid not in choices:
                        return []
                elif k == "file":
                    trace = self.selection.traces[0]
                    if str(trace.path) not in choices and trace.path.name not in choices:
                        return []
                elif k == "trace_id":
                    if self.selection.traces[0].trace_id not in choices:
                        return []
                elif k == "direction":
                    right = DIR[c[21]] if len(c) > 21 else DIR[sign]
                    left = DIR[c[22]] if len(c) > 22 else right
                    if right not in choices or left not in choices:
                        return []
                else:
                    actual = {
                        "half_cycle_id": h,
                        "cycle_id": cy,
                        "step_id": step,
                        "time_section_id": section,
                    }.get(k)
                    if k in {"mode", "step_role"}:
                        info = self._step_info(step)
                        actual = info.get("mode") if k == "mode" else info.get("role", "ordinary")
                    if actual is None:
                        raise ValueError(f"Unsupported scientific selection predicate: {k}")
                    if actual not in choices:
                        return []
                domains = [
                    (max(a, x), min(z, y))
                    for a, z in domains
                    for x, y in allowed
                    if min(z, y) > max(a, x)
                ]
                if not domains:
                    return []
        if self.selection.row_limit is not None and (self.head_end is None or sid > self.head_end):
            return []
        return domains

    def _clip(self, c):
        """Merge OR domains and clip original signal integrals, subtracting only declared exclusion portions without bridging removed rows."""
        from .io import expand_clauses

        *_, t0, i0, v0, t1, i1, v1, excluded, left_sid = c[:20]
        # Original interval domains are merged before integration; overlaps are counted once.
        domains = sorted(
            interval
            for clauses in expand_clauses(self.selection.clauses)
            for interval in self._bounds(c, clauses)
        )
        merged = []
        for a, z in domains:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(z, merged[-1][1]))
            else:
                merged.append((a, z))
        if not merged:
            return None
        vals = np.zeros(4)
        duration = 0.0
        for lo, hi in merged:
            piece = np.asarray(
                interval_parts(t0, i0, v0, t1, i1, v1, self.config.integration, lo, hi)
            )
            exc = np.zeros(4)
            for a, z in excluded:
                a = max(a, lo)
                z = min(z, hi)
                if z > a:
                    exc += np.asarray(
                        interval_parts(t0, i0, v0, t1, i1, v1, self.config.integration, a, z)
                    )
            if c[5] == "cycling":
                piece -= exc
            if c[5] == "excluded":
                piece = exc
            vals += piece
            duration += (t1 - t0) * (hi - lo)
        k = 0 if c[4] == 1 else 1
        return float(vals[k]), float(vals[k + 2]), duration

    def add(self, contributions):
        """Aggregate bounded per-interval contributions into disk-backed long metrics and time-weighted CC moments."""
        data = {}
        cc = {}
        for c in contributions:
            h, cy, step, section, sign, view, q, e, dt, qok, eok, *_ = c
            selected = (
                (q, e, dt)
                if not self.selection.clauses and self.selection.row_limit is None
                else self._clip(c)
            )
            for selected_flag, vals in ((0, (q, e, dt)), (1, selected)):
                if vals is None:
                    continue
                qq, ee, d = vals
                for scope, gid in (("global", 0), ("half_cycle", h), ("cycle", cy), ("step", step)):
                    key = (scope, gid, view, sign, selected_flag)
                    a = data.setdefault(key, [0.0, 0.0, 0.0, 0, 0])
                    a[0] += qq if math.isfinite(qq) else 0.0
                    a[1] += ee if math.isfinite(ee) else 0.0
                    a[2] += d
                    # Coverage is recorded once by the pipeline, including intervals
                    # with missing current that never generate contributions here.
                info = self._step_info(step)
                if (
                    len(c) > 20
                    and c[20] == step
                    and info.get("mode") == "CC"
                    and info.get("sign") == sign
                    and qok
                ):
                    duration = d
                    if c[18]:
                        from .io import expand_clauses

                        domains = (
                            [(0.0, 1.0)]
                            if not selected_flag
                            else [
                                interval
                                for clauses in expand_clauses(self.selection.clauses)
                                for interval in self._bounds(c, clauses)
                            ]
                        )
                        # Merge domains before accounting excluded durations.
                        merged = []
                        for x, y in sorted(domains):
                            if merged and x <= merged[-1][1]:
                                merged[-1] = (merged[-1][0], max(y, merged[-1][1]))
                            else:
                                merged.append((x, y))
                        excluded_duration = dt * sum(
                            max(0.0, min(z, y) - max(a, x)) for a, z in c[18] for x, y in merged
                        )
                        duration = (
                            excluded_duration
                            if view == "excluded"
                            else d - excluded_duration
                            if view == "cycling"
                            else d
                        )
                    if duration > 0:
                        moment = cc.setdefault((step, view, selected_flag), [0.0, 0.0])
                        moment[0] += sign * qq * 3600.0
                        moment[1] += duration
        self.db.executemany(
            "insert into amounts values(?,?,?,?,?,?,?,?,?,?) on conflict(scope,gid,view,sign,selected) do update set q=q+excluded.q,e=e+excluded.e,duration=duration+excluded.duration,q_invalid=q_invalid+excluded.q_invalid,e_invalid=e_invalid+excluded.e_invalid",
            [(*k, *a) for k, a in data.items()],
        )
        self.db.commit()

        self._add_cc(cc)

    def _add_cc(self, moments):
        """Merge bounded CC current-duration moments into the SQLite summary table."""
        if moments:
            self.db.executemany(
                "insert into cc_moments values(?,?,?,?,?) on conflict(step_id,view,selected) do update set integral_mA_s=integral_mA_s+excluded.integral_mA_s,duration_s=duration_s+excluded.duration_s",
                [(*key, *value) for key, value in moments.items()],
            )
            self.db.commit()

    def mark_invalid_groups(self, groups):
        """Batch invalidity writes; Q and E coverage stay independent."""
        rows = []
        for key, (q_count, e_count) in groups.items():
            h, cy, step = key[:3]
            views = (key[3],) if len(key) > 3 else ("measured", "cycling")
            selected_flags = (key[4],) if len(key) > 4 else (0, 1)
            for scope, gid in (("global", 0), ("half_cycle", h), ("cycle", cy), ("step", step)):
                if len(key) > 5 and scope != key[5]:
                    continue
                for view in views:
                    for sign in (-1, 1):
                        for selected in selected_flags:
                            rows.append((scope, gid, view, sign, selected, q_count, e_count))
        if rows:
            self.db.executemany(
                "insert into amounts values(?,?,?,?,?,0,0,0,?,?) "
                "on conflict(scope,gid,view,sign,selected) do update set "
                "q_invalid=q_invalid+excluded.q_invalid,e_invalid=e_invalid+excluded.e_invalid",
                rows,
            )
            self.db.commit()

    def add_arrays(self, inc, dt, owners, steps, cc_mask=None):
        """Full-view aggregation without Python tuples for every source interval."""
        valid = np.isfinite(inc[:, 0]) & (dt > 0)
        if not np.any(valid):
            return
        rows = []
        for k, sign in enumerate((1, -1)):
            for scope, groups in (
                ("global", np.zeros(len(dt), dtype=np.int64)),
                ("half_cycle", owners[k][0]),
                ("cycle", owners[k][1]),
                ("step", steps),
            ):
                unique, inverse = np.unique(groups[valid], return_inverse=True)
                q = np.bincount(inverse, weights=inc[valid, k])
                e = np.bincount(inverse, weights=np.nan_to_num(inc[valid, k + 2], nan=0.0))
                duration = np.bincount(inverse, weights=dt[valid])
                for gid, qq, ee, dd in zip(unique, q, e, duration):
                    for view in ("measured", "cycling", "excluded"):
                        for selected in (0, 1):
                            rows.append(
                                (
                                    scope,
                                    int(gid),
                                    view,
                                    sign,
                                    selected,
                                    float(qq) if view != "excluded" else 0.0,
                                    float(ee) if view != "excluded" else 0.0,
                                    float(dd),
                                    0,
                                    0,
                                )
                            )
        self.db.executemany(
            "insert into amounts values(?,?,?,?,?,?,?,?,?,?) "
            "on conflict(scope,gid,view,sign,selected) do update set "
            "q=q+excluded.q,e=e+excluded.e,duration=duration+excluded.duration",
            rows,
        )
        self.db.commit()

        if cc_mask is not None and np.any(valid & cc_mask):
            mask = valid & cc_mask
            unique, inverse = np.unique(steps[mask], return_inverse=True)
            integral = np.bincount(inverse, weights=(inc[mask, 0] - inc[mask, 1]) * 3600.0)
            duration = np.bincount(inverse, weights=dt[mask])
            self._add_cc(
                {
                    (int(step), view, selected): [float(q), float(d)]
                    for step, q, d in zip(unique, integral, duration)
                    for view in ("measured", "cycling")
                    for selected in (0, 1)
                }
            )

    def resolve_reference(self, index):
        """Resolve the fixed per-trace manual or measured branch capacity; missing, nonpositive, or partial support yields an explicit unavailable reference."""
        cfg = self.config
        if cfg.reference_capacity_mAh is not None:
            return {"value_mAh": cfg.reference_capacity_mAh, "source": "manual", "view": "cycling"}
        scope = "half_cycle" if cfg.reference_half_cycle_id is not None else "cycle"
        gid = (
            cfg.reference_half_cycle_id
            if scope == "half_cycle"
            else cfg.reference_cycle_id
            if cfg.reference_cycle_id is not None
            else 2
        )
        if scope == "half_cycle":
            with index.connect() as db:
                r = db.execute(
                    "select main_sign from runs where half_cycle_id=? and main_sign!=0 limit 1",
                    (gid,),
                ).fetchone()
            sign = r[0] if r else 0
        else:
            sign = (
                1
                if (
                    cfg.reference_direction
                    or ("discharge" if cfg.cycle_order == "charge_then_discharge" else "charge")
                )
                == "charge"
                else -1
            )
        r = self.db.execute(
            "select q,q_invalid from amounts where scope=? and gid=? and view='cycling' and sign=? and selected=0",
            (scope, gid, sign),
        ).fetchone()
        value = r[0] if r and r[0] > 0 and r[1] == 0 else None
        return {
            "value_mAh": value,
            "source": "measured_branch",
            "scope": scope,
            "group_id": gid,
            "direction": DIR.get(sign, "unknown"),
            "view": "cycling",
            "assignment_revision": index.revision,
            "reason": None if value else "reference_missing_partial_or_nonpositive",
        }

    def rows(self, index, trace_id, result_id):
        """Yield long metrics with units, coverage, selection scope, and reference-dependent validity from disk-backed aggregates."""
        self.db.row_factory = sqlite3.Row
        with index.connect() as db:
            # Join lookups remain on disk; branch IDs are not loaded into an unbounded dict.
            reference = (
                self.resolve_reference(index) if "c_rate" in getattr(self, "analyses", ()) else None
            )
            for r in self.db.execute(
                "select * from cc_moments where selected=1 order by step_id,view"
            ):
                info = self._step_info(r["step_id"])
                value = r["integral_mA_s"] / r["duration_s"]
                for metric, number, unit in (
                    ("cc_mean_current", value, "mA"),
                    (
                        "cc_mean_c_rate",
                        value / reference["value_mAh"]
                        if reference and reference["value_mAh"]
                        else None,
                        "h^-1",
                    ),
                ):
                    if metric == "cc_mean_c_rate" and reference is None:
                        continue
                    yield {
                        "trace_id": trace_id,
                        "result_id": result_id,
                        "assignment_revision": index.revision,
                        "scope": "step",
                        "group_id": r["step_id"],
                        "view": r["view"],
                        "direction": DIR.get(info.get("sign", 2), "unknown"),
                        "half_cycle_id": None,
                        "cycle_id": None,
                        "duration_observed_s": r["duration_s"],
                        "selection_partial": bool(
                            self.selection.clauses or self.selection.row_limit is not None
                        ),
                        "protocol_complete": None,
                        "metric": metric,
                        "value": number,
                        "unit": unit,
                        "quality": "observed_complete" if number is not None else "invalid",
                        "invalid_intervals": 0,
                        "reason": None if number is not None else "reference_unavailable",
                    }
            for r in self.db.execute(
                "select * from amounts where selected=1 order by scope,gid,view,sign"
            ):
                full = self.db.execute(
                    "select duration from amounts where scope=? and gid=? and view=? and sign=? and selected=0",
                    (r["scope"], r["gid"], r["view"], r["sign"]),
                ).fetchone()
                selection_partial = bool(full and abs(full[0] - r["duration"]) > 1e-9)
                branch = (
                    db.execute(
                        "select half_cycle_id,cycle_id,main_sign,time_section_id from runs where "
                        + ("half_cycle_id" if r["scope"] == "half_cycle" else "cycle_id")
                        + "=? and main_sign!=0 limit 1",
                        (r["gid"],),
                    ).fetchone()
                    if r["scope"] in {"half_cycle", "cycle"}
                    else None
                )
                base = {
                    "trace_id": trace_id,
                    "result_id": result_id,
                    "assignment_revision": index.revision,
                    "scope": r["scope"],
                    "group_id": r["gid"],
                    "view": r["view"],
                    "direction": DIR[r["sign"]],
                    "half_cycle_id": r["gid"] if r["scope"] == "half_cycle" else None,
                    "cycle_id": r["gid"]
                    if r["scope"] == "cycle"
                    else branch[1]
                    if branch
                    else None,
                    "duration_observed_s": r["duration"],
                    "selection_partial": selection_partial,
                    "protocol_complete": None,
                }
                for metric, col, unit, invalid in (
                    ("capacity", "q", "mAh", "q_invalid"),
                    ("energy", "e", "Wh", "e_invalid"),
                ):
                    quality = "partial" if r[invalid] or selection_partial else "observed_complete"
                    yield {
                        **base,
                        "metric": metric,
                        "value": r[col],
                        "unit": unit,
                        "quality": quality,
                        "invalid_intervals": r[invalid],
                        "reason": "missing_intervals_or_partial_selection"
                        if quality == "partial"
                        else None,
                    }
            for r in self.db.execute(
                "select distinct gid,view from amounts where scope='cycle' and selected=1 and gid>0 order by gid,view"
            ):
                ch = self.db.execute(
                    "select q,e,q_invalid,e_invalid from amounts where scope='cycle' and gid=? and view=? and sign=1 and selected=1",
                    (r["gid"], r["view"]),
                ).fetchone()
                dis = self.db.execute(
                    "select q,e,q_invalid,e_invalid from amounts where scope='cycle' and gid=? and view=? and sign=-1 and selected=1",
                    (r["gid"], r["view"]),
                ).fetchone()
                actual = db.execute(
                    "select count(distinct main_sign) from runs where cycle_id=? and main_sign in (-1,1)",
                    (r["gid"],),
                ).fetchone()[0]
                complete_selection = True
                for sign in (-1, 1):
                    durations = self.db.execute(
                        "select selected,duration from amounts where scope='cycle' and gid=? and view=? and sign=?",
                        (r["gid"], r["view"], sign),
                    ).fetchall()
                    values = {entry[0]: entry[1] for entry in durations}
                    complete_selection &= (
                        0 in values and 1 in values and abs(values[0] - values[1]) <= 1e-9
                    )
                for k, metric, invalid in (
                    (0, "coulombic_efficiency", 2),
                    (1, "energy_efficiency", 3),
                ):
                    good = (
                        actual == 2
                        and ch
                        and dis
                        and ch[k] > 0
                        and ch[invalid] == 0
                        and dis[invalid] == 0
                        and complete_selection
                    )
                    yield {
                        "trace_id": trace_id,
                        "result_id": result_id,
                        "assignment_revision": index.revision,
                        "scope": "cycle",
                        "group_id": r["gid"],
                        "view": r["view"],
                        "direction": "both",
                        "half_cycle_id": None,
                        "cycle_id": r["gid"],
                        "duration_observed_s": None,
                        "selection_partial": not complete_selection,
                        "protocol_complete": None,
                        "metric": metric,
                        "value": dis[k] / ch[k] if good else None,
                        "unit": "fraction",
                        "quality": "observed_complete" if good else "invalid",
                        "invalid_intervals": 0,
                        "reason": None
                        if good
                        else "missing_pair_partial_coverage_or_nonpositive_denominator",
                    }

    def close(self):
        """Release owned resources and stop further use. Plot sessions cancel producers/timers and close the widget; summary stores close SQLite handles."""
        self.db.close()
        if self._index_db is not None:
            self._index_db.close()
