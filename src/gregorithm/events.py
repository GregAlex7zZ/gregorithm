"""Event discovery, manual decisions and exclusions have separate APIs."""

import math
from dataclasses import dataclass, field, replace

import pyarrow as pa

from .config import digest
from .io import TableView
from .segmentation import DIR, assignment_config


@dataclass(frozen=True)
class EventConfig:
    """Immutable operational event-candidate criteria, separate from acceptance or resistance measurement. Defaults propose opposite-polarity pulses in both directions, time/sample support, relative brevity, current stability, and front-resolution checks. They require experimental calibration."""

    relation: str = "opposite"
    pulse_direction: str = "both"
    min_samples: int = 2
    min_duration_s: float = 0.1
    relative_duration: float = 0.1
    duration_range_s: tuple | None = None
    current_relative_rms: float = 0.02
    current_absolute_rms_mA: float = 0.0001
    max_front_fraction: float = 0.1

    def __post_init__(self):
        """Validate immutable constructor settings and reject unsupported or scientifically contradictory values."""
        if self.relation not in {"opposite", "same", "any"}:
            raise ValueError("relation")
        if self.pulse_direction not in {"charge", "discharge", "both"}:
            raise ValueError("pulse_direction")
        if self.min_samples < 1 or self.min_duration_s < 0:
            raise ValueError("support")
        if not isinstance(self.min_samples, int) or isinstance(self.min_samples, bool):
            raise ValueError("min_samples must be an integer")
        for name in (
            "min_duration_s",
            "relative_duration",
            "current_relative_rms",
            "current_absolute_rms_mA",
            "max_front_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(name)
        if self.duration_range_s is not None:
            lo, hi = self.duration_range_s
            if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi):
                raise ValueError("duration_range_s")


@dataclass(frozen=True)
class EventSelection:
    """Immutable event records plus decision/boundary revision. Use select(), decide(), and correct(); discovery never excludes automatically. accepted_time_ranges() supplies integration exclusions."""

    events: tuple = ()
    revision: str = "initial"
    parent_revision: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        """Return serializable configuration/provenance fields without materializing signal data."""
        keys = sorted({key for event in self.events for key in event if key != "result_id"})
        return {
            "version": 1,
            "events": [{key: event.get(key) for key in keys} for event in self.events],
            "revision": self.revision,
            "parent_revision": self.parent_revision,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d):
        """Reconstruct this configuration or decision object from its serialized dictionary; constructor validation still applies."""
        return cls(
            tuple(d["events"]), d["revision"], d.get("parent_revision"), d.get("metadata", {})
        )

    def select(self, event_id=None, **criteria):
        """Return an immutable filtered view/selection; repeated predicates intersect. Equality, lists, and min/max intervals are supported. This never truncates scientific computation implicitly.

        Usage: call on the EventSelection object; returned selections/revisions are independent of earlier views."""
        if event_id is not None:
            criteria["event_id"] = event_id
        return replace(
            self,
            events=tuple(
                e
                for e in self.events
                if all(
                    e.get(k) in (v if isinstance(v, (list, tuple)) else [v])
                    for k, v in criteria.items()
                )
            ),
        )

    def decide(self, event_id, decision, *, exclude=False, reason, origin="manual"):
        """Return revised event decisions. Exclusion requires accepted status and an explicit reason; discovery and resistance measurement are unchanged.

        Usage: call on the EventSelection object; returned selections/revisions are independent of earlier views."""
        if decision not in {"accepted", "rejected", "unreviewed"}:
            raise ValueError("decision")
        ids = set(event_id if isinstance(event_id, (list, tuple)) else [event_id])
        found = set()
        out = []
        for e in self.events:
            if e["event_id"] in ids:
                found.add(e["event_id"])
                if exclude and decision != "accepted":
                    raise ValueError("Only accepted events can be excluded")
                e = {
                    **e,
                    "decision": decision,
                    "excluded": exclude,
                    "decision_reason": reason,
                    "decision_origin": origin,
                }
            out.append(e)
        if found != ids:
            raise KeyError(ids - found)
        return EventSelection(
            tuple(out),
            digest({"events": out, "metadata": self.metadata}),
            self.revision,
            self.metadata,
        )

    def correct(self, event_id, *, reason, **boundaries):
        """Return a new event-boundary revision with an explicit reason. Changed sample boundaries require their timestamps; corrections disable automatic acceptance eligibility.

        Usage: call on the EventSelection object; returned selections/revisions are independent of earlier views."""
        if not reason:
            raise ValueError("An explicit correction reason is required")
        allowed = {
            "sample_t0",
            "sample_t1",
            "sample_t2",
            "sample_t3",
            "t0",
            "t1",
            "t2",
            "t3",
            "duration_s",
            "time_section_id",
        }
        if set(boundaries) - allowed:
            raise ValueError("Only declarative boundary edits supported")
        for key in ("t0", "t1", "t2", "t3"):
            if "sample_" + key in boundaries and key not in boundaries:
                raise ValueError("A changed sample boundary requires its explicit timestamp too")
        out = tuple(
            {
                **e,
                **boundaries,
                "correction_reason": reason,
                "auto_eligible": False,
                "duration_origin": "nominal"
                if "duration_s" in boundaries
                else e.get("duration_origin", "unknown"),
            }
            if e["event_id"] == event_id
            else e
            for e in self.events
        )
        if out == self.events:
            raise KeyError(event_id)
        for event in out:
            if event["event_id"] != event_id:
                continue
            sample_boundaries = [event.get("sample_" + key) for key in ("t0", "t1", "t2", "t3")]
            times = [event.get(key) for key in ("t0", "t1", "t2", "t3")]
            for values in (sample_boundaries, times):
                for j, (a, z) in enumerate(zip(values, values[1:])):
                    if a is not None and z is not None and (z < a or (j != 1 and z == a)):
                        raise ValueError("Event boundaries must retain rest-active-rest order")
            if any(value is not None and not math.isfinite(value) for value in times):
                raise ValueError("Finite event timestamps required")
            duration = event.get("duration_s")
            if duration is not None and (not math.isfinite(duration) or duration <= 0):
                raise ValueError("Positive finite event duration required")
        return EventSelection(
            out, digest({"events": out, "metadata": self.metadata}), self.revision, self.metadata
        )

    def accepted_ranges(self):
        """Return inclusive active sample ranges only for events explicitly accepted and excluded."""
        return tuple(
            (e["sample_t1"], e["sample_t2"])
            for e in self.events
            if e.get("excluded") and e["decision"] == "accepted"
        )

    def accepted_time_ranges(self):
        """Return original [t0,t3] exclusion intervals with acquisition-section identity; missing rest boundaries are rejected."""
        ranges = []
        for e in self.events:
            if e.get("excluded") and e["decision"] == "accepted":
                if e.get("t0") is None or e.get("t3") is None:
                    raise ValueError("Exclusion requires explicit t0/t3 boundaries")
                ranges.append((e["time_section_id"], e["t0"], e["t3"]))
        return tuple(ranges)

    def table(self, manifest=None):
        """Return a lazy long table retaining trace identity in each row; no wide pivot or all-file pandas materialization is implicit."""
        manifest = {"event_metadata": self.metadata, **(manifest or {})}

        def batches():
            """Yield bounded rows from the indexed segment/source table, retaining explicit scientific identity."""
            keys = sorted({key for event in self.events for key in event})
            schema = None
            for offset in range(0, len(self.events), 128):
                block = pa.RecordBatch.from_pylist(
                    [
                        {key: event.get(key) for key in keys}
                        for event in self.events[offset : offset + 128]
                    ]
                )
                schema = (
                    block.schema
                    if schema is None
                    else pa.unify_schemas([schema, block.schema], promote_options="permissive")
                )
            for offset in range(0, len(self.events), 128):
                block = pa.RecordBatch.from_pylist(
                    [
                        {key: event.get(key) for key in keys}
                        for event in self.events[offset : offset + 128]
                    ],
                    schema=schema,
                )
                if manifest.get("result_id"):
                    block = block.append_column(
                        "result_id", pa.array([manifest["result_id"]] * block.num_rows)
                    )
                yield block

        return TableView(batches, manifest)

    def auto_accept(self, profile="inverse_return"):
        """Apply only the explicitly requested inverse_return heuristic to eligible candidates, recording automatic decision provenance."""
        if profile != "inverse_return":
            raise ValueError("Explicit supported profile: inverse_return")
        result = self
        for e in self.events:
            if e.get("auto_eligible"):
                result = result.decide(
                    e["event_id"],
                    "accepted",
                    exclude=True,
                    reason="Explicit inverse_return heuristic",
                    origin="auto",
                )
        return result


def manual_event(
    trace,
    *,
    sample_t0,
    sample_t1,
    sample_t2,
    sample_t3=None,
    duration_s=None,
    reason="manual selection",
    config=None,
):
    """Define a diagnostic event by original sample IDs and retain a reason. Return an unreviewed EventSelection. Example: manual_event(trace, sample_t0=10, sample_t1=11, sample_t2=30, sample_t3=31). Cross-section or unordered boundaries are rejected.

    Parameters
    ----------
    trace
        BatteryTrace source handle; raw signals are never rewritten.
    sample_t0
        Original last-rest sample ID before pulse onset.
    sample_t1
        Original first active pulse sample ID.
    sample_t2
        Original last active pulse sample ID.
    sample_t3
        Original first post-pulse rest sample ID; required for exclusion.
    duration_s
        Optional nominal event duration in seconds; absence retains estimated/unknown origin.
    reason
        Required human-readable scientific/protocol justification retained in provenance.
    config
        Validated immutable AnalysisConfig or DQDVConfig as appropriate; None uses operational defaults."""
    from .dcir import fetch_samples

    indices = [sample_t0, sample_t1, sample_t2] + ([] if sample_t3 is None else [sample_t3])
    rows = fetch_samples(trace, indices, config)
    r0, r1, r2 = [rows[n] for n in (sample_t0, sample_t1, sample_t2)]
    r3 = rows.get(sample_t3)
    event = {
        "trace_id": trace.trace_id,
        "source_revision": trace.source_revision,
        "event_id": digest(
            {"trace": trace.trace_id, "source": trace.source_revision, "samples": indices}
        )[:32],
        "time_section_id": r1["time_section_id"],
        "sample_t0": sample_t0,
        "sample_t1": sample_t1,
        "sample_t2": sample_t2,
        "sample_t3": sample_t3,
        "t0": r0["time_s"],
        "t1": r1["time_s"],
        "t2": r2["time_s"],
        "t3": r3["time_s"] if r3 else None,
        "duration_s": duration_s
        if duration_s is not None
        else (r3["time_s"] - r0["time_s"] if r3 else None),
        "duration_origin": "nominal"
        if duration_s is not None
        else "estimated"
        if r3
        else "unknown",
        "pulse_direction": r1["direction"],
        "before_direction": "unknown",
        "after_direction": "unknown",
        "relation_to_before": "unknown",
        "candidate_status": "manual",
        "origin": "manual",
        "decision": "unreviewed",
        "excluded": False,
        "auto_eligible": False,
        "decision_reason": reason,
    }
    if not sample_t0 < sample_t1 <= sample_t2 or (sample_t3 is not None and sample_t3 <= sample_t2):
        raise ValueError("Invalid event sample order")
    if r0["time_section_id"] != r2["time_section_id"] or (
        r3 and r3["time_section_id"] != r0["time_section_id"]
    ):
        raise ValueError("Event crosses a time reset")
    result = EventSelection((event,), digest(event), metadata={"definition_origin": "manual"})
    trace._event_selection = result
    return result


def detect_events(trace, config=None, event_config=None):
    """Discover rest-pulse-rest candidates and unresolved excursions without accepting, excluding, or measuring them. Example: events = detect_events(trace); candidates = events.select(candidate_status="candidate"). Criteria are operational heuristics.

    Parameters
    ----------
    trace
        BatteryTrace source handle; raw signals are never rewritten.
    config
        Validated immutable AnalysisConfig or DQDVConfig as appropriate; None uses operational defaults.
    event_config
        EventConfig candidate criteria; independent of decisions and measurement settings."""
    from .config import AnalysisConfig
    from .dcir import pulse_statistics

    cfg = config or AnalysisConfig()
    ec = event_config or EventConfig()
    idx = trace.ensure_index(cfg)
    events = []
    with idx.connect() as db:
        for record in db.execute("select * from runs where sign in (-1,1) order by start"):
            pulse = dict(record)
            preceding = db.execute(
                "select * from runs where run_id=?", (pulse["run_id"] - 1,)
            ).fetchone()
            following = db.execute(
                "select * from runs where run_id=?", (pulse["run_id"] + 1,)
            ).fetchone()
            rest_before = (
                preceding
                if preceding
                and preceding["sign"] == 0
                and preceding["time_section_id"] == pulse["time_section_id"]
                else None
            )
            rest_after = (
                following
                if following
                and following["sign"] == 0
                and following["time_section_id"] == pulse["time_section_id"]
                else None
            )
            if rest_before is None and rest_after is None:
                continue
            before = (
                db.execute(
                    "select * from runs where run_id=?", (rest_before["run_id"] - 1,)
                ).fetchone()
                if rest_before
                else None
            )
            after = (
                db.execute(
                    "select * from runs where run_id=?", (rest_after["run_id"] + 1,)
                ).fetchone()
                if rest_after
                else None
            )
            if before and before["time_section_id"] != pulse["time_section_id"]:
                before = None
            if after and after["time_section_id"] != pulse["time_section_id"]:
                after = None
            bs = before["sign"] if before else 2
            ats = after["sign"] if after else 2
            ps = pulse["sign"]
            relation = (
                "opposite" if bs in (-1, 1) and bs == -ps else "same" if bs == ps else "unknown"
            )
            if ec.pulse_direction != "both" and DIR[ps] != ec.pulse_direction:
                continue
            if ec.relation != "any" and relation not in {ec.relation, "unknown"}:
                continue
            duration = pulse["t_end"] - pulse["t_start"]
            context = [
                x["t_end"] - x["t_start"]
                for x in (before, after)
                if x is not None and x["sign"] in (-1, 1)
            ]
            short = (
                (ec.duration_range_s[0] <= duration <= ec.duration_range_s[1])
                if ec.duration_range_s
                else bool(context) and all(duration <= ec.relative_duration * d for d in context)
            )
            supported = pulse["n"] >= ec.min_samples and duration >= ec.min_duration_s
            t0 = rest_before["t_end"] if rest_before else None
            t3 = rest_after["t_start"] if rest_after else None
            front0 = pulse["t_start"] - t0 if t0 is not None else None
            front1 = t3 - pulse["t_end"] if t3 is not None else None
            e = {
                "trace_id": trace.trace_id,
                "source_revision": trace.source_revision,
                "event_id": digest(
                    {
                        "trace": trace.trace_id,
                        "source": trace.source_revision,
                        "start": pulse["start"],
                        "end": pulse["end"],
                    }
                )[:32],
                "time_section_id": pulse["time_section_id"],
                "sample_t0": rest_before["end"] if rest_before else None,
                "sample_t1": pulse["start"],
                "sample_t2": pulse["end"],
                "sample_t3": rest_after["start"] if rest_after else None,
                "t0": t0,
                "t1": pulse["t_start"],
                "t2": pulse["t_end"],
                "t3": t3,
                "duration_s": t3 - t0 if t0 is not None and t3 is not None else None,
                "duration_origin": "estimated" if t0 is not None and t3 is not None else "unknown",
                "active_duration_s": duration,
                "duration_lower_s": duration,
                "duration_upper_s": t3 - t0 if t0 is not None and t3 is not None else None,
                "on_front_s": front0,
                "off_front_s": front1,
                "rest_duration_s": rest_before["t_end"] - rest_before["t_start"]
                if rest_before
                else None,
                "pulse_direction": DIR[ps],
                "before_direction": DIR[bs],
                "after_direction": DIR[ats],
                "relation_to_before": relation,
                "candidate_status": "candidate"
                if short and supported and rest_before and rest_after
                else "unresolved",
                "origin": "auto",
                "decision": "unreviewed",
                "excluded": False,
                "half_cycle_id": pulse["half_cycle_id"],
                "cycle_id": pulse["cycle_id"],
                "assignment_revision": idx.revision,
            }
            stats = pulse_statistics(trace, e, cfg)
            e.update(stats)
            stable = stats["current_rms_mA"] is not None and stats["current_rms_mA"] <= max(
                ec.current_absolute_rms_mA, ec.current_relative_rms * abs(stats["current_mean_mA"])
            )
            e["auto_eligible"] = bool(
                short
                and supported
                and relation == "opposite"
                and ats == bs
                and stable
                and stats["coverage_complete"]
                and front0 is not None
                and front1 is not None
                and max(front0, front1) <= ec.max_front_fraction * duration
            )
            events.append(e)
            if len(events) * 4096 > cfg.managed_memory_bytes // 8:
                from .diagnostics import BudgetExceeded

                raise BudgetExceeded("Event metadata budget exceeded; narrow detection selection")
    event_metadata = {
        "detection_config": ec.__dict__,
        "assignment_config_hash": digest(assignment_config(cfg)),
        "validation": "operational_proposal_not_experimental_calibration",
    }
    result = EventSelection(
        tuple(events),
        digest({"events": events, "metadata": event_metadata}),
        metadata=event_metadata,
    )
    trace._event_selection = result
    return result
