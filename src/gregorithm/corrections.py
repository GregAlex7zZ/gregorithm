"""Versioned, declarative manual step-mode corrections; raw signals never change."""

import sqlite3
from dataclasses import asdict, dataclass

from .config import AnalysisConfig, digest
from .segmentation import SegmentIndex


@dataclass(frozen=True)
class StepCorrection:
    """Inclusive original sample range, acquisition section, replacement regime, and required scientific reason. Corrections annotate mode/role without rewriting observed current or voltage."""

    sample_start: int
    sample_end: int
    mode: str
    reason: str
    time_section_id: int
    role: str = "ordinary"

    def __post_init__(self):
        """Validate immutable constructor settings and reject unsupported or scientifically contradictory values."""
        if self.sample_start < 0 or self.sample_end < self.sample_start:
            raise ValueError("Invalid original sample range")
        if self.mode not in {"CC", "CV", "rest", "variable", "unknown"}:
            raise ValueError("mode")
        if not self.reason:
            raise ValueError("A scientific/manual correction requires a reason")

    def to_dict(self):
        """Return serializable configuration/provenance fields without materializing signal data."""
        return asdict(self)


def apply(index, trace, corrections, config):
    """Copy and revise a SQLite step index from declarative edits, then publish only after database handles are closed on Windows.

    Parameters
    ----------
    index
        Disk-backed SegmentIndex captured for the scientific assignment revision.
    trace
        BatteryTrace source handle; raw signals are never rewritten.
    corrections
        Declarative StepCorrection objects/dictionaries with original inclusive support and reasons.
    config
        Validated immutable AnalysisConfig or DQDVConfig as appropriate; None uses operational defaults."""
    edits = sorted(corrections, key=lambda e: e["sample_start"])
    for left, right in zip(edits, edits[1:]):
        if left["sample_end"] >= right["sample_start"]:
            raise ValueError("Overlapping manual corrections require explicit consolidation")
    revision = digest({"parent_assignment_revision": index.revision, "step_corrections": edits})
    target = index.path.with_name(revision + ".sqlite")
    if target.exists():
        return SegmentIndex(target, revision, index.rows, index.trace_id, index.source_revision)
    temporary = target.with_suffix(".working.sqlite")
    if temporary.exists():
        temporary.unlink()
    with index.connect() as source, sqlite3.connect(temporary) as destination:
        source.backup(destination)
        destination.row_factory = sqlite3.Row
        destination.execute(
            "create table corrected_steps (step_id integer primary key,start integer,end integer,t_start real,t_end real,time_section_id integer,sign integer,mode text,reason text,i_level real,role text,parent_step_id integer)"
        )
        number = 0
        for record in source.execute("select * from steps order by start"):
            row = dict(record)
            cuts = {row["start"], row["end"] + 1}
            affected = []
            for edit in edits:
                if edit["sample_start"] <= row["end"] and edit["sample_end"] >= row["start"]:
                    if edit["time_section_id"] != row["time_section_id"]:
                        raise ValueError("A correction cannot cross time sections")
                    cuts.add(max(row["start"], edit["sample_start"]))
                    cuts.add(min(row["end"] + 1, edit["sample_end"] + 1))
                    affected.append(edit)
            cuts = sorted(cuts)
            for start, end_exclusive in zip(cuts[:-1], cuts[1:]):
                number += 1
                mode = row["mode"]
                reason = row["reason"]
                role = "ordinary"
                for edit in affected:
                    if edit["sample_start"] <= start <= edit["sample_end"]:
                        mode = edit["mode"]
                        reason = "manual: " + edit["reason"]
                        role = edit.get("role", "ordinary")
                t0 = row["t_start"]
                t1 = row["t_end"]
                if start != row["start"] or end_exclusive - 1 != row["end"]:
                    # Avoid recursion into trace.ensure_index while applying corrections.
                    from .io import filter_batch
                    from .segmentation import annotated_batches

                    clause = ({"sample_id": {"min": start, "max": end_exclusive - 1}},)
                    first = last = None
                    for block in annotated_batches(trace, index, config, clauses=clause):
                        block = filter_batch(block, clause)
                        if block.num_rows:
                            if first is None:
                                first = block.column("time_s")[0].as_py()
                            last = block.column("time_s")[-1].as_py()
                    t0 = first
                    t1 = last
                destination.execute(
                    "insert into corrected_steps values(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        number,
                        start,
                        end_exclusive - 1,
                        t0,
                        t1,
                        row["time_section_id"],
                        row["sign"],
                        mode,
                        reason,
                        row["i_level"],
                        role,
                        row["step_id"],
                    ),
                )
        destination.execute("drop table steps")
        destination.execute("alter table corrected_steps rename to steps")
        destination.execute("create index step_ranges on steps(start,end)")
        destination.commit()
    destination.close()
    temporary.replace(target)
    return SegmentIndex(target, revision, index.rows, index.trace_id, index.source_revision)


def correct_steps(trace, corrections, *, config=None):
    """Apply nonoverlapping declarative regime corrections and return a new assignment index. Example: correct_steps(trace, [StepCorrection(10,19,"CV","Protocol annotation",0)]). Earlier results retain their captured revision and raw signals never change.

    Parameters
    ----------
    trace
        BatteryTrace source handle; raw signals are never rewritten.
    corrections
        Declarative StepCorrection objects/dictionaries with original inclusive support and reasons.
    config
        Validated immutable AnalysisConfig or DQDVConfig as appropriate; None uses operational defaults."""
    config = config or AnalysisConfig()
    edits = tuple(
        c.to_dict() if isinstance(c, StepCorrection) else StepCorrection(**c).to_dict()
        for c in corrections
    )
    if any(e["sample_end"] >= trace.ensure_index(config).rows for e in edits):
        raise ValueError("Correction references samples outside the source")
    trace._mode_corrections = edits
    trace._index_config = None
    return trace.ensure_index(config)
