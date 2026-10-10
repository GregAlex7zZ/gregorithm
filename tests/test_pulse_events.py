"""Synthetic pulse-discovery tests; waveform matches are not protocol validation."""

from dataclasses import replace

import numpy as np
import pytest
from test_half_cycles import prepared

from gregorithm import MemoryBudget, PulseEventConfig, PulseProfile


@pytest.mark.parametrize("batch", [1, 3, 100])
def test_both_directions_boundaries_and_incomplete_events(tmp_path, batch):
    """Expose readable real boundaries and retain incomplete terminal opportunities."""
    source = prepared(
        tmp_path,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 0, 10, 10, 0, 0, -20, -20, -20, -20],
        batch_rows=batch,
    )
    events = source.events()
    try:
        frame = events.table()
        assert frame.event.tolist() == [1, 2]
        assert frame.direction.tolist() == ["charge", "discharge"]
        assert frame.candidate_status.tolist() == ["supported", "unresolved"]
        assert frame.loc[0, ["t0_s", "t1_s", "t2_s", "t3_s"]].tolist() == [1, 2, 3, 4]
        assert frame.relation_to_previous_active.tolist() == ["unknown", "opposite"]
        assert np.isnan(frame.t3_s.iloc[1])
        assert "following rest absent" in frame.candidate_reason.iloc[1]
        assert not {"sample_id", "trace_id", "result_id"} & set(frame.columns)
    finally:
        events.close()


def test_profile_full_brackets_stability_and_missing_current(tmp_path):
    """Require full duration-bracket containment and real whole-pulse current support."""
    source = prepared(
        tmp_path,
        range(15),
        [-1, -1, 0, 0, 0, 10, np.nan, 10.4, 10, 0, 0, 0, -1, -1, -1],
    )
    profile = PulseProfile(rest_before_s=3, pulse_s=4, rest_after_s=3)
    events = source.events(profile=profile)
    try:
        frame = events.table()
        pulse = frame[frame.direction == "charge"].iloc[0]
        assert pulse.pulse_duration_min_s == 3
        assert pulse.pulse_duration_max_s == 5
        assert pulse.has_missing_active_current
        assert pulse.profile_status == "unresolved"
        assert "missing active current" in pulse.profile_reason
    finally:
        events.close()


def test_profile_matched_not_matched_and_zero_rest(tmp_path):
    """Apply configurable tolerances inclusively and allow an absent nominal-zero rest."""
    source = prepared(tmp_path, range(13), [-1, -1, 0, 0, 0, 10, 10.5, 10, 0, 0, 0, -1, -1])
    matched = source.events(profile=PulseProfile(3, 3, 3, current_relative_range=0.05))
    try:
        pulse = matched.table().query("direction == 'charge'").iloc[0]
        assert pulse.profile_status == "matched"
        assert pulse.current_relative_range == pytest.approx(0.5 / 10.5)
    finally:
        matched.close()
    failed = source.events(profile=PulseProfile(3, 3, 3, current_relative_range=0.04))
    try:
        assert failed.table().query("direction == 'charge'").iloc[0].profile_status == "not_matched"
    finally:
        failed.close()
    terminal_root = tmp_path / "terminal"
    terminal_root.mkdir()
    terminal = prepared(terminal_root, [0, 1, 2], [5, 5, 5])
    zero = terminal.events(profile=PulseProfile(0, 2, 0))
    try:
        row = zero.table().iloc[0]
        assert row.profile_status == "unresolved"
        assert "duration boundary support" in row.profile_reason
    finally:
        zero.close()


def test_decisions_are_separate_and_views_are_bounded(tmp_path):
    """Layer review decisions without mutating discovery rows or measurement data."""
    source = prepared(tmp_path, range(8), [0, 0, 2, 2, 2, 0, 0, 0])
    budget = MemoryBudget(8 * 1024**2)
    events = source.events(memory_budget=budget)
    assert budget.used == 0
    try:
        original = events.table()
        decided = events.decide(1, "accepted", reason="known protocol", diagnostic_cycle_mark=True)
        assert original.decision.tolist() == ["unreviewed"]
        row = decided.table().iloc[0]
        assert row.decision == "accepted"
        assert row.diagnostic_cycle_mark
        assert events.statuses([]).table().empty
        with pytest.raises(ValueError, match="accepted"):
            events.decide(1, "rejected", reason="reviewed", diagnostic_cycle_mark=True)
        with pytest.raises(KeyError):
            events.decide(9, "accepted", reason="missing")
        iterator = events.iter_batches(1)
        next(iterator)
        with pytest.raises(ValueError, match="iterators"):
            events.close()
        iterator.close()
    finally:
        events.close()
    assert budget.used == 0


def test_configuration_direction_export_and_source_change(tmp_path):
    """Validate filters, immutable settings, export protection and source fingerprints."""
    source = prepared(tmp_path, range(10), [0, 0, 1, 1, 0, 0, -1, -1, 0, 0])
    only_charge = source.events(config=PulseEventConfig(pulse_direction="charge"))
    try:
        assert only_charge.table().direction.tolist() == ["charge"]
        destination = tmp_path / "events.parquet"
        only_charge.write_parquet(destination)
        assert destination.is_file()
        with pytest.raises(FileExistsError):
            only_charge.write_parquet(destination)
        changed = replace(source, read_options=replace(source.read_options, batch_rows=1))
        changed_events = changed.events()
        try:
            assert changed_events.table().shape[0] == 2
        finally:
            changed_events.close()
    finally:
        only_charge.close()
    with pytest.raises(ValueError):
        PulseEventConfig(min_real_samples=1)
    with pytest.raises(ValueError):
        PulseProfile(-1, 2, 3)


def test_accepted_diagnostic_mark_changes_only_cycle_interpretation(tmp_path):
    """Ignore a marked reversal for cycle creation while retaining all physical Q/E."""
    source = prepared(
        tmp_path,
        range(8),
        [1, 1, 0, -1, -1, 0, 1, 1],
        [4, 4, 4, 3.9, 3.9, 4, 4, 4],
    )
    events = source.events()
    assert events.table().direction.tolist() == ["charge", "discharge", "charge"]
    marked = events.decide(
        2,
        "accepted",
        reason="known diagnostic protocol",
        diagnostic_cycle_mark=True,
    )
    ordinary = source.capacity(include_energy=True)
    corrected = marked.capacity(include_energy=True)
    try:
        assert len(ordinary.table()) == 3
        frame = corrected.details().table()
        assert len(frame) == 1
        totals = source.totals()
        assert frame.charge_mAh.sum() == pytest.approx(totals.values["charge_mAh"])
        assert frame.discharge_mAh.sum() == pytest.approx(totals.values["discharge_mAh"])
        assert frame.charge_Wh.sum() == pytest.approx(totals.values["charge_Wh"])
        assert frame.discharge_Wh.sum() == pytest.approx(totals.values["discharge_Wh"])
        via_source = source.capacity(diagnostic_events=marked)
        try:
            assert via_source.table().shape[0] == 1
        finally:
            via_source.close()
    finally:
        ordinary.close()
        corrected.close()
        events.close()
