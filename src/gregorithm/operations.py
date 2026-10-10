"""Single-operation conveniences over the same batch/state engines."""

import uuid

import pyarrow as pa

from .config import AnalysisConfig
from .integration import integrated_batches
from .io import TableView


def segment(trace, *, config=None, events=None, cancellation=None, on_progress=None):
    """Build or reuse the disk-backed segmentation index. Example: index = segment(trace, config=AnalysisConfig()). Observed actions remain separate from confirmed cycle interpretation.

    Parameters
    ----------
    trace
        BatteryTrace source handle; raw signals are never rewritten.
    config
        Validated immutable AnalysisConfig or DQDVConfig as appropriate; None uses operational defaults.
    events
        Explicit EventSelection with source identity and decision/boundary revision.
    cancellation
        CancellationToken checked at cooperative boundaries; None creates a local token.
    on_progress
        Callable receiving serializable ProgressEvent records during execution."""
    return trace.ensure_index(config or AnalysisConfig(), events, cancellation, on_progress)


def integrate(trace, *, config=None, events=None, cancellation=None):
    """Return a lazy TableView of analytic point-signal integrals, preserving source columns. Example: integrate(trace).head(100).to_pandas(). Exclusions require an explicit accepted EventSelection.

    Parameters
    ----------
    trace
        BatteryTrace source handle; raw signals are never rewritten.
    config
        Validated immutable AnalysisConfig or DQDVConfig as appropriate; None uses operational defaults.
    events
        Explicit EventSelection with source identity and decision/boundary revision.
    cancellation
        CancellationToken checked at cooperative boundaries; None creates a local token."""
    config = config or AnalysisConfig()
    idx = trace.ensure_index(
        config, events if events and events.accepted_ranges() else None, cancellation=cancellation
    )
    manifest = {
        **trace.manifest(),
        "assignment_revision": idx.revision,
        "analysis": "integration",
        "config": config.to_dict(),
        "config_hash": config.config_hash,
        "result_id": uuid.uuid4().hex,
        "analysis_id": uuid.uuid4().hex,
        "selection_id": trace.select().selection_id,
        "parent_result_ids": [],
    }

    def batches():
        """Yield bounded rows from the indexed segment/source table, retaining explicit scientific identity."""
        for block in integrated_batches(trace, idx, config, events, cancellation):
            yield block.append_column(
                "result_id", pa.array([manifest["result_id"]] * block.num_rows)
            )

    return TableView(
        batches,
        manifest,
        memory_bytes=config.managed_memory_bytes,
    )


def summarize(source, **kwargs):
    """Return long capacity/energy/efficiency metrics from the shared pipeline. Example: summarize(trace).to_pandas(max_rows=200000). Keyword arguments are forwarded to analyze().

    Parameters
    ----------
    source
        Trace, selection, collection, or supported lazy result/view; see the return-scope contract.
    kwargs
        Keyword overrides forwarded to the documented configuration or shared pipeline."""
    from .analysis import analyze

    result = analyze(source, analyses=("capacity", "energy", "summary"), **kwargs)
    return result.table("metrics") if hasattr(result, "by_trace") else result.metrics


def calculate_c_rate(source, **kwargs):
    """Return analysis results with signed current/reference-capacity C-rate in h^-1. Example: calculate_c_rate(trace, config=AnalysisConfig(reference_capacity_mAh=1.0)). Missing references do not trigger fallback.

    Parameters
    ----------
    source
        Trace, selection, collection, or supported lazy result/view; see the return-scope contract.
    kwargs
        Keyword overrides forwarded to the documented configuration or shared pipeline."""
    from .analysis import analyze

    return analyze(source, analyses=("capacity", "c_rate"), **kwargs)
