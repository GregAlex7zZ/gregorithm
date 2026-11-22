"""Standalone long CSV/Parquet export for accepted current GITT/ICI results."""

from __future__ import annotations

from pathlib import Path

from gregorithm.contracts import ExportFormat, ExportReceipt, ExportRequest
from gregorithm.exporting import export_table
from gregorithm.progress import CancelToken, ProgressSink
from gregorithm.storage import open_result

from .persistence import GITT_ICI_ACCEPTED_TABLE_PATH, _manifest


class NoAcceptedResultsError(ValueError):
    """Raised before creating a destination when the published snapshot has no accepted D."""


def export_accepted_results(
    result_path: Path,
    destination: Path,
    format: ExportFormat,
    *,
    progress: ProgressSink | None = None,
    cancel: CancelToken | None = None,
) -> ExportReceipt:
    """Export accepted, current, finite D rows in the stable long schema."""
    info = open_result(result_path)
    row_counts = dict(_manifest(info.ref.path).row_counts)
    if row_counts.get(GITT_ICI_ACCEPTED_TABLE_PATH, 0) == 0:
        raise NoAcceptedResultsError("no accepted current diffusion result is available")
    return export_table(
        ExportRequest(result_path, GITT_ICI_ACCEPTED_TABLE_PATH, destination, format),
        progress=progress,
        cancel=cancel,
    )
