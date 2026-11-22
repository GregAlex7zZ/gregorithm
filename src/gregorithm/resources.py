"""Disk-space admission helpers for operations that write results.

Typed estimators (`estimate_import`, `estimate_export`) compute conservative byte counts from the
actual source. `require_free_space` compares such an estimate with the free space of the
destination volume before any output is staged.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from .contracts import OperationRequest, ResourceEstimate
from .diagnostics import Diagnostic, DiagnosticCode, DiagnosticKind


class ResourceEstimationError(RuntimeError):
    """Raised when a resource estimate is impossible or space is insufficient.

    The `diagnostic` attribute holds `ESTIMATE_INVALID_REQUEST` or `INSUFFICIENT_DISK_SPACE`.
    """

    def __init__(self, diagnostic: Diagnostic) -> None:
        """Store the diagnostic and use its message as text."""
        self.diagnostic = diagnostic
        super().__init__(diagnostic.message)


def estimate(request: OperationRequest) -> ResourceEstimate:
    """Reject generic resource estimates, which cannot be computed reliably.

    `OperationRequest` carries no typed source, schema or format information, so no conservative
    disk bound can be derived from it. Use the typed estimators `estimate_import` and
    `estimate_export` instead. This function always raises `ResourceEstimationError` with
    `ESTIMATE_INVALID_REQUEST`.
    """
    raise ResourceEstimationError(
        Diagnostic(
            DiagnosticCode.ESTIMATE_INVALID_REQUEST,
            DiagnosticKind.ERROR,
            (
                "This operation has no qualified generic resource model; use its "
                "typed source-inspecting estimator when available."
            ),
            operation=request.operation.value,
        )
    )


def require_free_space(
    root: Path,
    resource_estimate: ResourceEstimate,
    *,
    available_bytes: Callable[[Path], int] | None = None,
) -> int:
    """Check that `root` has room for `resource_estimate` and return free bytes.

    The nearest existing ancestor of `root` is probed (by default with `shutil.disk_usage`). A
    `ResourceEstimationError` with `INSUFFICIENT_DISK_SPACE` is raised when free space is below
    `resource_estimate.required_free_bytes`. Free space can change after the check; the result is
    an admission decision, not a reservation.
    """
    probe = available_bytes or (lambda path: shutil.disk_usage(path).free)
    nearest = root
    while not nearest.exists() and nearest != nearest.parent:
        nearest = nearest.parent
    available = int(probe(nearest))
    if available < resource_estimate.required_free_bytes:
        raise ResourceEstimationError(
            Diagnostic(
                DiagnosticCode.INSUFFICIENT_DISK_SPACE,
                DiagnosticKind.ERROR,
                "Insufficient incremental free space for the requested operation.",
            )
        )
    return available
