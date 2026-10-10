# Large files

Preparation, calculations and exports iterate over Arrow record batches. The normal API
does not add row identifiers and does not require a dictionary of full DataFrames.

- Reuse the prepared Parquet file for repeated analyses.
- Use `preview()` for a few rows and `iter_batches()` for point-level processing.
- Use `write_parquet()` for large derived results.
- Set `managed_memory_bytes` when coordinating multiple files.
- Start with one worker; increase workers only when memory and storage permit it.
- Plot reduction changes display points only and never changes scientific tables.

```python
from gregorithm import prepare_sources

cells = prepare_sources(["cell_a.csv", "cell_b.csv"], "prepared")
run = cells.run("capacity", workers=1, managed_memory_bytes=2 * 1024**3)
```

Files with the same complete basename are rejected because their human-readable keys
would collide.
