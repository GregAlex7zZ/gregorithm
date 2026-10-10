# User guide

Prepare a source once, then reuse the prepared Parquet file for calculations:

```python
from gregorithm import prepare_source

cell = prepare_source("cell.csv", "cell.prepared.parquet")
capacity = cell.capacity(include_energy=True)
capacity.table()
```

Required columns are `time_s`, `current_mA` and `voltage_V`. Use `battread` to convert
vendor exports or files with other column names before analysis. `ReadOptions` controls
low-level parsing of an already standardized text file; it does not infer scientific
column mappings. Positive current means charge and negative current means discharge.

Calculations return small result handles. Call `.table()` for bounded summaries,
`.plot()` for a final figure, `.plot_progressive()` for updates during computation and
`.write_parquet()` for persistent output.

See the [notebook](../../examples/getting_started.ipynb), [API recipes](../api/index.md),
[large-file guide](large-files.md) and [scientific methods](../scientific/methods.md).
