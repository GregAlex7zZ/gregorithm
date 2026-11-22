# Getting started

This page is for battery researchers who are new to Python packages. It takes about ten minutes.

## 1. Install Python and gregorithm

Install CPython 3.12 or newer from [python.org](https://www.python.org/downloads/). Then open a
terminal in a folder of your choice and create an isolated environment:

```console
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS/Linux) and
install the library from the extracted source directory. To follow the example notebooks, install
the optional `notebooks` extra:

```console
python -m pip install --upgrade pip
python -m pip install ".[notebooks]"
```

The `notebooks` extra adds JupyterLab, an IPython kernel and Matplotlib. They are only needed for
the examples; the library itself requires only NumPy and PyArrow.

## 2. Run the example notebooks

```console
python -m jupyter lab
```

JupyterLab opens in your browser but runs locally. Open `examples/notebooks`, start with
`01_getting_started_import.ipynb` and choose **Run → Run All Cells**. Continue in numerical order.
Each notebook is independent, uses a tiny synthetic dataset and writes its results to a temporary
folder that is deleted when the notebook kernel stops. See the [notebook index](../examples/README.md).

## 3. Prepare your own data

gregorithm needs three columns: elapsed time, current and voltage.

- Files with the header `time_s,current_a,voltage_v` or `time_s,current_mA,voltage_V` are
  recognized automatically.
- Any other CSV, TXT or Parquet layout is imported with an explicit `ColumnMapping` that names
  the three columns and their units (s, ms, min, h / A, mA, uA / V, mV). Elapsed-duration text
  such as `179:22:08.5` is also supported.
- **Positive current must mean charge.** gregorithm never inverts or infers the sign. If your
  instrument exports discharge as positive, convert the file first.

The source file is only read, never modified. Keep private data outside the repository folder.

## 4. Analyse

Follow the [user guide](user-guide.md) for the complete workflow:

1. inspect and import the source;
2. elaborate it (cycles, half-cycles, steps and cumulative capacity/energy);
3. compute Capacity, efficiency and C-rate;
4. detect, review and measure DCIR pulses, and optionally recalculate cycles;
5. compute dQ/dV curves;
6. export tables and record provenance.

## Reading results responsibly

- Always report units, the gregorithm version, parameters and result fingerprints.
- Treat diagnostics as part of the result. A `null` value is *unavailable*, not zero.
- DCIR candidates must be reviewed before they are accepted and measured.
- dQ/dV smoothing is an explicit, parameterized transformation; raw values are always kept.
- The synthetic examples teach the API; they do not validate a laboratory method or instrument.

The scientific rules are described in [Scientific methods](methods/README.md).
