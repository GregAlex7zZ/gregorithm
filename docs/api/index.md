# API recipes

## Capacity, energy and cycles

```python
capacity = cell.capacity(include_energy=True)
capacity.table()
capacity.cycles([2, 3]).table()
capacity.summary().table()
capacity.normalize(mass_g=0.005, area_cm2=1.13).table()
```

## Optional step interpretation

```python
steps = cell.steps()
steps.table()
steps.quantities(include_energy=True).table()
steps.types("CC").c_rate(capacity=capacity).table()
steps.measurements().preview()
```

The source gains a `step_type` column only through the explicit `measurements()` view.

## Pulse discovery and DCIR

```python
events = cell.events()
events.table()
events.dcir(delay_s=10).table()
```

Discovery, acceptance and resistance measurement are separate operations. One delay is
evaluated per DCIR call. Manual boundaries are supported through `ManualPulse`.

## dQ/dV

```python
curves = cell.dqdv(cycles=[2])
curves.table()
curves.outcomes()
curves.plot()
```

The corrected historical profile is the default. A fixed voltage-step profile is an
explicit alternative. CV support is always excluded.

## Progressive plotting

```python
session = cell.plot_progressive(y="voltage_V")
session.start()
```

The point limit applies to the complete figure and affects visualization only.
