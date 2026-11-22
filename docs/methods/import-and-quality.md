# Import and data quality

## Units and sign

Canonical columns are `time_s` (s), `current_a` (A) and `voltage_v` (V), stored as float64.
Source units are converted with exact factors: time s, ms (1/1000), min (60), h (3600); current
A, mA (1/1000), uA (1/1 000 000); voltage V, mV (1/1000).

**Positive current is always charge, negative current is discharge.** There is no option to
invert the sign and no inference from step labels or other columns. Convert sources that use the
opposite convention before import.

Elapsed time may be numeric or duration text `H+:MM:SS[.fraction]` (hours may exceed 23; minutes
and seconds are 00–59; the fraction separator is the chosen decimal separator). It is elapsed
time, not a clock time: `179:22:08` means 645 728 s. Fractions are never truncated.

## Column recognition

The headers `time_s,current_a,voltage_v` and `time_s,current_mA,voltage_V` are recognized
directly. Every other layout needs an explicit `ColumnMapping`; units are never guessed. Unmapped
columns are not copied. The source file is never modified.

## Row cleaning (in this order)

1. Remove every row whose current or voltage is missing, non-numeric or non-finite.
2. Validate time only on the remaining rows. Missing or invalid time aborts the import and
   reports the source row.
3. For equal timestamps keep the first surviving row and discard the later ones (warning).
4. A time smaller than the previous retained time aborts the import (no sorting or splitting).
5. At least two rows must remain.

Removed rows simply do not exist for later analyses: their neighbours become adjacent. Irregular
spacing and long gaps receive no special treatment. Import warnings are reported once in the
import report and are not propagated to later results as quality flags.

## Example

A row at 2 min, +250 mA, 3700 mV (units min/mA/mV) becomes 120.0 s, +0.25 A, 3.7 V: a charge
observation. A later row with the same time is discarded as a duplicate; a later row with a
smaller time aborts the import. A row with `I = NaN` is removed *before* the time checks, so its
time can never cause an error or reserve a timestamp.
