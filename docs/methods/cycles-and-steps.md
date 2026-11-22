# Cycles, half-cycles and steps

## Rest

With rest threshold ε ≥ 0 (default 0 A), a point is rest when `|I| ≤ ε`. The measured current is
never changed, even when it lies within the threshold.

## Half-cycles

- Points before the first non-rest point have half-cycle 0 and cycle 0 (leading rest).
- The first non-rest point starts half-cycle 1 with its sign as direction.
- A new half-cycle starts only after **three consecutive** points of the opposite sign; all three
  belong to the new half-cycle. A rest point resets the count.
- Shorter opposite-sign runs stay in the current half-cycle, with their measured current.
- Rest after active current keeps the current half-cycle label; rest never starts a half-cycle.

Example: the signs `+ + + − − 0 −` do not confirm a discharge, while `+ + + − − 0 − − −` confirm
it at the first of the last three negatives.

## Cycles

Half-cycles are paired according to `cycle_order` (default charge → discharge). If half-cycle 1
has the selected first direction, half-cycles (1, 2) form cycle 1, (3, 4) cycle 2, and so on.
Otherwise half-cycle 1 alone is cycle 1 (incomplete) and (2, 3) form cycle 2. A final cycle with
only its first phase is incomplete; this says nothing about whether the experiment finished.

## Steps: rest, CC, CV, other

Classification runs separately in each uninterrupted region of non-rest points with the same
half-cycle, in time order:

1. Take the next three unassigned points. They are **CC** if
   `max(I) − min(I) ≤ cc_relative_tolerance × |mean(I)|` (default 1 %). The CC segment then grows
   one point at a time while the whole segment still satisfies the inequality; the first
   violating point starts a new search.
2. Otherwise they start **CV** if `max(V) − min(V) ≤ cv_absolute_tolerance_v` (default 1 mV) and
   the current is not stable: each of the three currents differs from the mean current of the
   preceding CC segment by more than the CC tolerance or, without a preceding CC segment, the
   three currents fail the CC test.
3. Once CV is confirmed, every later point of the region is CV (CV is latched).
4. If neither applies, the window moves forward by one point; points never assigned are **Other**.

CC has priority over CV, and CC and CV need at least three consecutive points. As a consequence, a
`CC → CV → CC` sequence without rest is classified `CC → CV → CV`.

Stored codes: 0 rest, 1 CC, 2 CV, 3 other. The CC segments found here are the only CC definition
used by C-rate, DCIR and dQ/dV.
