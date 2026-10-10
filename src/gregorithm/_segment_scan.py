"""Reference/JIT-identical bounded scan of observed actions and temporal windows."""

import numpy as np


def scan(t, i, v, signs, row, section, previous_t, ri, rf, wi, wf, window_s, splits):
    """Scan bounded original data while carrying the containing operation state across batches.

    Parameters
    ----------
    t
        One bounded float64 timestamp batch in seconds.
    i
        Matching point current batch in mA.
    v
        Matching point voltage batch in V.
    signs
        Bounded operation-specific value described by the function contract and API reference.
    row
        Bounded operation-specific value described by the function contract and API reference.
    section
        Bounded operation-specific value described by the function contract and API reference.
    previous_t
        Bounded operation-specific value described by the function contract and API reference.
    ri
        Bounded operation-specific value described by the function contract and API reference.
    rf
        Bounded operation-specific value described by the function contract and API reference.
    wi
        Bounded operation-specific value described by the function contract and API reference.
    wf
        Bounded operation-specific value described by the function contract and API reference.
    window_s
        Bounded operation-specific value described by the function contract and API reference.
    splits
        Bounded operation-specific value described by the function contract and API reference."""
    n = len(t)
    run_i = np.empty((n + 1, 5), dtype=np.int64)
    run_f = np.empty((n + 1, 6), dtype=np.float64)
    win_i = np.empty((n + 1, 5), dtype=np.int64)
    win_f = np.empty((n + 1, 6), dtype=np.float64)
    nr = nw = 0
    for j in range(n):
        tt = t[j]
        ii = i[j]
        vv = v[j]
        ss = signs[j]
        reset = row > 0 and (not np.isfinite(previous_t) or not np.isfinite(tt) or tt < previous_t)
        if reset:
            if ri[0] >= 0:
                run_i[nr] = ri
                run_f[nr] = rf
                nr += 1
                ri[0] = -1
            if wi[0] >= 0:
                win_i[nw] = wi
                win_f[nw] = wf
                nw += 1
                wi[0] = -1
            section += 1
        split = False
        if len(splits):
            k = np.searchsorted(splits, row)
            split = k < len(splits) and splits[k] == row
        if ri[0] >= 0 and (ri[3] != ss or split):
            run_i[nr] = ri
            run_f[nr] = rf
            nr += 1
            ri[0] = -1
        if wi[0] >= 0 and (wi[3] != ss or tt - wf[0] >= window_s):
            win_i[nw] = wi
            win_f[nw] = wf
            nw += 1
            wi[0] = -1
        for kind in range(2):
            ints = ri if kind == 0 else wi
            floats = rf if kind == 0 else wf
            if ints[0] < 0:
                ints[0] = row
                ints[1] = row
                ints[2] = section
                ints[3] = ss
                ints[4] = 1
                floats[0] = tt
                floats[1] = tt
                floats[2] = ii if np.isfinite(ii) else np.inf
                floats[3] = ii if np.isfinite(ii) else -np.inf
                floats[4] = vv if np.isfinite(vv) else np.inf
                floats[5] = vv if np.isfinite(vv) else -np.inf
            else:
                ints[1] = row
                ints[4] += 1
                floats[1] = tt
                if np.isfinite(ii):
                    floats[2] = min(floats[2], ii)
                    floats[3] = max(floats[3], ii)
                if np.isfinite(vv):
                    floats[4] = min(floats[4], vv)
                    floats[5] = max(floats[5], vv)
        previous_t = tt
        row += 1
    return run_i[:nr], run_f[:nr], win_i[:nw], win_f[:nw], row, section, previous_t


_FAST = None


def backend(name):
    """Select the reference or cached accelerated scanner without changing scientific state rules.

    Parameters
    ----------
    name
        Registry/table/preset name as documented for the containing API."""
    global _FAST
    if name == "numpy":
        return scan
    try:
        import numba
    except ImportError:
        if name == "numba":
            raise
        return scan
    if _FAST is None:
        _FAST = numba.njit(cache=True, nogil=True, fastmath=False)(scan)
    return _FAST
