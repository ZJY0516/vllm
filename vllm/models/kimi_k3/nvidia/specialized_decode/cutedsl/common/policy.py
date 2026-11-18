# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-batch policy tables."""

import os


def _remap_rows():
    """K31_S30K=<q> (verify rows per request, e.g. 6 for k = 5, 7 for k = 6): the
    rows of q-token verify batches up to 64 that no k = 4 batch has (multiples of q,
    not of 5). Every batch table gives such a row the entry of the nearest multiple
    of 5 (its k = 4 neighbour), or none when that row has none. Rows that are
    multiples of 5 keep their entries. Default off (no remapped rows)."""
    try:
        q = int(os.environ.get("K31_S30K", "0") or 0)
    except ValueError:
        return ()
    if q <= 1 or q == 5:
        return ()
    return tuple(r for r in range(q, 65, q) if r % 5)


REMAP_ROWS = _remap_rows()


def batch_table(spec):
    """Per-batch table of integers ("1-2:96,3-8:120"), tuples ("16:64.16.4.8") or names
    ("16-64:trtllm"); later entries override earlier ones."""
    table = {}
    for item in filter(None, spec.split(",")):
        span, value = item.split(":")
        lo, _, hi = span.partition("-")
        if "." in value:
            value = tuple(map(int, value.split(".")))
        elif value.lstrip("-").isdigit():
            value = int(value)
        table.update({b: value for b in range(int(lo), int(hi or lo) + 1)})
    for row in REMAP_ROWS:
        near = 5 * ((row + 2) // 5)
        if near in table:
            table[row] = table[near]
        else:
            table.pop(row, None)
    return table
