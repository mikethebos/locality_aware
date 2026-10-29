#!/usr/bin/env python3
"""Parse srun --cpu-bind=verbose output; print runs of consecutive ranks whose
CPU bit increments by 1, one run per line:  <rank range>  <bit range>"""
import re, sys

PAT = re.compile(r'task\s+(\d+)\s+\d+\s+\[\d+\]:\s+mask\s+(0x[0-9a-fA-F]+)')

def bits(mask):
    v, out, i = int(mask, 16), [], 0
    while v:
        if v & 1:
            out.append(i)
        v >>= 1
        i += 1
    return tuple(out)

def rng(a, b):
    return str(a) if a == b else f"{a}-{b}"

def fmt(first, last):
    if len(first) == 1:
        return rng(first[0], last[0])
    return ",".join(rng(a, b) for a, b in zip(first, last))

src = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
rows = {}
for line in src:
    m = PAT.search(line)
    if m:
        rows[int(m.group(1))] = bits(m.group(2))

start = prev_rank = None
for rank in sorted(rows):
    b = rows[rank]
    if start is not None and rank == prev_rank + 1 and \
       len(b) == len(prev_bits) and all(x == y + 1 for x, y in zip(b, prev_bits)):
        pass
    else:
        if start is not None:
            print(rng(start[0], prev_rank), fmt(start[1], prev_bits))
        start = (rank, b)
    prev_rank, prev_bits = rank, b
if start is not None:
    print(rng(start[0], prev_rank), fmt(start[1], prev_bits))
