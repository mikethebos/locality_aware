#!/usr/bin/env python3
"""Parse srun --cpu-bind=verbose output and print (world rank, cpu bit on), sorted by rank.

Usage: bind_bits.py LOGFILE      (or pipe via stdin: cat log | bind_bits.py)
Multi-bit masks print as a comma-separated list of bits.
"""
import re, sys

PAT = re.compile(r'task\s+(\d+)\s+\d+\s+\[\d+\]:\s+mask\s+(0x[0-9a-fA-F]+)')

def bits(mask):
    v, out, i = int(mask, 16), [], 0
    while v:
        if v & 1:
            out.append(i)
        v >>= 1
        i += 1
    return out

def main():
    src = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
    rows = {}
    for line in src:
        m = PAT.search(line)
        if m:
            rows[int(m.group(1))] = bits(m.group(2))
    for rank in sorted(rows):
        print(rank, ",".join(map(str, rows[rank])))

    print("unique cpuids:", len(set([cpu for cpus in rows.values() for cpu in cpus])))

if __name__ == "__main__":
    main()
