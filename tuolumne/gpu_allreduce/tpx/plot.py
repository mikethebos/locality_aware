import os
import sys
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def find_nnodes(fn):
    match = re.search(r'allreduce_N(\d+)\s*', fn)
    if match:
        node_count = int(match.group(1))
    return node_count

def update(times, curr_size, local_times, size, procs_per_gpu, func=min):
    times[procs_per_gpu][curr_size] = func(local_times)
    return times, size, []

def find_size(l):
    match = re.search(r'0:\s+Testing Size\s+(\d+)', l)
    if match:
        size = int(match.group(1))
    return size

def find_timings(l):
    matches = re.findall(r'^\s*\d+:\s*(.+?):\s+([-+]?\d*\.\d+e[-+]\d+)', l)
    out = []
    for k, v in matches:
        out.append((k, float(v)))
    return out

def find_besttimes(lines):
    times = {1: {}}
                
    curr_size = 0
    procs_per_gpu = 1  # Results MUST begin with 1 PPG case
    local_times = []
    for l in lines:
        if "0: Testing Size " in l:
            size = find_size(l)
            if size != curr_size and curr_size != 0:
                times, curr_size, local_times = update(times, curr_size, local_times, size, procs_per_gpu, func=lambda x: x)
            if curr_size == 0:
                curr_size = size
        for (k, v) in find_timings(l):
            local_times.append((k, v))
            
    times, curr_size, local_times = update(times, curr_size, local_times, curr_size, procs_per_gpu, func=lambda x: x)
    return times

def push_sizes_in(times):
    out = {}
    for procs_per_gpu in times.keys():
        out[procs_per_gpu] = {}
        for size in times[procs_per_gpu].keys():
            for (k, v) in times[procs_per_gpu][size]:
                if k not in out[procs_per_gpu].keys():
                    out[procs_per_gpu][k] = []
                out[procs_per_gpu][k].append((size, v))
    return out

def is_size_in(lp, s):
    return s in [x[0] for x in lp]

def filter_big_sizes(times, max_size):
    out = {}
    for procs_per_gpu in times.keys():
        out[procs_per_gpu] = {}
        for size in times[procs_per_gpu].keys():
            if size <= max_size:
                out[procs_per_gpu][size] = times[procs_per_gpu][size]
    return out

if __name__ == "__main__":
    if "NO_TITLE" in os.environ.keys() and int(os.environ["NO_TITLE"]) == 1:
        plt.title = lambda *args, **kwargs: None
    
    fn_in = sys.argv[1]
    with open(fn_in, 'r') as f:
        lines = f.readlines()
    
    fn_out = fn_in + "_plot.pdf"

    locality_lines = lines
    locality_times = find_besttimes(locality_lines)
    
    locality_times = filter_big_sizes(locality_times, 1024)
    
    locality_times = push_sizes_in(locality_times)
    
    pdf = PdfPages(fn_out)
    
    ppg = 1
    data = locality_times[ppg]

    keys = [
        "PMPI Allreduce Time",
        "MPIL GPU-Aware Recursive Doubling Time",
        "MPIL CopyToCPU Recursive Doubling Time",
        "MPIL CopyToCPU Node-Aware Dissemination Time",
        "MPIL CopyToCPU NUMA-Aware Dissemination Time",
    ]

    def to_dict(series):
        # assumes (x, y) pairs
        return {x: y for x, y in series}

    # --- convert PMPI baseline to dict ---
    base_dict = to_dict(data["PMPI Allreduce Time"])
    x_vals = sorted(base_dict.keys())

    # align all methods on PMPI x-axis
    def aligned_series(method):
        d = to_dict(data[method])
        return np.array([d.get(x, np.nan) for x in x_vals])

    base = np.array([base_dict[x] for x in x_vals])

    # =========================================================
    # PLOT 1: TIMINGS
    # =========================================================
    plt.figure()

    for k in keys:
        y = aligned_series(k)
        plt.plot(x_vals, y, label=k.replace("MPIL ", "").replace("CopyToCPU ", "C2C ").replace(" Time", ""))

    plt.title(f"Allreduce Timings\n{find_nnodes(fn_in)} nodes, PPG={ppg}")
    plt.xlabel("Num doubles")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.tight_layout()
    pdf.savefig(plt.gcf())

    # =========================================================
    # PLOT 2: SPEEDUP vs PMPI
    # =========================================================
    plt.figure()

    for k in keys[1:]:
        y = aligned_series(k)
        speedup = base / y
        plt.plot(x_vals, speedup, label=k.replace("MPIL ", "").replace("CopyToCPU ", "C2C ").replace(" Time", ""))

    plt.title(f"Allreduce Speedup vs PMPI\n{find_nnodes(fn_in)} nodes, PPG={ppg}")
    plt.xlabel("Num doubles")
    plt.xscale("log")
    # plt.yscale("log")
    plt.ylim((0.5, 1.5))
    plt.ylabel("Speedup (PMPI / method)")
    plt.legend()
    plt.tight_layout()
    pdf.savefig(plt.gcf())
    
    pdf.close()