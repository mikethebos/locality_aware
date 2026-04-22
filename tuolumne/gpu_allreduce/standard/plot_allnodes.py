import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import StrMethodFormatter

def find_nnodes(fn):
    """Extracts node count from filename matching 'allreduce_N(\d+)'."""
    match = re.search(r'allreduce_N(\d+)\s*', fn)
    if match:
        return int(match.group(1))
    return None

def update(times, curr_size, local_times, size, procs_per_gpu):
    times[procs_per_gpu][curr_size] = local_times
    return times, size, []

def find_size(l):
    match = re.search(r'0:\s+Testing Size\s+(\d+)', l)
    if match:
        return int(match.group(1))
    return None

def find_timings(l):
    matches = re.findall(r'^\s*\d+:\s*(.+?):\s+([-+]?\d*\.\d+e[-+]\d+)', l)
    return [(k, float(v)) for k, v in matches]

def find_besttimes(lines):
    times = {1: {}}
    curr_size = 0
    procs_per_gpu = 1
    local_times = []
    for l in lines:
        if "0: Testing Size " in l:
            size = find_size(l)
            if size is not None:
                if size != curr_size and curr_size != 0:
                    times, curr_size, local_times = update(times, curr_size, local_times, size, procs_per_gpu)
                if curr_size == 0:
                    curr_size = size
        for (k, v) in find_timings(l):
            local_times.append((k, v))
    
    if curr_size != 0:
        times, _, _ = update(times, curr_size, local_times, curr_size, procs_per_gpu)
    return times

def push_sizes_in(times):
    out = {}
    for ppg in times:
        out[ppg] = {}
        for size in times[ppg]:
            for (k, v) in times[ppg][size]:
                if k not in out[ppg]:
                    out[ppg][k] = []
                out[ppg][k].append((size, v))
    return out

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_nodes_min.py <input_directory> [prefix] [suffix]")
        sys.exit(1)

    plt.rcParams['axes.labelsize'] = 'large'

    dir_in = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) >= 3 else ""
    suffix = sys.argv[3] if len(sys.argv) >= 4 else ".out"

    node_data = {}
    all_seen_sizes = set()

    for fn in os.listdir(dir_in):
        if fn.startswith(prefix) and fn.endswith(suffix):
            nn = find_nnodes(fn)
            if nn is not None:
                with open(os.path.join(dir_in, fn), 'r') as f:
                    lines = f.readlines()
                    raw_times = find_besttimes(lines)
                    pushed_times = push_sizes_in(raw_times)
                    node_data[nn] = pushed_times
                    
                    if 1 in pushed_times:
                        for method in pushed_times[1]:
                            for size, _ in pushed_times[1][method]:
                                all_seen_sizes.add(size)

    if not node_data:
        print(f"No valid files found in {dir_in}")
        sys.exit(1)

    # CHANGE: Logic to find the smallest size
    target_size = min(all_seen_sizes)
    sorted_nodes = sorted(node_data.keys())
    ppg = 1

    keys = [
        "PMPI Allreduce Time",
        "MPIL GPU-Aware Recursive Doubling Time",
        "MPIL CopyToCPU Recursive Doubling Time",
        "MPIL CopyToCPU Node-Aware Dissemination Time",
        "MPIL CopyToCPU NUMA-Aware Dissemination Time",
        "MPIL CopyToCPU RADIX-4 Dissemination Time",
    ]

    # 3. Filter for nodes that actually contain data for our target_size
    # We'll use the baseline (PMPI) as the gatekeeper
    nodes_with_data = []
    for nn in sorted_nodes:
        data = node_data[nn].get(ppg, {})
        pmpi_series = data.get("PMPI Allreduce Time", [])
        if any(s == target_size for s, v in pmpi_series):
            nodes_with_data.append(nn)

    # 4. Extract Y-values using ONLY the nodes that have data
    plot_series = {k: [] for k in keys}
    for nn in nodes_with_data:
        data = node_data[nn].get(ppg, {})
        for k in keys:
            series = data.get(k, [])
            # Extract the value for the target size
            val = next((v for s, v in series if s == target_size), np.nan)
            plot_series[k].append(val)

    fn_out = os.path.basename(os.path.normpath(os.path.abspath(dir_in))) + "_node_scaling_min_size.pdf"
    pdf = PdfPages(fn_out)
    
    # Timing Plot
    plt.figure()
    for k in keys:
        label = k.replace("MPIL ", "").replace("CopyToCPU ", "C2C ").replace(" Time", "")
        plt.plot(nodes_with_data, plot_series[k], label=label)

    plt.title(f"Allreduce Timings vs Nodes\nSmallest Size ({target_size} doubles), PPG={ppg}")
    plt.xlabel("Nodes")
    plt.ylabel("Time (s)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xticks(nodes_with_data, labels=[str(n) for n in nodes_with_data])
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    pdf.savefig(bbox_inches="tight")

    # Speedup Plot
    plt.figure()
    base_times = np.array(plot_series["PMPI Allreduce Time"])
    for k in keys[1:]:
        y = np.array(plot_series[k])
        speedup = base_times / y
        label = k.replace("MPIL ", "").replace("CopyToCPU ", "C2C ").replace(" Time", "")
        plt.plot(nodes_with_data, speedup, label=label)

    plt.title(f"Allreduce Speedup vs PMPI\nSmallest Size ({target_size} doubles), PPG={ppg}")
    plt.xlabel("Nodes")
    plt.ylabel("Speedup (PMPI / Method)")
    plt.xscale("log")
    plt.xticks(nodes_with_data, labels=[str(n) for n in nodes_with_data])
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    pdf.savefig(bbox_inches="tight")

    pdf.close()
    print(f"Generated {fn_out} for the smallest size: {target_size}")