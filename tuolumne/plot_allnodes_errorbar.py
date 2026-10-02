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

# assume sizes pushed in
def combine_times(li_times):
    out = {}
    for times in li_times:
        for procs_per_gpu in times.keys():
            if procs_per_gpu not in out.keys():
                out[procs_per_gpu] = {}
            for k in times[procs_per_gpu].keys():
                if k not in out[procs_per_gpu].keys():
                    out[procs_per_gpu][k] = {}
                for (size, time) in times[procs_per_gpu][k]:
                    if size not in out[procs_per_gpu][k].keys():
                        out[procs_per_gpu][k][size] = []
                    out[procs_per_gpu][k][size].append(time)
    return out

def reduce_combined_times(combined_times, func):
    out = {}
    for procs_per_gpu in combined_times.keys():
        out[procs_per_gpu] = {}
        for k in combined_times[procs_per_gpu].keys():
            out[procs_per_gpu][k] = []
            for size in combined_times[procs_per_gpu][k].keys():
                out[procs_per_gpu][k].append((size, func(combined_times[procs_per_gpu][k][size])))
    return out

def get_min_max_error_diffs(li_local_times):    
    return (np.mean(li_local_times) - np.min(li_local_times), np.max(li_local_times) - np.mean(li_local_times))

def min_max_error_diff_reduced_times_to_np(times_error_diffs):
    out = {}
    for procs_per_gpu in times_error_diffs.keys():
        out[procs_per_gpu] = {}
        for k in times_error_diffs[procs_per_gpu].keys():
            out[procs_per_gpu][k] = np.zeros((3, len(times_error_diffs[procs_per_gpu][k])))
            for ind, (size, (min_time_diff, max_time_diff)) in enumerate(times_error_diffs[procs_per_gpu][k]):
                (out[procs_per_gpu][k])[0, ind] = size
                (out[procs_per_gpu][k])[1, ind] = min_time_diff
                (out[procs_per_gpu][k])[2, ind] = max_time_diff
    return out

def min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(array_2d_ref, lp_avg_ref, array_2d_new, lp_avg_new):
    new_2d_vals = np.copy(array_2d_new)
    new_2d_vals[1, :] = np.array([y[1] for y in lp_avg_new]) - array_2d_new[1, :]
    new_2d_vals[2, :] = np.array([y[1] for y in lp_avg_new]) + array_2d_new[2, :]
    
    ref_2d_vals = np.copy(array_2d_ref)
    ref_2d_vals[1, :] = np.array([y[1] for y in lp_avg_ref]) - array_2d_ref[1, :]
    ref_2d_vals[2, :] = np.array([y[1] for y in lp_avg_ref]) + array_2d_ref[2, :]
    
    mask = np.isin(ref_2d_vals[0, :], new_2d_vals[0, :])
    ref_2d_vals = ref_2d_vals[:, mask]
    
    speedups = np.copy(ref_2d_vals)
    speedups[1, :] = ref_2d_vals[1, :] / new_2d_vals[2, :]  # worst speedup is old min / new max
    speedups[2, :] = ref_2d_vals[2, :] / new_2d_vals[1, :]  # best speedup is old max / new min
    
    avg_speedups = (np.array([y[1] for y in lp_avg_ref]))[mask] / np.array([y[1] for y in lp_avg_new])
    
    speedup_diffs = np.copy(speedups)
    speedup_diffs[1, :] = avg_speedups - speedups[1, :]
    speedup_diffs[2, :] = speedups[2, :] - avg_speedups
    
    return speedup_diffs, speedup_diffs[0, :], avg_speedups

# assume sizes pushed in
def find_max_size_in_all_li_times(li_times):
    def extract_sizes(times):
        all_sizes = None
        for subdict in times.values():
            for inner_list in subdict.values():
                current = {pair[0] for pair in inner_list}
                if all_sizes is None:
                    all_sizes = current
                else:
                    all_sizes = all_sizes.intersection(current)
        return all_sizes if all_sizes is not None else set()

    common_sizes = None
    for times in li_times:
        sizes = extract_sizes(times)
        if common_sizes is None:
            common_sizes = sizes
        else:
            common_sizes = common_sizes.intersection(sizes)

    return max(common_sizes) if common_sizes else None

def num_node_combined_times_pairs_push_num_nodes_in_at_size(node_comb_times_pairs, target_size):
    out = {}
    for num_nodes, combined_times in node_comb_times_pairs:
        for procs_per_gpu in combined_times.keys():
            if procs_per_gpu not in out.keys():
                out[procs_per_gpu] = {}
            for k in combined_times[procs_per_gpu].keys():
                if k not in out[procs_per_gpu].keys():
                    out[procs_per_gpu][k] = {}
                for size in combined_times[procs_per_gpu][k].keys():
                    if size == target_size:
                        local_times = combined_times[procs_per_gpu][k][size]
                        out[procs_per_gpu][k][num_nodes] = local_times
    return out

if __name__ == "__main__":
    if len(sys.argv) >= 2 and (sys.argv[1] == "-h" or sys.argv[1] == "--help"):
        print("Usage: python plot_allnodes_errorbar.py <input_directory>,<input_directory> [prefix] [suffix]")
        sys.exit(1)

    plt.rcParams['axes.labelsize'] = 'large'

    dir_in = sys.argv[1] if len(sys.argv) >= 2 \
        else "gpu_allreduce/cpx/rocm7.2.1_cray-mpich9.1.0,cpu_allreduce/rocm7.2.1_cray-mpich9.1.0"
    prefix = sys.argv[2] if len(sys.argv) >= 3 else ""
    suffix = sys.argv[3] if len(sys.argv) >= 4 else ".out"

    if "," in dir_in:
        dirs_in = dir_in.split(",")
    else:
        dirs_in = [dir_in]

    node_data = {"gpu": {}, "cpu": {}}

    for dir_in in dirs_in:
        for fn in os.listdir(dir_in):
            if fn.startswith(prefix) and fn.endswith(suffix):
                nn = find_nnodes(fn)
                if nn is not None:
                    with open(os.path.join(dir_in, fn), 'r') as f:
                        lines = f.readlines()
                        raw_times = find_besttimes(lines)
                        pushed_times = push_sizes_in(raw_times)
                        run_type = "gpu"
                        if "cpu" in dir_in:
                            run_type = "cpu"
                        node_data[run_type].setdefault(nn, []).append(pushed_times)

    if not node_data:
        print(f"No valid files found in {dir_in}")
        sys.exit(1)

    target_size = 1
    sorted_nodes = {"gpu": sorted(node_data["gpu"].keys()),
                                     "cpu": sorted(node_data["cpu"].keys())}
    ppg = 1

    gpu_keys = [
        "PMPI Allreduce Time",
        "MPIL GPU-Aware Recursive Doubling Time",
        "MPIL CopyToCPU Recursive Doubling Time",
        "MPIL CopyToCPU Node-Aware Dissemination Time",
        "MPIL CopyToCPU NUMA-Aware Dissemination Time",
    ]
    cpu_keys = [
        "PMPI_Allreduce Time",
        "MPIL Recursive Doubling Allreduce Time",
        "MPIL Node-Aware Dissemination Allreduce Time",
        "MPIL NUMA-Aware Dissemination Allreduce Time"
    ]

    # 3. Filter for nodes that actually contain data for our target_size
    gpu_nodes_with_data, cpu_nodes_with_data = [], []
    for nn in sorted_nodes["gpu"]:
        ok = True
        for run in range(len(node_data["gpu"][nn])):
            data = node_data['gpu'][nn][run].get(ppg, {})
            series = data.get("MPIL CopyToCPU NUMA-Aware Dissemination Time", [])
            if not any(s == target_size for s, v in series):
                ok = False
                break
        if ok:
            gpu_nodes_with_data.append(nn)
    for nn in sorted_nodes["cpu"]:
        ok = True
        for run in range(len(node_data["cpu"][nn])):
            data = node_data['cpu'][nn][run].get(ppg, {})
            series = data.get("MPIL NUMA-Aware Dissemination Allreduce Time", [])
            if not any(s == target_size for s, v in series):
                ok = False
                break
        if ok:
            cpu_nodes_with_data.append(nn)

    # 4. Extract Y-values using ONLY the nodes that have data
    plot_series_gpu, plot_series_cpu = {k: [] for k in gpu_keys}, {k: [] for k in cpu_keys}
    for nn in gpu_nodes_with_data:
        # MIKE stopped here
        data = node_data["gpu"][nn].get(ppg, {})
        for k in gpu_keys:
            series = data.get(k, [])
            # Extract the value for the target size
            import pdb;pdb.set_trace()
            val = next((v for s, v in series if s == target_size), np.nan)
            plot_series_gpu[k].append(val)

    fn_out = os.path.basename(os.path.normpath(os.path.abspath(dir_in))) + "_node_scaling_min_size.pdf"
    pdf = PdfPages(fn_out)
    
    # Timing Plot
    plt.figure()
    for k in gpu_keys:
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
    for k in gpu_keys[1:]:
        y = np.array(plot_series[k])
        speedup = base_times / y
        print("Speedup of " + k + ": " + str(speedup))
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