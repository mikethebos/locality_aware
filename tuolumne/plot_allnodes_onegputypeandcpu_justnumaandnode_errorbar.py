import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import StrMethodFormatter

# How CPU series are drawn: "o" = dots only, "o--" = dots joined by a dashed line
CPU_FMT = "--"
CAPSIZE = 3

def find_nnodes(fn):
    r"""Extracts node count from filename matching 'allreduce_N(\d+)'."""
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

def find_timings(l, alg_colon=True):
    if alg_colon:
        matches = re.findall(r'^\s*\d+:\s*(.+?):\s+([-+]?\d*\.\d+e[-+]\d+)', l)
    else:
        matches = re.findall(r'^\s*\d+:\s*(.+?)\s+([-+]?\d*\.\d+e[-+]\d+)', l)
    return [(k, float(v)) for k, v in matches]

def find_besttimes(lines, alg_colon=True):
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
        for (k, v) in find_timings(l, alg_colon=alg_colon):
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

    no_title = False
    if "NO_TITLE" in os.environ.keys() and int(os.environ["NO_TITLE"]) == 1:
        plt.title = lambda *args, **kwargs: None
        no_title = True

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
                        alg_colon = True
                        run_type = "gpu"
                        if "cpu" in dir_in:
                            alg_colon = False
                            run_type = "cpu"
                        raw_times = find_besttimes(lines, alg_colon=alg_colon)
                        pushed_times = push_sizes_in(raw_times)
                        node_data[run_type].setdefault(nn, []).append(pushed_times)

    if not node_data["gpu"] and not node_data["cpu"]:
        print(f"No valid files found in {', '.join(dirs_in)}")
        sys.exit(1)

    target_size = 1
    sorted_nodes = {"gpu": sorted(node_data["gpu"].keys()),
                                     "cpu": sorted(node_data["cpu"].keys())}
    ppg = 1

    gpu_keys = [
        "PMPI Allreduce Time",
        # "MPIL GPU-Aware Recursive Doubling Time",
        # "MPIL CopyToCPU Recursive Doubling Time",
        "MPIL CopyToCPU Node-Aware Dissemination Time",
        "MPIL CopyToCPU NUMA-Aware Dissemination Time",
    ]
    cpu_keys = [
        "PMPI_Allreduce Time",
        # "MPIL Recursive Doubling Allreduce Time",
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

    if not gpu_nodes_with_data and not cpu_nodes_with_data:
        print(f"No nodes have data for size {target_size}")
        sys.exit(1)
    all_nodes = sorted(set(gpu_nodes_with_data) | set(cpu_nodes_with_data))

    # 4. Mean and (min, max) error bars at target_size for every node count
    #    (same pipeline as the locality script: combine runs -> push nodes in -> reduce)
    def build_series(run_type, nodes):
        """Returns ({key: [(nodes, mean), ...]}, {key: 3 x n array of nodes / mean-min / max-mean})."""
        if not nodes:
            return {}, {}
        pairs = [(nn, combine_times(node_data[run_type][nn])) for nn in nodes]
        at_size = num_node_combined_times_pairs_push_num_nodes_in_at_size(pairs, target_size)
        avg = reduce_combined_times(at_size, np.mean)
        err = min_max_error_diff_reduced_times_to_np(reduce_combined_times(at_size, get_min_max_error_diffs))
        return avg[ppg], err[ppg]

    gpu_avg, gpu_err = build_series("gpu", gpu_nodes_with_data)
    cpu_avg, cpu_err = build_series("cpu", cpu_nodes_with_data)
    runs = [("gpu", gpu_keys, gpu_avg, gpu_err),
            ("cpu", cpu_keys, cpu_avg, cpu_err)]

    # 5. Styling: GPU = lines, CPU = dots. A CPU method shares its color with the
    #    GPU "CopyToCPU" version of the same algorithm so the two are easy to compare.
    colors = {k: f"C{i}" for i, k in enumerate(gpu_keys)}
    cpu_to_gpu_key = {
        "PMPI_Allreduce Time": "PMPI Allreduce Time",
        # "MPIL Recursive Doubling Allreduce Time": "MPIL CopyToCPU Recursive Doubling Time",
        "MPIL Node-Aware Dissemination Allreduce Time": "MPIL CopyToCPU Node-Aware Dissemination Time",
        "MPIL NUMA-Aware Dissemination Allreduce Time": "MPIL CopyToCPU NUMA-Aware Dissemination Time",
    }
    colors.update({ck: colors[gk] for ck, gk in cpu_to_gpu_key.items()})

    def short_label(k):
        label = k.replace("PMPI_Allreduce", "PMPI Allreduce")
        if label.startswith("MPIL "):
            label = label[len("MPIL "):].replace(" Allreduce Time", " Time")
        label = label.replace("CopyToCPU ", "C2C ")
        return label.replace(" Time", "")

    def has_data(avg, k):
        return k in avg and len(avg[k]) > 0

    def draw(run_type, k, x, y, yerr):
        plt.errorbar(x, y, yerr=np.clip(yerr, 0, None),
                     fmt=CPU_FMT if run_type == "cpu" else "-",
                     color=colors.get(k), capsize=CAPSIZE,
                     label=f"{short_label(k)} ({run_type.upper()})")

    def format_node_axis():
        plt.xlabel("Nodes")
        plt.xscale("log")
        plt.xticks(ticks=all_nodes)
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.gca().xaxis.set_minor_formatter("")

    def finish_fig(title, top=0.85):
        fig, ax = plt.gcf(), plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        fig.tight_layout(rect=[0, 0, 1, top])          # reserve headroom (figure coords)

        sorted_legend = sorted(
            zip(handles, labels), 
            key=lambda x: x[0].lines[0].get_color()
        )

        sorted_handles, sorted_labels = zip(*sorted_legend)

        fig.legend(sorted_handles, sorted_labels, loc="lower center", frameon=False,
                ncol=int(np.ceil(len(sorted_handles) / 2)),
                bbox_to_anchor=(0.5, top - 0.04))           # figure coords for fig.legend, 0.04 to decrease space between legend and plot
        if not no_title:
            fig.suptitle(title, y=0.995, va="top")          # title above the legend

    fn_out = "justnumaandnode_node_scaling_min_size_errorbar.pdf"
    add_prefix = None
    gpu_exists, cpu_exists = False, False
    for dir in dirs_in:
        if "gpu" in dir:
            add_prefix = dir + os.path.sep
            gpu_exists = True
        elif "cpu" in dir:
            cpu_exists = True
    if add_prefix == None:
        add_prefix = dirs_in[0] + os.path.sep
    if gpu_exists and cpu_exists:
        add_prefix += "onegputypeandcpu_"
    elif gpu_exists:
        add_prefix += "onegputype_"
    elif cpu_exists:
        add_prefix += "cpu_"
    fn_out = add_prefix + fn_out
    pdf = PdfPages(fn_out)
    
    # Timing Plot
    plt.figure()
    for run_type, keys, avg, err in runs:
        if not avg:
            continue
        for k in keys:
            if not has_data(avg, k):
                print(f"Warning: no {run_type} data for '{k}' at size {target_size}")
                continue
            draw(run_type, k, [x for x, _ in avg[k]], [y for _, y in avg[k]], err[k][1:, :])

    format_node_axis()
    plt.ylabel("Time (s)")
    plt.yscale("log")
    finish_fig(f"Allreduce Timings vs Nodes\nSmallest Size ({target_size} doubles), PPG={ppg}")
    pdf.savefig(bbox_inches="tight")

    # Speedup Plot (each run type is compared against its own PMPI baseline)
    base_run_type = None
    for run_type, keys, avg, err in runs: # compare to gpu baseline
        if run_type == "gpu":
            base_run_type = run_type
            base_key = keys[0]
            base_err = err[base_key]
            base_avg = avg[base_key]
            break
    if base_run_type == None:
        run_type, keys, avg, err = runs[0] 
        base_run_type = run_type
        base_key = keys[0]
        base_err = err[base_key]
        base_avg = avg[base_key]
    
    plt.figure()
    for run_type, keys, avg, err in runs:
        if not avg:
            continue
        if not has_data({base_key: base_avg}, base_key):
            print(f"Warning: no {base_run_type} baseline '{base_key}', skipping {base_run_type} speedups")
            continue
        for k in keys[:]:
            if not has_data(avg, k):
                continue
            if [x for x, _ in base_avg] != [x for x, _ in avg[k]]:
                print(f"Warning: '{k}' and baseline '{base_key}' cover different node counts, skipping")
                continue
            speedup_error_diffs, speedup_x, speedup_avg = min_max_error_diff_reduced_time_nps_to_speedup_error_diffs(
                base_err, base_avg, err[k], avg[k])
            print(f"Speedup of {k} ({run_type.upper()}) at nodes {speedup_x}: {speedup_avg}")
            draw(run_type, k, speedup_x, speedup_avg, speedup_error_diffs[1:, :])

    format_node_axis()
    plt.ylabel("Speedup (PMPI / Method)")
    finish_fig(f"Allreduce Speedup vs {base_run_type.upper()} {base_key}\n"
           f"Smallest Size ({target_size} doubles), PPG={ppg}")
    pdf.savefig(bbox_inches="tight")

    pdf.close()
    print(f"Generated {fn_out} for the smallest size: {target_size}")