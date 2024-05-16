import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

th_run_type = "Threaded CC Neighbor_alltoallv"

def check(it, li):
    for lii in li:
        if lii in it:
            return True
    return False

fn = sys.argv[1]

include_list = []

if len(sys.argv) > 2:
    for i in range(2, len(sys.argv)):
        include_list.append(sys.argv[i])

with open(fn, 'r') as f:
    lines = f.readlines()
    
xs_list = {}
ys_list = {}
    
thread_count = 0
units_per_proc = 0
for line in lines:
    if "Unthreaded" in line:
        thread_count = 1
    elif "Running" in line and "threads" in line:
        thread_count = int(line.split(" ")[-2])
    elif "(around)" in line:
        units_per_proc = int(line.split(" ")[-1])
    elif "Time" in line and "e-" in line:
        run_type = line.split(" Time ")[0]
        if len(include_list) > 0 and not check(run_type, include_list):
            continue
        if run_type not in xs_list:
            xs_list[run_type] = {}
            ys_list[run_type] = {}
        if thread_count not in xs_list[run_type].keys():
            xs_list[run_type][thread_count] = []
            ys_list[run_type][thread_count] = []
        xs_list[run_type][thread_count].append(units_per_proc)
        ys_list[run_type][thread_count].append(float(line.split(" ")[-1]))

if len(include_list) == 0:
    pdf = PdfPages(sys.argv[1] + ".pdf")
    for run_type in xs_list.keys():
        if "Threaded" in run_type:
            continue
        fig = plt.figure()
        plt.loglog(xs_list[run_type][1], ys_list[run_type][1], label=run_type)
        for thread_count in xs_list[th_run_type].keys():
            plt.loglog(xs_list[th_run_type][thread_count], ys_list[th_run_type][thread_count], label=str(thread_count) + " threads")
        plt.title("Threaded vs " + run_type)
        plt.xlabel("Floating point numbers")
        plt.ylabel("Time (s)")
        plt.legend()
        plt.tight_layout()
        pdf.savefig(fig)
    pdf.close()