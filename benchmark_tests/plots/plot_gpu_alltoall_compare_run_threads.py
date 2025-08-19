import sys
from copy import deepcopy

import matplotlib.pyplot as plt

fn = sys.argv[1]

with open(fn, "r") as f:
    lines = f.readlines()
    
times = {}

test_type = ""
test_size = -1
for line in lines:
    if "Running" in line and "Test" in line:
        test_type = line.removeprefix("Running ").strip()
    elif "Testing Size" in line:
        test_size = int(line.removeprefix("Testing Size ").strip())
    elif " Time " in line and ("e-" in line or "e+" in line):
        specific_type = line[:line.find(" Time ")].strip()
        t = float(line[line.find(" Time ") + len(" Time "):].strip())
        comb_type = (test_type, specific_type)
        if comb_type not in times.keys():
            times[comb_type] = []
        times[comb_type].append((test_size, t))
        
# print(times)
        
new_fn = fn + "_run_plot.png"
best_xs = []
best_ys = []
plt.figure()
for (test_type, specific_type) in times.keys():
    title = ""
    suffix = ""
    if "PMPI" in specific_type:
        continue
    if "Unthreaded" in test_type:
        title += "Unthreaded: "
    elif " threads" in test_type:
        if "40 Threads" not in specific_type:
            continue
        if "without launches" in test_type:
            suffix += "Without Launches: "
        elif "with launches" in test_type:
            suffix += "With Launches: "
            
    if title == "" and suffix == "":
        continue
    
    title += suffix + specific_type
    
    xs = []
    ys = []
    for (test_size, t) in times[(test_type, specific_type)]:
        xs.append(test_size)
        ys.append(t)
        if "Unthreaded" in test_type and len(best_xs) > 0 and "PMPI" not in specific_type:
            if t < best_ys[best_xs.index(test_size)]:
                best_ys[best_xs.index(test_size)] = t
        
    if "Unthreaded" in test_type and "PMPI" not in specific_type:
        if len(best_xs) == 0:
            best_xs = deepcopy(xs)
            best_ys = deepcopy(ys)
        
    plt.loglog(xs, ys, label=title)
    
plt.legend()
plt.tight_layout()
plt.savefig(new_fn, dpi=500)

print("best times:", best_ys)

new_fn = fn + "_run_plot_speedup.png"
plt.figure()
for (test_type, specific_type) in times.keys():
    title = ""
    suffix = ""
    if "PMPI" in specific_type:
        continue
    if "Unthreaded" in test_type:
        continue
    elif " threads" in test_type:
        if "40 Threads" not in specific_type:
            continue
        if "without launches" in test_type:
            suffix += "Without Launches: "
        elif "with launches" in test_type:
            suffix += "With Launches: "
            
    if title == "" and suffix == "":
        continue
    
    title += suffix + specific_type
    
    xs = []
    ys = []
    for (test_size, t) in times[(test_type, specific_type)]:
        xs.append(test_size)
        ys.append(best_ys[best_xs.index(test_size)] / t)
    
    print(title + " x:", xs)
    print(title + " y:", ys)
    
    plt.loglog(xs, ys, label=title)
    
plt.legend()
plt.tight_layout()
plt.savefig(new_fn, dpi=500)