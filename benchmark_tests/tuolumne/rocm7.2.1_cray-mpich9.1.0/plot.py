import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
import pyfancyplot as plt
import glob

class PingPong:
    active_procs = ""
    sizes = ""
    times = ""

    def __init__(self, num_active):
        self.active_procs = num_active
        self.sizes = list()
        self.times = list()

    def add_timing(self, size, time):
        if size in self.sizes:
            idx = self.sizes.index(size)
            self.times[idx] = time
        else:
            self.sizes.append(size)
            self.times.append(time)

## TODO : Change if new architecture
num_sockets = 4
GPN = 4

alltoall_fn_prefix = "alltoall_plus_copy_differenttypes"

active_method = False
curr_ppg = -1
curr_size = -1
curr_times = []
nodes = [2,4,8,16]
alltoall_1ppg = list() 
alltoall_2ppg = list()
alltoall_4ppg = list()
alltoall_8ppg = list() #

for n in nodes:

    alltoall_1ppg.append(PingPong(1))
    alltoall_2ppg.append(PingPong(1))
    alltoall_4ppg.append(PingPong(1))
    alltoall_8ppg.append(PingPong(1)) #

    files = glob.glob(alltoall_fn_prefix + "_n%d.*.out"%n)
    print(files)
    for fn in files:
        file = open(fn, 'r')
        for line in file:

            if "Processes Per GPU" in line:
                if active_method and curr_size != -1 and len(curr_times) > 0:
                    method.add_timing(curr_size, min(curr_times))
                    curr_times = []
                active_method = True
                if "1 Processes Per GPU" in line:
                    method = alltoall_1ppg[-1]
                    curr_ppg = 1
                    curr_size = -1
                elif "2 Processes Per GPU" in line:
                    method = alltoall_2ppg[-1]
                    curr_ppg = 2
                    curr_size = -1
                elif "4 Processes Per GPU" in line:
                    method = alltoall_4ppg[-1]
                    curr_ppg = 4
                    curr_size = -1
                elif "8 Processes Per GPU" in line:
                    method = alltoall_8ppg[-1]
                    curr_ppg = 8
                    curr_size = -1                
                else:
                    active_method = False
                    curr_ppg = -1
                    curr_size = -1

            elif active_method and "Size" in line:
                if active_method and curr_size != -1 and len(curr_times) > 0:
                    method.add_timing(curr_size, min(curr_times))
                    curr_times = []
                splitting = line.split(' (Per')
                curr_size = (int)(splitting[0].rsplit(' ')[-1])
            elif active_method and curr_size != -1:
                if "STD MPS COPY:" in line or "STD:" in line or "NB MPS COPY:" in line or "NB:" in line or "PW MPS COPY:" in line or "PW:" in line or "NB SPLIT MPS COPY:" in line or "PW SPLIT MPS COPY:" in line:
                    # if curr_size <= 8192 or curr_size >= 10**6.0:
                        # continue
                    splitting = line.split(": ")
                    time = (float)(splitting[-1].split('\n')[0])
                    curr_times.append(time)
                    
        if active_method and curr_size != -1 and len(curr_times) > 0:
            method.add_timing(curr_size, min(curr_times))
            curr_times = []

        file.close()

# Standard alltoall Times
plt.add_luke_options()
plt.line_plot(alltoall_1ppg[-1].times, alltoall_1ppg[-1].sizes, color='r', label = "1 PPG")
plt.line_plot(alltoall_2ppg[-1].times, alltoall_2ppg[-1].sizes, color='b', label = "2 PPG")
plt.line_plot(alltoall_4ppg[-1].times, alltoall_4ppg[-1].sizes, color='g', label = "4 PPG")
plt.line_plot(alltoall_8ppg[-1].times, alltoall_8ppg[-1].sizes, color='y', label = "8 PPG") #
plt.add_anchored_legend(ncol=3)
plt.add_labels("Num Floats", "Time (Seconds)")
plt.set_scale('log', 'log')
plt.save_plot(alltoall_fn_prefix + "_somemethods_ppg_best.pdf")

# Speedup at max size
one_ppg_sizes = alltoall_1ppg[-1].sizes
one_ppg_times = alltoall_1ppg[-1].times
max_ppg_sizes = alltoall_2ppg[-1].sizes
max_ppg_times = alltoall_2ppg[-1].times
intersect = set(one_ppg_sizes).intersection(max_ppg_sizes)
max_size = max(intersect)
print(alltoall_fn_prefix + "_somemethods_ppg best 2 ppg: speedup at " + str(max_size) + ": " + str(one_ppg_times[one_ppg_sizes.index(max_size)] / max_ppg_times[max_ppg_sizes.index(max_size)]))
one_ppg_sizes = alltoall_1ppg[-1].sizes
one_ppg_times = alltoall_1ppg[-1].times
max_ppg_sizes = alltoall_4ppg[-1].sizes
max_ppg_times = alltoall_4ppg[-1].times
intersect = set(one_ppg_sizes).intersection(max_ppg_sizes)
max_size = max(intersect)
print(alltoall_fn_prefix + "_somemethods_ppg best 4 ppg: speedup at " + str(max_size) + ": " + str(one_ppg_times[one_ppg_sizes.index(max_size)] / max_ppg_times[max_ppg_sizes.index(max_size)]))

one_ppg_sizes = alltoall_1ppg[-1].sizes
one_ppg_times = alltoall_1ppg[-1].times
max_ppg_sizes = alltoall_8ppg[-1].sizes
max_ppg_times = alltoall_8ppg[-1].times
intersect = set(one_ppg_sizes).intersection(max_ppg_sizes)
max_size = max(intersect)
print(alltoall_fn_prefix + "_somemethods_ppg best 8 ppg: speedup at " + str(max_size) + ": " + str(one_ppg_times[one_ppg_sizes.index(max_size)] / max_ppg_times[max_ppg_sizes.index(max_size)]))

# Standard alltoall Times
plt.add_luke_options()
plt.line_plot([alltoall_1ppg[i].times[-1] for i in range(len(nodes))], nodes, color='r', label = "1 PPG")
plt.line_plot([alltoall_2ppg[i].times[-1] for i in range(len(nodes))], nodes, color='b', label = "2 PPG")
plt.line_plot([alltoall_4ppg[i].times[-1] for i in range(len(nodes))], nodes, color='g', label = "4 PPG")
plt.line_plot([alltoall_8ppg[i].times[-1] for i in range(len(nodes))], nodes, color='y', label = "8 PPG") #
plt.add_anchored_legend(ncol=3)
plt.add_labels("Nodes", "Time (Seconds)")
plt.set_scale('linear', 'log')
plt.set_xticks(nodes, nodes)
plt.save_plot(alltoall_fn_prefix + "_somemethods_ppg_nodes_best.pdf")

active_method = False
curr_ppg = -1
curr_size = -1
nodes = [2,4,8,16]
alltoall_1ppg = list() 
alltoall_2ppg = list()
alltoall_4ppg = list()
alltoall_8ppg = list()

for n in nodes:

    alltoall_1ppg.append(PingPong(1))
    alltoall_2ppg.append(PingPong(1))
    alltoall_4ppg.append(PingPong(1))
    alltoall_8ppg.append(PingPong(1))

    files = glob.glob(alltoall_fn_prefix + "_n%d.*.out"%n)
    print(files)
    for fn in files:
        file = open(fn, 'r')
        for line in file:

            if "Processes Per GPU" in line:
                active_method = True
                if "1 Processes Per GPU" in line:
                    method = alltoall_1ppg[-1]
                    curr_ppg = 1
                    curr_size = -1
                elif "2 Processes Per GPU" in line:
                    method = alltoall_2ppg[-1]
                    curr_ppg = 2
                    curr_size = -1
                elif "4 Processes Per GPU" in line:
                    method = alltoall_4ppg[-1]
                    curr_ppg = 4
                    curr_size = -1
                elif "8 Processes Per GPU" in line:
                    method = alltoall_8ppg[-1]
                    curr_ppg = 8
                    curr_size = -1
                else:
                    active_method = False
                    curr_ppg = -1
                    curr_size = -1

            elif active_method and "Size" in line:
                splitting = line.split(' (Per')
                curr_size = (int)(splitting[0].rsplit(' ')[-1])
            elif active_method and curr_size != -1:
                if "STD MPS COPY:" in line or "STD:" in line:
                    # if curr_size <= 8192 or curr_size >= 10**7:
                        # continue
                    splitting = line.split(": ")
                    time = (float)(splitting[-1].split('\n')[0])
                    method.add_timing(curr_size, time)

        file.close()

# Standard alltoall Times
plt.add_luke_options()
plt.line_plot(alltoall_1ppg[-1].times, alltoall_1ppg[-1].sizes, color='r', label = "1 PPG")
plt.line_plot(alltoall_2ppg[-1].times, alltoall_2ppg[-1].sizes, color='b', label = "2 PPG")
plt.line_plot(alltoall_4ppg[-1].times, alltoall_4ppg[-1].sizes, color='g', label = "4 PPG")
plt.line_plot(alltoall_8ppg[-1].times, alltoall_8ppg[-1].sizes, color='y', label = "8 PPG")
plt.add_anchored_legend(ncol=3)
plt.add_labels("Num Floats", "Time (Seconds)")
plt.set_scale('log', 'log')
plt.save_plot(alltoall_fn_prefix + "_somemethods_ppg.pdf")

# Speedup at max size
one_ppg_sizes = alltoall_1ppg[-1].sizes
one_ppg_times = alltoall_1ppg[-1].times
max_ppg_sizes = alltoall_2ppg[-1].sizes
max_ppg_times = alltoall_2ppg[-1].times
intersect = set(one_ppg_sizes).intersection(max_ppg_sizes)
max_size = max(intersect)
print(alltoall_fn_prefix + "_somemethods_ppg 2 ppg: speedup at " + str(max_size) + ": " + str(one_ppg_times[one_ppg_sizes.index(max_size)] / max_ppg_times[max_ppg_sizes.index(max_size)]))
one_ppg_sizes = alltoall_1ppg[-1].sizes
one_ppg_times = alltoall_1ppg[-1].times
max_ppg_sizes = alltoall_4ppg[-1].sizes
max_ppg_times = alltoall_4ppg[-1].times
intersect = set(one_ppg_sizes).intersection(max_ppg_sizes)
max_size = max(intersect)
print(alltoall_fn_prefix + "_somemethods_ppg 4 ppg: speedup at " + str(max_size) + ": " + str(one_ppg_times[one_ppg_sizes.index(max_size)] / max_ppg_times[max_ppg_sizes.index(max_size)]))
one_ppg_sizes = alltoall_1ppg[-1].sizes
one_ppg_times = alltoall_1ppg[-1].times
max_ppg_sizes = alltoall_8ppg[-1].sizes
max_ppg_times = alltoall_8ppg[-1].times
intersect = set(one_ppg_sizes).intersection(max_ppg_sizes)
max_size = max(intersect)
print(alltoall_fn_prefix + "_somemethods_ppg 8 ppg: speedup at " + str(max_size) + ": " + str(one_ppg_times[one_ppg_sizes.index(max_size)] / max_ppg_times[max_ppg_sizes.index(max_size)]))

# Standard alltoall Times
plt.add_luke_options()
plt.line_plot([alltoall_1ppg[i].times[-1] for i in range(len(nodes))], nodes, color='r', label = "1 PPG")
plt.line_plot([alltoall_2ppg[i].times[-1] for i in range(len(nodes))], nodes, color='b', label = "2 PPG")
plt.line_plot([alltoall_4ppg[i].times[-1] for i in range(len(nodes))], nodes, color='g', label = "4 PPG")
plt.line_plot([alltoall_8ppg[i].times[-1] for i in range(len(nodes))], nodes, color='y', label = "8 PPG")
plt.add_anchored_legend(ncol=3)
plt.add_labels("Nodes", "Time (Seconds)")
plt.set_scale('linear', 'log')
plt.set_xticks(nodes, nodes)
plt.save_plot(alltoall_fn_prefix + "_somemethods_ppg_nodes.pdf")

active_method = False
curr_ppg = -1
curr_size = -1
nodes = [2,4,8,16]
alltoall_1ppg = list() 
alltoall_2ppg = list()
alltoall_4ppg = list()
alltoall_8ppg = list()

for n in nodes:

    alltoall_1ppg.append(PingPong(1))
    alltoall_2ppg.append(PingPong(1))
    alltoall_4ppg.append(PingPong(1))
    alltoall_8ppg.append(PingPong(1))

    files = glob.glob(alltoall_fn_prefix + "_n%d.*.out"%n)
    print(files)
    for fn in files:
        file = open(fn, 'r')
        for line in file:

            if "Processes Per GPU" in line:
                active_method = True
                if "1 Processes Per GPU" in line:
                    method = alltoall_1ppg[-1]
                    curr_ppg = 1
                    curr_size = -1
                elif "2 Processes Per GPU" in line:
                    method = alltoall_2ppg[-1]
                    curr_ppg = 2
                    curr_size = -1
                elif "4 Processes Per GPU" in line:
                    method = alltoall_4ppg[-1]
                    curr_ppg = 4
                    curr_size = -1
                elif "8 Processes Per GPU" in line:
                    method = alltoall_8ppg[-1]
                    curr_ppg = 8
                    curr_size = -1
                else:
                    active_method = False
                    curr_ppg = -1
                    curr_size = -1

            elif active_method and "Size" in line:
                splitting = line.split(' (Per')
                curr_size = (int)(splitting[0].rsplit(' ')[-1])
            elif active_method and curr_size != -1:
                if "PW MPS COPY:" in line or "PW:" in line:
                    # if curr_size <= 8192 or curr_size >= 10**7:
                        # continue
                    splitting = line.split(": ")
                    time = (float)(splitting[-1].split('\n')[0])
                    method.add_timing(curr_size, time)

        file.close()

# Standard alltoall Times
plt.add_luke_options()
plt.line_plot(alltoall_1ppg[-1].times, alltoall_1ppg[-1].sizes, color='r', label = "1 PPG")
plt.line_plot(alltoall_2ppg[-1].times, alltoall_2ppg[-1].sizes, color='b', label = "2 PPG")
plt.line_plot(alltoall_4ppg[-1].times, alltoall_4ppg[-1].sizes, color='g', label = "4 PPG")
plt.line_plot(alltoall_8ppg[-1].times, alltoall_8ppg[-1].sizes, color='y', label = "8 PPG")
plt.add_anchored_legend(ncol=3)
plt.add_labels("Num Floats", "Time (Seconds)")
plt.set_scale('log', 'log')
plt.save_plot(alltoall_fn_prefix + "_somemethods_ppg_pw.pdf")

# Speedup at max size
one_ppg_sizes = alltoall_1ppg[-1].sizes
one_ppg_times = alltoall_1ppg[-1].times
max_ppg_sizes = alltoall_2ppg[-1].sizes
max_ppg_times = alltoall_2ppg[-1].times
intersect = set(one_ppg_sizes).intersection(max_ppg_sizes)
max_size = max(intersect)
print(alltoall_fn_prefix + "_somemethods_ppg PW 2 ppg: speedup at " + str(max_size) + ": " + str(one_ppg_times[one_ppg_sizes.index(max_size)] / max_ppg_times[max_ppg_sizes.index(max_size)]))
one_ppg_sizes = alltoall_1ppg[-1].sizes
one_ppg_times = alltoall_1ppg[-1].times
max_ppg_sizes = alltoall_4ppg[-1].sizes
max_ppg_times = alltoall_4ppg[-1].times
intersect = set(one_ppg_sizes).intersection(max_ppg_sizes)
max_size = max(intersect)
print(alltoall_fn_prefix + "_somemethods_ppg PW 4 ppg: speedup at " + str(max_size) + ": " + str(one_ppg_times[one_ppg_sizes.index(max_size)] / max_ppg_times[max_ppg_sizes.index(max_size)]))
one_ppg_sizes = alltoall_1ppg[-1].sizes
one_ppg_times = alltoall_1ppg[-1].times
max_ppg_sizes = alltoall_8ppg[-1].sizes
max_ppg_times = alltoall_8ppg[-1].times
intersect = set(one_ppg_sizes).intersection(max_ppg_sizes)
max_size = max(intersect)
print(alltoall_fn_prefix + "_somemethods_ppg PW 8 ppg: speedup at " + str(max_size) + ": " + str(one_ppg_times[one_ppg_sizes.index(max_size)] / max_ppg_times[max_ppg_sizes.index(max_size)]))

# Standard alltoall Times
plt.add_luke_options()
plt.line_plot([alltoall_1ppg[i].times[-1] for i in range(len(nodes))], nodes, color='r', label = "1 PPG")
plt.line_plot([alltoall_2ppg[i].times[-1] for i in range(len(nodes))], nodes, color='b', label = "2 PPG")
plt.line_plot([alltoall_4ppg[i].times[-1] for i in range(len(nodes))], nodes, color='g', label = "4 PPG")
plt.line_plot([alltoall_8ppg[i].times[-1] for i in range(len(nodes))], nodes, color='y', label = "8 PPG")
plt.add_anchored_legend(ncol=3)
plt.add_labels("Nodes", "Time (Seconds)")
plt.set_scale('linear', 'log')
plt.set_xticks(nodes, nodes)
plt.save_plot(alltoall_fn_prefix + "_somemethods_ppg_nodes_pw.pdf")

active_method = False
curr_ppg = -1
curr_size = -1
nodes = [2,4,8,16]
alltoall_1ppg = list() 
alltoall_2ppg = list()
alltoall_4ppg = list()
alltoall_8ppg = list()

for n in nodes:

    alltoall_1ppg.append(PingPong(1))
    alltoall_2ppg.append(PingPong(1))
    alltoall_4ppg.append(PingPong(1))
    alltoall_8ppg.append(PingPong(1))

    files = glob.glob(alltoall_fn_prefix + "_n%d.*.out"%n)
    print(files)
    for fn in files:
        file = open(fn, 'r')
        for line in file:

            if "Processes Per GPU" in line:
                active_method = True
                if "1 Processes Per GPU" in line:
                    method = alltoall_1ppg[-1]
                    curr_ppg = 1
                    curr_size = -1
                elif "2 Processes Per GPU" in line:
                    method = alltoall_2ppg[-1]
                    curr_ppg = 2
                    curr_size = -1
                elif "4 Processes Per GPU" in line:
                    method = alltoall_4ppg[-1]
                    curr_ppg = 4
                    curr_size = -1
                elif "8 Processes Per GPU" in line:
                    method = alltoall_8ppg[-1]
                    curr_ppg = 8
                    curr_size = -1
                else:
                    active_method = False
                    curr_ppg = -1
                    curr_size = -1

            elif active_method and "Size" in line:
                splitting = line.split(' (Per')
                curr_size = (int)(splitting[0].rsplit(' ')[-1])
            elif active_method and curr_size != -1:
                if "NB MPS COPY:" in line or "NB:" in line:
                    # if curr_size <= 8192 or curr_size >= 10**7:
                        # continue
                    splitting = line.split(": ")
                    time = (float)(splitting[-1].split('\n')[0])
                    method.add_timing(curr_size, time)

        file.close()

# Standard alltoall Times
plt.add_luke_options()
plt.line_plot(alltoall_1ppg[-1].times, alltoall_1ppg[-1].sizes, color='r', label = "1 PPG")
plt.line_plot(alltoall_2ppg[-1].times, alltoall_2ppg[-1].sizes, color='b', label = "2 PPG")
plt.line_plot(alltoall_4ppg[-1].times, alltoall_4ppg[-1].sizes, color='g', label = "4 PPG")
plt.line_plot(alltoall_8ppg[-1].times, alltoall_8ppg[-1].sizes, color='y', label = "8 PPG")
plt.add_anchored_legend(ncol=3)
plt.add_labels("Num Floats", "Time (Seconds)")
plt.set_scale('log', 'log')
plt.save_plot(alltoall_fn_prefix + "_somemethods_ppg_nb.pdf")

# Speedup at max size
one_ppg_sizes = alltoall_1ppg[-1].sizes
one_ppg_times = alltoall_1ppg[-1].times
max_ppg_sizes = alltoall_2ppg[-1].sizes
max_ppg_times = alltoall_2ppg[-1].times
intersect = set(one_ppg_sizes).intersection(max_ppg_sizes)
max_size = max(intersect)
print(alltoall_fn_prefix + "_somemethods_ppg NB 2 ppg: speedup at " + str(max_size) + ": " + str(one_ppg_times[one_ppg_sizes.index(max_size)] / max_ppg_times[max_ppg_sizes.index(max_size)]))
one_ppg_sizes = alltoall_1ppg[-1].sizes
one_ppg_times = alltoall_1ppg[-1].times
max_ppg_sizes = alltoall_4ppg[-1].sizes
max_ppg_times = alltoall_4ppg[-1].times
intersect = set(one_ppg_sizes).intersection(max_ppg_sizes)
max_size = max(intersect)
print(alltoall_fn_prefix + "_somemethods_ppg NB 4 ppg: speedup at " + str(max_size) + ": " + str(one_ppg_times[one_ppg_sizes.index(max_size)] / max_ppg_times[max_ppg_sizes.index(max_size)]))
one_ppg_sizes = alltoall_1ppg[-1].sizes
one_ppg_times = alltoall_1ppg[-1].times
max_ppg_sizes = alltoall_8ppg[-1].sizes
max_ppg_times = alltoall_8ppg[-1].times
intersect = set(one_ppg_sizes).intersection(max_ppg_sizes)
max_size = max(intersect)
print(alltoall_fn_prefix + "_somemethods_ppg NB 8 ppg: speedup at " + str(max_size) + ": " + str(one_ppg_times[one_ppg_sizes.index(max_size)] / max_ppg_times[max_ppg_sizes.index(max_size)]))

# Standard alltoall Times
plt.add_luke_options()
plt.line_plot([alltoall_1ppg[i].times[-1] for i in range(len(nodes))], nodes, color='r', label = "1 PPG")
plt.line_plot([alltoall_2ppg[i].times[-1] for i in range(len(nodes))], nodes, color='b', label = "2 PPG")
plt.line_plot([alltoall_4ppg[i].times[-1] for i in range(len(nodes))], nodes, color='g', label = "4 PPG")
plt.line_plot([alltoall_8ppg[i].times[-1] for i in range(len(nodes))], nodes, color='y', label = "8 PPG")
plt.add_anchored_legend(ncol=3)
plt.add_labels("Nodes", "Time (Seconds)")
plt.set_scale('linear', 'log')
plt.set_xticks(nodes, nodes)
plt.save_plot(alltoall_fn_prefix + "_somemethods_ppg_nodes_nb.pdf")

curr_ppg = -1
curr_size = -1
nodes = [2,4,8,16]
alltoall_1ppg_sys = list() 
alltoall_1ppg_pw = list()
alltoall_1ppg_nb = list()

for n in nodes:

    alltoall_1ppg_sys.append(PingPong(1))
    alltoall_1ppg_pw.append(PingPong(1))
    alltoall_1ppg_nb.append(PingPong(1))

    files = glob.glob(alltoall_fn_prefix + "_n%d.*.out"%n)
    print(files)
    for fn in files:
        file = open(fn, 'r')
        for line in file:

            if "Processes Per GPU" in line:
                if "1 Processes Per GPU" in line:
                    curr_ppg = 1
                    curr_size = -1
                else:
                    curr_ppg = -1
                    curr_size = -1

            elif curr_ppg != -1 and "Size" in line:
                splitting = line.split(' (Per')
                curr_size = (int)(splitting[0].rsplit(' ')[-1])
            elif curr_ppg != -1 and curr_size != -1:
                if "STD:" in line or "PW:" in line or "NB:" in line:
                    # if curr_size <= 8192 or curr_size >= 10**7:
                        # continue
                    splitting = line.split(": ")
                    time = (float)(splitting[-1].split('\n')[0])
                    if "STD:" in line:
                        alltoall_1ppg_sys[-1].add_timing(curr_size, time)
                    elif "PW:" in line:
                        alltoall_1ppg_pw[-1].add_timing(curr_size, time)
                    elif "NB:" in line:
                        alltoall_1ppg_nb[-1].add_timing(curr_size, time)

        file.close()

# Standard alltoall Times
plt.add_luke_options()
plt.line_plot(alltoall_1ppg_sys[-1].times, alltoall_1ppg_sys[-1].sizes, color='r', label = "Sys")
plt.line_plot(alltoall_1ppg_pw[-1].times, alltoall_1ppg_pw[-1].sizes, color='b', label = "Pw")
plt.line_plot(alltoall_1ppg_nb[-1].times, alltoall_1ppg_nb[-1].sizes, color='g', label = "Nb")
plt.add_anchored_legend(ncol=3)
plt.add_labels("Num Floats", "Time (Seconds)")
plt.set_scale('log', 'log')
plt.save_plot(alltoall_fn_prefix + "_somemethods_1ppg.pdf")

# Speedup at max size
one_ppg_sizes = alltoall_1ppg_sys[-1].sizes
one_ppg_times = alltoall_1ppg_sys[-1].times
max_ppg_sizes = alltoall_1ppg_pw[-1].sizes
max_ppg_times = alltoall_1ppg_pw[-1].times
intersect = set(one_ppg_sizes).intersection(max_ppg_sizes)
max_size = max(intersect)
print(alltoall_fn_prefix + "_somemethods_1ppg Pw: speedup at " + str(max_size) + ": " + str(one_ppg_times[one_ppg_sizes.index(max_size)] / max_ppg_times[max_ppg_sizes.index(max_size)]))
one_ppg_sizes = alltoall_1ppg_sys[-1].sizes
one_ppg_times = alltoall_1ppg_sys[-1].times
max_ppg_sizes = alltoall_1ppg_nb[-1].sizes
max_ppg_times = alltoall_1ppg_nb[-1].times
intersect = set(one_ppg_sizes).intersection(max_ppg_sizes)
max_size = max(intersect)
print(alltoall_fn_prefix + "_somemethods_1ppg Nb: speedup at " + str(max_size) + ": " + str(one_ppg_times[one_ppg_sizes.index(max_size)] / max_ppg_times[max_ppg_sizes.index(max_size)]))

# Standard alltoall Times
plt.add_luke_options()
plt.line_plot([alltoall_1ppg_sys[i].times[-1] for i in range(len(nodes))], nodes, color='r', label = "Sys")
plt.line_plot([alltoall_1ppg_pw[i].times[-1] for i in range(len(nodes))], nodes, color='b', label = "Pw")
plt.line_plot([alltoall_1ppg_nb[i].times[-1] for i in range(len(nodes))], nodes, color='g', label = "Nb")
plt.add_anchored_legend(ncol=3)
plt.add_labels("Nodes", "Time (Seconds)")
plt.set_scale('linear', 'log')
plt.set_xticks(nodes, nodes)
plt.save_plot(alltoall_fn_prefix + "_somemethods_1ppg_nodes.pdf")

active_method = False
curr_ppg = -1
curr_size = -1
curr_times = []
nodes = [2,4,8,16]
alltoall_1ppg_best = list() 
alltoall_multippg = dict()
alltoall_multippg_pw = dict()
alltoall_multippg_nb = dict()
alltoall_multippg_pwsplit = dict()
alltoall_multippg_nbsplit = dict()

for i, n in enumerate(nodes):

    alltoall_1ppg_best.append(PingPong(1))

    files = glob.glob(alltoall_fn_prefix + "_n%d.*.out"%n)
    print(files)
    for fn in files:
        file = open(fn, 'r')
        for line in file:

            if "Processes Per GPU" in line:
                if active_method and curr_size != -1 and len(curr_times) > 0 and method is not None:
                    method.add_timing(curr_size, min(curr_times))
                    curr_times = []
                active_method = True
                if "1 Processes Per GPU" in line:
                    method = alltoall_1ppg_best[-1]
                    curr_ppg = 1
                    curr_size = -1
                elif "2 Processes Per GPU" in line:
                    method = None
                    if 2 not in alltoall_multippg_pw.keys():
                        alltoall_multippg_pw[2] = []
                        alltoall_multippg[2] = []
                        alltoall_multippg_nb[2] = []
                        alltoall_multippg_pwsplit[2] = []
                        alltoall_multippg_nbsplit[2] = []
                    if len(alltoall_multippg_pw[2]) < i + 1:
                        alltoall_multippg_pw[2].append(PingPong(1))
                        alltoall_multippg[2].append(PingPong(1))
                        alltoall_multippg_nb[2].append(PingPong(1))
                        alltoall_multippg_pwsplit[2].append(PingPong(1))
                        alltoall_multippg_nbsplit[2].append(PingPong(1))
                    curr_ppg = 2
                    curr_size = -1
                elif "4 Processes Per GPU" in line:
                    method = None
                    if 4 not in alltoall_multippg_pw.keys():
                        alltoall_multippg_pw[4] = []
                        alltoall_multippg[4] = []
                        alltoall_multippg_nb[4] = []
                        alltoall_multippg_pwsplit[4] = []
                        alltoall_multippg_nbsplit[4] = []
                    if len(alltoall_multippg_pw[4]) < i + 1:
                        alltoall_multippg_pw[4].append(PingPong(1))
                        alltoall_multippg[4].append(PingPong(1))
                        alltoall_multippg_nb[4].append(PingPong(1))
                        alltoall_multippg_pwsplit[4].append(PingPong(1))
                        alltoall_multippg_nbsplit[4].append(PingPong(1))
                    curr_ppg = 4
                    curr_size = -1
                elif "8 Processes Per GPU" in line:
                    method = None
                    if 8 not in alltoall_multippg_pw.keys():
                        alltoall_multippg_pw[8] = []
                        alltoall_multippg[8] = []
                        alltoall_multippg_nb[8] = []
                        alltoall_multippg_pwsplit[8] = []
                        alltoall_multippg_nbsplit[8] = []
                    if len(alltoall_multippg_pw[8]) < i + 1:
                        alltoall_multippg_pw[8].append(PingPong(1))
                        alltoall_multippg[8].append(PingPong(1))
                        alltoall_multippg_nb[8].append(PingPong(1))
                        alltoall_multippg_pwsplit[8].append(PingPong(1))
                        alltoall_multippg_nbsplit[8].append(PingPong(1))
                    curr_ppg = 8
                    curr_size = -1
                else:
                    method = None
                    active_method = False
                    curr_ppg = -1
                    curr_size = -1

            elif active_method and "Size" in line:
                if active_method and curr_size != -1 and len(curr_times) > 0 and method is not None:
                    method.add_timing(curr_size, min(curr_times))
                    curr_times = []
                splitting = line.split(' (Per')
                curr_size = (int)(splitting[0].rsplit(' ')[-1])
            elif active_method and curr_size != -1:
                if "STD:" in line or "NB:" in line or "PW:" in line:
                    # if curr_size <= 8192 or curr_size >= 10**6.0:
                        # continue
                    splitting = line.split(": ")
                    time = (float)(splitting[-1].split('\n')[0])
                    curr_times.append(time)
                elif "STD MPS COPY:" in line or "NB MPS COPY:" in line or "NB SPLIT MPS COPY:" in line or "PW MPS COPY:" in line or "PW SPLIT MPS COPY:" in line:
                    splitting = line.split(": ")
                    time = (float)(splitting[-1].split('\n')[0])
                    if "STD MPS COPY:" in line:
                        alltoall_multippg[curr_ppg][-1].add_timing(curr_size, time)
                    elif "NB MPS COPY:" in line:
                        alltoall_multippg_nb[curr_ppg][-1].add_timing(curr_size, time)
                    elif "NB SPLIT MPS COPY:" in line:
                        alltoall_multippg_nbsplit[curr_ppg][-1].add_timing(curr_size, time)
                    elif "PW MPS COPY:" in line:
                        alltoall_multippg_pw[curr_ppg][-1].add_timing(curr_size, time)
                    elif "PW SPLIT MPS COPY:" in line:
                        alltoall_multippg_pwsplit[curr_ppg][-1].add_timing(curr_size, time)
                    
        if active_method and curr_size != -1 and len(curr_times) > 0 and method is not None:
            method.add_timing(curr_size, min(curr_times))
            curr_times = []

        file.close()

for ppg in alltoall_multippg_pw.keys():
    # Standard alltoall Times
    plt.add_luke_options()
    plt.line_plot(alltoall_1ppg_best[-1].times, alltoall_1ppg_best[-1].sizes, label = "1 PPG, Best")
    plt.line_plot(alltoall_multippg_pw[ppg][-1].times, alltoall_multippg_pw[ppg][-1].sizes, label = str(ppg) + " PPG, Pw")
    plt.line_plot(alltoall_multippg_pwsplit[ppg][-1].times, alltoall_multippg_pwsplit[ppg][-1].sizes, label = str(ppg) + " PPG, Pw Split")
    plt.line_plot(alltoall_multippg_nb[ppg][-1].times, alltoall_multippg_nb[ppg][-1].sizes, label = str(ppg) + " PPG, Nb")
    plt.line_plot(alltoall_multippg_nbsplit[ppg][-1].times, alltoall_multippg_nbsplit[ppg][-1].sizes, label = str(ppg) + " PPG, Nb Split")
    plt.line_plot(alltoall_multippg[ppg][-1].times, alltoall_multippg[ppg][-1].sizes, label = str(ppg) + " PPG, Sys")
    plt.add_anchored_legend(ncol=3)
    plt.add_labels("Num Floats", "Time (Seconds)")
    plt.set_scale('log', 'log')
    plt.save_plot(alltoall_fn_prefix + "_somemethods_ppg_best1_vs_all" + str(ppg) + "ppg.pdf")
    
    # Standard alltoall Times
    plt.add_luke_options()
    plt.line_plot([alltoall_1ppg_best[i].times[-1] for i in range(len(nodes))], nodes, label = "1 PPG, Best")
    plt.line_plot([alltoall_multippg_pw[ppg][i].times[-1] for i in range(len(nodes))], nodes, label = str(ppg) + " PPG, Pw")
    plt.line_plot([alltoall_multippg_pwsplit[ppg][i].times[-1] for i in range(len(nodes))], nodes, label = str(ppg) + " PPG, Pw Split")
    plt.line_plot([alltoall_multippg_nb[ppg][i].times[-1] for i in range(len(nodes))], nodes, label = str(ppg) + " PPG, Nb")
    plt.line_plot([alltoall_multippg_nbsplit[ppg][i].times[-1] for i in range(len(nodes))], nodes, label = str(ppg) + " PPG, Nb Split")
    plt.line_plot([alltoall_multippg[ppg][i].times[-1] for i in range(len(nodes))], nodes, label = str(ppg) + " PPG, Sys")
    plt.add_anchored_legend(ncol=3)
    plt.add_labels("Nodes", "Time (Seconds)")
    plt.set_scale('linear', 'log')
    plt.set_xticks(nodes, nodes)
    plt.save_plot(alltoall_fn_prefix + "_somemethods_ppg_nodes_best1_vs_all" + str(ppg) + "ppg.pdf")