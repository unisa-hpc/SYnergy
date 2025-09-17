# Test for estimating the frequency change ovrhead
## freq_overhead.cpp
This file run two kernels with different approaches APP, KERNEL or PHASE. This script reproduce the plot in the IPDPS paper Figure XXX
### How to run


## freq_overhead_test.cpp
This program estimate the time spent in frequency change.
STEP 1 Run a dummy kernel N times without changing the frequency. 
STEP 2 Run a dummy kernl N time by changing the freq. from X to Y and Y-> X. 
STEP 3 After the freq. change the program wait unti the frequency is changed. (polling_freq)
### How to run
```
./build/samples/freq_overhead_test  <num_runs> <polling_time_us> <n_kernels> <freq1> <freq2>
```
#### Script args:
- `num_runs` defines the number of runs.
- `polling_time_us` defines the waiting time before checking the current frequency.
- `n_kernels` define the number of kernels submitted to the queue.
- `freq1` and `freq2` define the frequencies used during STEP 2. 


## run_freq_overhead_test.py
This script runs the freq_overhead_test.cpp by selecting freq1 and freq2 between MIN frequency and MAX frequency.
We want to test how much the distance between frequency impact on the frequency change overhead. 
The script start with the maximum distance freq1=MIN and freq2=MAX and ends when freq1 and fre2 are close to each other.
### How to run
```
samples/freq_overhead/run_freq_ovrhead_test.py --output-log=$(pwd)/ 
```
#### Script args:
- `--num-runs` number of runs.
- `--num-kernels` number of kernels executed for each run
- `--query-freq-path` path to the executable query_freq for having the available frequency on the target hardware
- `--output-log-path` path to the directory where the log file will be stored
- `--overhead-script-path` path to the script freq_overhead_test

