#!/bin/bash
#PBS -N freq_scaling_test
#PBS -A  xxx
#PBS -l select=1:ngpus=1
#PBS -l walltime=00:20:00
#PBS -l filesystems=home
#PBS -o xxx/SYnergy/pbs-out/output.txt
#PBS -e xxx/SYnergy/pbs-out/error.txt
#PBS -q debug

# Load required modules or source oneAPI environment
source xxx/scripts/aurora/env/set_env.sh

# /home/lcarpent/energy-workspace/SYnergy/build/samples/matrix_mul  
export  ZES_ENABLE_SYSMAN=1
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT


NUM_RUNS=1
STEP=1400
MIN_FREQ=200
MAX_FREQ=1600
EXE=/samples/mat_mul

/home/lcarpent/energy-workspace/SYnergy/scripts/aurora/run/run_freq_scaling.sh $NUM_RUNS $MIN_FREQ $MAX_FREQ $STEP $EXE