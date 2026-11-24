# num of runs
NUM_RUNS=$1

# Values fixed for Intel Max 1550
MIN_FREQ_MHZ=$2
MAX_FREQ_MHZ=$3
STEP=$4 # Use multiple of 50 since this is the step for Intel Max 1550. 
EXE=$5

SCRIPT_DIR=$(dirname "$(readlink -f "$0")") 
BUILD_DIR="$SCRIPT_DIR/../../../build/" 
LOG_DIR="$SCRIPT_DIR/../../../logs/"
NUM_GPUS_PER_NODE=12


echo $LOG_DIR
echo $BUILD_DIR
echo $SCRIPT_DIR
if [ ! -d "${LOG_DIR}" ]; then
    # Create the directory
    mkdir -p "${LOG_DIR}"
    echo "Directory created: ${LOG_DIR}"
else
    echo "Directory already exists: ${LOG_DIR}"
    rm -rf ${LOG_DIR}/*
fi


for ((freq=$MIN_FREQ_MHZ; freq<=$MAX_FREQ_MHZ; freq+=$STEP)); do
    echo "----------------------------------------"
    echo "Running frequency: ${freq} MHz"
    echo "----------------------------------------"
    for ((j=0; j<$NUM_GPUS_PER_NODE; j++)); do
        geopmwrite GPU_CORE_FREQUENCY_MAX_CONTROL gpu_chip ${j} 1600e6
        geopmwrite GPU_CORE_FREQUENCY_MIN_CONTROL gpu_chip ${j} 1600e6 
        geopmwrite GPU_CORE_FREQUENCY_MIN_CONTROL gpu_chip ${j} ${freq}e6
        geopmwrite GPU_CORE_FREQUENCY_MAX_CONTROL gpu_chip ${j} ${freq}e6
        echo "GPU ${j} start energy: "
        geopmread  GPU_CORE_ENERGY gpu_chip ${j} 
    done

    for ((i=0; i<$NUM_RUNS; i++)); do
        # Create difrecotory for log files
        if [ ! -d "${LOG_DIR}/freq-${freq}/run-${i}/" ]; then
            # Create the directory
            mkdir -p "${LOG_DIR}/freq-${freq}/run-${i}/"
            echo "Directory created: ${LOG_DIR}/freq-${freq}/run-${i}/"
        else
            echo "Directory already exists: ${LOG_DIR}/freq-${freq}/run-${i}/"
            rm -rf ${LOG_DIR}/*
        fi

        echo "Run $i at ${freq} MHz"
        "${BUILD_DIR}/${EXE}" "${freq}" "${LOG_DIR}/freq-${freq}/run-${i}/"  > "${LOG_DIR}/freq-${freq}/run-${i}/output.log"
    done
done
