#!/bin/bash

SUPPORTED_ARCHS=" intel cuda "
arch="cuda"
frequencies=""
def_core=""
def_mem=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch=*)
      arch="${1#*=}"
      shift
      ;;
    *)
      echo "Invalid argument: $1"
      return 1 2>/dev/null
      exit 1
      ;;
  esac
done

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [[ $SUPPORTED_ARCHS != *"$arch"* ]]; then
  echo "Unsupported architecture: $arch"
  return 1 2>/dev/null
  exit 1
fi

if [[ "$arch" = "intel" ]]; then
  intel_gpu_frequency -d
  output=$(intel_gpu_frequency)
  # find min and max frequencies
  min=$(echo "$output" | grep "min" | awk '{print $2}')
  max=$(echo "$output" | grep "max" | awk '{print $2}')
  # for each frequency between min and max, increment i by 50
  for ((i=min; i<=max; i+=50)); do
    frequencies="$frequencies $i"
  done
  # trim leading whitespace
  frequencies=$(echo "$frequencies" | sed -e 's/^[[:space:]]*//')
elif [[ "$arch" = "cuda" ]]; then
  # read frequencies with nvidia-smi
  output=$(nvidia-smi -q -d SUPPORTED_CLOCKS)
  # find min and max frequencies
  frequencies=$(nvidia-smi -i 0 --query-supported-clocks=gr --format=csv,noheader,nounits)

  nvsmi_out=$(nvidia-smi  -q | grep "Default Applications Clocks" -A 2 | tail -n +2)
  def_core=$(echo $nvsmi_out | awk '{print $3}')
  def_mem=$(echo $nvsmi_out | awk '{print $7}')
fi

# run the benchmark for each frequency
for freq in $frequencies; do
  echo "[*] Running benchmark for frequency $freq"
  if [[ "$arch" = "intel" ]]; then
    intel_gpu_frequency --set $freq
  elif [[ "$arch" = "cuda" ]]; then
    nvidia-smi -ac $def_mem,$freq > /dev/null
  fi
  echo "Non-setting frequency..."
  $SCRIPT_DIR/freq_overhead
  if [[ "$arch" = "intel" ]]; then
    intel_gpu_frequency -d
  elif [[ "$arch" = "cuda" ]]; then
    nvidia-smi -rac > /dev/null
  fi
  echo "Setting frequency..."
  $SCRIPT_DIR/freq_overhead $freq
done

if [[ "$arch" = "intel" ]]; then
  intel_gpu_frequency -d
elif [[ "$arch" = "cuda" ]]; then
  nvidia-smi -rac
fi