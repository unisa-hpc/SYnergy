#!/bin/bash

SUPPORTED_ARCHS=" intel cuda "
arch="cuda"
first_iters="1"
second_iters="1"
first_freq="1110"
second_freq="645"
app_freq="705"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform=*)
      arch="${1#*=}"
      shift
      ;;
    --first-iters=*)
      first_iters="${1#*=}"
      shift
      ;;
    --second-iters=*)
      second_iters="${1#*=}"
      shift
      ;;
    --first-freq=*)
      first_freq="${1#*=}"
      shift
      ;;
    --second-freq=*)
      second_freq="${1#*=}"
      shift
      ;;
    --app-freq=*)
      app_freq="${1#*=}"
      shift
      ;;
    -h | --help)
      help
      return 0 2>/dev/null
      exit 0
      ;;
    *)
      echo "Invalid argument: $1"
      echo "Use --help for more information."
      return 1 2>/dev/null
      exit 1
      ;;
  esac
done

function help {
  echo "Usage: run.sh [options]"
  echo "Options:"
  echo "  --platform=<arch>      Architecture to run the sample on. Supported values: $SUPPORTED_ARCHS"
  echo "  --first-iters=<value>  Number of iterations for the first frequency (default: 1)"
  echo "  --second-iters=<value> Number of iterations for the second frequency (default: 1)"
  echo "  --first-freq=<value>   First frequency to set (default: 1110)"
  echo "  --second-freq=<value>  Second frequency to set (default: 645)"
  echo "  --app-freq=<value>     Application frequency to set (default: 705)"
}

function reset_freq {
  if [[ "$arch" = "intel" ]]; then
    intel_gpu_frequency -d
  elif [[ "$arch" = "cuda" ]]; then
    nvidia-smi -rac > /dev/null
    nvidia-smi -rgc > /dev/null
  fi
}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [[ $SUPPORTED_ARCHS != *"$arch"* ]]; then
  echo "Unsupported architecture: $arch"
  return 1 2>/dev/null
  exit 1
fi

echo "Running freq_overhead for $arch" > ./output.log
for n_iters in 4 8 16; do
  echo "Running freq_overhead for $n_iters iterations" >> ./output.log
  echo "Policy: app" >> ./output.log
  $SCRIPT_DIR/freq_overhead app 10 $n_iters 1024 2048 $app_freq $app_freq $first_iters $second_iters >> ./output.log
  echo "Policy: phase" >> ./output.log
  $SCRIPT_DIR/freq_overhead phase 10 $n_iters 1024 2048 $first_freq $second_freq $first_iters $second_iters >> ./output.log
  echo "Policy: kernel" >> ./output.log
  $SCRIPT_DIR/freq_overhead kernel 10 $n_iters 1024 2048 $first_freq $second_freq $first_iters $second_iters >> ./output.log
done

reset_freq