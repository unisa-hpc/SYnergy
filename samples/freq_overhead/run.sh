#!/bin/bash

SUPPORTED_ARCHS=" intel cuda rocm "
arch="cuda"
first_iters="1"
second_iters="1"
first_freq1="1110"
first_freq2="1117"
second_freq1="645"
second_freq2="652"
app_freq="715"
n_runs=1
n_iters=128

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel_iters=*)
      n_iters="${1#*=}"
      shift
      ;;
    --runs=*)
      n_runs="${1#*=}"
      shift
      ;;
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

function reset_freq {
  if [[ "$arch" = "intel" ]]; then
    intel_gpu_frequency -d
  elif [[ "$arch" = "cuda" ]]; then
    nvidia-smi -rac > /dev/null
    nvidia-smi -rgc > /dev/null
  elif [[ "$arch" = "rocm" ]]; then
    rocm-smi --device=0 --setperflevel auto > /dev/null
  fi
}

function init_scaling {
  if [[ "$arch" = "rocm" ]]; then
    rocm-smi --device=0 --setperflevel manual > /dev/null
  fi
}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [[ $SUPPORTED_ARCHS != *"$arch"* ]]; then
  echo "Unsupported architecture: $arch"
  return 1 2>/dev/null
  exit 1
fi

init_scaling

echo "Running freq_overhead for $arch" > ./output.log
echo "Running freq_overhead for $n_iters iterations" >> ./output.log
echo "Policy: app" >> ./output.log
reset_freq
$SCRIPT_DIR/freq_overhead app $n_runs $n_iters 1024 1024 $app_freq $app_freq $app_freq $app_freq $first_iters $second_iters >> ./output.log
init_scaling
echo "Policy: kernel" >> ./output.log
$SCRIPT_DIR/freq_overhead kernel $n_runs $n_iters 1024 1024 $first_freq1 $first_freq2 $second_freq1 $second_freq2 $first_iters $second_iters >> ./output.log

reset_freq