#!/bin/bash

SUPPORTED_ARCHS=" intel cuda "
arch="cuda"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform=*)
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

function reset_freq {
  if [[ "$arch" = "intel" ]]; then
    intel_gpu_frequency -d
  elif [[ "$arch" = "cuda" ]]; then
    nvidia-smi -rac > /dev/null
  fi
}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [[ $SUPPORTED_ARCHS != *"$arch"* ]]; then
  echo "Unsupported architecture: $arch"
  return 1 2>/dev/null
  exit 1
fi

echo "Running freq_overhead for $arch" > ./output.log
for n_iters in 8 16 32; do
  echo "Running freq_overhead for $n_iters iterations" >> ./output.log
  echo "Policy: app" >> ./output.log
  $SCRIPT_DIR/freq_overhead app 10 $n_iters 1024 524288 1087 1087 >> ./output.log
  echo "Policy: phase" >> ./output.log
  $SCRIPT_DIR/freq_overhead phase 10 $n_iters 1024 524288 1117 997 >> ./output.log
  echo "Policy: kernel" >> ./output.log
  $SCRIPT_DIR/freq_overhead kernel 10 $n_iters 1024 524288 1117 997 >> ./output.log
done

reset_freq