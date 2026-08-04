#!/usr/bin/env bash
# Build gpu_burn for whatever Jetson this is. No hardcoded -arch.
set -euo pipefail
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

NVCC=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)
[[ -x $NVCC ]] || { echo "ERROR: nvcc not found. Install CUDA toolkit (JetPack ships it at /usr/local/cuda)." >&2; exit 1; }
echo "nvcc: $($NVCC --version | tail -1)"

# Prefer -arch=native (CUDA 11.5+). Fall back to querying the device, then to a
# safe multi-arch fatbinary.
if $NVCC -arch=native -O3 -o "$DIR/gpu_burn" "$DIR/gpu_burn.cu" 2>/dev/null; then
	echo "built with -arch=native"
else
	CC=$(python3 - <<'PY' 2>/dev/null || true
import ctypes
for lib in ("libcudart.so","libcudart.so.13","libcudart.so.12","libcudart.so.11.0"):
    try: rt=ctypes.CDLL(lib); break
    except OSError: rt=None
if rt:
    maj=ctypes.c_int(); min_=ctypes.c_int()
    # cudaDevAttrComputeCapabilityMajor=75, Minor=76
    if rt.cudaDeviceGetAttribute(ctypes.byref(maj),75,0)==0 and \
       rt.cudaDeviceGetAttribute(ctypes.byref(min_),76,0)==0:
        print(f"{maj.value}{min_.value}")
PY
)
	if [[ -n ${CC:-} ]]; then
		echo "detected compute capability ${CC:0:1}.${CC:1}"
		$NVCC -arch="sm_${CC}" -O3 -o "$DIR/gpu_burn" "$DIR/gpu_burn.cu"
	else
		echo "falling back to multi-arch build (Xavier/Orin)"
		$NVCC -gencode arch=compute_72,code=sm_72 \
		      -gencode arch=compute_87,code=sm_87 \
		      -O3 -o "$DIR/gpu_burn" "$DIR/gpu_burn.cu"
	fi
fi
echo "built: $DIR/gpu_burn"
