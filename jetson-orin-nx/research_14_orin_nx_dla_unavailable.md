# DLA Hardware Initializes but Is Unusable Through TensorRT on This Jetson Orin NX Software Stack

**Platform:** Seeed Studio reComputer J4012 — NVIDIA Jetson Orin NX 16GB (module P3767-300) on a J401 carrier board
**Software stack:** JetPack 7.2, L4T R39.2.0, CUDA 13.2, TensorRT 10.16.2.10
**Date:** August 2026

## Abstract

This board's two DLA (Deep Learning Accelerator) cores are widely assumed to be usable "for free" alongside the GPU on Jetson Orin hardware — most public benchmarks either use them successfully or don't mention them at all. This paper attempted to build a DLA-targeted TensorRT engine specifically to characterize concurrent DLA+GPU power and utilization during LLM inference, and found the DLA path does not work on this JetPack 7.2 / TensorRT 10.16.2.10 configuration at all. The kernel driver initializes both DLA cores cleanly at boot with no errors, and `jetson_clocks` reports both online and clocked. TensorRT itself, however, reports zero available DLA cores when asked to build any DLA-targeted engine, and no DLA compiler package exists anywhere in this JetPack release's apt repository to remedy it — the required component appears to be entirely absent from this software configuration, not merely misconfigured. This is reported as a negative result: the concurrent DLA+GPU utilization study this paper set out to run could not be performed on this hardware/software combination as currently configured.

## 1. What Was Attempted

A small ONNX classification model (ResNet-18, downloaded directly from the ONNX model zoo, since no ONNX model already existed locally and this board has no `torch`/`torchvision` installed for local export) was used as a minimal target — the goal was DLA *utilization* during a concurrent GPU workload, not a specific detection or classification benchmark, so any small standard model sufficed. `trtexec` was used to attempt a DLA-targeted engine build:

```
trtexec --onnx=resnet18.onnx --useDLACore=0 --int8 --allowGPUFallback --saveEngine=resnet18_dla.engine
```

This failed immediately during engine configuration:

```
[E] Cannot create DLA engine, 0 not available
[E] Network And Config setup failed
[E] Building engine failed
```

Retried with `--fp16` instead of `--int8` (DLA supports both, so this ruled out a precision-specific issue) and with `sudo` (ruled out a permissions issue) — both retries failed identically.

## 2. Isolating the Cause: Kernel Driver Works, TensorRT Doesn't See It

`dmesg` shows both DLA cores initializing successfully at boot, with no errors:

```
[   10.714680] nvdla 15880000.nvdla0: Adding to iommu group 37
[   10.738612] nvdla 15880000.nvdla0: syncpt_unit_base 60000000 syncpt_unit_size 4000000 size 10000
[   10.758399] nvdla 158c0000.nvdla1: Adding to iommu group 39
[   10.956140] nvdla 158c0000.nvdla1: syncpt_unit_base 60000000 syncpt_unit_size 4000000 size 10000
```

`sudo jetson_clocks --show` independently confirms both cores online and clocked:

```
DLA0_CORE:   Online=1 MinFreq=0 MaxFreq=1228800000 CurrentFreq=1228800000
DLA1_CORE:   Online=1 MinFreq=0 MaxFreq=1228800000 CurrentFreq=1228800000
```

Device nodes for both cores' control channels exist (`/dev/nvhost-ctrl-nvdla0`, `/dev/nvhost-ctrl-nvdla1`). The kernel-level driver stack is unambiguously present and functioning. The failure is entirely at the TensorRT/userspace level: TensorRT's own DLA-core-count query returns zero, which is what `trtexec`'s "Cannot create DLA engine, 0 not available" error actually reports — not a hardware absence, a software one.

Checking for the missing component: `find / -iname "*nvdla*compiler*"` and `find / -iname "libnvdla*"` locate only `/usr/lib/aarch64-linux-gnu/nvidia/libnvdla_runtime.so` — a runtime library, not a compiler. `dpkg -l | grep -i dla` shows only `libcudla-13-2`/`libcudla-dev-13-2` (CUDA-DLA interop libraries, not the DLA engine compiler itself) installed. Searching the apt repository directly — `apt-cache search nvidia | grep -iE 'dla|compiler'` — returns no DLA-compiler package at all, under any name, in this JetPack 7.2 release's package index.

## 3. Conclusion

The DLA compiler component TensorRT needs to build a DLA-targeted engine is not merely uninstalled — it does not appear to exist as an installable package anywhere in this JetPack 7.2 configuration's apt repository, despite the kernel driver, device nodes, and clock management for both DLA cores all functioning correctly. This is worth documenting because most public Jetson benchmarking content either successfully uses DLA or silently doesn't mention trying — a reader assuming "the DLA cores are there, so DLA-targeted inference is available" on a JetPack 7.2 / TensorRT 10.16.2.10 board would hit this exact wall. Whether this is specific to this board's JetPack revision, a broader R39.2.0 issue, or specific to this unit's history of package modifications (this board has had a substantial desktop-package strip applied, documented in a companion paper) was not distinguished here — a clean-image JetPack 7.2 install would be needed to isolate which. The concurrent DLA+GPU utilization study this paper set out to perform — running a DLA workload alongside GPU-side LLM decode to characterize whether the two draw from a shared power/current budget — could not be conducted as a result, and is left as future work contingent on resolving this gap.

## Evidence

`trtexec` failure logs (both `--int8` and `--fp16` attempts), `dmesg` DLA driver initialization output, `jetson_clocks --show` DLA status, and the `dpkg`/`apt-cache` search results confirming no DLA compiler package is available, in `data/`, prefixed `orinnx-20260811-campaign4-`.
