# Engine and Plugin ABI Breakage Across Three TensorRT-Edge-LLM Point Releases on the Same Jetson Orin NX

**Platform:** Seeed Studio reComputer J4012 — NVIDIA Jetson Orin NX 16GB (module P3767-300) on a J401 carrier board
**Second host:** Windows 11 desktop, RTX 5090 (32607 MiB VRAM, driver 610.62), WSL2 Ubuntu 24.04.1 LTS
**Software stack:** JetPack 7.2, L4T R39.2.0, CUDA 13.2, TensorRT-Edge-LLM (versions v0.8.0, v0.9.1/`7f061f2`, v0.10.0/`71dd1ba`)
**Date:** August 2026

## Abstract

TensorRT-Edge-LLM (NVIDIA's edge inference framework, used throughout this project's Orin NX benchmark series) requires quantization and ONNX export to run on a separate x86 host with an NVIDIA GPU, while engine build and inference run on the Jetson device itself. In the course of scoping a new speculative-decoding benchmark campaign, this project encountered three distinct, incompatible TensorRT-Edge-LLM versions in active use across a single evening: v0.8.0 (the version that originally built this project's existing baseline engines), v0.9.1 (the version the Orin device's C++ runtime happened to be built at), and v0.10.0 (the current release, freshly cloned on the x86 export host). None of the three versions could reliably consume artifacts produced by either of the others. This paper documents two independent, reproducible breakages this caused, and reports a genuine upstream defect discovered and filed as a result (NVIDIA/TensorRT-Edge-LLM#178). The practical finding: a TensorRT-Edge-LLM deployment split across two hosts (as the framework's own documentation requires) needs the same commit pinned on both sides, deliberately and explicitly — "latest" on the export host and "whatever was built months ago" on the edge device are not assumed-compatible, and nothing in the tooling warns you until deserialization or compilation fails.

## 1. Breakage One: Plugin ABI Drift Between v0.8.0 Engines and v0.9.1/v0.10.0 Runtimes

This project's existing Qwen3-1.7B-Instruct baseline engine (`~/tensorrt-edgellm-workspace/Qwen3-1.7B-Instruct/engine/llm.engine`), used in prior published benchmarks, was built with TensorRT-Edge-LLM v0.8.0. Re-running it against the (newer) runtime binaries currently on the device — first v0.10.0's `llm_build`/`llm_inference`, then v0.9.1's — failed identically both times:

```
[WARNING] [version.cpp:103:checkVersion] Model version 0.8.0 does not match runtime version 0.9.1. Consider re-exporting or re-building.
...
[ERROR] [TensorRT] IPluginRegistry::getCreator: Error Code 4: API Usage Error
  (Cannot find plugin: Int4GroupwiseGemmPlugin, version: 1, namespace:.
   In getCreatorInternal at /_src/runtime/dispatch/pluginRegistry.cpp:436)
[ERROR] [TensorRT] IRuntime::deserializeCudaEngineV2: Error Code 1: Serialization
  (Serialization assertion creator failed. Cannot deserialize plugin since
   corresponding IPluginCreatorInterface not found in Plugin Registry)
```

The runtime does emit a version-mismatch warning (a real courtesy the framework provides), but the warning is non-fatal and easy to miss in verbose startup logs; the actual failure surfaces two log lines later as an opaque TensorRT plugin-registry error that gives no indication the root cause is a version skew rather than a corrupt file or a build misconfiguration. The plugin name itself changed between releases — v0.10.0's export path for this same model produces an `Int4GroupwiseGemmPluginV2` node (see Section 2), suggesting the plugin was versioned as part of ordinary development between v0.8.0 and v0.10.0, not a compatibility mechanism.

**Fix applied:** the original ONNX export (`Qwen3-1.7B-Instruct/onnx/llm/`) was still present on disk. Re-running `llm_build` against the *current* runtime's compiled binary, from that same ONNX, produced a working engine immediately — the ONNX itself was not the problem, only the previously-serialized `.engine` file. This confirms the breakage is purely at the compiled-engine layer, not the export layer; keeping ONNX artifacts around as a re-buildable source of truth, rather than only the final `.engine`, is what made recovery a one-command fix instead of a re-quantize-from-scratch.

## 2. Breakage Two: A v0.10.0 Runtime Bug That Blocks Every Engine Build on Orin

Independently of the v0.8.0 issue above, attempting a *fresh* export+build at the current v0.10.0 release — for a new EAGLE3 speculative-decoding engine, using ONNX freshly quantized and exported on the RTX 5090 x86 host at the same v0.10.0 commit — failed during `llm_build` on the Orin device with a compile-time C++ error, not a runtime deserialization error:

```
cuteDslFMHAV2Runner.h:127:37: error: 'fmha_v2_d64_Kernel_Module_t' was not declared in this scope
  127 |     static detail::LazyKernelModule<fmha_v2_d64_Kernel_Module_t> sLLM_d64;
compilation terminated due to -Wfatal-errors.
```

v0.10.0's own CHANGELOG explains the mechanism: *"Replaced the legacy embedded-cubin FMHA-v2 backend with CuTe DSL FMHA-v2 and removed the checked-in FMHA-v2 cubin artifacts."* The new header (`cpp/kernels/contextAttentionKernels/cuteDslFMHAV2Runner.h`) unconditionally declares kernel-module types (`fmha_v2_d64`, `d128`, `d256`, `d512`, and paged/sliding-window/bidirectional variants) that are meant to be supplied by a prebuilt CuTe DSL artifact checked into the repository per target architecture. Inspecting `cpp/kernels/cuteDSLArtifact/aarch64/sm_87/` (the Jetson Orin target) directly confirms it contains no `fmha_v2_*` headers at all — only GEMM, GDN, MoE, and FFPA kernel families are present for this architecture at this commit. Per `cpp/CMakeLists.txt`, "fmha is always linked" — this is not an optional kernel group, so the omission breaks `edgellmCore` compilation entirely, for every model, on this platform, not only spec-decode.

This was filed upstream with full reproduction steps: **[NVIDIA/TensorRT-Edge-LLM#178](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/178)**.

### 2.1 Workaround attempted and abandoned: pinning to the pre-migration commit

The commit immediately before this CuTe DSL migration (`7f061f2`, tagged `v0.9.1`) predates the FMHA backend replacement and builds cleanly on Orin — the C++ runtime rebuild at this commit succeeded without modification. However, re-running the *Python* export side at this same older commit, using a freshly created virtual environment (`pip install .`, unpinned dependencies, as the project's own installation docs specify), failed for an unrelated reason:

```
huggingface_hub.errors.HfUriError: Invalid HF URI
  'hf://datasets/cnn_dailymail@.../.huggingface.yaml'.
  Repository id must be 'namespace/name', got 'cnn_dailymail'.
```

The v0.9.1-era calibration code references the `cnn_dailymail` dataset by its legacy short-form Hugging Face identifier. Current (2026) `datasets`/`huggingface_hub` packages, installed fresh from PyPI with no version pin, no longer resolve short-form dataset IDs — HF deprecated that resolution path since this repository commit was written. Attempting to fix this by pinning `huggingface_hub` to an older, compatible release then broke a *different* dependency in the opposite direction: `pip install .`'s own unpinned `transformers` (resolved to a current release) requires newer `huggingface_hub` symbols (`is_offline_mode`) than the pinned-old version provides:

```
ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'
```

This is dependency rot compounding version skew: the repository code, the calibration dataset it downloads, and the Python package ecosystem it installs from are three independently moving targets, and pinning any one of them to match an eight-month-old commit does not guarantee the other two still agree with each other. This workaround was abandoned as a dead end rather than pursued further (e.g., by also pinning `transformers` and re-deriving a fully consistent dependency set for that specific commit) — the effort to fully reconstruct a working v0.9.1-era Python environment exceeded the value of avoiding the wait for an upstream fix to #178.

## 3. Conclusion

Three TensorRT-Edge-LLM versions were live in this project within a single evening — not through any deliberate multi-version testing, but simply as the ordinary consequence of one host (the Orin device) having been set up months prior and another (a newly-provisioned x86 export host) being freshly cloned at "latest" per the framework's own documented setup instructions. Both cross-version failures encountered were silent or misleading at the point of failure: the v0.8.0-engine-on-v0.9.1-runtime case buries the real cause behind a non-fatal warning and an opaque plugin-registry error two lines later; the v0.10.0 CuTe DSL case is a genuine upstream packaging gap with no warning at all, only a compile failure. Neither is discoverable by reading the top-level usage documentation, which describes the quantize→export→transfer→build→infer pipeline as if version is a non-issue.

The practical recommendation, and the reason this is worth a standalone note rather than folding into a results-focused paper: **treat the export host and the edge device as one deployment with one pinned commit, not two independently-updatable systems.** Cloning "latest" on a freshly-provisioned x86 host without checking what commit the edge device's C++ runtime was actually built from — which nothing in the framework surfaces automatically — is enough to reproduce either failure mode in this paper. Retaining ONNX exports (not only compiled `.engine` files) as a durable, re-buildable artifact, as this project already did opportunistically, is what made Section 1's breakage a one-command recovery instead of a full re-quantization; that practice is worth adopting deliberately rather than by accident.

## Evidence

Build logs for both breakages (`/tmp/build_base.log`, `/tmp/make.log`, `/tmp/make2.log` equivalents), the `llm_inference` deserialization failure log, the filed upstream issue (NVIDIA/TensorRT-Edge-LLM#178, full text below), and the abandoned-workaround pip install logs, in `data/`, prefixed `pikoi-orinnx-20260814-version-churn-`.

### Filed upstream issue text (NVIDIA/TensorRT-Edge-LLM#178)

> **Title:** [Bug] v0.10.0: prebuilt sm_87 (Jetson Orin) CuteDSL artifact missing fmha_v2_d* kernel modules, breaks all LLM engine builds
>
> See Section 2 above for the full technical content; the filed issue additionally includes exact reproduction commands (`git checkout v0.10.0`, the platform-recommended `cmake`/`make` invocation from the installation docs) and full system information (JetPack 7.2, CUDA 13.2, SM87, commit `71dd1bae032e70771265917ec74d3ff4cad07a10`).
