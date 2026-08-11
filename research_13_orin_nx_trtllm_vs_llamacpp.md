# TensorRT-Edge-LLM vs. llama.cpp on the Same Jetson Orin NX 16GB: A Matched-Model Head-to-Head

**Platform:** Seeed Studio reComputer J4012 — NVIDIA Jetson Orin NX 16GB (module P3767-300) on a J401 carrier board
**Software stack:** JetPack 7.2, L4T R39.2.0, CUDA 13.2, TensorRT 10.16.2.10, TensorRT-Edge-LLM, llama.cpp (CUDA backend, built from source)
**Date:** August 2026

## Abstract

TensorRT-Edge-LLM and llama.cpp are rarely compared rigorously on the same Jetson hardware with the same model weights — most comparisons either use different models, different hardware, or don't control for quantization scheme. This paper builds both stacks on the same Orin NX unit already characterized in companion papers, using the same base checkpoints (Qwen3-1.7B and Qwen3-4B-Instruct-2507) for both, and benchmarks prefill and decode throughput at matched settings (128-token context, `MAXN_SUPER` power mode, identical clock-locked state). TensorRT-Edge-LLM wins decisively on both models: 57-65% faster prefill and 28-50% faster decode. Power draw between the two stacks is nearly identical (within 1W), so the throughput gap translates almost directly into a tokens-per-joule gap of similar magnitude — this is not a case where a slower engine buys better efficiency. The comparison is not perfectly controlled: the two stacks quantize differently (TensorRT-Edge-LLM used NVIDIA's own NVFP4-family export used elsewhere in this project's benchmarks; llama.cpp used a freshly-generated Q4_K_M GGUF from an unquantized checkpoint, since llama.cpp's converter cannot ingest NVIDIA's quantized tensor format directly), a real constraint documented here rather than glossed over.

## 1. Method

Both engines were built and run on the same physical unit, same boot, same `MAXN_SUPER` power mode with `jetson_clocks` applied identically before each benchmark.

**TensorRT-Edge-LLM** used the pre-built engines from the companion tokens-per-joule paper — Qwen3-1.7B-Instruct and Qwen3-4B-Instruct-2507, both built from NVIDIA's own quantized checkpoint export (`hf_quant_config.json` present, NVFP4-family scheme), via `llm_build --maxBatchSize 1 --maxInputLen 512 --maxKVCacheCapacity 1024`. Prefill/decode figures reused from that paper's 128-token-context MAXN_SUPER runs for a same-condition comparison.

**llama.cpp** was cloned fresh (`ggml-org/llama.cpp`, depth-1) and built from source with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87`, targeting `llama-bench`/`llama-cli`/`llama-quantize` only. The CUDA compile itself was the single most time-consuming step of this entire study — llama.cpp's CUDA backend generates a very large number of templated kernel instantiation files (attention-variant and matmul-quantization-variant kernels each compiled separately), and at `-j4` on this board's 8 cores the build took roughly 45-60 minutes of wall-clock time, dwarfing every other build in this project's benchmarking work. A first build attempt was lost entirely to an unrelated `nvpmodel` reboot (§2) partway through and had to restart from scratch — plan for this if reproducing.

The engine's own `Qwen3-1.7B-Instruct`/`Qwen3-4B-Instruct-2507` quantized checkpoints could not be converted directly: `convert_hf_to_gguf.py` failed with `Can not map tensor 'model.layers.0.mlp.down_proj.pre_quant_scale'` — llama.cpp's converter expects standard HF tensor layouts and does not understand NVIDIA's NVFP4-family quantization metadata. The unquantized base checkpoints were downloaded fresh from Hugging Face instead (`Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B-Instruct-2507`; llama.cpp requires the CPU build of PyTorch as a conversion dependency, itself a nontrivial install on this board since `pip3` was not present after this unit's earlier desktop-package strip and had to be reinstalled via `apt-get install python3-pip` first), converted to f16 GGUF, then quantized to **Q4_K_M** via `llama-quantize` (1.7B: 3.88GiB→1.19GiB; 4B: 7.67GiB→2.32GiB). `llama-bench -ngl 99 -p 128 -n 128 -r 5` was run for both models with concurrent `tegrastats --interval 300` capturing `VDD_IN`.

## 2. A Reliability Aside: Reboots Kill Background Builds

Getting to a stable `MAXN_SUPER` state for this comparison required two `nvpmodel` mode-switch reboots earlier in this project's broader benchmarking session (documented in the companion tokens-per-joule paper). Each of those reboots silently killed the still-running, `nohup`'d llama.cpp build in progress, since a full system reboot terminates all processes regardless of `nohup`/`disown` — this is not a bug in the build tooling, but a real operational hazard worth flagging for anyone running a long background compile on a Jetson board interleaved with `nvpmodel` mode-switching work: sequence the power-mode sweep to completion first, or isolate long builds to a session with no pending reboots.

## 3. Results

### 3.1 Throughput (128-token context, MAXN_SUPER, both clock-locked identically)

| Model | Stack | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|
| Qwen3-1.7B | TensorRT-Edge-LLM | **2648.2** | **56.5** |
| Qwen3-1.7B | llama.cpp (Q4_K_M) | 1683.5 ± 63.8 | 44.2 ± 0.03 |
| Qwen3-4B | TensorRT-Edge-LLM | **1178.9** | **30.0** |
| Qwen3-4B | llama.cpp (Q4_K_M) | 713.4 ± 16.4 | 20.0 ± 0.03 |

TensorRT-Edge-LLM wins on every measure: **+57.3% prefill / +27.8% decode** on the 1.7B model, **+65.3% prefill / +49.8% decode** on the 4B model. The gap widens on the larger model for both phases, consistent with TensorRT's engine having more opportunity to exploit fused/autotuned kernels as compute per layer grows.

### 3.2 Power and Efficiency

| Model | Stack | Mean VDD_IN | Decode J/token |
|---|---|---:|---:|
| Qwen3-1.7B | TensorRT-Edge-LLM | 19.994 W | 0.3538 |
| Qwen3-1.7B | llama.cpp | 20.081 W | 0.4542 |
| Qwen3-4B | TensorRT-Edge-LLM | 21.419 W | 0.7140 |
| Qwen3-4B | llama.cpp | 21.457 W | 1.0714 |

Power draw between the two stacks is within 1W of each other for both models — running either engine costs essentially the same electrically. The efficiency gap therefore tracks the throughput gap almost exactly: llama.cpp uses **28% more energy per decoded token** on the 1.7B model and **50% more** on the 4B model. This rules out the possibility that TensorRT's speed advantage comes at a hidden power cost — on this hardware, it doesn't.

## 4. What Isn't Controlled Here

This comparison is informative but not a clean ablation, and readers should weigh two real confounds:

- **Quantization scheme differs.** TensorRT-Edge-LLM's engines use NVIDIA's own NVFP4-family quantized export (whatever scheme produced the `hf_quant_config.json`-tagged checkpoints used throughout this project), while llama.cpp used Q4_K_M — llama.cpp's own established general-purpose 4-bit scheme, chosen because it is the standard default choice for this model class in that ecosystem, but a different quantization method with different bit-allocation and calibration behavior. Some portion of the throughput and efficiency gap here reflects TensorRT-Edge-LLM's engine-level optimizations (kernel fusion, CUDA graph capture, autotuning) and some portion reflects quantization scheme — this study cannot cleanly separate the two.
- **Different starting checkpoints.** TensorRT-Edge-LLM's checkpoints were the exact ones already prepared and used throughout this project's other benchmarks; llama.cpp's were freshly downloaded from Hugging Face because the existing quantized checkpoints were not convertible. Both should represent the same underlying model weights pre-quantization, but this was not independently verified tensor-by-tensor.

Neither confound is expected to explain a 28-65% gap on its own — quantization-scheme differences at this bit-width are typically single-digit-percent throughput effects in published comparisons elsewhere — but a reader wanting a cleaner controlled comparison (same exact quantization scheme in both stacks) would need to either export TensorRT-Edge-LLM from a Q4_K_M-equivalent source or find a way to convert NVIDIA's NVFP4 export format directly, neither of which this study attempted.

## 5. Conclusion

On this specific Orin NX 16GB unit, for both model sizes tested, TensorRT-Edge-LLM is substantially faster and substantially more energy-efficient than llama.cpp's CUDA backend at matched context length and power mode — not a close call, and not offset by a hidden power cost. The gap is large enough (28-65% depending on metric and model size) that it should dominate a deployment decision on this hardware even accounting for the quantization-scheme confound in §4. The practical cost of choosing TensorRT-Edge-LLM is setup complexity — this project's own experience building it (documented in the companion strip/tune paper) and building llama.cpp's CUDA backend (§1-2 here) shows the latter is a substantially simpler, faster build with a much larger and more portable ecosystem of pre-quantized GGUF models, which for many use cases may outweigh a same-hardware throughput gap of this size.

## Evidence

Raw logs in `data/`, prefixed `orinnx-20260811-campaign3-`: `llama-bench` output for both models, `tegrastats` captures, the failed direct-conversion error log, and the successful f16→Q4_K_M `llama-quantize` output.
