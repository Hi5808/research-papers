# WSL2 GPU Passthrough Has No Overhead Tax for LLM Inference on This RTX 5090 — It's Measurably Faster

**Platform:** Windows 11 desktop, RTX 5090 (32607 MiB VRAM, driver 610.62), CUDA UMD 13.3
**Compared environments:** native Windows Python 3.12.10 + CUDA-enabled PyTorch, vs. WSL2 Ubuntu 24.04.1 LTS + identical PyTorch build
**Model/method:** Qwen3-1.7B, fp16, single-batch, HuggingFace `transformers` `generate()`, greedy decode
**Date:** August 2026

## Abstract

WSL2's GPU passthrough overhead for CUDA workloads is widely assumed to be "near-zero" in general discussion, but is rarely measured for LLM-specific inference (attention/KV-cache access patterns differ from typical GPU benchmark workloads like games or CUDA samples). This paper runs an identical prefill/decode benchmark — same model, same precision, same exact PyTorch build (`2.13.0+cu130`), same GPU — under native Windows and under WSL2 on the same physical machine, and finds not "near-zero overhead" but a **measurable WSL2 advantage**: 44% faster prefill (6393 vs. 4440 tok/s, mean of 3 runs each) and 9% faster decode (39.8 vs. 36.6 tok/s). This is the opposite of the commonly assumed direction. A real, stated confound is not eliminated here: native Windows was running its normal desktop session (compositor, background services) during measurement, while WSL2 is headless — some or all of the gap may reflect GPU/scheduler contention from the native desktop environment rather than a WSL2-specific advantage. This is reported as a controlled same-hardware comparison with that confound named explicitly, not as a clean isolation of "WSL2 passthrough overhead" as a mechanism in the abstract.

## 1. Methodology

Both environments used the identical PyTorch wheel (`torch==2.13.0+cu130`, `pip install torch --index-url https://download.pytorch.org/whl/cu130`), confirmed to match by version string before each run — an initial mismatch (WSL2 environment had drifted to `2.12.0+cu130` due to unrelated dependency-resolution churn from other work on this host earlier the same session) was caught and corrected before any comparison numbers were trusted. Both ran the same benchmark script: `AutoModelForCausalLM.from_pretrained(..., dtype=torch.float16, device_map="cuda:0")`, a 3-iteration warmup, then 10 timed prefill-only forward passes (128-token fixed prompt, no KV cache reuse) and 5 timed `generate()` calls (128 new tokens, greedy, `use_cache=True`), with `torch.cuda.synchronize()` bracketing every timed region to avoid measuring asynchronous CUDA queue depth instead of actual completion time.

Three full runs of the script (each run reloading the model fresh) were taken per environment to check for run-to-run noise before drawing any conclusion — this mattered: the first native run's prefill figure (3730 tok/s) differed from the second (4882 tok/s) by 31%, large enough that a single-run comparison would have been unreliable in either direction.

## 2. Results

| Metric | Native Windows (n=3) | WSL2 (n=3) | WSL2/Native ratio |
|---|---|---|---|
| Prefill tok/s (128 tok) | mean 4440, stdev 621 | mean 6393, stdev 138 | **1.44x** |
| Decode tok/s (128 new tok, greedy) | mean 36.6, stdev 1.0 | mean 39.8, stdev 2.5 | **1.09x** |
| Peak allocated VRAM | 4172 MiB | 4172 MiB | identical |

Two things stand out beyond the headline direction. First, **WSL2's prefill numbers are far more consistent run-to-run** (stdev 138, ~2% of mean) than native's (stdev 621, ~14% of mean) — not just faster on average, but noticeably more stable, which argues against the gap being a fluke of one lucky WSL2 run. Second, the decode-phase gap (9%) is much smaller than the prefill-phase gap (44%) — prefill is a single large batched matrix-multiply-heavy forward pass, while decode is a sequence of small, latency-sensitive single-token steps; whatever mechanism produces native Windows' overhead evidently matters far more for large-batch throughput than for small-batch latency-bound work.

## 3. What This Does and Doesn't Establish

**Does not establish:** that WSL2's paravirtualized GPU passthrough layer is inherently faster than a native driver path in general. The native Windows environment during these runs was a normal interactive desktop session — display compositor (DWM) active, background services running, the same GPU serving the desktop's own rendering — while the WSL2 environment is headless with no competing GPU consumers. Some or all of the measured gap plausibly reflects this contention difference rather than anything about the passthrough mechanism itself. This confound was not isolated (e.g., by testing native Windows from a minimal/Safe-Mode-like session, or by disabling the desktop compositor) — doing so is the natural next step if a cleaner mechanistic claim is wanted.

**Does establish:** for the practical, realistic case most people are actually in — a normal Windows desktop machine, using WSL2 as a Linux environment for ML tooling (as this project does for the TensorRT-Edge-LLM export pipeline; see the companion papers on the speculative-decoding campaign) — there is no throughput tax to worry about. If anything, running inference workloads from inside WSL2 on this specific system was faster and more consistent than running the identical code natively. The common assumption that WSL2 costs you GPU throughput did not hold on this hardware for this workload.

## 4. Conclusion

Same GPU, same exact PyTorch build, same model, same precision — WSL2 outperformed native Windows on both prefill (44%) and decode (9%) throughput for Qwen3-1.7B inference, with substantially lower run-to-run variance in prefill. This contradicts the common assumption that WSL2 passthrough imposes a throughput cost, at least for this hardware/workload combination. The most likely explanation, not confirmed here, is desktop-session GPU/scheduler contention on the native side rather than a genuine passthrough-layer speed advantage — a real confound stated plainly rather than glossed over, consistent with this project's practice elsewhere (see the TensorRT-Edge-LLM-vs-llama.cpp and Orin-vs-RK3588 papers' own precision/setup caveats). Whatever the mechanism, the practical takeaway for anyone doing ML work on a Windows+WSL2 machine is that WSL2 is not the throughput compromise it's often assumed to be — on this system, it was the better-performing and more consistent choice of the two, not a fallback.

## Evidence

Benchmark script (`wsl_compare_bench.py`), and all six raw run logs (3 native, 3 WSL2) — including the version-drift discovery and correction — in `data/`, prefixed `pikoi-20260814-wsl2-passthrough-`.
