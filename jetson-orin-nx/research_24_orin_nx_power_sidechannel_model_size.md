# The Power Side-Channel Goes Further: Identifying Which Model Size Is Running, Not Just What Kind of Task

**Platform:** Seeed Studio reComputer J4012 — NVIDIA Jetson Orin NX 16GB (module P3767-300) on a J401 carrier board
**Software stack:** JetPack 7.2, L4T R39.2.0, MAXN_SUPER power mode, TensorRT-Edge-LLM v0.9.1
**Follow-up to:** paper 23 (workload-type power side-channel)
**Date:** August 2026

## Abstract

Paper 23 showed that four coarse workload *types* (idle, CPU stress, disk I/O, LLM decode) are trivially distinguishable from this board's power draw alone, using nothing but mean/variance/range statistics. This paper asks a harder question: holding the task constant (decode inference against an identical prompt) and varying only the *model size*, does the power signature still carry enough information to identify which model is running? Two TensorRT-Edge-LLM engines — Qwen3-1.7B-Instruct and Qwen3-4B-Instruct-2507, both freshly built at the same runtime version, both int4-class quantization, same input prompt — were sampled 4 times each under identical conditions. **Result: 8/8 trials correctly classified, 100% accuracy**, using the same simple leave-one-out nearest-neighbor approach as paper 23. The two models occupy tightly clustered, cleanly separated regions of feature space: 1.7B consistently shows stdev ~747-749mA and a top-decile peak current of ~3690mA across all four trials; 4B consistently shows stdev ~547-552mA and a top-decile peak of ~3126mA. Counter-intuitively, the *smaller* model produces the *larger* power swings — a real, mechanistically plausible finding (not just noise) discussed in §3, not the direction a naive "bigger model, bigger power spikes" intuition would predict.

## 1. Method

Both engines were built fresh at the identical TensorRT-Edge-LLM v0.9.1 commit used throughout this project's recent recovery work (see paper 15's version-churn findings — using matched versions specifically to avoid confounding the model-size comparison with any build/runtime difference). Both received the same treatment: `llm_build --maxBatchSize 1 --maxInputLen 1024 --maxKVCacheCapacity 4096`, then `llm_inference` against the identical prompt file used throughout this project's Qwen3 benchmarking (`input_qwen.json`), with power sampled at the same ~20Hz effective rate established in paper 23. Four independent trials per model, alternating 1.7B/4B/1.7B/4B/... to avoid any systematic drift-over-time confound between the two classes. Feature extraction and classification used the same leave-one-out nearest-neighbor method as paper 23, with one added feature (mean of the top 10% of samples, to specifically capture peak burst intensity) alongside mean, stdev, and range.

## 2. Results

| Model | Mean current (mA) | Stdev | Range | Top-10% mean (mA) |
|---|---|---|---|---|
| Qwen3-1.7B | 1783-1842 | 747-749 | 2856-2896 | 3682-3712 |
| Qwen3-4B | 1737-1744 | 547-552 | 2448-2496 | 3117-3133 |

All 8 leave-one-out classifications were correct. Notably, the *mean* current is nearly identical between the two models (1.7B: ~1800mA, 4B: ~1741mA — a difference too small to reliably discriminate on its own), while stdev, range, and peak burst current differ by roughly 35-40% between classes and are consistent to within a few percent across all four trials of each model — the actual discriminating signal lives entirely in the *shape* of the power trace's variability, not its average level.

## 3. Why the Smaller Model Shows Bigger Swings — A Plausible Mechanism, Not Yet Proven

The intuitive prediction would be that the larger (4B) model, doing more compute per decoded token, draws a higher peak current. The data shows the opposite. A plausible explanation, consistent with this project's own prior findings (papers 9, 11, 12 established that this board's decode-phase workload is generally not compute-saturating and rarely trips throttling, unlike prefill): Qwen3-1.7B generates tokens faster than Qwen3-4B (fewer parameters, less compute per layer per token), so within the same fixed 8-second sampling window it completes more discrete decode steps. Each individual decode step likely has a similar underlying "burst of GPU activity, then a shorter idle/memory-bound gap" pattern regardless of model size (autoregressive decoding is inherently one-token-at-a-time), but the *smaller* model's bursts are more tightly packed together — less idle time diluting each burst's contribution to the trace's overall variance — while the larger model's slower per-token cadence spreads its bursts further apart, and each individual burst, while representing more total work, may not produce a higher *instantaneous* peak current if the additional compute is more evenly pipelined across more of the model's layers rather than concentrated into a sharper spike. This mechanism is plausible and consistent with the measured data but was not independently verified here — confirming it would require higher-resolution power sampling than this board's INA3221 update rate allows (per paper 23's own sampling-rate finding), or a direct measurement of per-token decode latency alongside the power trace to check whether burst frequency, not burst amplitude, is really the dominant factor.

## 4. What This Adds Beyond Paper 23

Paper 23 established that *what kind* of workload is running (idle vs. CPU-bound vs. I/O-bound vs. LLM inference) is trivially visible in this board's power draw. This paper establishes something meaningfully harder: *within the same workload category* (LLM decode specifically), *which specific model* is running is also visible, with equally clean separation and equally small sample sizes. This matters for the side-channel's real-world significance — an observer who already knows "this device is doing LLM inference" (from paper 23's result) can plausibly go one step further and infer *which model*, without any access beyond the power rail, which is a meaningfully more specific and more concerning capability than workload-type classification alone (e.g., distinguishing a smaller, less capable model from a larger, more capable one running the same class of task could reveal information about a deployment's actual configuration or capacity).

## 5. Limitations

Same caveats as paper 23, restated for this specific test: two model sizes only (a real deployment scenario might involve many more candidate models, which could plausibly reduce separability as classes multiply), small trial count (4 per class), identical prompt/task held constant deliberately to isolate the model-size variable — a more realistic scenario with varying prompts and task types simultaneously was not tested, and the mechanistic explanation in §3 is a plausible hypothesis, not a confirmed cause.

## Evidence

Raw power traces for all 8 trials (2 models × 4 trials), the collection harness (`model_sidechannel_collect.sh`), and the classification script (`model_sidechannel_classify.py`), in `data/`, prefixed `orinnx-20260815-model-sidechannel-`.
