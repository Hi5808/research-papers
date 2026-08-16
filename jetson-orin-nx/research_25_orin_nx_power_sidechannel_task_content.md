# The Power Side-Channel Has a Ceiling: Response Length Leaks, Task Content Doesn't (At Least Not This Easily)

**Platform:** Seeed Studio reComputer J4012 — NVIDIA Jetson Orin NX 16GB (module P3767-300) on a J401 carrier board
**Software stack:** JetPack 7.2, L4T R39.2.0, MAXN_SUPER power mode, TensorRT-Edge-LLM v0.9.1, Qwen3-1.7B-Instruct (single model, held constant)
**Follow-up to:** papers 23 (workload type) and 24 (model size)
**Date:** August 2026

## Abstract

Papers 23 and 24 established that this board's power draw perfectly (100%) discriminates workload *type* and model *size* using nothing but coarse trace statistics. This paper pushes the same method one step further and asks whether it can identify *which specific task or prompt content* was given to an otherwise-identical model — the natural next question after "what kind of thing is running" and "which model is it," namely "what is it actually doing." The result is a genuine, informative negative: **response length is trivially visible and highly discriminating** (a short one-word-answer prompt is unmistakable against anything that generates a full response, ~3.2s vs. ~11.2-11.9s wall time, correctly classified in all trials), but **once multiple prompts produce similarly-long responses, this project's simple statistical method cannot reliably tell them apart** — three distinct task types (creative writing, code generation, step-by-step math reasoning), all of which happened to generate responses near the same 512-token output ceiling, were correctly classified only 3/9 times (67% wrong) by the leave-one-out nearest-neighbor classifier used throughout this line of work, and dropped further to near-chance when duration was excluded from the feature set entirely. The honest conclusion: this project's power side-channel has a real ceiling, and it sits at "how much was generated," not "what was generated" — a meaningfully different and more limited privacy exposure than papers 23-24 might have suggested was coming next.

## 1. Method

Four prompts, same model (Qwen3-1.7B-Instruct, same engine as papers 23-24), same runtime, run to natural completion (not a fixed sampling window as in prior papers — the power sampler ran until the `llm_inference` process actually exited, so response length differences show up directly as trace-length differences, which is itself the variable under test):

- **short_factual**: "What is the capital of France? Answer in one word."
- **long_creative**: a 400-word short-story-writing prompt
- **code_generation**: a request for a documented Python quicksort implementation
- **math_reasoning**: a step-by-step word problem requiring worked arithmetic

Three trials per prompt. Features: overall mean/stdev/max current, early-window mean, late-window mean (to capture any within-trace shape change over the response's duration), and wall-clock duration. Classification: the same leave-one-out nearest-neighbor method as papers 23-24, run twice — once with duration included as a feature, once with it explicitly excluded, to separate "how much signal comes from length alone" from "how much comes from the trace's actual shape."

## 2. Results

| Prompt | Wall time (s) | Mean current (mA) | Stdev | Max |
|---|---|---|---|---|
| short_factual | 3.20-3.25 | 1671-1724 | 168-171 | 2008-2064 |
| math_reasoning | 11.15-11.17 | 3686-3736 | 1264-1271 | 4608-4656 |
| long_creative | 11.66-11.70 | 3669-3736 | 1262-1277 | 4568-4648 |
| code_generation | 11.76-11.88 | 3714-3757 | 1244-1267 | 4584-4664 |

`short_factual`'s duration and every power statistic are dramatically different from the other three (roughly 3.6x shorter, less than half the mean current, an order of magnitude lower variance) — this one is trivially separable by inspection alone. The other three prompts, by contrast, occupy almost the same numeric range on every feature: durations span only 11.15-11.88s across all three types (a ~6% spread, smaller than this project's own measured run-to-run noise in earlier papers), and mean/stdev/max current overlap substantially between them.

**With duration included as a feature**: 6/12 correct (50%) — `short_factual` (3/3) and `math_reasoning` (3/3, apparently the fastest of the three long responses by a small, consistent margin) classified correctly every time; `long_creative` and `code_generation` were confused with each other and with `math_reasoning` in all 6 of their combined trials.

**With duration explicitly excluded**: 3/12 correct (25%, at chance for 4 classes) — only `short_factual` remained identifiable (from its power-level and variance statistics alone, independent of trace length), and every trial of the other three prompts was misclassified. This isolates the finding cleanly: essentially all of the signal that made the "with duration" case better than chance came from response length, not from anything about the power trace's shape reflecting the actual content or task type being computed.

## 3. Why This Ceiling Makes Sense

Autoregressive decoding, at the level of the coarse (~20Hz) telemetry this board's INA3221 sensor and this project's sampling method can access, looks structurally similar regardless of *what* is being generated — every decode step is one forward pass through the same fixed model architecture, and the GPU-level compute pattern of "compute the next token's logits" is not obviously different in kind between generating a line of Python, a sentence of prose, or a line of arithmetic, only in *how many* such steps occur before the response ends (which is exactly the variable that did leak clearly). This is consistent with, and a natural limiting case of, papers 23-24's own findings: paper 23 distinguished categories with genuinely different underlying compute *mechanisms* (CPU-bound busy-loop vs. GPU decode vs. disk I/O), and paper 24 distinguished model sizes that differ in the actual amount of computation per token — both are differences in the *shape* of the underlying work. Distinguishing prompts that all run the identical model doing the identical *kind* of per-token computation, differing only in the semantic content of what's being predicted, is a fundamentally finer-grained signal this method and this sensor's resolution do not appear to carry, at least not at the sample sizes and feature simplicity tested here.

## 4. What This Does and Doesn't Rule Out

**Does not rule out**: that task-content side-channel leakage is impossible on this hardware. This test used a small number of prompts (4), a small trial count (3 each), and deliberately simple statistical features chosen for consistency with papers 23-24's methodology, not because they represent the ceiling of what's extractable. Higher-resolution power sampling (beyond what this board's sensor update rate provides, per paper 23's own measurement), frequency-domain analysis rather than simple time-domain statistics, or correlating against known per-token timing patterns for specific vocabulary (a much more sophisticated attack than attempted here) might reveal signal this simple method cannot see. This paper tested the same easy, cheap method that worked cleanly for papers 23-24 and found it does not generalize to this harder question — that is a real finding about the limits of *this specific method*, not a proof about the limits of *this specific side-channel*.

**Does establish**: that response *length* is a real, easily-observed leak independent of content — an observer with only power-rail access to this board can reliably tell whether a given LLM request produced a short or long response, which is itself non-trivial information (e.g., distinguishing a refused/deflected short answer from a fully-engaged long one, or a simple lookup from a complex generation task) even without knowing what the content was. And it establishes a genuine, tested negative result for the harder question at this sensor resolution and this feature simplicity — worth documenting plainly rather than only publishing the positive results from papers 23-24 and leaving the natural next question unanswered.

## 5. Conclusion

Extending this project's power side-channel line of work from "what kind of workload" (paper 23) and "which model" (paper 24) to "which specific task or content," the simple method that worked perfectly for the first two questions hits a real ceiling on the third: response length leaks cleanly and by a wide margin, but content/task-type among similarly-long responses does not, at this board's sensor resolution and this feature set. This is reported as a genuine negative result, not a failure to find something that's there — the honest bound on what a cheap, simple power side-channel actually exposes about this board's LLM inference: *how much* was generated, not *what*.

## Evidence

Raw power traces for all 12 trials (4 prompts × 3 trials, sampled to natural completion), the 4 prompt input files, the collection harness (`prompt_sidechannel_collect.sh`), and the classification script (`prompt_sidechannel_classify.py`, run both with and without the duration feature), in `data/`, prefixed `orinnx-20260815-prompt-sidechannel-`.
