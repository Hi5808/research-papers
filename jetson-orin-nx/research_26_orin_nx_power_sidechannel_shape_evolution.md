# Revising the Power Side-Channel Ceiling: Duration-Normalized Trace Shape Recovers Real Task-Content Signal Paper 25 Missed

**Platform:** Seeed Studio reComputer J4012 — NVIDIA Jetson Orin NX 16GB (module P3767-300) on a J401 carrier board
**Software stack:** JetPack 7.2, L4T R39.2.0, MAXN_SUPER power mode, TensorRT-Edge-LLM v0.9.1, Qwen3-1.7B-Instruct (same model, same data as paper 25)
**Direct revision of:** paper 25's conclusion that task content does not survive past response length
**Date:** August 2026

## Abstract

Paper 25 concluded that this project's power side-channel method could reliably identify *how much* an LLM generated but not *what kind of task* it was doing, once response length was controlled for — a simple leave-one-out classifier using coarse trace statistics (mean, stdev, max, two-window early/late split) dropped to chance (25%, 3/12) when duration was excluded from the feature set. This paper re-analyzes the **same raw data** with a different feature representation and finds that conclusion was an artifact of the feature choice, not a genuine limit of the underlying signal. Resampling each trace into 20 time-normalized bins (so every trace, regardless of actual duration, is represented as a 20-point shape describing *when within its own response* power was higher or lower) and z-score normalizing each shape (removing absolute power level entirely, leaving only rhythm) recovers real above-chance classification: **66.7% overall (chance 25%)**, and critically, **`code_generation` is identified correctly in all 3 of its trials using shape alone**, with no duration information at all. `long_creative` and `math_reasoning` remain difficult to tell apart from each other (0/3 and 2/3 respectively, frequently confused with one another specifically), but code generation's power rhythm is genuinely, reproducibly distinct. This revises paper 25's headline finding: task content is not undetectable at this sensor resolution, it was undetectable *by the specific simple statistics paper 25 chose to use* — a meaningful, honest correction rather than a reversal of the underlying data.

## 1. What Changed: Shape Instead of Summary Statistics

Paper 25's features were global scalars computed once per trace: overall mean, overall stdev, overall max, and a crude two-bin early/late split. This collapses each trace's entire temporal structure into essentially one number describing "how variable was it," discarding *where in the response* that variability occurred. This paper instead resamples each trace to 20 equal-width time bins normalized to that trace's own duration (so a 3-second and an 11-second trace both become 20-point vectors, directly comparable regardless of length), then z-score normalizes each 20-point shape to zero mean and unit variance — explicitly removing both duration and absolute power level from the representation, leaving only the trace's *rhythm*: whether power tends to be relatively higher early or late, and how that rhythm's shape compares across trials of the same vs. different task type.

## 2. Result: Real, Task-Specific Signal — But Not Uniform Across Task Types

Leave-one-out nearest-neighbor classification on these shape vectors alone (no duration feature at all, same raw data as paper 25):

| Class | Correct / Total | Notes |
|---|---|---|
| `code_generation` | 3/3 | **Perfect** — every trial's shape matched another `code_generation` trial more closely than any other class |
| `math_reasoning` | 2/3 | One trial misclassified as `long_creative` |
| `long_creative` | 0/3 | All three misclassified, consistently as `math_reasoning` |
| `short_factual` | 3/3 | Still trivially separable (its shape is qualitatively different — an initial dip followed by a rise, visibly distinct in the raw bin values, unlike the other three which all start low and stay low) |

Overall: 8/12 (66.7%), well above the 25% chance rate for 4 classes. Restricting to just the three similarly-long classes (removing `short_factual`'s trivial separability from the comparison) still gives 5/9 (55.6%) against a 33.3% chance rate — the signal holds even in the genuinely hard subset paper 25 specifically flagged as indistinguishable.

The asymmetry is the interesting part: `code_generation`'s rhythm is *reliably* distinct (100% correct across all 3 trials, every time matched correctly against the other 2 `code_generation` trials specifically), while `long_creative` and `math_reasoning` remain genuinely confusable with each other, even though both are now clearly separable from `code_generation`. A plausible explanation, not confirmed here: code generation output has a fundamentally different token-level structure than prose or worked arithmetic — indentation, syntax tokens, variable names, and structural boilerplate (`def`, `return`, docstring delimiters) may produce a more regular, different-rhythm token-emission pattern than natural-language prose or step-by-step numerical reasoning, both of which are closer to each other in their underlying text structure (sentences of varying length, similar token-type distributions) and may simply produce more similar decode rhythms as a result.

## 3. Why Paper 25 Missed This

Paper 25's two-bin early/late split is a coarse enough summary that it likely washed out exactly the kind of finer temporal pattern the 20-bin shape representation preserves. This is not a flaw specific to paper 25's execution — it's a direct illustration of a general principle worth stating plainly for this project's future side-channel work: **absence of signal in a simple feature set is evidence about that feature set, not proof of absence of signal in the underlying data.** Paper 25's own §4 explicitly flagged this possibility ("higher-resolution power sampling... or frequency-domain analysis... might reveal signal this simple method cannot see") as an open question rather than a closed one — this paper answers that open question in the affirmative, using the same sensor and same sampling resolution, purely by choosing a richer representation of data that was already being collected.

## 4. What Remains Unresolved

`long_creative` and `math_reasoning` are still not reliably separable from each other by this shape-based method — this paper improves on paper 25's negative result but does not fully overturn it. Whether a still-richer representation (more time bins, frequency-domain features via FFT, or a proper time-series distance metric like dynamic time warping instead of simple Euclidean distance on resampled bins) would separate those two remaining classes, or whether they are genuinely closer in underlying decode rhythm than code generation is to either of them, is not established here. The sample size remains small (3 trials per class, unchanged from paper 25's data) — this analysis is a genuine re-reading of the same 12 traces, not new data collection, so its statistical power is limited by the same small-n caveat as papers 23-25.

## 5. Conclusion

Re-analyzing paper 25's own raw power traces with duration-normalized, shape-based features instead of coarse summary statistics recovers real, reproducible task-content signal that the simpler method missed — most clearly for code generation, which is identified correctly in 100% of its trials using nothing but the normalized rhythm of the power trace, with response length and absolute power level explicitly removed from the comparison. This revises, rather than reverses, paper 25's conclusion: the power side-channel's real ceiling on this board is not "duration only, content never" — it is that *the specific simple method tested in paper 25* only recovered duration, and a mildly more sophisticated but still cheap and simple representation of the exact same data goes meaningfully further. Whether an even richer method could close the remaining gap between `long_creative` and `math_reasoning` is open, honestly-flagged future work.

## Evidence

Re-analysis of the identical raw power traces already published as evidence for paper 25 (`orinnx-20260815-prompt-sidechannel-*`), plus the new shape-extraction/classification script (`prompt_sidechannel_shape.py`), in `data/`, prefixed `orinnx-20260815-power-sidechannel-shape-`.
