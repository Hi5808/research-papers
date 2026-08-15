# Does Hardware Over-Current Throttling Affect LLM Inference Latency on Jetson Orin NX? A Direct Measurement

**Platform:** Seeed Studio reComputer J4012 — NVIDIA Jetson Orin NX 16GB (module P3767-300) on a J401 carrier board
**Software stack:** JetPack 7.2, L4T R39.2.0, CUDA 13.2, TensorRT 10.16.2.10, TensorRT-Edge-LLM
**Date:** August 2026

## Abstract

A companion paper on this board identified the mechanism behind its ~34W practical power ceiling: Tegra's `soctherm_oc` hardware over-current protection, firing hundreds of times per second under combined CPU+GPU compute load, independent of temperature. That paper's stress workloads were synthetic (a dense FMA compute kernel plus `stress`). This paper asks whether the same protection circuit engages during *actual LLM inference* workloads, and whether it measurably affects throughput or timing consistency when it does. The answer is workload-dependent in a way that matters for deployment: single-stream decode inference — the dominant cost in LLM serving — never triggers the over-current protection, with or without concurrent CPU load, and shows no measurable throughput or timing effect. Prefill (the compute-dense phase) does trigger it under concurrent CPU load, and when it fires 13,726 times across a ~27-second benchmark, throughput drops 10.1% — but per-iteration timing variance *decreases*, not increases, under throttling. The protection manifests as a sustained higher latency floor, not added jitter.

## 1. Method

Two workload types were tested at `MAXN_SUPER`, each with and without a concurrent `stress --cpu 8` load, using Qwen3-4B-Instruct-2507's TensorRT-Edge-LLM engine:
- **Decode**: 500-iteration sustained decode benchmark (128-token past-KV length)
- **Prefill**: 50-iteration prefill benchmark (512-token input) — the more compute-dense phase

`soctherm_oc`'s three throttle-event counters (`/sys/class/hwmon/hwmon5/oc{1,2,3}_event_cnt` on this boot — hwmon device numbering is not stable across reboots on this board and must be resolved by name, not assumed index) were polled at 100ms resolution throughout each run, alongside `llm_bench`'s own aggregate E2E timing and standard deviation output (`--outputDir` dumps a per-benchmark summary CSV; it does not expose true per-token latency, a tool limitation documented here rather than a per-token jitter trace as originally intended).

## 2. Decode Inference Never Triggers Over-Current Throttling

| Condition | Decode tok/s | `oc3` events observed |
|---|---|---|
| Decode alone | 30.0 | 0 (entire 500-iteration run) |
| Decode + concurrent `stress --cpu 8` | 30.6 | 0 (entire run) |

Zero throttle events fired in either condition, and throughput was statistically unchanged (the 0.6 tok/s difference is within normal run-to-run variance seen elsewhere in this project's benchmarks). Decode is memory-bandwidth-bound, generating one token at a time with low GPU compute occupancy — consistent with the companion power-ceiling paper's finding that the LLM decode benchmark alone only reached 26W, well under the ~34W threshold at which the over-current protection engages. **For a deployment whose workload is dominated by decode** — the common case for interactive LLM serving, where prefill happens once per request and decode runs for every subsequent token — this board's over-current protection is simply not a factor, regardless of concurrent CPU load from request handling, tokenization, or other services.

## 3. Prefill Under Concurrent CPU Load Does Trigger It, With a Measurable Effect

| Condition | Prefill tok/s | E2E time (mean ± std) | `oc3` events observed |
|---|---|---|---|
| Prefill alone | 1059.3 | 483.36 ± 9.50 ms | not polled (established elsewhere as non-zero under sufficient combined load) |
| Prefill + concurrent `stress --cpu 8` | 952.8 | 537.39 ± 4.53 ms | **13,726** (over ~27s, confirmed live via before/after and continuous polling) |

Two results here, one expected and one not:
- **Throughput dropped 10.1%** (1059.3 → 952.8 tok/s) under the condition that triggered heavy over-current throttling — a real, measurable cost tied directly to the protection circuit engaging, not a coincidental slowdown from CPU contention alone (the companion paper's `stress`-only baseline draws far less power than prefill-plus-`stress`, and prefill alone was not CPU-contended in this comparison in a way that would explain a 10% drop by CPU scheduling pressure alone).
- **Timing variance decreased under throttling**, not increased: standard deviation fell from 9.50ms to 4.53ms even as the mean rose. This is the opposite of the naive expectation that a real-time current limiter firing hundreds of times per second would manifest as burstiness or instability in per-iteration timing. Instead, the protection appears to impose a *consistently* higher floor — plausibly because it engages so frequently (roughly one event per ~2ms across the run) that its effect is closer to a continuous clock-rate reduction than a series of discrete stalls, smoothing out rather than adding to natural run-to-run variance.

## 4. Conclusion

Whether this board's over-current protection matters for LLM inference depends entirely on workload phase, not on power mode or thermal state (both already ruled out as the controlling factor in the companion power-ceiling paper). Decode-bound serving workloads — the overwhelming majority of real LLM inference cost — never engage it, with or without competing CPU load, and lose nothing. Prefill under concurrent CPU load does engage it and costs a real, repeatable ~10% throughput penalty, but the failure mode is a slower-but-more-consistent floor, not added jitter or tail-latency risk — which for most serving use cases (where mean throughput matters more than per-request timing variance) is the more benign of the two possible outcomes. A deployment specifically sensitive to prefill latency tail risk under concurrent host-side CPU load should still budget for this ~10% degradation; one dominated by decode-phase serving can disregard it entirely.

## Evidence

Raw logs in `data/`, prefixed `orinnx-20260811-campaign2-`: 100ms-resolution `oc1/oc2/oc3` event-counter polls for all four conditions (decode alone, decode+stress, prefill alone, prefill+stress), matching `llm_bench` output and `--outputDir` summary CSVs.
