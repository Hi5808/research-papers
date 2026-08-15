# CUDA MPS Provides No Measurable Throughput Benefit on Jetson Orin NX's Integrated GPU — and a Cold-Start Artifact Almost Said Otherwise

**Platform:** Seeed Studio reComputer J4012 — NVIDIA Jetson Orin NX 16GB (module P3767-300) on a J401 carrier board
**Software stack:** JetPack 7.2, L4T R39.2.0, CUDA 13.2, TensorRT-Edge-LLM v0.9.1 runtime
**Workload:** concurrent `llm_inference` processes against a pre-built Qwen3-1.7B-Instruct engine, MAXN_SUPER power mode
**Date:** August 2026

## Abstract

NVIDIA documents CUDA MPS (Multi-Process Service) as supported on Jetson since CUDA 12.5, but no public benchmark appears to confirm whether it actually delivers a throughput benefit on an Orin-class *integrated* GPU — MPS's traditional value proposition (better SM utilization when multiple processes time-share a discrete GPU's compute resources) may not transfer to an iGPU with a fundamentally different sharing/context model, and Jetson's `nvidia-smi` reports `Compute Mode: N/A` here, unlike the `EXCLUSIVE_PROCESS` mode discrete GPUs normally require for MPS. This paper measures concurrent-process wall-clock throughput for TensorRT-Edge-LLM inference with and without MPS enabled, at two concurrency levels (3 and 6 simultaneous processes). The first measurement appeared to show MPS was 18% faster — but this was a cold-start artifact: the very first run of the session (no-MPS) paid TensorRT engine deserialization and cache-warming costs that the immediately following MPS run did not. Once both conditions were measured from an equally warm state, the difference vanished entirely: 6.24s vs. 6.25s mean wall time at N=3 (stdev 0.12 vs 0.08), and 9.15s vs. 9.10s at N=6. **MPS provides no measurable throughput benefit for this workload on this hardware, at either concurrency level tested** — a clean negative result, and a cautionary tale about trusting a single comparison run without controlling for warm-up state.

## 1. MPS Setup on Orin's iGPU

`nvidia-cuda-mps-control` and `nvidia-cuda-mps-server` are present and start without error on this board — no Jetson-specific setup differences were needed beyond the standard `CUDA_MPS_PIPE_DIRECTORY`/`CUDA_MPS_LOG_DIRECTORY` environment variables. One notable divergence from typical discrete-GPU MPS guides: those normally require first setting `nvidia-smi -c EXCLUSIVE_PROCESS`, but this board's `nvidia-smi -q` reports `Compute Mode: N/A` — the compute-mode concept doesn't apply to this iGPU, and MPS was started and used successfully without ever setting it. Server engagement was verified directly from the control daemon's log: three concurrent client processes each logged `NEW CLIENT ... Server already exists`, confirming all three were correctly routed through a single shared MPS server process rather than three independent CUDA contexts.

## 2. The Cold-Start Artifact That Almost Produced a False Positive

The very first measurement taken this session — 3 concurrent processes, no MPS, executed before any other `llm_inference` invocation in the session — took **8.72s** wall time. The MPS condition, measured immediately after (with the same engine file, same prompt, same power mode), took **7.13s** — an apparent 18% improvement that would have been a plausible, publishable-looking result on its own. Two further no-MPS runs (now with the engine's disk-backed weights and TensorRT deserialization already warm in the page cache from the first run) measured 6.35s and 6.26s — both faster than the *first* MPS run, immediately signaling the initial comparison was confounded by warm-up state rather than reflecting a genuine MPS effect. This is reported here specifically as a methodology note: a naive single-run "before/after" comparison here would have shipped a wrong conclusion in the opposite direction from what a full, warm-state-controlled measurement shows.

## 3. Controlled Results: No Measurable Difference

With all runs taken from an equally warm state (each condition's first run discarded, or all runs taken with cache already warm from a prior run of either condition):

| Concurrency | Condition | Wall time samples (s) | Mean | Stdev |
|---|---|---|---|---|
| N=3 | No MPS | 6.35, 6.26, 6.11 | 6.24 | 0.12 |
| N=3 | MPS | 6.34, 6.21, 6.20 | 6.25 | 0.08 |
| N=6 | No MPS | 8.87, 9.43 | 9.15 | 0.40 |
| N=6 | MPS | 10.31, 7.88 | 9.10 | 1.72 |

At N=3, the two conditions are statistically indistinguishable (means within 0.01s of each other, well inside each condition's own run-to-run variance). At N=6, the means remain close (9.15 vs 9.10) but MPS shows notably *higher* variance (stdev 1.72 vs 0.40) — not enough samples here to say whether that's a real property of MPS under this workload or noise from only two runs per condition at the higher concurrency level; this specific variance question is flagged as open rather than concluded.

## 4. Why This Is Plausible, Not Surprising in Retrospect

MPS's classical benefit comes from letting multiple processes share a single GPU context and submit work concurrently rather than serializing through independent contexts with time-sliced scheduling overhead between them — valuable on datacenter GPUs with many SMs where several small kernels can genuinely co-reside and execute in parallel across different SM partitions. Orin's iGPU (per this project's own companion tokens-per-joule paper) has vastly fewer resources — 4 active TPCs at MAXN_SUPER, shared with CPU cores over a unified memory bus, no true multi-tenant SM-partitioning hardware, and TensorRT-Edge-LLM's `llm_inference` workload here is itself already close to fully utilizing available compute per the companion soctherm_oc-throttling paper's prefill findings. Under those conditions there is little idle SM capacity left for MPS's concurrency mechanism to actually exploit — the processes are likely already effectively serialized by real compute contention regardless of context-sharing overhead, which would explain why removing that overhead (MPS's actual mechanism) produces no measurable change.

## 5. Conclusion

CUDA MPS starts and operates correctly on this board's Orin NX iGPU under JetPack 7.2/CUDA 13.2, with one real setup divergence from discrete-GPU guides (no `EXCLUSIVE_PROCESS` compute mode available or needed). But it provides no measurable concurrent-throughput benefit for TensorRT-Edge-LLM inference at either 3 or 6 simultaneous processes — a negative result, plausible given the iGPU's limited spare compute capacity relative to a datacenter GPU's typical MPS use case, and one this paper very nearly reported the opposite of due to an uncontrolled cold-start comparison in its first measurement. Anyone benchmarking MPS (or any GPU-sharing mechanism) on Jetson hardware should explicitly discard or separately account for the first invocation in any session, not just here — TensorRT engine deserialization and page-cache warming costs are large enough on this hardware to fully explain an 18% "effect" that isn't real.

## Evidence

`mps_concurrency_test.sh` (the test harness), all raw per-process timing files, and MPS control daemon logs confirming server-sharing, in `data/`, prefixed `orinnx-20260815-mps-igpu-`.
