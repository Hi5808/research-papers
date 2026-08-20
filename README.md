# Research Papers

Technical research and findings from production ML systems work.

## Contact Me

Have questions about these findings or interested in consulting? Get in touch:

📧 **Email:** [ahkt808@proton.me](mailto:ahkt808@proton.me)

---

## Edge AI Deployment on Constrained Hardware

Production deployments of vision-language models and real-time detection on a Seeed Studio reComputer J3011 (NVIDIA Jetson Orin Nano 8GB, J401 carrier) — production edge hardware rather than an NVIDIA developer kit.

### 1. Vision-Language Models on Edge Hardware: JetPack 7.2 Deployment Patterns

Deployment of Qwen2-VL-2B on Jetson Orin Nano with unified-memory OOM patterns, CUDA 13.2 SBSA unification enabling upstream wheels, and pipeline architecture for near-real-time captioning on 8GB devices.

**Key findings:**
- CUDA 13.2 SBSA unification reverses historical Jetson-ecosystem lag
- Unified-memory OOM pattern and fix (device_map='cuda:0', low_cpu_mem_usage=True)
- Native transformers models superior to custom-code models for stability

**Platform:** reComputer J3011 (Jetson Orin Nano 8GB) | **Date:** July 2026

[Read full paper →](research_1_vlm_edge_deployment.md)

---

### 2. TensorRT Inference Optimization on Jetson Orin: The 1.7x Speedup Pattern

YOLOv8n object detection optimization achieving 1.7x speedup over PyTorch (30ms → 17ms) through kernel autotuning and layer fusion on Compute Capability 8.7.

**Key findings:**
- PyTorch baseline already real-time capable (33 FPS)
- TensorRT kernel autotuning is per-GPU, not portable across compute capabilities
- Speedup valuable for headroom and multi-model concurrency, not hard real-time requirements
- Capture/encode becomes bottleneck once inference reaches 17ms

**Platform:** reComputer J3011 (Jetson Orin Nano 8GB) | **Date:** July 2026

[Read full paper →](research_2_tensorrt_optimization.md)

---

### 3. Unified-Memory Multi-Model Concurrency: The Memory Accounting Problem

Running VLM + TensorRT detector simultaneously on 8GB unified memory: process consolidation, quantization kernel compatibility, and the gap between model size and actual process memory under concurrent GPU workloads.

**Key findings:**
- Per-process CUDA context overhead (~1-1.5GB) dominates model size in multi-model scenarios
- int8 vs int4 quantization hit different hardware-compatibility paths (int8 failed on CC 8.7, int4 NF4 succeeded)
- Baseline system load (GUI, apps) creates invisible floor in development environments
- Achieved 29 FPS detection + 5-7s captioning concurrently with 116MB memory margin

**Platform:** reComputer J3011 (Jetson Orin Nano 8GB) | **Date:** July 2026

[Read full paper →](research_3_unified_memory_multimodel.md)

---

## Small Model Fine-Tuning and Evaluation

Building and evaluating multi-model fine-tuning pipelines: failure modes in distillation, and rigorous methodology for honest model comparison.

### 4. Multi-Teacher Synthetic Data Distillation: Failure Modes in Small Model Fine-Tuning

Seven concrete bugs found while building a 7-model IT-operations diagnostic suite via multi-teacher LLM distillation, three of which caused a degraded model to ship with no error signal anywhere in the pipeline.

**Key findings:**
- Template-fill failures are silent, not loud — missing fill values write literal placeholder text into training data undetected
- Word-prefix deduplication hashing can collapse 84% of a short-output dataset as false duplicates
- Teacher model *fit* matters more than teacher model *scale* — a general chat model produced fabricated WMI logic for code-generation tasks that passed superficial review
- Unsorted checkpoint globbing can silently export an undertrained model with a "success" exit code at every pipeline stage

**Platform:** NVIDIA RTX 5090 32GB, AMD Ryzen 7 9700X | **Date:** July 2026

[Read full paper →](research_4_multi_teacher_distillation_pitfalls.md)

---

### 5. Does Domain-Specialist Fine-Tuning Beat a Generalist? Four Rounds of Methodology Correction to Find Out

Testing whether narrow domain fine-tuning outperforms a generalist model of the same size (1.5B) across four IT-operations domains. The final answer is a clean **yes across all four domains** — but it took a broken exact-keyword test, a corrected-but-shallow judge-based test finding no effect, a deeper judge-based test revealing a domain-dependent effect, and a proportionally-dosed deep-training-data intervention (which first had to fail once, on linux, before the dosing bug was caught) to actually detect it.

**Key findings:**
- Exact-keyword test matching produces false negatives on genuinely correct answers using valid alternative implementations
- 3-question smoke tests produce non-reproducible pass/fail results from LLM sampling variance alone, even at low temperature
- LLM-as-judge scoring with a single-dimension rubric + Wilson confidence intervals recovers a statistically honest comparison
- Test *depth*, not just sample size, determines whether an evaluation can detect a real effect — a shallow test set found nothing; adding multi-hop scenarios revealed a real, domain-dependent effect the shallow test was structurally incapable of surfacing
- Training data interventions must be dosed *proportionally* to each dataset's size, not as a fixed absolute count — the same fixed addition that reversed networking's result (50%→85% pass rate) initially made linux *worse*, purely because linux's larger pre-existing dataset diluted the same absolute addition
- Final result: all four domains show a clear specialist advantage (55-85% pass rate vs. 30-70% for the generalist) once evaluation and training-data dosing were both corrected with matching rigor

**Platform:** NVIDIA RTX 5090 32GB, AMD Ryzen 7 9700X | **Date:** July 2026

[Read full paper →](research_5_specialist_vs_generalist_evaluation.md)

---

## Board Bring-Up and Recovery

Hardware-level bring-up and recovery work: flashing tool regressions, device-discovery methodology, and OS provisioning under real-world constraints.

### 6. Recovering a Rockchip RK3588 Board from Maskrom Under a Non-Elevated Windows Session: An Empirical Debugging Study

Recovering a Radxa ROCK 5B+ from an unrecognized maskrom-mode USB state on a non-elevated Windows host — device discovery, non-elevated toolchain assembly, OS image selection, and a flashing-tool regression root-caused by diffing two closed-source binaries.

**Key findings:**
- A device-discovery query filtered on class and status structurally excluded the target device, which by definition has neither — a selection-effect bug, not a detection failure
- RKDevTool v2.96's `err=995` write failure (a widely-reported, unresolved community bug) was root-caused via binary string-diffing against a working v2.86, with no source access or debugger, and corroborated against the tool's own logs
- A USB-level mode-switch (maskrom → loader) does not survive `usbip` forwarding: the transition re-enumerates the device, which invalidates `usbipd`'s share state and hangs the transfer indefinitely
- Vendor-official was the stalest OS option by ~2 years; freshness was checkable from release metadata in seconds, but the freshest OS then landed on the wrong side of a Python ABI break against the target SDK's wheels — freshness must be checked for mutual stack compatibility, not maximized per component
- NPU/GPU/codec driver inclusion was confirmed directly from the shipped image's kernel config (no mount, no elevation) rather than from contradictory forum reports

**Platform:** Radxa ROCK 5B+ (Rockchip RK3588) | **Date:** July 2026

[Read full paper →](research_6_rk3588_maskrom_recovery.md)

---

### 7. Fan Curve Tuning on Jetson Orin: The Thermal-Margin Encoding Trap

An undocumented encoding in NVIDIA's `nvfancontrol` configuration causes fan-curve edits to be applied inverted, plus measured thermal results from a retuned curve under combined CPU+GPU load and a benchmark-methodology failure mode in `nvpmodel`.

**Key findings:**
- With `TMARGIN ENABLED`, the fan curve's temperature column is *margin below the 105 °C limit*, not degrees Celsius — an operator targeting 60 °C who writes `60` actually sets 45 °C, inverting their model of every subsequent edit
- The encoding is verifiable on a live system three independent ways (kernel trip-point complements, reductio on the Celsius reading, and live RPM interpolation), though the interpolation test is degenerate at exactly half the limit temperature
- The stock `quiet` profile holds the fan fully off until Tj 35 °C and below 4000 RPM until 94 °C — acoustically sensible, but it introduces unmeasured thermal variance into benchmarks
- A retuned curve held Tj to 57 °C under sustained combined CPU+GPU load at 25 W, but reached exactly 60 °C at MAXN_SUPER with the fan saturated at its 6000 RPM ceiling — the target is met at the boundary, not within it
- At full clocks the equilibrium is set by the fan's maximum rather than the curve's shape, so no further profile tuning can improve that operating point; the remaining levers are power cap or ambient
- `nvpmodel -m 2` returned success, logged as applied, and silently reverted — it requires a reboot to take effect. Only the achieved clocks revealed the first run had executed at the previous power cap; Jetson benchmarks must gate collection on achieved clocks, not requested mode

**Platform:** reComputer J3011 (Jetson Orin Nano 8GB) | **Date:** August 2026

[Read full paper →](research_7_jetson_fan_curve_thermal.md) · Raw data: [25 W](data/thermal-20260803-25W-combined-load.csv) · [MAXN_SUPER](data/thermal-20260803-MAXN_SUPER-combined-load.csv)

---

### 8. Characterizing an Unbranded NVMe SSD: The Dead Sensor and the Span-Dependent Cache

A full non-destructive characterization of a 1 TB SSD of unknown provenance, performed in place as the live boot device of a Jetson Orin NX — covering what the firmware reports, what it gets wrong, and what only measurement reveals.

**Key findings:**
- The drive is untraceable — generic model string, placeholder serial, and a PCI subsystem ID identical to the device ID because no OEM ever programmed one — but the controller is positively identified as a DRAM-less Silicon Motion SM2263XT via two independent sources, with non-zero `HMPRE` settling the EN/XT ambiguity that pci.ids cannot
- The drive's temperature sensor is **frozen**: one unchanging value across 9,191 samples and 2.8 hours of sustained load, while the SoC sensor logged into the same CSV moved through 191 distinct values. Its zeroed thermal counters are therefore not evidence of good thermal behaviour — they would read zero on a burning drive
- Sustained sequential write holds 573 MB/s for exactly 107.8 GiB and then falls 4.01x to 143 MB/s as the SLC cache exhausts, at 39 °C — cache exhaustion, not throttling
- 4K random read is span-dependent as a 64 MiB host-memory-buffer implies, but only mildly: a 300 GiB working set is ~20% slower than 64 GiB
- ext4 costs 3.2% on sequential reads but **2.5x on 4K random reads** — the raw device delivered 269k IOPS against 109k through the filesystem, over a *larger* span, so span cannot explain it
- The 128 KiB maximum I/O size is the *host's* limit, not the drive's: Identify reports `MDTS=6` (256 KiB), inverting the usual assumption that such caps are a drive property
- The sampling harness built for this campaign was found to overstate throughput by 32% by dividing counter deltas by a nominal interval it never actually achieved — caught only because fio measured the same quantity independently, and revealed by peak samples that exceeded the drive's own PCIe link

**Platform:** reComputer J4012 (Jetson Orin NX 16GB on a J401 carrier) | **Date:** August 2026

[Read full paper →](research_8_nvme_characterization.md) · Raw data: [SLC write profile](data/nvme-20260804-slc-write-profile.csv) · [full sampler run](data/nvme-20260804-sampler-full-run.csv) · [SMART pre/post](data/nvme-20260804-smart-pre.json)

---

### 9. Desktop Strip and Performance Tuning on Jetson Orin NX 16GB: Reproducing the Orin Nano Result on a Different Module

Reproduces the prior Orin Nano package-strip and clock/power/fan tuning study on a reComputer J4012 (Orin NX 16GB) unit on the same J401 carrier family, testing whether the methodology transfers within the same module family rather than being an artifact of one board.

**Key findings:**
- The 135-package removal batch, the ported `max65` fan-curve profile, and both systemd drop-in fixes (`nvidia-cdi-refresh.service`, `nvpmodel.service`) all reproduced without modification
- `nvpmodel` mode indices are **not portable across modules in the same family**: `MAXN_SUPER` is mode 2 on the Orin Nano SKU but mode 0 on this Orin NX SKU — applying the Nano's mode number verbatim would have silently selected the wrong power profile
- Qwen3-1.7B-Instruct prefill throughput came out 35% higher (2732.9 vs 2025 tok/s), consistent with the NX's higher clocks and larger core count; decode stayed flat (63.5 vs 62.2 tok/s), consistent with a memory-bandwidth-bound phase
- **Correction caught by a follow-up stress test:** the sustained decode run's 69.6 °C peak did not actually exercise the `max65` fan curve — `nvfancontrol.service` doesn't hot-reload, and had not been restarted since the profile was written. After restarting it, live PWM polling confirmed the curve engages exactly as designed (0→255 at the 65 °C crossing)
- That same follow-up found a naive CPU+GPU compute stress test undershoots the module's real power ceiling (17.7W vs the LLM benchmark's 26W) — SM occupancy alone doesn't reach the module's power envelope, reproducing a finding from the companion Nano fan-curve paper
- **Eight workload/ordering combinations were tried to reach the 40W `MAXN_SUPER` nameplate figure — none did.** Best result was 34.3W using a staged GPU-first, CPU-cores-one-at-a-time technique matching NVIDIA staff's own documented guidance on a near-identical developer forum report. An external-fan test ruled out thermal throttling (power stayed flat at 34.07W despite running 11°C cooler) — and reading the board's hardware current-monitoring registers directly identified the real mechanism: Tegra's `soctherm_oc` over-current protection fired 7,158 times in a 35-second stress test (confirmed via a live before/after counter read), tied to an INA3221-configured current threshold on the input rail that works out to almost exactly 40W. The limit isn't user-adjustable — it lives in NVIDIA's closed BPMP firmware, not the device tree

**Platform:** reComputer J4012 (Jetson Orin NX 16GB, J401 carrier) | **Date:** August 2026

[Read full paper →](research_9_orin_nx_strip_perf_tuning.md) · Raw data: [strip removal log](data/orinnx-20260810-strip-removal.log) · [sustained decode benchmark](data/orinnx-20260810-decode500-bench.log) · [tegrastats capture](data/orinnx-20260810-tegrastats-decode500.csv)

---

### 10. Four Benchmark Campaigns on Jetson Orin NX 16GB: What Nobody Had Published Yet

An overview tying together four targeted benchmark campaigns on the same Orin NX unit, chosen specifically to fill gaps a literature check confirmed were open rather than re-measuring numbers that already exist elsewhere.

**Key findings:**
- Three campaigns produced substantive new findings (tokens-per-joule, OC-throttle/latency correlation, TensorRT vs. llama.cpp); the fourth (DLA) produced a clean, worth-knowing negative result
- Two reliability traps recur across all four: `nvpmodel -m` into 10W/15W silently fails without an interactive reboot confirmation over SSH, and this board intermittently loses its GPU `devfreq` governor node across reboots
- A background build does not survive a `nvpmodel`-triggered reboot even with `nohup`/`disown` — sequence reboot-requiring work before long unattended builds, not concurrently

**Platform:** reComputer J4012 (Jetson Orin NX 16GB, J401 carrier) | **Date:** August 2026

[Read full paper →](research_10_orin_nx_benchmark_campaign_overview.md)

---

### 11. Tokens-per-Joule Across All Five nvpmodel Power Modes on Jetson Orin NX 16GB

Measures decode throughput and energy-per-token for Qwen3-1.7B/4B across 10W, 15W, 25W, 40W, and MAXN_SUPER, finding the opposite efficiency curve from the published Orin Nano result.

**Key findings:**
- Efficiency *degrades* from 10W through 25W and only improves at 40W — the actual Pareto-optimal mode on this hardware — not a mid-tier mode like the Nano's published 25W
- Mechanism: the per-mode GPU clock/TPC table is non-monotonic (25W runs 4 TPCs at a lower clock than 10W/15W's 2 TPCs; full GPU throughput only unlocks at 40W)
- Two of five `nvpmodel -m` mode-switch attempts silently failed and left the board at its prior mode with no error, requiring a reboot-and-reverify procedure to get clean data

**Platform:** reComputer J4012 (Jetson Orin NX 16GB, J401 carrier) | **Date:** August 2026

[Read full paper →](research_11_orin_nx_tokens_per_joule.md)

---

### 12. Does Hardware Over-Current Throttling Affect LLM Inference Latency on Jetson Orin NX?

Correlates this board's `soctherm_oc` over-current protection (previously found to cause its ~34W power ceiling) with actual LLM inference behavior, rather than only synthetic stress workloads.

**Key findings:**
- Decode inference never triggers the protection, with or without concurrent CPU load — decode-dominated serving workloads can disregard it entirely
- Prefill under concurrent CPU load does trigger it (13,726 events in ~27s) and costs a real 10.1% throughput drop
- Counter-intuitively, per-iteration timing variance *decreased* under throttling (9.50ms→4.53ms stddev) — a sustained latency floor, not added jitter

**Platform:** reComputer J4012 (Jetson Orin NX 16GB, J401 carrier) | **Date:** August 2026

[Read full paper →](research_12_orin_nx_oc_throttle_latency.md)

---

### 13. TensorRT-Edge-LLM vs. llama.cpp on the Same Jetson Orin NX 16GB

Builds both inference stacks from source on the same board and benchmarks the same Qwen3 models at matched settings — a comparison rarely done with both variables controlled simultaneously.

**Key findings:**
- TensorRT-Edge-LLM wins decisively: 57-65% faster prefill, 28-50% faster decode, depending on model size
- Power draw is nearly identical between stacks (within 1W), so the efficiency gap tracks the throughput gap almost exactly — not a speed-for-power tradeoff
- A real, disclosed confound: llama.cpp can't ingest NVIDIA's NVFP4-family quantized checkpoints directly, so the two stacks compare different quantization schemes, not just different engines

**Platform:** reComputer J4012 (Jetson Orin NX 16GB, J401 carrier) | **Date:** August 2026

[Read full paper →](research_13_orin_nx_trtllm_vs_llamacpp.md)

---

### 14. DLA Hardware Initializes but Is Unusable Through TensorRT on This Jetson Orin NX Software Stack

Attempted to build a DLA-targeted TensorRT engine for a concurrent DLA+GPU utilization study; documents why that study could not be run.

**Key findings:**
- DLA cores initialize cleanly at the kernel level (both online, clocked, no driver errors) but TensorRT reports zero available DLA cores on this JetPack 7.2 / TensorRT 10.16.2.10 build
- No DLA compiler package exists anywhere in this JetPack release's apt repository to fix it — the required component appears entirely absent, not merely misconfigured
- Reported as a negative result: most Jetson benchmarking content assumes DLA "just works" once the cores are present; here it demonstrably doesn't

**Platform:** reComputer J4012 (Jetson Orin NX 16GB, J401 carrier) | **Date:** August 2026

[Read full paper →](research_14_orin_nx_dla_unavailable.md)

---

### 30. Bringing audio.cpp to NVIDIA Jetson Orin: A Full-Catalog CUDA Bring-Up

A maintainer asked r/JetsonNano for help getting their C++/GGML audio inference runtime (audio.cpp — TTS, ASR, voice conversion, music generation) running on Jetson Orin, with no Orin hardware of their own to test on. This paper documents the full bring-up: laptop-side prep on a discrete RTX 5060 (catching a real CMake architecture-precedence bug before it ever reached real hardware), then a complete 40-model-family benchmark run natively on both an Orin Nano 8GB and an Orin NX 16GB.

**Key findings:**
- Orin NX: 40/40 model families ran clean, zero failures — full parity with the discrete-GPU baseline
- Orin Nano: 34/40 clean after root-causing all 9 initial failures down to real stderr — 5 are a straightforward 8GB capacity ceiling, 2 were fixed with a real, previously-unsurfaced ggml build flag (`-DGGML_CUDA_NO_VMM=ON`, closing a 32GB virtual-address-reservation failure last documented as a Jetson issue back in 2023, confirmed to cost a real ~2% throughput regression on the 16GB NX where it isn't needed), 1 was fixed by a reboot (boot-persistent Tegra NVMAP allocator fragmentation), and 1 remains genuinely unfixable (a single 8GB allocation request that exceeds the board's entire memory pool outright)
- No garbled or corrupted output occurred anywhere, on either board, at any point — directly resolving the SM 8.7 CUDA-correctness risk flagged by prior community reports before ever touching real hardware
- Findings and the working fix were reported upstream: [github.com/0xShug0/audio.cpp/issues/12](https://github.com/0xShug0/audio.cpp/issues/12#issuecomment-5359377035)

**Platform:** reComputer J3011 (Jetson Orin Nano 8GB) + reComputer J4012 (Jetson Orin NX 16GB), JetPack 7.2 | **Date:** August 2026

[Read full paper →](audiocpp-jetson-orin/research_30_audiocpp_jetson_orin_bringup.md) · Raw data: [RTX 5060 benchmark](audiocpp-jetson-orin/benchmark_results_rtx5060.csv) · [Orin Nano benchmark](audiocpp-jetson-orin/benchmark_results_orin.csv) · [Orin NX benchmark](audiocpp-jetson-orin/benchmark_results_orinnx.csv) · [family size manifest](audiocpp-jetson-orin/family_manifest.csv) · Scripts: [bring-up](audiocpp-jetson-orin/bringup.sh) · [telemetry wrapper](audiocpp-jetson-orin/run_with_telemetry.sh) · [Jetson benchmark runner](audiocpp-jetson-orin/jetson_bench_runner.py) · [laptop benchmark runner](audiocpp-jetson-orin/laptop_benchmark_runner.py)

---

## About

Papers 1-3 document research from production deployments on a Seeed Studio reComputer J3011 (Jetson Orin Nano 8GB on a J401 carrier), emphasizing practical systems-level challenges in edge AI deployment. Papers 4-5 document findings from building and evaluating a multi-model small-LLM fine-tuning pipeline, emphasizing failure modes that produce valid-looking but degraded artifacts, and the evaluation rigor required to trust A/B comparisons between fine-tuned models. Paper 6 documents hardware-level board bring-up, covering device-discovery methodology and firmware-flashing tool regressions under real-world host constraints. Paper 7 returns to the Jetson platform at the thermal and power-management layer, documenting a configuration-encoding trap and the measurement discipline required for trustworthy thermal benchmarks. Paper 8 extends that measurement discipline to storage, characterizing an unbranded NVMe SSD non-destructively on a live system and treating the instrument itself as something requiring validation. Paper 9 returns to the strip-and-tune methodology of the early Nano papers, testing it on an Orin NX 16GB to separate what generalizes across a module family from what is board-specific and must be re-derived. Papers 10-14 are a follow-on benchmark campaign on that same Orin NX unit, deliberately targeting gaps a literature check confirmed were open — power-mode efficiency, hardware throttling under real inference workloads, a matched-model inference-stack comparison, and DLA availability — with paper 10 as the overview and 11-14 as the individual campaigns, including one reported negative result.

## Citation

If you reference these findings, please cite the individual papers with their full titles and dates.
