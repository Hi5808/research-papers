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

**Platform:** reComputer J401 carrier (Jetson Orin NX 16GB) | **Date:** August 2026

[Read full paper →](research_8_nvme_characterization.md) · Raw data: [SLC write profile](data/nvme-20260804-slc-write-profile.csv) · [full sampler run](data/nvme-20260804-sampler-full-run.csv) · [SMART pre/post](data/nvme-20260804-smart-pre.json)

---

## About

Papers 1-3 document research from production deployments on a Seeed Studio reComputer J3011 (Jetson Orin Nano 8GB on a J401 carrier), emphasizing practical systems-level challenges in edge AI deployment. Papers 4-5 document findings from building and evaluating a multi-model small-LLM fine-tuning pipeline, emphasizing failure modes that produce valid-looking but degraded artifacts, and the evaluation rigor required to trust A/B comparisons between fine-tuned models. Paper 6 documents hardware-level board bring-up, covering device-discovery methodology and firmware-flashing tool regressions under real-world host constraints. Paper 7 returns to the Jetson platform at the thermal and power-management layer, documenting a configuration-encoding trap and the measurement discipline required for trustworthy thermal benchmarks. Paper 8 extends that measurement discipline to storage, characterizing an unbranded NVMe SSD non-destructively on a live system and treating the instrument itself as something requiring validation.

## Citation

If you reference these findings, please cite the individual papers with their full titles and dates.
