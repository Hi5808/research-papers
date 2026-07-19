# Research Papers

Technical research and findings from production ML systems work.

## Contact Me

Have questions about these findings or interested in consulting? Get in touch:

📧 **Email:** [ahkt808@proton.me](mailto:ahkt808@proton.me)

---

## Edge AI Deployment on Constrained Hardware

Production deployments of vision-language models and real-time detection on NVIDIA Jetson Orin Nano 8GB.

### 1. Vision-Language Models on Edge Hardware: JetPack 7.2 Deployment Patterns

Deployment of Qwen2-VL-2B on Jetson Orin Nano with unified-memory OOM patterns, CUDA 13.2 SBSA unification enabling upstream wheels, and pipeline architecture for near-real-time captioning on 8GB devices.

**Key findings:**
- CUDA 13.2 SBSA unification reverses historical Jetson-ecosystem lag
- Unified-memory OOM pattern and fix (device_map='cuda:0', low_cpu_mem_usage=True)
- Native transformers models superior to custom-code models for stability

**Platform:** NVIDIA Jetson Orin Nano 8GB | **Date:** July 2026

[Read full paper →](research_1_vlm_edge_deployment.md)

---

### 2. TensorRT Inference Optimization on Jetson Orin: The 1.7x Speedup Pattern

YOLOv8n object detection optimization achieving 1.7x speedup over PyTorch (30ms → 17ms) through kernel autotuning and layer fusion on Compute Capability 8.7.

**Key findings:**
- PyTorch baseline already real-time capable (33 FPS)
- TensorRT kernel autotuning is per-GPU, not portable across compute capabilities
- Speedup valuable for headroom and multi-model concurrency, not hard real-time requirements
- Capture/encode becomes bottleneck once inference reaches 17ms

**Platform:** NVIDIA Jetson Orin Nano 8GB | **Date:** July 2026

[Read full paper →](research_2_tensorrt_optimization.md)

---

### 3. Unified-Memory Multi-Model Concurrency: The Memory Accounting Problem

Running VLM + TensorRT detector simultaneously on 8GB unified memory: process consolidation, quantization kernel compatibility, and the gap between model size and actual process memory under concurrent GPU workloads.

**Key findings:**
- Per-process CUDA context overhead (~1-1.5GB) dominates model size in multi-model scenarios
- int8 vs int4 quantization hit different hardware-compatibility paths (int8 failed on CC 8.7, int4 NF4 succeeded)
- Baseline system load (GUI, apps) creates invisible floor in development environments
- Achieved 29 FPS detection + 5-7s captioning concurrently with 116MB memory margin

**Platform:** NVIDIA Jetson Orin Nano 8GB | **Date:** July 2026

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

## About

Papers 1-3 document research from production deployments on NVIDIA Jetson Orin Nano 8GB hardware, emphasizing practical systems-level challenges in edge AI deployment. Papers 4-5 document findings from building and evaluating a multi-model small-LLM fine-tuning pipeline, emphasizing failure modes that produce valid-looking but degraded artifacts, and the evaluation rigor required to trust A/B comparisons between fine-tuned models. Paper 6 documents hardware-level board bring-up, covering device-discovery methodology and firmware-flashing tool regressions under real-world host constraints.

## Citation

If you reference these findings, please cite the individual papers with their full titles and dates.
