# Research Papers

Technical research and findings from production ML systems work: edge AI deployment on constrained hardware, and small-model fine-tuning pipelines.

## Contact Me

Have questions about these findings or interested in edge AI consulting? Get in touch:

📧 **Email:** [ahkt808@proton.me](mailto:ahkt808@proton.me)

---

## Papers

### Edge AI Deployment (NVIDIA Jetson Orin Nano)

### 1. Vision-Language Models on Edge Hardware: JetPack 7.2 Deployment Patterns

Deployment of Qwen2-VL-2B on Jetson Orin Nano with unified-memory OOM patterns, CUDA 13.2 SBSA unification enabling upstream wheels, and pipeline architecture for near-real-time captioning on 8GB devices.

**Key findings:**
- CUDA 13.2 SBSA unification reverses historical Jetson-ecosystem lag
- Unified-memory OOM pattern and fix (device_map='cuda:0', low_cpu_mem_usage=True)
- Native transformers models superior to custom-code models for stability

[Read full paper →](research_1_vlm_edge_deployment.md)

---

### 2. TensorRT Inference Optimization on Jetson Orin: The 1.7x Speedup Pattern

YOLOv8n object detection optimization achieving 1.7x speedup over PyTorch (30ms → 17ms) through kernel autotuning and layer fusion on Compute Capability 8.7.

**Key findings:**
- PyTorch baseline already real-time capable (33 FPS)
- TensorRT kernel autotuning is per-GPU, not portable between CC generations
- Speedup valuable for headroom and multi-model concurrency, not hard real-time requirements
- Capture/encode becomes bottleneck once inference reaches 17ms

[Read full paper →](research_2_tensorrt_optimization.md)

---

### 3. Multi-Model Concurrency on Unified-Memory Edge Hardware: The Memory Accounting Problem

Running VLM + TensorRT detector simultaneously on 8GB unified memory: process consolidation, quantization kernel compatibility, and the gap between model size and actual process memory under concurrent GPU workloads.

**Key findings:**
- Per-process CUDA context overhead (~1-1.5GB) dominates model size in multi-model scenarios
- int8 vs int4 quantization hit different hardware-compatibility paths (int8 failed on CC 8.7, int4 NF4 succeeded)
- Baseline system load (GUI, apps) creates invisible floor in development environments
- Achieved 29 FPS detection + 5-7s captioning concurrently with 116MB memory margin

[Read full paper →](research_3_unified_memory_multimodel.md)

---

### Small Model Fine-Tuning Pipelines

### 4. Multi-Teacher Synthetic Data Distillation: Failure Modes in Small Model Fine-Tuning

Seven concrete bugs found while building a 7-model IT-operations diagnostic suite via multi-teacher LLM distillation, three of which caused a degraded model to ship with no error signal anywhere in the pipeline.

**Key findings:**
- Template-fill failures are silent, not loud — missing fill values write literal placeholder text into training data undetected
- Word-prefix deduplication hashing can collapse 84% of a short-output dataset as false duplicates
- Teacher model *fit* matters more than teacher model *scale* — a general chat model produced fabricated WMI logic for code-generation tasks that passed superficial review
- Unsorted checkpoint globbing can silently export an undertrained model with a "success" exit code at every pipeline stage

[Read full paper →](research_4_multi_teacher_distillation_pitfalls.md)

---

### 5. Does Domain-Specialist Fine-Tuning Beat a Generalist? A Negative Result and an Evaluation Methodology Fix

Testing whether narrow domain fine-tuning outperforms a generalist model of the same size (1.5B) across four IT-operations domains — and the evaluation methodology bug that would have produced a confidently wrong answer if not caught first.

**Key findings:**
- Exact-keyword test matching produces false negatives on genuinely correct answers using valid alternative implementations
- 3-question smoke tests produce non-reproducible pass/fail results from LLM sampling variance alone, even at low temperature
- LLM-as-judge scoring with a single-dimension rubric + Wilson confidence intervals recovers a statistically honest comparison
- Result: no statistically detectable specialist advantage over a generalist at 1.5B scale and current data volumes — a genuine negative result, not a broken test

[Read full paper →](research_5_specialist_vs_generalist_evaluation.md)

---

## About

Papers 1-3 document research from production deployments on NVIDIA Jetson Orin Nano 8GB hardware during July 2026, emphasizing practical systems-level challenges in edge AI deployment. Papers 4-5 document findings from building and evaluating a multi-model small-LLM fine-tuning pipeline, emphasizing failure modes that produce valid-looking but degraded artifacts, and the evaluation rigor required to trust A/B comparisons between fine-tuned models.

## Citation

If you reference these findings, please cite the individual papers with their full titles and dates.
