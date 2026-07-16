# Research Papers: Edge AI Deployment on NVIDIA Jetson Orin Nano

Technical research and findings from production edge AI deployments on constrained hardware.

## Contact Me

Have questions about these findings or interested in edge AI consulting? Get in touch:

📧 **Email:** [ahkt808@proton.me](mailto:ahkt808@proton.me)

---

## Papers

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

## About

These papers document research from production deployments on NVIDIA Jetson Orin Nano 8GB hardware during July 2026. Findings emphasize practical systems-level challenges in edge AI deployment, from ecosystem lag and unified-memory constraints to hardware-specific kernel compatibility issues not visible in standard documentation.

## Citation

If you reference these findings, please cite the individual papers with their full titles and dates.
