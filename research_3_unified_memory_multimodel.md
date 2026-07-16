# Multi-Model Concurrency on Unified-Memory Edge Hardware: The Memory Accounting Problem

**Platform:** NVIDIA Jetson Orin Nano 8GB  
**Software stack:** JetPack 7.2, L4T R39.2, CUDA 13.2, PyTorch 2.13.0+cu130, TensorRT 10.16.2, bitsandbytes 0.49.2  
**Date:** July 2026

## Abstract

Running two GPU models concurrently on 8GB Orin is a systems problem, not merely a "does it fit" arithmetic exercise. Three independent factors each contributed comparable-magnitude memory pressure: per-process CUDA context overhead, non-ML baseline system load, and quantization-kernel compatibility. Naive summation of model sizes produces incorrect capacity estimates.

## Key Finding #1: The CUDA Context Tax

Two models in separate processes, each paying its own fixed CUDA context overhead:

```
Model A (VLM fp16): 4.4GB weights + 1.4GB context/runtime = 5.8GB
Model B (TensorRT): 0.2GB model + 1.2GB context/runtime = 1.4GB
─────────────────────────────────────────────────────
Separate processes: ~7.2GB (leaves only 200MB free on 8GB device)
```

**Naive approach:** Sum model sizes → "do they fit?" → Yes → deploy.  
**Reality:** Process consolidation reduces overhead:

```
Single process, both models: 4.4GB + 0.2GB weights + 1.5GB shared context = 6.1GB
# Saves: ~1GB vs. separate processes
```

**Practical implication:** For multi-model workloads on edge hardware, consolidate into a single process sharing one CUDA context. This saves an entire model's context overhead (~1-1.5GB).

## Key Finding #2: The Baseline System Load Floor

During interactive development on Orin (running desktop GUI, Firefox, Claude Code, etc.), baseline system memory was consuming ~3.5GB before any ML model loaded:

```
Total system RAM: 7.4GB
Baseline (GUI + apps): 3.5GB
Available for ML: 3.9GB
```

A production image (headless, no GUI) would recover this entirely. But in development, you hit an invisible floor that capacity planning easily overlooks.

**Practical implication:** Benchmark multi-model deployments on the actual target image (production stack, not dev environment). What "fits" in dev may not fit in production if your dev environment is fatter than deployment.

## Key Finding #3: Quantization Kernel Compatibility Is Not Fungible

We attempted int8 quantization to close a 330MB memory gap. The model loaded successfully:

```python
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, quantization_config=bnb_config, device_map='cuda:0'
)
# ✓ Loads successfully, memory footprint halved as expected
```

But inference crashed:

```
RuntimeError: cublasLt ran into an error!
# shapeA=torch.Size([3840, 1280]) shapeB=torch.Size([1564, 1280])
```

**Root cause:** bitsandbytes' int8 matrix-multiply path depends on cuBLASLt kernels not validated against Orin (CC 8.7) + CUDA 13.2 at time of writing. The int8 and int4 codepaths use different kernels; int4 (NF4) worked:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, quantization_config=bnb_config, device_map='cuda:0'
)
# ✓ Loads and infers successfully
```

**Practical implication:** "Quantization" is not a monolithic choice. int8 and int4 hit different kernel paths; different hardware/software combinations validate different paths. End-to-end inference testing (not just model loading) is essential—a model that loads may still crash on first inference.

## Results: Concurrent Execution

Both models running simultaneously:

| Component | Performance |
|---|---|
| YOLO detection | 29.1 FPS, 19.8ms inference |
| VLM captioning | 5-7s/frame, comparable quality to standalone fp16 |
| Final memory state | 116MB free, ~1GB swap (stable, no further OOM) |

Sample concurrent caption: *"A man wearing a dark hoodie sits in a cluttered room with a poster on the wall."* — consistent content with independent frame-by-frame inference.

## Reproduction Pattern

```bash
# Single-process consolidation with int4 quantization (required on this hardware)
pip install bitsandbytes accelerate

python3 -c "
import torch
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForImageTextToText.from_pretrained(
    'Qwen/Qwen2-VL-2B-Instruct',
    quantization_config=bnb_config,
    device_map='cuda:0',
    low_cpu_mem_usage=True,
)

# Load TensorRT detector in same process, same CUDA context
from ultralytics import YOLO
yolo = YOLO('yolov8n.engine')

# Both models now share context; run inference in separate threads
"
```

## Conclusion

Concurrent multi-model deployment on edge hardware requires accounting for: (1) per-process CUDA overhead (consolidate to one process), (2) the true baseline system load on your target image (measure in production mode), and (3) end-to-end testing of quantization choices (kernel compatibility varies by type and hardware). "Does it fit" requires measurement, not arithmetic.
