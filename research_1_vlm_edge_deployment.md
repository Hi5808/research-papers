# Vision-Language Models on Edge Hardware: JetPack 7.2 Deployment Patterns

**Platform:** NVIDIA Jetson Orin Nano 8GB  
**Software stack:** JetPack 7.2, L4T R39.2, CUDA 13.2, PyTorch 2.13.0+cu130, Transformers 5.13.1  
**Date:** July 2026

## Abstract

Modern vision-language models can achieve near-real-time inference on constrained edge hardware (8GB Orin Nano) using standard upstream PyTorch/Transformers packages, but deployment requires navigating software-ecosystem lag and a unified-memory-specific OOM failure mode not present on discrete-GPU systems.

## Key Finding #1: CUDA 13.2 SBSA Unification

**The Problem:** Standard ML wheel-building practice targets either cloud GPU systems or Jetson-specific L4T builds — rarely both. JetPack 7.2 was released before container ecosystems (`jetson-containers`) had published compatible builds, blocking the standard cloud-ML deployment path.

**The Solution:** NVIDIA's CUDA 13.2 unified Jetson Orin onto the server-class ARM SBSA (Server Base System Architecture) CUDA toolkit. This means standard `pip install torch` wheels (built for generic aarch64+CUDA) work directly:

```python
import torch
torch.cuda.is_available()  # True
torch.cuda.get_device_name(0)  # 'Orin'
```

**Implication:** When official Jetson container support lags, check whether the underlying platform has been unified with better-supported architectures. This reverses the historical pattern where Jetson always required specialized builds.

## Key Finding #2: The Unified-Memory OOM Pattern

**The Problem:** Naive model loading fails with CUDA OOM despite the model fitting comfortably in 8GB:

```python
model = AutoModelForImageTextToText.from_pretrained(model_id).to('cuda')
# RuntimeError: CUDA OOM ... (free: 104MB, total: 7.9GB)
```

**Root Cause:** On unified-memory devices, `.from_pretrained()` materializes the full model on CPU, then `.to('cuda')` copies it to GPU — both copies coexist briefly in the *same physical memory pool*, nearly doubling peak usage. On discrete-GPU systems (CPU RAM ≠ VRAM) this is transparent; on unified memory it exhausts the pool.

**The Fix:**
```python
model = AutoModelForImageTextToText.from_pretrained(
    model_id, dtype=torch.float16,
    device_map='cuda:0', low_cpu_mem_usage=True,
)
```

This avoids the CPU materialization step entirely. **Critical pattern for any transformers-based workflow on Jetson or similar ARM SBSA edge hardware.**

## Key Finding #3: Model Selection Under Version Skew

Vision-language model ecosystems vary widely in how they handle library version changes:

- **Custom code models** (e.g., moondream2 with `trust_remote_code=True`): Ship internal modeling code that references specific transformers attributes. When the library renames or removes those attributes, the model breaks silently at import time, with no workaround except waiting for the model author to update their code.

- **Native transformers models** (e.g., Qwen2-VL-2B-Instruct using `AutoModelForImageTextToText`): Integrated into the transformers library itself. Library changes are coordinated with model support — breakage is rare and release notes highlight incompatibilities.

**Selection criteria:** Prefer native models unless the custom model offers unique capabilities. The stability payoff is substantial on edge hardware where intervention options are limited.

## Results

| Metric | Value |
|---|---|
| Model load time (cached) | 11.8s |
| Single-image inference (150 tokens) | 23.9s |
| Live captioning cadence | 5-9s/frame |
| Inference quality | Consistent, hallucination-free on static scenes |

## Reproduction

```bash
# PyTorch + transformers + pillow, from standard indices
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install transformers accelerate

# Load model with unified-memory pattern
python3 -c "
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
model = AutoModelForImageTextToText.from_pretrained(
    'Qwen/Qwen2-VL-2B-Instruct', dtype=torch.float16,
    device_map='cuda:0', low_cpu_mem_usage=True,
)
"
```

## Conclusion

JetPack 7.2 edge deployment is viable without Jetson-specific tooling, provided you: (1) recognize platform unification opportunities, (2) account for unified-memory OOM patterns in model loading, and (3) select models that integrate natively with evolving library ecosystems.
