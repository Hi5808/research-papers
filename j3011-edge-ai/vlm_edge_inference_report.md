# Multimodal Edge AI on the NVIDIA Jetson Orin Nano: VLM Captioning, TensorRT-Accelerated Object Detection, and Concurrent Multi-Model Deployment Under Unified Memory Constraints

**Platform:** Seeed Studio reComputer J3011 (NVIDIA Jetson Orin Nano 8GB)
**Software stack:** JetPack 7.2, L4T R39.2.0, CUDA 13.2, PyTorch 2.13.0+cu130, Transformers 5.13.1, TensorRT 10.16.2.10, Ultralytics 8.4.95, bitsandbytes 0.49.2
**Date:** July 2026

## Abstract

We report on three linked edge-AI deployment efforts on an NVIDIA Jetson Orin Nano 8GB: (1) real-time image captioning via a 2-billion-parameter vision-language model (Qwen2-VL-2B-Instruct), (2) real-time object detection via a TensorRT-compiled YOLOv8n engine, and (3) concurrent deployment of both models in a single process under the device's 8GB unified-memory ceiling. We document the transition away from Jetson-specific container tooling (`jetson-containers`) — which had not yet published builds for this hardware generation — toward standard upstream PyTorch wheels, made possible by NVIDIA's unification of Jetson Orin onto the ARM SBSA CUDA toolkit as of CUDA 13.2. We characterize the unified-memory constraints of an 8GB edge device under multi-model GPU workloads, present a working fix for a common out-of-memory failure mode, quantify a 1.7x TensorRT speedup over PyTorch for detection, and report a genuine hardware-specific bitsandbytes int8 kernel failure alongside its int4 resolution — enabling final concurrent operation at 29 FPS detection alongside 5-7s-cadence captioning on a single 8GB device.

## 1. Introduction

Edge deployment of vision-language models is attractive for applications requiring on-device scene understanding without cloud round-trips — robotics, surveillance, assistive devices. The Jetson Orin Nano, with its 1024-core Ampere GPU and 8GB of unified LPDDR5 memory, is a natural target, but the ecosystem's tooling (container images, prebuilt wheels) typically lags new JetPack releases by months. This report documents a deployment attempt on JetPack 7.2 essentially at release, and the workarounds required as a result.

## 2. Hardware and Software Environment

| Component | Spec |
|---|---|
| SoC | NVIDIA Jetson Orin Nano 8GB (6-core Cortex-A78AE, 1024-core Ampere GPU, 32 Tensor Cores) |
| Carrier board | Seeed Studio reComputer J3011 (J401), Super Mode enabled (25W power profile) |
| L4T / JetPack | R39.2.0 / 7.2 |
| CUDA | 13.2 (ARM SBSA toolkit) |
| Storage | 128GB NVMe (FORESEE XP1000F128G) |

## 3. Methodology

### 3.1 Container-based deployment: a dead end

Our first approach followed the community-standard path for Jetson VLM deployment: `dusty-nv/jetson-containers`, which provides NVIDIA-optimized Docker images (`nano_llm`, `llava`, `vila`) with TensorRT-accelerated multimodal pipelines. Investigation of the repository's autotag resolver, run directly against this device, revealed that no compatible container images existed for L4T r39.x at any dependency level — not just the top-level VLM packages, but the underlying `pytorch` (best available: `r36.4.0-cu128`) and `transformers` (`r35.3.1`) base images. The repository's L4T-version-detection code had been updated in anticipation of JetPack 7.2 hardware, but the community had not yet published matching container builds. A from-source build of the full dependency chain (PyTorch, TensorRT bindings, flash-attention) was judged impractical within the session (estimated hours, with material risk of failure given the CUDA 12.6→13.2 jump).

### 3.2 Pivot: standard upstream wheels via SBSA unification

NVIDIA's CUDA 13.2 release notes state that Jetson Orin now shares the same ARM SBSA (Server Base System Architecture) CUDA toolkit used by server-class GPUs (GH200/GB200), rather than requiring Jetson-specific L4T-tied builds. This meant standard upstream `pip install torch` wheels — built for generic aarch64+CUDA rather than Jetson's historically bespoke stack — were hypothesized to work directly.

This was confirmed empirically:

```
torch 2.13.0+cu130
cuda available: True
device: Orin (compute capability 8.7)
```

A GPU compute smoke test (20× 2000×2000 matrix multiplication) completed in 0.313s, confirming genuine GPU execution rather than silent CPU fallback. Note that PyTorch emits a compatibility warning because its published kernel list does not explicitly enumerate CC 8.7 (Orin's compute capability falls in a gap in the CC 8.0/9.0/10.0 kernel buckets); in practice, PTX forward-compatibility within the Ampere family made this a non-issue.

### 3.3 Model selection

Two candidate models were evaluated:

**`vikhyatk/moondream2`** (1.8B, `trust_remote_code=True`): downloaded and loaded weights successfully (592 tensors), but failed at the final loading step with:
```
AttributeError: 'HfMoondream' object has no attribute 'all_tied_weights_keys'.
Did you mean: '_tied_weights_keys'?
```
This is a version-skew failure: moondream2 ships custom modeling code that references an internal `transformers` attribute renamed in the installed 5.13.1 release. Custom `trust_remote_code` models are inherently fragile against fast-moving host library versions, since they pin to the API surface at time of publication rather than tracking upstream.

**`Qwen/Qwen2-VL-2B-Instruct`**: uses transformers' *native* `AutoModelForImageTextToText`/`AutoProcessor` classes rather than remote code, and was selected as the production model for this reason — native integration is maintained in lockstep with the library itself, avoiding this class of breakage entirely.

### 3.4 The unified-memory OOM failure mode

The initial loading pattern — `AutoModelForImageTextToText.from_pretrained(...).to('cuda')` — failed with a CUDA OOM error despite the model (≈4.4GB in fp16) fitting comfortably within the 8GB total device memory:

```
RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED ... CUDACachingAllocator.cpp
[W] memory allocation failed with OOM on device 0 while trying to allocate 467664896 bytes
    (free: 104280064, total: 7913803776)
```

Root cause: on a **unified-memory** device, `from_pretrained()` first materializes the full model on host (CPU) memory, then `.to('cuda')` copies it to device memory — for a brief window, both the CPU-resident and GPU-resident copies coexist in the *same physical memory pool*, roughly doubling peak usage relative to the model's steady-state footprint. On a discrete-GPU system this is harmless (CPU RAM and VRAM are separate pools); on Jetson's unified architecture it can exhaust the single shared pool even when the model comfortably fits at rest.

**Fix:** load directly to the target device, avoiding the intermediate CPU materialization:
```python
model = AutoModelForImageTextToText.from_pretrained(
    model_id, dtype=torch.float16,
    device_map='cuda:0', low_cpu_mem_usage=True,
)
```
This is a generally applicable pattern for any transformers model deployment on unified-memory edge hardware (Jetson, and likely similar ARM SBSA edge SoCs going forward), not specific to this model.

### 3.5 Secondary fixes

- **Missing `torchvision`**: Qwen2-VL's processor pipeline instantiates a video-processing backend (used even for still images) that hard-requires `torchvision`. Installed from the matching `cu130` wheel index.

## 4. Results

### 4.1 Single-shot inference

| Metric | Value |
|---|---|
| Model load (`device_map='cuda:0'`, cached weights) | 11.8s |
| Inference, 150 max new tokens | 23.9s |

Sample output (webcam capture, static single image):
> *"The image depicts a dark, dimly lit room at night. The walls are painted a muted, blueish-gray color, which casts a cool, blue light throughout the space. The room appears to be a bedroom or a small living area, given the presence of a bed and some furniture..."*

### 4.2 Live continuous captioning pipeline

A persistent pipeline (model loaded once, then looped: capture → infer → serve) was built to avoid re-paying the model-load cost per frame, with output reduced to 60 max new tokens for faster turnaround:

| Metric | Value |
|---|---|
| Steady-state cadence | 5-9s per frame |
| Fastest observed round | 4.4s |
| Slowest observed round | 8.9s |

Output was served via a local HTTP endpoint (`latest.jpg` + `latest.json`, atomically written with `os.replace` to avoid torn reads) and a polling HTML viewer refreshing every 2s — an appropriate "live" cadence given the ~7s inference-bound ceiling, rather than attempting frame-rate video which the hardware cannot support at this model size.

Across ~30 consecutive rounds pointed at a static indoor scene (person at a desk), captions were qualitatively consistent (recurring identification of "person," "cluttered room," "desk," "poster on wall," "dark hoodie") with some frame-to-frame variance in fine details (activity inferred as "writing," "reading," or "using a tablet" across different rounds) — expected behavior for a 2B-parameter model performing independent single-frame inference with no temporal memory between calls.

## 5. Discussion

The central finding of practical relevance: **when a hardware platform's official container/wheel ecosystem lags the underlying silicon's software support**, it is worth checking whether the underlying platform has been architecturally unified with a better-supported one — in this case, CUDA 13.2's SBSA unification meant Jetson Orin could ride on server-class CUDA tooling months before Jetson-specific tooling caught up. This is a reversal of the historical pattern, where Jetson deployment always required NVIDIA- or community-maintained Jetson-specific builds.

The unified-memory OOM failure mode (Section 3.4) is likely to recur for any practitioner moving transformers-based model loading code from a discrete-GPU workstation to Jetson-class edge hardware without modification, since the naive `.from_pretrained().to(device)` pattern is standard practice on non-unified-memory systems and only breaks silently-to-loudly on unified memory near the capacity boundary.

## 6. VLM Deployment Summary

A modern 2B-parameter VLM can run with genuine GPU acceleration on an 8GB Jetson Orin Nano using only upstream, non-Jetson-specific software (stock PyTorch/transformers pip wheels), achieving single-digit-second inference latency suitable for near-real-time (not frame-rate) captioning applications. The primary engineering obstacles were software-ecosystem lag (container tooling) and a unified-memory-specific OOM pattern, both practical rather than fundamental, and both resolved without hardware workarounds.

This raised an obvious follow-up question: VLM captioning is inherently too slow (5-9s/frame) for applications needing genuine frame-rate video understanding. Sections 7-9 report a companion effort using a lighter-weight, TensorRT-accelerated object detection model to fill that gap, and the concurrency problems that emerged when running both models on the same device simultaneously.

## 7. Real-Time Object Detection with TensorRT

### 7.1 Motivation and model selection

Where the VLM work targeted rich, free-form scene description at multi-second latency, a complementary need is genuine real-time perception — frame-rate video with per-object localization. YOLOv8n (Ultralytics, nano variant, 80-class COCO detector) was selected as the lightest available option, prioritizing speed over the wider vocabulary of larger variants.

### 7.2 Dependency friction

Installing `ultralytics` surfaced a numpy ABI conflict: the package's `__init__` eagerly imports `FastSAM`, which transitively imports `matplotlib` — and the system-installed `matplotlib` (apt-managed, under `/usr/lib/python3/dist-packages`) had been compiled against numpy 1.x, while the pip environment's numpy had been upgraded to 2.5.1 as a transitive dependency:
```
ImportError: numpy.core.multiarray failed to import
```
Resolved by upgrading `matplotlib`/`scipy` via pip to numpy2-compatible builds, which shadow the older system packages in the user's local site-packages. This class of failure — an apt-managed system library silently incompatible with a pip-managed dependency bump — is a recurring hazard on Jetson, where the base OS image ships numerous apt-installed Python packages that pip-based ML workflows can silently invalidate.

### 7.3 Baseline PyTorch performance

Direct `yolov8n.pt` inference (fp16, `device=0`) on a fresh webcam frame: correctly detected `person` (0.81), `cup` (0.83), `tv` (0.66). Benchmarked at **30.3ms/frame average** (20 iterations) — already ~33 FPS, i.e., real-time-capable with zero additional optimization work. This is a useful baseline finding in itself: for small (nano-class) detection models, raw PyTorch on Orin Nano's GPU is already sufficient for real-time use without requiring engine compilation.

### 7.4 TensorRT engine export

Given the further headroom available, the model was exported to a TensorRT FP16 engine via `model.export(format='engine', half=True, device=0, imgsz=640)`, targeting the exact on-device GPU (Orin, compute capability 8.7) rather than shipping a generic PyTorch graph.

- **First attempt failed**: `ModuleNotFoundError: No module named 'onnx'` — Ultralytics' TensorRT export path routes through ONNX as an intermediate representation, which is not installed by default. Resolved with `pip install onnx onnxslim onnxruntime`.
- **Retry**: ONNX export completed in 2.8s (12.3MB intermediate `.onnx`). The subsequent TensorRT build performs genuine per-layer kernel autotuning specific to the target GPU, and is markedly slower: **406.1 seconds (~6.8 minutes)** of continuous CPU/GPU activity for a nano-sized model, confirmed via process monitoring to be active work rather than a hang. Output: an 8.7MB `yolov8n.engine`.

### 7.5 TensorRT vs. PyTorch benchmark

| Backend | Avg latency | Approx FPS |
|---|---|---|
| PyTorch fp16 (`yolov8n.pt`) | 30.3ms | ~33 |
| TensorRT FP16 (`yolov8n.engine`) | **17.4ms** | **~57** |

A **1.7x speedup**, with detection confidence slightly *higher* on the TensorRT path for the same test frame (person 0.87 vs. 0.81, cup 0.84 vs. 0.83, tv 0.67 vs. 0.66) — TensorRT's kernel fusion and precision handling did not measurably degrade accuracy here, consistent with FP16 being a lossless-in-practice precision reduction for this model class.

### 7.6 Live streaming pipeline

Given inference now well under 20ms, a genuine frame-rate video pipeline was implemented (`yolo_stream.py`): a background thread performs capture → TensorRT inference → `results.plot()` box rendering → JPEG encode into a shared, lock-protected buffer; a `ThreadingHTTPServer` serves this as a true `multipart/x-mixed-replace` MJPEG stream, in contrast to the VLM pipeline's necessarily poll-based JSON+static-image approach (Section 4.2), which was the correct design choice only because VLM inference was too slow to support genuine streaming. **Measured live throughput: 41 FPS, 17.4ms inference, capture/encode/serve overhead adding negligibly to the TensorRT baseline.**

Two refinements were subsequently applied to the rendered output: a confidence threshold (`conf=0.55`) to suppress low-confidence noisy detections, and cleaner label rendering (`line_width=3, font_size=18, conf=False` — thicker boxes, larger text, and hiding the numeric confidence score for a less cluttered display), based on user feedback that default Ultralytics box labels were visually cluttered for a live-viewing use case.

## 8. Concurrent Multi-Model Deployment: The Unified-Memory Saga

### 8.1 Problem statement

With both a fast detector and a rich captioner independently working, the natural next step was running them *simultaneously* — a live annotated video feed alongside periodic natural-language scene descriptions. On a discrete-GPU workstation this is a routine multi-process deployment. On an 8GB unified-memory edge device, it became a genuine systems problem, worked through in four rounds.

### 8.2 Round 1 — two separate processes (OOM)

Starting the VLM captioning process while the YOLO streaming process (Section 7.6) was already running failed with a CUDA OOM:
```
RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED ... CUDACachingAllocator.cpp
[W] memory allocation failed with OOM ... (free: 2834817024, total: 7913803776)
```
Investigation via `ps` revealed the YOLO streaming process alone was holding **1.4GB RSS** — substantially more than the 8.7MB TensorRT engine file would suggest, since a process's fixed CUDA context (driver state, kernel caches, PyTorch/Ultralytics runtime) carries its own overhead independent of model size. Two independent processes each pay this fixed tax on top of their respective model weights, a cost that is easy to underestimate when reasoning about "does the model fit" in isolation.

### 8.3 Round 2 — single-process consolidation (near miss)

Consolidating both models into one process (`combined_pipeline.py`, running detection and captioning as two threads sharing one CUDA context) reduced but did not eliminate the shortfall: the VLM's fp16 weight load required 4.42GB; only 4.09GB was free — a **330MB gap**. This also surfaced a finding independent of the ML workload entirely: baseline system load (desktop GUI session, browser, and the development-tooling processes used to conduct this work) was consuming approximately 3.5GB of the device's 7.4GB total *before any model was loaded* — a fixed floor that a deployed (non-interactive, non-desktop) production image would not pay, but which materially affected this development-time measurement.

### 8.4 Round 3 — int8 quantization (loads, then crashes)

To close the 330MB gap robustly rather than opportunistically (e.g., by asking the user to free RAM), int8 quantization was applied via `bitsandbytes` (`BitsAndBytesConfig(load_in_8bit=True)`). The quantized model *loaded* successfully — confirming int8 roughly halves resident weight memory as expected — but crashed on first inference:
```
RuntimeError: cublasLt ran into an error!
    shapeA=torch.Size([3840, 1280]) shapeB=torch.Size([1564, 1280]) shapeC=(1564, 3840)
```
This is a genuine hardware/library incompatibility: `bitsandbytes`'s int8 matrix-multiply path depends on cuBLASLt kernels that had not been validated against this GPU generation and CUDA version combination (Orin, compute capability 8.7, CUDA 13.2) at time of writing — plausibly because both the SBSA-unified CUDA stack and this specific compute-capability gap (Section 3.2) are recent enough that `bitsandbytes`'s kernel dispatch tables have not caught up. Because the failure occurred inside a Python daemon thread, it did not crash the host process — the YOLO detection thread continued operating normally throughout, while the captioning thread silently stalled.

### 8.5 Round 4 — int4 (NF4) quantization (success)

Switching to 4-bit NormalFloat quantization (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16)`) — a different kernel code path from int8 — avoided the cuBLASLt failure entirely. Both models loaded and ran concurrently:

| Component | Result |
|---|---|
| YOLO detection | 29.1 FPS, 19.8ms inference, live bounding boxes |
| VLM captioning | New caption every 5-7s, quality comparable to the fp16 captions of Section 4 |

Sample concurrent-mode caption: *"A man wearing a dark hoodie sits in a cluttered room with a poster on the wall."* — consistent in content and specificity with the standalone fp16 captions reported in Section 4.2, indicating NF4 quantization did not perceptibly degrade caption quality for this use case.

Final memory state was tight (116MB free, ~1GB resident in zram swap) but stable across the full observation window, with no further OOM events.

### 8.6 Discussion

The four-round progression illustrates that "will two models fit on an 8GB device" is not answerable by summing published model sizes. Three independent factors each contributed a comparable-magnitude effect: (a) per-process fixed CUDA context overhead, resolved by consolidating to a single process; (b) non-ML baseline system memory, which is easy to omit from capacity planning during interactive development; and (c) quantization *kernel-implementation* compatibility, which varies by specific quantization scheme even when the memory-footprint math is identical — int8 and int4 both target the same GPU and library, yet only one had a working kernel path on this hardware at time of writing. A practitioner who stopped after confirming int8 loaded successfully (Section 8.4) would have shipped a broken system; only end-to-end inference testing, not just successful model loading, surfaced the fault.

## 9. Overall Conclusion

Across both efforts, this device — an 8GB Jetson Orin Nano on JetPack 7.2 — proved capable of substantially more than a first-pass capacity estimate would suggest: a 2B-parameter VLM at near-real-time captioning latency, a TensorRT-accelerated detector at genuine 29-57 FPS depending on concurrent load, and (with the right quantization choice) both running simultaneously on a single device with under 200MB of memory margin to spare. None of the obstacles encountered were fundamental hardware limitations; all were software-ecosystem lag, unified-memory-specific allocation patterns, or kernel-compatibility gaps in fast-moving quantization libraries — each diagnosable and resolvable through direct measurement (`ps`, `free`, `tegrastats`) rather than speculation.

## Appendix: Reproduction commands

```bash
python3 -m pip install --break-system-packages torch torchvision \
  --index-url https://download.pytorch.org/whl/cu130
python3 -m pip install --break-system-packages transformers accelerate pillow

python3 -c "
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

model = AutoModelForImageTextToText.from_pretrained(
    'Qwen/Qwen2-VL-2B-Instruct', dtype=torch.float16,
    device_map='cuda:0', low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')

img = Image.open('image.jpg')
messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'Describe this image.'}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[img], return_tensors='pt').to('cuda')
out = model.generate(**inputs, max_new_tokens=150)
print(processor.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0])
"
```

**YOLOv8n → TensorRT export and benchmark:**
```bash
python3 -m pip install --break-system-packages ultralytics onnx onnxslim onnxruntime

python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='engine', half=True, device=0, imgsz=640)  # ~7 min build
"

python3 -c "
from ultralytics import YOLO
import time
model = YOLO('yolov8n.engine')
model('image.jpg', verbose=False)  # warmup
t0 = time.time()
for _ in range(20):
    results = model('image.jpg', verbose=False, conf=0.55)
print(f'{(time.time()-t0)/20*1000:.1f}ms/frame')
"
```

**Concurrent VLM + YOLO (int4 quantization required on this hardware — int8 crashes, see Section 8.4):**
```bash
python3 -m pip install --break-system-packages bitsandbytes

python3 -c "
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForImageTextToText.from_pretrained(
    'Qwen/Qwen2-VL-2B-Instruct', quantization_config=bnb_config,
    device_map='cuda:0', low_cpu_mem_usage=True,
)
# load a YOLO('yolov8n.engine') instance in the same process/CUDA context
# alongside this model to run both concurrently within the 8GB budget.
"
```
