# Project Log: Live YOLO Object Detection on Jetson Orin Nano

Running log for the real-time webcam object detection pipeline. Appended chronologically as work happens.

---

## 2026-07-15

**Goal:** Live streamed object detection feed from the USB webcam, using the best available performance path on this hardware (as opposed to the VLM project's ~7s/frame cadence — YOLO should support genuine real-time/frame-rate video).

**00:00 — Environment setup**
- Installed `ultralytics` (8.4.95) via pip.
- Hit an import crash: `ultralytics.models` eagerly imports `FastSAM` → `SegmentationPredictor` → `matplotlib`, and system `matplotlib` (apt-installed, `/usr/lib/python3/dist-packages`) was built against numpy 1.x while our pip environment has numpy 2.5.1 (pulled in as a torch/ultralytics dependency) → `ImportError: numpy.core.multiarray failed to import`.
  - **Fix:** `pip install --break-system-packages -U matplotlib scipy` to get numpy2-compatible wheels shadowing the system ones in `~/.local`.

**00:05 — Baseline PyTorch inference**
- `yolov8n.pt` (nano, smallest/fastest COCO model), fp16, `device=0`.
- First run on a stale cached snapshot: 0 detections (bad test image, not a model issue).
- Fresh webcam frame: correctly detected `cup` (0.83), `person` (0.81), `tv` (0.66).
- Benchmark: **~30.3ms/frame average** over 20 runs in raw PyTorch fp16 — already ~33 FPS, i.e. real-time-capable without any further optimization.

**00:10 — TensorRT export (best-performance path)**
- User asked for best-available performance, so exporting `yolov8n.pt` → ONNX → TensorRT FP16 engine (`yolov8n.engine`), targeting this exact GPU (Orin, CC 8.7) rather than running generic PyTorch.
- First attempt failed: `ModuleNotFoundError: No module named 'onnx'` (ultralytics' TensorRT export path routes through ONNX as an intermediate representation).
  - **Fix:** `pip install --break-system-packages onnx onnxslim onnxruntime`.
- Retry: ONNX export succeeded (2.8s, 12.3MB `.onnx`). TensorRT engine build in progress as of this entry — this step does per-layer kernel autotuning for the specific GPU and is the slow part (observed taking several minutes on this hardware for prior TensorRT builds).

**Status: TensorRT engine build in progress.** Next steps once it completes:
1. Benchmark `.engine` inference speed vs. the 30.3ms PyTorch baseline.
2. Build the live MJPEG/streaming pipeline with bounding boxes drawn, served over local HTTP (bound to `127.0.0.1` per the same network-exposure constraint as the VLM project — will ask before opening to LAN).
3. Confirm sustained frame rate against the webcam's actual native capture rate (capture itself, not just inference, may become the bottleneck at this speed).

**00:20 — TensorRT engine benchmark**
- Build completed: 406.1s (~6.8 min) of genuine per-layer kernel autotuning (confirmed via `ps` — not a hang, just how long TensorRT engine generation takes for even a nano model on this hardware). Output: `~/yolov8n.engine`, 8.7MB, FP16.
- Benchmark (20 iterations, `/tmp/fresh_snap.jpg`):

| Backend | Avg latency | Approx FPS |
|---|---|---|
| PyTorch fp16 (`yolov8n.pt`) | 30.3ms | ~33 |
| TensorRT FP16 (`yolov8n.engine`) | **17.4ms** | **~57** |

- **1.7x speedup** from the TensorRT engine, with detection quality intact or slightly better (person 0.87, cup 0.84, tv 0.67 vs. PyTorch's 0.81/0.83/0.66).
- Decision: use `yolov8n.engine` as the inference backend for the live streaming pipeline — at 17.4ms/frame, inference is no longer the bottleneck; webcam capture rate will likely be the limiting factor instead.

**Next:** build the live detection pipeline (capture → TensorRT inference → draw boxes → serve), bound to localhost per the same network-exposure policy as the VLM project.

**00:2X — Live streaming pipeline built and running**
- `~/yolo_stream.py`: single process, background thread does capture → TensorRT inference → `results[0].plot()` box drawing → JPEG encode into a shared buffer; `ThreadingHTTPServer` serves a true `multipart/x-mixed-replace` MJPEG stream at `/stream` (not polling like the VLM project — inference is fast enough now for genuine smooth video), plus `/stats` (JSON: fps/infer_ms/n_dets) and `/` (viewer page).
- Bound to `127.0.0.1:8001` (localhost-only, same network-exposure policy as before).
- **Confirmed live: 41 FPS, 17.4ms inference, 4 objects detected** at first measurement — capture+serve overhead is negligible on top of the 17.4ms TensorRT inference baseline; this is genuine real-time video, unlike the VLM pipeline's ~7s snapshot cadence.
- Viewer: `http://localhost:8001/`

## Combined pipeline: the unified-memory saga

User requested VLM captions restarted alongside the live YOLO stream. This triggered a multi-round memory-fitting problem on the 8GB unified-memory Orin Nano:

**Round 1 — two separate processes (fp16 VLM + running YOLO stream process):**
OOM. `free -h` at OOM time showed only 2.8GB free — the YOLO stream process alone (with its own CUDA context, PyTorch/TensorRT runtime, ultralytics) was holding ~1.4GB RSS, well above the ~200MB assumed for "just the TensorRT engine." Two separate Python processes each pay their own fixed CUDA-context tax (~1-1.5GB) on top of model weights.

**Round 2 — consolidated into a single process (`combined_pipeline.py`, fp16 VLM):**
Closer, but still OOM: needed 4.42GB for VLM weights, only 4.09GB free (~330MB short). Confirmed baseline system load (desktop GUI + Firefox + Claude Code's own processes) alone eats ~3.5GB of the 7.4GB total — a real, non-negotiable floor that isn't ML-workload-related at all.

**Round 3 — int8 quantization via bitsandbytes:**
Model *loaded* successfully (memory fit fine — int8 roughly halves the weight footprint). But inference crashed:
```
RuntimeError: cublasLt ran into an error!
```
A genuine bitsandbytes int8 CUDA kernel incompatibility with this hardware/software combination (Orin CC 8.7 + CUDA 13.2, likely too new for bitsandbytes' cublasLt int8 path to have been validated against). The failure was isolated to a daemon thread, so the process didn't crash outright — YOLO kept running fine, only captions got stuck on "loading...".

**Round 4 — int4 (nf4) quantization via bitsandbytes — SUCCESS:**
Different kernel code path than int8, avoided the cublasLt bug entirely. Both models loaded and ran concurrently:
- YOLO: 29.1 FPS, 19.8ms inference, live bounding boxes
- VLM: captions every 5-7s, quality-comparable to the fp16 captions from earlier in the project (correctly identifying "man," "dark hoodie," "cluttered room," "poster on wall")

Final memory state: tight (116MB free, ~1GB into swap) but stable — no further OOMs observed across ~8 consecutive caption rounds.

**Takeaway:** on an 8GB unified-memory edge device, running two GPU models concurrently is a real engineering problem, not just a "does it fit" checkbox — process consolidation, quantization *type* (int8 vs int4 hit different hardware-compatibility outcomes on this exact silicon), and honest accounting of non-ML baseline memory (desktop environment, dev tools) all mattered. The combined pipeline is served at `http://localhost:8002/`.
