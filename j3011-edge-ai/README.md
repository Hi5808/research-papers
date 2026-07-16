# J3011 Edge AI Project: VLM + Real-Time Object Detection

Complete implementation and project logs for concurrent deployment of Qwen2-VL-2B (vision-language model) and YOLOv8n (object detection) on NVIDIA Jetson Orin Nano 8GB.

## Files

### Python Implementation

- **`combined_pipeline.py`** — Unified inference pipeline running both VLM and YOLO detector in a single process with shared CUDA context. Handles concurrent frame capture, detection inference, VLM captioning, and web serving.

- **`yolo_stream.py`** — Standalone real-time YOLO object detection with MJPEG streaming. TensorRT-accelerated inference served over HTTP with `/stream` endpoint for live video feed and `/stats` for performance metrics.

### Documentation & Logs

- **`vlm_edge_inference_report.md`** — Comprehensive technical report covering container ecosystem lag, CUDA 13.2 SBSA unification, unified-memory OOM patterns, model selection criteria, and concurrent multi-model deployment engineering. Primary research documentation.

- **`yolo_project_log.md`** — Chronological project log documenting TensorRT engine building, performance benchmarking (1.7x speedup, 30ms → 17ms), and multi-round memory-fitting problem solving on 8GB unified memory.

### Configuration & Models

- **`nvfancontrol.conf.reference`** — Fan control configuration for sustained thermal performance under continuous inference load.

- **`yolov8n.engine`** — Pre-compiled TensorRT FP16 inference engine for YOLOv8n on Orin (Compute Capability 8.7). ~8.7MB, ready for deployment. Built via `ultralytics export format='engine'`.

## Key Results

- **YOLO detection:** 29.1 FPS, 19.8ms inference (TensorRT optimized)
- **VLM captioning:** 5-7s per frame (Qwen2-VL-2B, int4 quantized)
- **Concurrent operation:** Both models running simultaneously with 116MB memory margin
- **Boot time:** 13.7s (minimal Yocto OS, optimized JetPack 7.2)

## Architecture

**Single-process consolidation** with int4 NF4 quantization and shared CUDA context enables dual-model inference on 8GB:

```
GPU Memory (8GB unified):
├── Qwen2-VL-2B (int4)  → ~2.2GB
├── YOLOv8n engine      → ~0.2GB
├── Shared CUDA context → ~1.5GB
├── PyTorch/TensorRT runtime → ~2GB
└── Free margin         → ~116MB (stable)
```

Process consolidation saves ~1GB compared to separate processes (each paying independent CUDA context tax).

## Deployment

Both scripts are designed for localhost-only serving (127.0.0.1) per the development constraint of avoiding network exposure during active development. For production deployment, update host binding and add appropriate authentication.

## Related Research

See the research papers in the parent directory for detailed analysis:
- Vision-Language Models on Edge Hardware
- TensorRT Performance Optimization  
- Unified-Memory Multi-Model Concurrency
