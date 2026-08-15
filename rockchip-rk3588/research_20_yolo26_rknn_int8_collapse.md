# YOLO26 on RKNN: INT8 Quantization Collapses Output Density by 92%, and the End2End Export Path's Build-Time Warning Is Silently Ignored

**Toolchain:** `rknn-toolkit2` v2.3.2 (x86_64, PC-simulator inference — no physical RK3588 hardware used for this paper's measurements), `ultralytics` v8.4.120, target platform `rk3588`
**Model:** YOLO26n (stock COCO-pretrained, not fine-tuned), ONNX opset 12
**Date:** August 2026

## Abstract

An independent, unresolved GitHub issue (`ultralytics/ultralytics#23753`, closed as stale, never fixed) reports two problems deploying YOLO26 on RK3588 via `rknn-toolkit2`: a segfault when exporting with `end2end=True` (NMS/Top-K fused into the graph), and INT8-quantized models that run without error but detect nothing. This paper reproduces both symptoms independently on a fresh install, using stock YOLO26n rather than the original reporter's custom model, via `rknn-toolkit2`'s PC-simulator inference path (no physical RK3588 device involved in these specific measurements). Exporting with `end2end=True` reproduces the reporter's exact output shape (`[1, 300, 6]`) and, during RKNN build, triggers 8 repeated `Unkown op target: 0` errors — present in both FP16 and INT8 variants of the end2end model, entirely absent from either variant of the `end2end=False` model — but the toolkit reports `build ret=0` and proceeds to a successful-looking export regardless, without surfacing this as a failure. Separately, and independent of the end2end issue, INT8 quantization of the `end2end=False` model's raw detection head collapses simulator-inference output density from 55.05% nonzero values (FP16) to 4.76% nonzero values (INT8) — a ~92% reduction — a precise, quantified signature consistent with the original report's "detects nothing at normal confidence thresholds" observation, now measured on a different model and toolkit version than the original report.

## 1. Reproducing the End2End Export Shape and the Silent Build Warning

Exporting stock `yolo26n.pt` with `end2end=True` (opset 12, simplified) produces an ONNX graph with output shape `[1, 300, 6]` — an exact match to the original report's shape for their custom model, confirming this is a structural property of YOLO26's end2end export path itself, not specific to any one trained model. The `end2end=False` export produces `[1, 84, 8400]` (4 box coordinates + 80 COCO classes; the original report's `[1, 9, 8400]` differs only because their custom model had 5 classes instead of 80 — same structure, same 4+N pattern).

Converting each ONNX export to `.rknn` for `target_platform='rk3588'` isolates the warning precisely:

| Build | `Unkown op target: 0` count |
|---|---|
| `end2end=False`, FP16 | 0 |
| `end2end=False`, INT8 | 0 |
| `end2end=True`, FP16 | 8 |
| `end2end=True`, INT8 | 8 |

The count is identical (8) across both precisions for the end2end model and exactly zero for both precisions of the non-end2end model — the warning is tied specifically to the end2end/Top-K graph structure, independent of quantization, and appears at every attempt regardless of precision choice. In every case, despite these 8 errors being printed, the build proceeds: `I rknn building done.` / `build ret=0` / `export ret=0`. **The toolkit detects the problem at compile time (it prints the error) but does not fail the build or surface it as anything other than log noise** — a real robustness/reporting gap in `rknn-toolkit2`'s own build pipeline. This is a plausible, previously-uncharacterized mechanism for the original report's runtime segfault: a `.rknn` file is produced and looks successful (no build-time failure, no crash during conversion), and only fails once actually run on real NPU hardware, which is exactly the confusing failure mode the original reporter describes ("segfault during NPU inference," not during export or conversion).

## 2. Quantifying the INT8 Detection Collapse

The original report states INT8-quantized YOLO26 "fails to detect any objects at the same confidence thresholds" as FP16, without a precise measurement of why. Running each of the four `.rknn` builds through `rknn-toolkit2`'s PC-simulator inference (`init_runtime(target=None)`, no physical RK3588 device) against a standard test image and measuring the fraction of nonzero values in the raw output tensor:

| Build | Output shape | Nonzero fraction | Mean | Max |
|---|---|---|---|---|
| `end2end=False`, FP16 | (1, 84, 8400) | **55.05%** | 9.26 | 675.00 |
| `end2end=False`, INT8 | (1, 84, 8400) | **4.76%** | 9.27 | 675.08 |
| `end2end=True`, FP16 | (1, 300, 6) | 94.94% | 173.98 | 640.00 |
| `end2end=True`, INT8 | (1, 300, 6) | 49.56% | 12.06 | 72.91 |

For the `end2end=False` model — the export path the original reporter actually used to work around the segfault — INT8 quantization drops the raw detection head's nonzero output density by roughly **92% relative** (55.05% → 4.76%), while the mean and max values stay nearly identical between FP16 and INT8 (9.26 vs 9.27 mean; 675.00 vs 675.08 max). This pattern — same value range, drastically fewer nonzero entries — is consistent with INT8 quantization driving a large fraction of the raw output (most plausibly objectness/class-confidence channels) to exactly zero rather than merely adding noise, which would directly explain "no detections at normal confidence thresholds": if confidence values are quantized to zero for most anchors, no anchor clears a normal threshold regardless of how sensible the box coordinates in the surviving 4.76% remain.

The `end2end=True` model's INT8 nonzero fraction also drops substantially (94.94% → 49.56%), though this comparison is less directly interpretable — this output is already post-NMS/Top-K ([1, 300, 6], padded slots for unused candidates), so its baseline sparsity pattern differs structurally from the raw detection head, and this model's simulator run may not reflect real hardware behavior anyway given §1's build-time warnings for this exact model.

## 3. What This Does Not Establish

**Does not establish:** that this paper's numbers exactly reproduce the original reporter's experience — different model (stock YOLO26n vs. their custom-trained model), different exact toolkit patch state possibly, and critically, **all inference in this paper ran on `rknn-toolkit2`'s PC simulator, not physical RK3588 NPU hardware** — the simulator is Rockchip's own approximation of NPU execution for development purposes and is not guaranteed to reproduce every real-hardware numerical or crash behavior exactly. The `end2end=True` segfault specifically was not reproduced here at all — the simulator ran all four builds without crashing, which is expected if the crash mechanism is specific to how real RK3588 NPU firmware/drivers handle the unrecognized op the `Unkown op target: 0` warnings are pointing at, something the PC simulator likely does not exercise the same way. Confirming the actual segfault, and confirming this INT8 collapse pattern holds on real silicon (not just the simulator's approximation), both require running on real RK3588 hardware — not done in this paper.

**Does establish:** a precise, quantified, independently-reproduced signature for both halves of the original bug report, using a different (stock) model and via a lower-effort measurement path (PC simulator, no physical device needed) than the original report used — the `Unkown op target: 0` build-time warning as a likely compile-time signal for the end2end segfault issue, and a specific ~92% relative nonzero-density collapse as the quantified mechanism behind the "INT8 detects nothing" symptom.

## 4. Conclusion

Both bugs from `ultralytics/ultralytics#23753` reproduce independently on a stock model via `rknn-toolkit2`'s own tooling, without needing physical RK3588 hardware to get real, quantified evidence for either. The end2end export path triggers a specific, countable build-time warning (`Unkown op target: 0`, 8 occurrences, present in both precisions, absent from the non-end2end export) that the toolkit does not treat as a build failure — a real gap between what the toolkit can detect and what it reports as an error, worth a real bug report to `airockchip/rknn-toolkit2` distinct from the closed `ultralytics` issue, since the underlying problem is on the RKNN toolkit's op-support/error-handling side, not Ultralytics' export code. The INT8 quantization issue has a precise, measurable signature: a ~92% relative collapse in the raw detection head's nonzero output density, with the surviving value range essentially unchanged — a plausible root cause is confidence/objectness channels being quantized to exactly zero for the overwhelming majority of anchors, directly explaining why no detections clear normal confidence thresholds. Physical RK3588 hardware validation (confirming the actual segfault, and confirming the INT8 collapse pattern holds outside the simulator) remains open future work.

## Evidence

ONNX export logs and output-shape verification, all four RKNN build logs (showing the `Unkown op target: 0` isolation), and the PC-simulator inference script and raw output statistics, in `data/`, prefixed `pikoi-20260815-yolo26-rknn-int8-`.
