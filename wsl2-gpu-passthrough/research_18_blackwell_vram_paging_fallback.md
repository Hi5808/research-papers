# CUDA Allocations Past Physical VRAM Silently Succeed on Windows and WSL2 Alike on This RTX 5090 — Not a WSL2-Specific Bug

**Platform:** Windows 11 desktop, RTX 5090 (32607 MiB VRAM, driver 610.62), CUDA UMD 13.3, Blackwell (sm_120)
**Compared environments:** native Windows Python 3.12.10 + PyTorch `2.13.0+cu130`, vs. WSL2 Ubuntu 24.04.1 LTS + identical PyTorch build
**Date:** August 2026

## Abstract

This paper set out to reproduce and quantify a live, cross-filed upstream bug (`microsoft/WSL#40401`, with linked issues in `pytorch/pytorch`, `vllm-project/vllm`, and `sgl-project/sglang`) reporting ~16GiB of CUDA driver context overhead on Blackwell (sm_120) GPUs under WSL2, invisible to `torch.cuda.mem_get_info()`, causing real-world OOM in inference engines despite reported free memory. Part of that bug reproduced immediately on this RTX 5090: a `PROT_NONE` virtual memory reservation exceeding 100GB appears the instant `torch.cuda.init()` runs, invisible to both `mem_get_info()` and `nvidia-smi`. But the practical follow-up question — does this reservation actually cost usable VRAM, and is WSL2 worse than native Windows for it — produced a different and more interesting answer than expected. Allocating sequential 3GB blocks past the point where `nvidia-smi` shows VRAM fully exhausted (~31.8GB used) continued succeeding without error on **both** native Windows and WSL2, identically, up to 45GB total (140% of physical VRAM) — with every block's data independently correct on readback (no corruption, no aliasing). This is Windows' long-standing GPU memory paging fallback (allocations beyond VRAM transparently backed by system RAM), which WSL2 simply inherits via its GPU-PV passthrough layer rather than exhibiting independently. The original bug report's real-world OOM is not explained by this mechanism and remains a distinct, unresolved issue — likely specific to the large-scale multi-engine fragmentation pattern (many concurrent Mamba-state-cache allocations) that this paper's simpler sequential-block test does not reproduce.

## 1. Confirming the Reported Hidden Reservation

Running the minimal repro from the upstream bug thread — `torch.cuda.init(); x = torch.zeros(10, device="cuda:0")` — on this RTX 5090 under WSL2 immediately produces:

```
=== large anon PROT_NONE mappings in /proc/self/maps (>1GB) ===
111.92GB  204e00000-1e00000000 ---p 00000000 00:00 0
8.06GB  7740b4000000-7742b8000000 ---p 00000000 00:00 0
```

on a card with 32607 MiB of physical VRAM — a reservation over 3x the card's actual capacity. `torch.cuda.mem_get_info()` reported only a 0.12GB drop, and `nvidia-smi` showed 1279 MiB used, both effectively blind to this. This independently confirms the mechanism reported in the upstream thread on different (consumer, 32GB) hardware than the original reporters' 96GB RTX PRO 6000 and 12GB RTX 5070 cards, at the identical driver version (610.62) as one of those reports.

## 2. Does the Reservation Actually Cost Usable VRAM? Single-Block Test: No

A sequential single-tensor allocation test — allocate a tensor, verify, free, repeat at increasing size — reached **30GB** of single-contiguous-block allocation on this 32GB card with no OOM and no discrepancy between `mem_get_info()`'s predictions and actual outcomes, at every step, identically on native Windows and WSL2 (both reached exactly 30GB before the test's final step). The 111GB virtual reservation from §1, being `PROT_NONE` (unbacked address space, not committed memory), does not by itself reduce real usable capacity for straightforward allocation patterns.

## 3. Multi-Block Test: The Actual, More Interesting Finding

A second test held 15 sequential 3GB blocks *simultaneously* (not freed between allocations, closer to a real KV-cache pool's access pattern) and queried `nvidia-smi` after each. Results, identical on both native Windows and WSL2:

| Block # | Cumulative claimed | `nvidia-smi` memory.used |
|---|---|---|
| 9 | 27.0 GB | 28689–28751 MiB |
| 10 | 30.0 GB | 31761–31823 MiB (physical ceiling reached) |
| 11 | 33.0 GB | 31772–31825 MiB (**unchanged** — no more physical VRAM consumed) |
| 15 | 45.0 GB | 31823–31825 MiB (still unchanged) |

`nvidia-smi` correctly reports physical VRAM capping at ~31.8GB from block 10 onward — it is not lying. But **every allocation from block 11 through 15 still succeeds**, and a readback verification pass (each block filled with a unique value 1.0 through 15.0 before the next was allocated) confirmed **all 15 blocks held correct, distinct, uncorrupted data** — 45GB of apparently-valid, independently-addressable tensor data on a 32GB card, roughly 140% of physical capacity. This is not silent corruption or address aliasing; it is genuine oversubscription with a real (if unidentified in this test) backing store — almost certainly system RAM, consistent with another commenter's independent finding on the same upstream bug thread of `vmmemWSL`'s Windows-side commit charge ballooning far beyond what guest-visible GPU memory usage would predict.

## 4. This Is Not a WSL2 Bug — It's Windows, and WSL2 Just Inherits It

The critical result: **native Windows and WSL2 produced identical numbers at every step of both tests**, down to allocation-by-allocation `nvidia-smi` readings within single-digit MiB of each other. This means the paging-past-physical-VRAM behavior is not something WSL2's GPU-PV virtualization layer introduces — it is standard Windows WDDM driver behavior (a long-documented "shared GPU memory" / system-memory fallback path that Windows' graphics driver model has supported for years, originally for display/compositor workloads, now evidently also engaged by CUDA allocations on this driver/architecture combination), and WSL2 simply passes it through unchanged rather than exhibiting an independent WSL2-specific quirk. A reader coming from native Linux CUDA — where exceeding physical VRAM normally raises a clean, immediate OOM rather than silently degrading into host-RAM-backed allocations — would find both Windows environments behave unexpectedly the same way, not find WSL2 specifically worse than "real" Windows.

## 5. What This Does Not Explain

This mechanism does not explain the upstream bug report's actual real-world failure (`torch.OutOfMemoryError` in vLLM/SGLang loading Mamba-hybrid models despite 50+GiB reported free on a 96GB card) — if anything, this paper's data suggests Windows/WSL2 should tend toward *silent slowdown via host-RAM paging* rather than a hard OOM in that scenario, the opposite of what was reported. The likely explanation, not tested here: the original bug's allocation pattern (many concurrent, variably-sized Mamba state-cache blocks, not this paper's simple sequential same-size blocks) hits allocator fragmentation against the huge `PROT_NONE` virtual reservation from §1 in a way this paper's test does not reproduce — a genuine remaining gap, worth a follow-up test using an actual fragmenting, variable-size, concurrent allocation pattern rather than this paper's simpler sequential probe.

This paper also did not measure the performance cost of the host-RAM-backed blocks in §3 — accessing "GPU" memory that's actually resident in system RAM over the paging fallback path would be expected to be dramatically slower than real VRAM access, but no throughput test was run against the oversubscribed blocks to confirm or quantify this.

## 6. Conclusion

The specific ~16GiB-hidden-overhead phenomenon reported in `microsoft/WSL#40401` reproduces on this RTX 5090 (a ~112GB virtual reservation, even larger relative to this card's 32GB capacity than the original report's ratio) — but does not, in the two allocation patterns tested here, translate into WSL2-specific practical VRAM loss: single-block and multi-block allocation ceilings were identical between native Windows and WSL2 down to near-single-digit-MiB precision, both silently oversubscribing physical VRAM by 40%+ via what is very likely Windows' own GPU memory paging fallback rather than anything specific to WSL2's virtualization layer. The practically useful finding for anyone deploying LLM inference on a Windows+WSL2 Blackwell machine: **WSL2 is not the worse-behaved of the two environments here** — whatever risk this paging fallback poses (unmeasured performance cliff when exceeding physical VRAM, and the original bug report's unexplained real OOM under a different, fragmenting allocation pattern) applies equally to native Windows, not as a WSL2-specific cost as the upstream bug's WSL-team-targeted filing might suggest to a reader skimming only the issue title.

## Evidence

Three benchmark scripts (`wsl_minimal_repro.py`, `wsl_alloc_ceiling.py`, `wsl_multi_block2.py`) and all raw run logs (native + WSL2, each script) in `data/`, prefixed `pikoi-20260815-blackwell-vram-paging-`.
