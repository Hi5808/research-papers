# Ollama's Install Script Falsely Warns of No GPU Support on JetPack 7.2 — The Binary's Own CUDA Detection Works Fine

**Platform:** Seeed Studio reComputer J4012 — NVIDIA Jetson Orin NX 16GB (module P3767-300) on a J401 carrier board
**Software stack:** JetPack 7.2, L4T R39.2.0, CUDA 13.2
**Ollama version tested:** 0.32.13 (generic `ollama-linux-arm64` build)
**Date:** August 2026

## Abstract

Public forum reports describe Ollama and similar LLM-runtime installers silently falling back to CPU-only inference on JetPack 7.2 Jetson hardware, with unreliable workarounds (`OLLAMA_IGPU_ENABLE`, manual library paths) needed to force GPU use. This paper traces the actual root cause in Ollama's official `install.sh`: its JetPack-version detection only recognizes L4T `R36` (JetPack 6) and `R35` (JetPack 5) in `/etc/nv_tegra_release`, printing `"Unsupported JetPack version detected. GPU may not be supported"` for this board's `R39` (JetPack 7.2) and skipping a JetPack-specific package download branch entirely. This looks, from the warning alone, exactly like the forum reports' silent-CPU-fallback failure mode. It is not. Manually extracting Ollama's plain generic `linux-arm64` build (bypassing the installer script's package-selection logic, but not its underlying binary) and running real inference produced **`GR3D_FREQ` at 98%, `ollama ps` reporting `100% GPU`**, and correct GPU-accelerated generation — the binary's own internal CUDA runtime selection (trying a `cuda_v12` build first, cleanly skipping it because Orin's compute capability 8.7 isn't in that build's compiled architecture list, then falling through to a `cuda_v13` build that does support it) works correctly on JetPack 7.2 independent of the install script's stale detection logic. The alarming warning is real and worth fixing upstream, but it does not describe an actual functional failure on this board's specific setup.

## 1. The Install Script's JetPack Detection Is Genuinely Stale

`install.sh`'s JetPack-specific logic:

```bash
if [ -f /etc/nv_tegra_release ] ; then
    if grep R36 /etc/nv_tegra_release > /dev/null ; then
        download_and_extract "https://ollama.com/download" "$OLLAMA_INSTALL_DIR" "ollama-linux-${ARCH}-jetpack6"
    elif grep R35 /etc/nv_tegra_release > /dev/null ; then
        download_and_extract "https://ollama.com/download" "$OLLAMA_INSTALL_DIR" "ollama-linux-${ARCH}-jetpack5"
    else
        warning "Unsupported JetPack version detected.  GPU may not be supported"
    fi
fi
```

This board's `/etc/nv_tegra_release` begins `# R39 (release), REVISION: 2.0, ...` — JetPack 7.2's L4T major version. `R39` matches neither `R36` nor `R35`, so this branch's `else` fires unconditionally for every JetPack 7.x device, regardless of whether that device's GPU is actually supported by Ollama's underlying inference libraries. The comment above this block ("Check for NVIDIA JetPack systems with additional downloads") indicates this branch exists to fetch *extra*, JetPack-specific packages on top of the base generic build that the script already downloads earlier for every platform — meaning even when this branch does nothing, the base `ollama-linux-${ARCH}` package the script installs regardless of JetPack detection is still present and, per §2 below, already GPU-capable on its own for this board.

## 2. The Generic Build Already Handles This Board Correctly

Extracting the plain `ollama-linux-arm64.tar.zst` package (no JetPack-specific variant, the same base package the install script downloads for every Linux/aarch64 target before ever reaching the JetPack-detection branch above) and starting `ollama serve` produces:

```
level=INFO source=llama_server.go:299 msg="skipping CUDA device — compute capability not in compiled architectures" device=Orin cc=870 archs="[500 520 600 610 700 750 800 860 890 900 1000 1200]" libDirs="[.../lib/ollama /.../lib/ollama/cuda_v12]"
level=INFO source=types.go:32 msg="inference compute" id=0 filter_id=0 library=CUDA compute=8.7 name=CUDA0 description=Orin libdirs=ollama,cuda_v13 driver=13.2 pci_id=0000:00:00.0 type=iGPU total="15.2 GiB" available="13.5 GiB"
```

The package ships two bundled CUDA runtime builds, `cuda_v12` and `cuda_v13`. The `cuda_v12` build's compiled architecture list (`[500 520 600 610 700 750 800 860 890 900 1000 1200]`, i.e. compute capabilities 5.0 through 12.0) includes `860` (desktop/datacenter Ampere) but not `870` (Jetson Orin's specific Ampere variant) — a clean, correctly-logged skip, not a crash or silent failure. The `cuda_v13` build's architecture list does include Orin's `cc=8.7`, is selected instead, and correctly reports the iGPU (`description=Orin`, `driver=13.2` — matching this board's actual CUDA 13.2 install) with 13.5 GiB available for inference.

## 3. Confirmed With Real Inference, Not Just Log Claims

Pulling a small model (`qwen2.5:0.5b`) and running generation while capturing `tegrastats` at 300ms intervals showed `GR3D_FREQ` (GPU utilization) reaching **98%** during generation, with CPU cores at or near 0% utilization in the same samples — VDD_IN power draw climbing from an idle ~7.5W to 16.9W, consistent with genuine GPU compute activity rather than CPU-bound work. `ollama ps` after the run independently reported:

```
NAME            SIZE      PROCESSOR    CONTEXT
qwen2.5:0.5b    481 MB    100% GPU     4096
```

Both signals — hardware telemetry and the application's own accounting — agree: this is real, complete GPU-accelerated inference on JetPack 7.2, not a silent fallback.

## 4. What This Does and Doesn't Resolve

**Does not establish:** that every report of JetPack 7.x GPU fallback in public forums is mistaken. Those reports may involve different installer versions, different models (larger models exceeding the reported 13.5 GiB iGPU-available figure could plausibly hit a different failure mode), different JetPack 7.0/7.1 point releases, or genuinely different root causes (e.g. the env-var workarounds `OLLAMA_IGPU_ENABLE`/`GGML_BACKEND_PATH` mentioned in those threads suggest some setups needed manual intervention this board's setup did not require) — this paper tested one specific board, one specific Ollama version, one small model, and did not attempt to reproduce a failure, only to check whether one occurs by default.

**Does establish:** for this exact board (JetPack 7.2, L4T R39.2.0, CUDA 13.2) and this Ollama version, the install script's `"Unsupported JetPack version detected"` warning is a false alarm as far as actual GPU functionality goes — a user seeing that warning and concluding their Jetson Orin NX can't run GPU-accelerated Ollama inference on JetPack 7.2 would be wrong, at least for the tested model size and installation method (manual generic-package extraction, bypassing only the script's package-selection branch, not its underlying binaries).

## 5. Conclusion

Ollama's install script has a real, fixable staleness bug — its JetPack-version detection predates JetPack 7's release and doesn't recognize L4T R39 — but the warning it prints overstates the actual impact for this board. The inference binary's own runtime CUDA-architecture selection (trying `cuda_v12`, cleanly falling through to `cuda_v13` on failure) already handles Jetson Orin's compute capability correctly, and real GPU-accelerated inference works out of the box once the generic package is in place, independent of whatever the install script's shell-level JetPack detection decides. This is a case where reading the actual failure mode (a stale `grep` pattern in a shell script) rather than trusting the alarming warning message it produces avoided writing off a working GPU path as broken. The install script itself remains worth flagging upstream as a documentation/UX bug, distinct from any functional GPU-support gap.

## Evidence

`install.sh` source (relevant JetPack-detection excerpt above), full `ollama serve` startup log showing the `cuda_v12`→`cuda_v13` fallback, `tegrastats` capture during inference (`GR3D_FREQ` samples reaching 98%), and `ollama ps` output, in `data/`, prefixed `orinnx-20260815-ollama-jetpack7-`.
