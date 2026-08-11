# Tokens-per-Joule Across All Five nvpmodel Power Modes on Jetson Orin NX 16GB

**Platform:** Seeed Studio reComputer J4012 — NVIDIA Jetson Orin NX 16GB (module P3767-300) on a J401 carrier board
**Software stack:** JetPack 7.2, L4T R39.2.0, CUDA 13.2, TensorRT 10.16.2.10, TensorRT-Edge-LLM
**Date:** August 2026

## Abstract

A published tokens-per-joule Pareto curve exists for the Orin Nano across `nvpmodel` power modes, identifying 25W as the efficiency-optimal setting for that module. No equivalent measurement existed for the Orin NX. This paper measures Qwen3-1.7B-Instruct and Qwen3-4B-Instruct-2507 decode throughput and energy-per-token across all five `nvpmodel` modes on this board (10W, 15W, 25W, 40W, MAXN_SUPER) and finds the opposite result from the Nano: efficiency does not peak at a mid-tier mode. It gets *worse* from 10W through 25W, then improves sharply at 40W — the best efficiency point on this hardware — before falling slightly at MAXN_SUPER. The mechanism is a non-monotonic GPU configuration table: 10W and 15W run 2 active TPCs at 612MHz, 25W runs 4 TPCs at a *lower* 408MHz, and only 40W/MAXN_SUPER unlock 4 TPCs at 1173MHz. Getting this measurement required first working around a documented `nvpmodel` reliability bug — two of five mode-switch attempts silently failed and left the board at its prior mode without error, requiring a reboot-and-reverify procedure rather than trusting the command's exit status.

## 1. Method

For each of the five `nvpmodel` modes, `jetson_clocks` was run to lock clocks to that mode's ceiling, then `jetson_clocks --show` was captured to record actually-achieved CPU/GPU/EMC frequencies (not assumed from the mode name). For both Qwen3-1.7B-Instruct and Qwen3-4B-Instruct-2507 (both pre-built TensorRT-Edge-LLM engines, `--maxKVCacheCapacity 1024 --maxInputLen 512`), a 10-iteration prefill benchmark (128-token input) and a 200-iteration sustained decode benchmark (128-token past-KV length) were run via `llm_bench`, with `tegrastats --interval 300` capturing `VDD_IN` concurrently through the decode window. Energy per token was computed as mean `VDD_IN` (W) divided by measured decode tok/s.

## 2. A Reliability Bug Had to Be Worked Around First

The first sweep attempt set each mode via `nvpmodel -m <id>` and proceeded immediately to benchmarking. Post-hoc verification (`nvpmodel -q` logged after every switch) showed the 10W and 15W switches had **silently failed**: both left the board at `MAXN_SUPER`, with no error returned by the command. This is the exact failure mode already documented in this project's companion fan-curve paper — `nvpmodel -m` can report success and silently revert without a reboot. Investigating further: `nvpmodel -m 1` (10W) actually prompts an interactive confirmation (`DO YOU WANT TO REBOOT NOW? enter YES/yes to confirm:`) when the target mode requires a reboot, and a non-interactive SSH invocation has no TTY to answer it — the command fails with `NVPM ERROR: bad input!` and the mode never changes, silently, unless the caller pipes an explicit confirmation and then waits out an actual reboot.

The corrected procedure — `printf '<password>\nyes\n' | sudo -S nvpmodel -m <id>`, followed by a reboot and a live re-check of `nvpmodel -q` — successfully applied 10W and 15W. A secondary, intermittent issue surfaced across these reboots: on roughly two of five boots, the GPU's `devfreq` governor node (`/sys/class/devfreq/17000000.gpu`) was absent immediately after boot, causing `nvpmodel.service` to fail and `jetson_clocks --show` to report `Error! GPU frequency scaling not supported!`. This cleared on a subsequent reboot each time it occurred; the underlying cause (a driver-load race specific to this board's software stack) was not root-caused further, but readers attempting to reproduce clock-locked measurements on this or a similar unit should verify `/sys/class/devfreq/` contains a `.gpu` entry before trusting any `jetson_clocks` output, not just check its exit code.

Notably, not every mode transition required this treatment: 15W→25W and 25W→MAXN_SUPER both applied immediately without a reboot prompt. Only transitions into 10W and 15W from a higher mode triggered the interactive/reboot path in this sweep — worth budgeting for if reproducing this measurement, since it is not a fixed cost per mode switch.

## 3. Results

### 3.1 Achieved clocks per mode (post `jetson_clocks`, verified live)

| Mode | CPU cores online | CPU freq | GPU freq | Active TPCs |
|---|---|---|---|---|
| 10W | 4 | 1190 MHz | 612 MHz | 2 |
| 15W | 4 | 1421 MHz | 612 MHz | 2 |
| 25W | (not separately re-verified beyond GPU/TPC) | — | **408 MHz** | **4** |
| 40W | 8 | 1984 MHz | 1173 MHz | 4 |
| MAXN_SUPER | 8 | 1984 MHz | 1173 MHz | 4 |

25W's GPU configuration is the counter-intuitive one: lower clock (408 MHz) than 10W/15W (612 MHz), but double the active TPC count (4 vs 2). 40W and MAXN_SUPER share an identical GPU configuration — consistent with the companion power-ceiling paper's finding that this board's real achievable draw plateaus well below the MAXN_SUPER nameplate figure, so the two modes do not meaningfully differ under real workloads.

### 3.2 Decode throughput and energy per token

| Mode | Qwen3-1.7B tok/s | Qwen3-1.7B mean VDD_IN | Qwen3-1.7B J/token | Qwen3-4B tok/s | Qwen3-4B mean VDD_IN | Qwen3-4B J/token |
|---|---|---|---|---|---|---|
| 10W | 25.1 | 8.821 W | 0.3514 | 12.6 | 9.324 W | 0.7400 |
| 15W | 25.5 | 10.635 W | 0.4171 | 12.9 | 11.192 W | 0.8676 |
| 25W | 26.3 | 13.287 W | 0.5052 | 13.8 | 14.224 W | 1.0307 |
| 40W | 58.1 | 17.890 W | **0.3079** | 30.1 | 20.074 W | **0.6670** |
| MAXN_SUPER | 56.5 | 19.994 W | 0.3538 | 30.0 | 21.419 W | 0.7140 |

**40W is the Pareto-optimal mode for both models on this hardware** — lowest joules-per-token of any mode, and also higher throughput than every mode below it. This is the opposite shape from the published Orin Nano curve, where 25W was the efficiency sweet spot. The mechanism is directly visible in §3.1: efficiency *degrades* monotonically from 10W to 25W because CPU/EMC power keeps rising across those three modes while GPU throughput barely moves (2 TPCs @ 612MHz through 10W/15W, then 4 TPCs @ a *lower* 408MHz at 25W — a wash in effective GPU throughput, not a gain) — so each step spends more power for almost no more tok/s. Only at 40W does GPU configuration actually unlock (4 TPCs @ 1173MHz), and throughput more than doubles while power increases only ~35% over 25W, producing the efficiency jump. MAXN_SUPER draws more power than 40W for no decode throughput gain (identical GPU config, per §3.1), making it slightly less efficient than 40W on both models — consistent with the companion power-ceiling paper's finding that this board cannot actually reach the current draw the MAXN_SUPER nameplate implies under ordinary workloads.

## 4. Conclusion

The Orin Nano's published finding — that a mid-tier `nvpmodel` mode is Pareto-optimal for LLM inference efficiency — does not transfer to the Orin NX. This board's efficiency-optimal mode is 40W, and the three lower-numbered modes (10W/15W/25W) are strictly worse choices for both throughput and energy-per-token than 40W on this hardware, because NVIDIA's per-mode GPU clock/TPC table for this SoC bin does not scale GPU throughput smoothly with the named wattage — it holds GPU configuration flat or even trades clock for TPC count non-monotonically across the low tiers, only unlocking full throughput at 40W. A deployment choosing a power mode by name alone, or by extrapolating from Nano guidance, would very likely select an inferior mode on this hardware. Measuring the actual per-mode clock table (§3.1), not just the wattage label, was necessary to explain — and to trust — the throughput/efficiency numbers in §3.2.

## Evidence

Raw logs in `data/`, prefixed `orinnx-20260811-campaign1-`: per-mode `nvpmodel` set/verify logs, `jetson_clocks --show` output, prefill/decode `llm_bench` logs, and `tegrastats` captures for all 10 mode×model combinations, including the discarded pre-fix 10W/15W runs retained alongside the corrected reruns as evidence of the reliability bug in §2.
