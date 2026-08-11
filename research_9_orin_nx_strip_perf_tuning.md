# Desktop Strip and Performance Tuning on Jetson Orin NX 16GB: Reproducing the Orin Nano Result on a Different Module

**Platform:** Seeed Studio reComputer J401 — NVIDIA Jetson Orin NX 16GB (module P3767-300) on the same J401 carrier used for the Orin Nano study
**Software stack:** JetPack 7.2, L4T R39.2.0, CUDA 13.2, TensorRT 10.16.2.10, Ubuntu 24.04.4 LTS
**Date:** August 2026

> **Platform identification note.** As with the Orin Nano unit in prior papers,
> the device tree `model` string reads `NVIDIA Jetson Orin NX Engineering
> Reference Developer Kit Super`, which is generic BSP text, not a real devkit
> identifier — this is a reComputer J401 carrier. Confirmed via
> `/etc/nv_boot_control.conf`:
> ```
> TNSPEC 3767-300-0000-H.2-1-0-recomputer-orin-j401-
> ```
> the `300` variant identifying this as an Orin NX module, versus `301`/`0003`
> for the Orin Nano used in the earlier strip/tune and fan-curve papers. Both
> modules share the P3767 family and, notably, the **same fan config filename**
> (`nvfancontrol_p3767_0000.conf`) — confirming the earlier finding that this
> filename is not SKU-specific.

## Abstract

A prior study on an Orin Nano 8GB (same J401 carrier) established that stripping desktop packages, locking clocks via `jetson_clocks`, switching to `MAXN_SUPER` power mode, and applying an aggressive fan curve produced an 8.9x boot-time improvement with no measured throttling under sustained inference load. This paper reproduces that exact procedure — the same 135-package removal batch, the same systemd persistence pattern, the same `max65` fan-margin profile — on an Orin NX 16GB unit, to test whether the methodology transfers across modules in the same family rather than being an artifact of one board. It largely does, with one required correction: **`nvpmodel` mode indices are not portable across modules in the same family** — `MAXN_SUPER` is mode 2 on this Orin Nano SKU but mode 0 on this Orin NX SKU, and applying the Nano's mode number verbatim would have silently set the wrong power profile. Inference throughput on Qwen3-1.7B-Instruct came out 35% higher on prefill (memory-bandwidth- and compute-bound) and statistically unchanged on decode (2732.9 vs 2025 tok/s prefill; 63.5 vs 62.2 tok/s decode), consistent with the NX's higher clocks and larger core count but similar per-token memory-bandwidth ceiling. Sustained decode load pushed junction temperature past the Nano's 65 °C fan-trigger threshold (peak 69.6 °C at 26 W), the first live confirmation of the `max65` profile actually engaging under this study's methodology — the Nano paper's own soak never naturally reached its trigger point.

## 1. Method: What Was Reproduced Verbatim vs. What Required Re-Derivation

The original Orin Nano procedure was reconstructed from the project's raw evidence
logs (`dryrun.log`, `strip-removal.log`, `autoremove.log`), not just its published
paper prose, since the paper itself did not inline the exact package list. That
exact 135-package list was replayed against the Orin NX unmodified:

```
accountsservice colord colord-data eog evince evolution-data-server firefox fwupd
fwupd-signed gdm3 gnome-shell gnome-session-bin libreoffice-core nautilus
network-manager-gnome(auto-removed only) snapd thunderbird totem ubiquity
x11-apps xorg xserver-xorg-core xwayland yaru-theme-gtk [... full 135-package
list identical to the Nano run, see evidence/dryrun.log]
```

A dry-run (`apt-get remove --purge -s`) was run first and inspected for cascades
into `ssh`, networking, kernel, CUDA, or TensorRT packages — none occurred; the
only incidental removal was `network-manager-gnome`, the GUI applet, not core
`network-manager`. This matches the Nano run's result exactly, on a different
module and package-database state. The dry-run reported **135 to remove**, the
same count as the original — even though the NX unit's installed package set
was not derived from the exact same image build, and had accumulated updates
independently up to the time of this study.

What did **not** transfer without correction:

- **`nvpmodel` mode numbers.** The Nano paper used `nvpmodel -m 2` for
  `MAXN_SUPER`. On this NX unit, `nvpmodel -q` lists `MAXN_SUPER` as **mode 0**,
  with `10W`/`15W`/`25W`/`40W` occupying modes 1–4. The NX shipped from Seeed
  already defaulting to 40W (mode 4). Running `nvpmodel -m 2` verbatim would
  have silently selected the NX's `15W` mode instead of `MAXN_SUPER` — a
  power-mode table is per-module, not per-family, and must be read live
  (`nvpmodel -q`) rather than assumed from a prior paper on a related SKU.
- **Clock targets.** `jetson_clocks` locked this NX to 8 CPU cores at
  **1984 MHz** and GPU at **1173 MHz**, versus the Nano's 6 cores at 1728 MHz
  and GPU at 1020 MHz — expected given the NX's larger core count and higher
  rated clocks, but a number that has to be read post-lock (`jetson_clocks
  --show`), not assumed equal across modules.
- **`nvidia-cdi-refresh.service`/`nvpmodel.service` failures.** Both were
  observed in a failed state on this unit before any tuning was applied —
  `nvpmodel.service` failed even at first boot, before package stripping. The
  same two systemd drop-in overrides documented in the Nano paper
  (`Restart=on-failure`/`RestartSec=5` for the CDI refresh unit; an `ExecStart`
  override piping `yes` into `nvpmodel.sh` for the nvpmodel unit) resolved both
  on this unit without modification — this part of the fix generalizes.

What transferred unmodified:

- The `max65` fan profile's config **path and margin semantics** — same file,
  same `TMARGIN ENABLED`/`GROUP_MAX_TEMP 105` encoding documented in the
  companion fan-curve paper. The Nano's profile (full PWM once actual
  temperature reaches 65 °C, i.e. margin ≤ 40) was ported as a literal
  file copy plus one line change (`FAN_DEFAULT_PROFILE max65`), since it did
  not exist on this NX unit's stock config and had to be added rather than
  edited.
- The two systemd overrides above.
- The `jetson-clocks-autostart.service` persistence pattern
  (`Requires=nvpmodel.service`, oneshot `ExecStart=/usr/bin/jetson_clocks`,
  `RemainAfterExit=yes`) — installed and enabled identically.

## 2. Package Strip Results

| Metric | Before | After |
|---|---|---|
| Installed packages (`dpkg -l \| grep ^ii`) | ~2202 total dpkg entries | 1817 |
| Boot target | `graphical.target` | `multi-user.target` |
| `systemctl --failed` | 1 unit (`nvpmodel.service`, pre-existing) | 0 (after drop-in fixes) |

The removal itself completed in a single batch with exit code 0, matching the
original methodology's "single batch removal rather than staged increments."
Full logs: `evidence/dryrun.log`, `evidence/strip-removal.log`,
`evidence/autoremove.log`.

## 3. Clock, Power, and Fan Tuning Results

| Setting | Value |
|---|---|
| Power mode | `MAXN_SUPER` (mode 0 on this SKU) |
| CPU cores locked | 8 cores @ 1984 MHz |
| GPU locked | 1173 MHz |
| Fan profile | `max65` (custom, ported from Nano study) |
| Failed systemd units after fixes | 0 |

## 4. Inference Benchmark: Qwen3-1.7B-Instruct

Reproduced the two-stage TensorRT-Edge-LLM workflow from the qwen-orin project
scripts (`01_export_x86.sh` → `04_run_orin.sh`): ONNX export happens on an x86
host with a discrete GPU (already done for this checkpoint, reused from prior
work — not re-exported for this study), engine build and inference run natively
on-device. Built via
`cmake -DEMBEDDED_TARGET=jetson-orin -DCUDA_CTK_VERSION=13.2 -DENABLE_CUTE_DSL=ALL`,
engine via `llm_build --maxBatchSize 1 --maxInputLen 512 --maxKVCacheCapacity
1024` (SM 8.7 applies to both Orin Nano and Orin NX, so build flags were
unchanged from the Nano study).

One build-time correction was required that the Nano paper did not document:
`EDGELLM_PLUGIN_PATH` (needed for the framework's custom ONNX ops) is set by
the build script into `~/.bashrc`, which a non-interactive SSH command does
not source — the first engine-build attempt failed with `ERROR: Cannot open
plugin library` until the variable was exported explicitly in the same shell
invocation.

Correctness check: inference on the same prompt used in the Nano paper ("What
is the capital of the United States?") produced a correct, well-formed answer
— `evidence/benchmark/output_qwen.json`.

| Metric | Orin Nano (prior study) | Orin NX (this study) | Delta |
|---|---|---|---|
| Prefill (128-token context, tok/s) | 2025 | 2732.9 | +35% |
| Decode (CUDA-graph, steady state, tok/s) | 62.2 | 63.5 (10-iter) / 63.6 (500-iter) | ~flat |
| Peak power during sustained decode | not measured in Nano paper | 26007 mW (VDD_IN) | — |
| Peak junction temp during sustained decode | ~59 °C (never triggered 65 °C fan) | 69.6 °C (fan triggered) | — |

The prefill gain is consistent with the NX's larger core count and higher
clocks (1984 MHz vs 1728 MHz CPU, 1173 MHz vs 1020 MHz GPU) — prefill is
compute-bound and scales with SM throughput. Decode is memory-bandwidth-bound
per-token and stayed essentially flat, which is the expected signature of a
bandwidth ceiling rather than a compute one; this is not a like-for-like
memory-bandwidth comparison (module RAM sizes differ: 16GB LPDDR5 vs the
Nano's 8GB), and the decode figures should not be read as a bandwidth
measurement, only as a throughput one.

Notably, this study's sustained 500-iteration decode run is the first time
across this project's papers that the `max65` fan curve was observed
triggering live — the Nano study's own soak test held at ~59 °C and never
naturally reached its 65 °C threshold. On this NX unit under the same decode
workload, junction temperature peaked at 69.6 °C, crossing the trigger and
confirming the curve activates as designed under sufficient sustained load
(visible in `evidence/benchmark/tegrastats_decode500.log` as a power/thermal
drop partway through the capture, coincident with the fan engaging).

## 5. Conclusion

The strip-and-tune methodology transfers across modules in the same P3767
family with two required, not optional, corrections: re-reading `nvpmodel -q`
for the correct `MAXN_SUPER` mode index rather than reusing a prior board's
number, and re-reading achieved clock values post-lock rather than assuming
parity. Everything else — the package list, the fan-curve porting approach,
the two systemd drop-in fixes, and the TensorRT-Edge-LLM build/benchmark
pipeline — reproduced without modification. The inference throughput
comparison is consistent with what the hardware specs predict (compute-bound
phase scales with the NX's larger core count and higher clocks; bandwidth-
bound phase does not), which is itself a useful sanity check that the tuning
was applied correctly on both boards — a tuning bug that left clocks at
default would likely have shown up as a larger, not smaller, prefill gap.

## Evidence

Raw logs for every step in this paper are in `data/`, prefixed `orinnx-20260810-`:
- `strip-dryrun.log`, `strip-removal.log`, `strip-autoremove.log` — package removal
- `system-state-after.log` — post-tuning `nvpmodel`/`jetson_clocks`/systemd snapshot
- `engine-build.log` — corrected TensorRT engine build (with the `EDGELLM_PLUGIN_PATH` fix)
- `qwen3-1.7b-output.json` — inference correctness check output
- `prefill-decode10-bench.log` — 10-iteration prefill/decode benchmark
- `decode500-bench.log`, `tegrastats-decode500.csv` — sustained 500-iteration decode run with concurrent power/thermal capture
