# Desktop Strip and Performance Tuning on Jetson Orin NX 16GB: Reproducing the Orin Nano Result on a Different Module

**Platform:** Seeed Studio reComputer J4012 — NVIDIA Jetson Orin NX 16GB (module P3767-300) on a J401 carrier board, same carrier family used for the Orin Nano study. The reComputer J4012 designation matters for this paper's thermal and power-delivery findings: it identifies the specific enclosure and cooling solution these results were measured in, which is not interchangeable with an NVIDIA reference devkit or a different reComputer SKU on the same carrier.
**Software stack:** JetPack 7.2, L4T R39.2.0, CUDA 13.2, TensorRT 10.16.2.10, Ubuntu 24.04.4 LTS
**Date:** August 2026

> **Platform identification note.** As with the Orin Nano unit in prior papers,
> the device tree `model` string reads `NVIDIA Jetson Orin NX Engineering
> Reference Developer Kit Super`, which is generic BSP text, not a real devkit
> identifier — this is a Seeed Studio **reComputer J4012** on a J401 carrier
> board (the carrier designation and the reComputer product SKU are distinct;
> J401 is the carrier, J4012 is the enclosed product this paper's thermal and
> power results were actually measured in). Confirmed via
> `/etc/nv_boot_control.conf`:
> ```
> TNSPEC 3767-300-0000-H.2-1-0-recomputer-orin-j401-
> ```
> the `300` variant identifying this as an Orin NX module, versus `301`/`0003`
> for the Orin Nano used in the earlier strip/tune and fan-curve papers. Both
> modules share the P3767 family and, notably, the **same fan config filename**
> (`nvfancontrol_p3767_0000.conf`) — confirming the earlier finding that this
> filename is not SKU-specific. The reComputer J4012 enclosure (case, stock
> heatsink+fan, airflow path) is what §7's power-ceiling and external-cooling
> results are specific to — they do not transfer to a bare module, an NVIDIA
> reference devkit, or a different reComputer SKU on the same J401 carrier.

## Abstract

A prior study on an Orin Nano 8GB (same J401 carrier) established that stripping desktop packages, locking clocks via `jetson_clocks`, switching to `MAXN_SUPER` power mode, and applying an aggressive fan curve produced an 8.9x boot-time improvement with no measured throttling under sustained inference load. This paper reproduces that exact procedure — the same 135-package removal batch, the same systemd persistence pattern, the same `max65` fan-margin profile — on an Orin NX 16GB unit, to test whether the methodology transfers across modules in the same family rather than being an artifact of one board. It largely does, with one required correction: **`nvpmodel` mode indices are not portable across modules in the same family** — `MAXN_SUPER` is mode 2 on this Orin Nano SKU but mode 0 on this Orin NX SKU, and applying the Nano's mode number verbatim would have silently set the wrong power profile. Inference throughput on Qwen3-1.7B-Instruct came out 35% higher on prefill (memory-bandwidth- and compute-bound) and statistically unchanged on decode (2732.9 vs 2025 tok/s prefill; 63.5 vs 62.2 tok/s decode), consistent with the NX's higher clocks and larger core count but similar per-token memory-bandwidth ceiling. Sustained decode load pushed junction temperature past the Nano's 65 °C fan-trigger threshold (peak 69.6 °C at 26 W) — **but §6 found this ran on a stale fan profile**, because `nvfancontrol.service` does not hot-reload its config and had not been restarted since the `max65` profile was written. A follow-up combined CPU+GPU stress test caught this, and after restarting the service the curve was confirmed live and working (PWM 0→255 at the 65 °C crossing). That same follow-up also found the module's true power ceiling is higher than a naive compute-only stress test suggests: it peaked at only 17.7W against the LLM benchmark's 26W, reproducing a finding from the companion Nano fan-curve paper that SM occupancy alone does not reach the module's real power envelope — tensor-core and DRAM-bandwidth-heavy workloads do. See §6 for the full correction.

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
| Peak junction temp during sustained decode | ~59 °C (never triggered 65 °C fan) | 69.6 °C (see §6 correction: fan curve was stale at time of this run) | — |

The prefill gain is consistent with the NX's larger core count and higher
clocks (1984 MHz vs 1728 MHz CPU, 1173 MHz vs 1020 MHz GPU) — prefill is
compute-bound and scales with SM throughput. Decode is memory-bandwidth-bound
per-token and stayed essentially flat, which is the expected signature of a
bandwidth ceiling rather than a compute one; this is not a like-for-like
memory-bandwidth comparison (module RAM sizes differ: 16GB LPDDR5 vs the
Nano's 8GB), and the decode figures should not be read as a bandwidth
measurement, only as a throughput one.

This study's sustained 500-iteration decode run pushed junction temperature to
69.6 °C, above the Nano study's 65 °C fan-trigger point (whose own soak test
held at ~59 °C and never naturally reached that threshold) — but **this run
did not actually exercise the `max65` profile**. As found in §6, the
`nvfancontrol` service was still running on the stock `quiet` profile at this
point in the study; the custom profile had been written to disk but the
service had not yet been restarted to pick it up. The live confirmation that
`max65` engages as designed came later, from the dedicated follow-up stress
test in §6, not from this benchmark run.

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

## 6. Correction: `nvfancontrol` Does Not Hot-Reload, and a Naive Compute Stress Test Undershoots the Module's Real Power Ceiling

Two issues surfaced after this paper's initial publication, from a follow-up
combined CPU+GPU stress test run to establish the board's actual peak power
draw (the LLM benchmark's 26W had been assumed to be near the ceiling, but
was never verified against a dedicated stress workload).

**The fan curve appeared not to be working.** A first combined-stress run
(`stress --cpu 8` for 90s + the same `gpu_stress.cu` FMA-loop kernel used in
the companion Nano fan-curve paper, both concurrent, `tegrastats` at 500ms)
pushed junction temperature to a peak of **74.5 °C** — well past the `max65`
profile's 65 °C trigger — with no sign of the fan curve having engaged.

Root cause: `nvfancontrol.service` reads its config **once, at service
start**, and does not hot-reload on file changes. In §3's procedure, the
service had already started at boot (20:29:57) using the stock `quiet`
profile; the `max65` profile was written into the config file afterward
(20:34:23) but the running service never picked it up. `systemctl restart
nvfancontrol.service` is a required step after installing a custom profile —
editing the file alone is not sufficient, and nothing in the service's status
output distinguishes "config edited, not yet applied" from "config applied."
This is a mechanical gap in this paper's own procedure, not a property of the
`max65` profile itself.

After restarting the service, live `pwm1` sysfs polling under the same
combined stress workload confirmed the profile drives the fan exactly as
designed: PWM jumped from 0 to **255 (full)** the moment junction temperature
crossed into the trigger band (~73.6 °C, margin ≤ 40 relative to
`GROUP_MAX_TEMP 105`), then settled to **217 (≈85%)**, holding sustained
temperature flat around 73 °C rather than continuing to climb — direct,
live confirmation of the margin-based curve engaging correctly
(`data/orinnx-20260810-tegrastats-stress-postfanfix.csv`).

**The naive compute stress test undershoots the module's real power ceiling.**
Even with the fan curve confirmed working, the combined `stress`+`gpu_stress.cu`
workload peaked at only **17.7W** `VDD_IN` — lower than the LLM decode
benchmark's 26W from §4, despite pinning all 8 CPU cores and the GPU at
effectively 100% occupancy for 90 seconds. This reproduces, on this module, a
finding already reported in the companion fan-curve paper on the Nano: *"the
module's power envelope is reachable only by a load that exercises tensor
cores and DRAM, not by SM occupancy alone."* `gpu_stress.cu`'s kernel does
simple FMA arithmetic entirely in registers — no tensor-core instructions, no
meaningful DRAM traffic — so despite 99% reported GPU occupancy it does not
exercise the power-hungry paths that real inference workloads do. The LLM
benchmark's 26W, not the stress test's 17.7W, is the better available estimate
of this board's actual peak draw under this study; a true power ceiling would
require a workload that combines CPU saturation with tensor-core-heavy GPU
work (e.g. concurrent LLM decode + `stress`), which was not attempted here.

Corrected data: `tegrastats-stress-precorrection.csv` (pre-fix run, fan not
engaged) and `tegrastats-stress-postfanfix.csv` (post-fix run, fan confirmed
engaging) are both retained in `data/` for comparison rather than overwritten,
since the delta between them is itself evidence of the failure mode.

## 7. Addendum: The `max65` Curve Had a Hunting Bug, and the Fan's Real Ceiling Is Not What the Config Declares

Two further issues surfaced from continued testing after §6's initial fix.

**The `max65` profile as first written caused audible fan hunting.** It
encoded a single-degree cliff — full PWM (255) at margin ≤ 40 (actual ≥ 65 °C),
PWM 0 at margin ≥ 41 (actual ≤ 64 °C) — with nothing in between, unlike
NVIDIA's stock profiles, which ramp gradually across wide margin bands
specifically to avoid this. A live poll at idle with junction temperature
essentially flat (66.4–67.2 °C, never crossing back below 65 °C) showed PWM
cycling 0 → 64 → 255 → 242 → 0 rather than holding steady — the close-loop
RPM governor was chasing a target that itself swung from fully off to fully
on across a single degree of thermal noise, including noise induced by the
fan's own cooling effect. The profile was rewritten with a graduated ramp
spread across a ~30 °C band (255 at margin ≤ 30, stepping down through 217,
140, 70, to 0 at margin ≥ 61) mirroring the stock profile's shape. After the
rewrite, PWM held at a constant value (118) across the same 3 °C of thermal
drift that previously caused full-scale cycling, and stepped smoothly (118 →
171 → 214) as sustained load pushed temperature up through a 15 °C range —
hunting eliminated.

**The fan's declared 255 (100%) ceiling is not achievable.** With
`nvfancontrol.service` stopped entirely (`systemctl is-active` confirmed
`inactive`, no process running) and PWM written directly to the kernel sysfs
interface (`/sys/class/hwmon/hwmon1/pwm1`), commanding 255 did not hold: it
decayed unassisted to 187 after 1 second, then to 88 after 2 more, and
stayed there. Nothing else was writing to the file. Low and mid values (0,
128) held exactly as written. This means the fan's real sustained maximum is
approximately **88/255 (≈34% duty)**, not the 100% the config's `RPM 6000`
declares — a hardware- or firmware-level limit sitting below the
`nvfancontrol` layer entirely, independent of any curve authored in the
config file. This directly explains why the graduated `max65` profile's
close-loop control plateaued around PWM 214–217 rather than reaching 255
during the sustained combined-stress tests in §6 and below: it was correctly
converging toward the fan's actual achievable ceiling, not failing to reach a
software-declared one. This is a sharper, board-local instance of a finding
already reported at the config level in the companion Nano fan-curve paper —
that the declared ceiling is a config value rather than a hardware limit —
now confirmed by bypassing the config layer entirely.

**A dedicated attempt to reach the nominal 40W power mode's namesake wattage
did not get there, across five independent workload combinations.**

| Workload | Peak `VDD_IN` | Peak Tj |
|---|---|---|
| LLM decode benchmark alone (§4) | 26.0 W | 69.6 °C |
| CPU+GPU compute stress alone (§6, `gpu_stress.cu` FMA kernel) | 17.7 W | 74.5 °C |
| CPU stress + single 1.7B prefill loop | 32.8 W | 83 °C |
| CPU stress + GPU burn kernel + single 1.7B prefill loop | 32.6 W | 83.1 °C |
| CPU stress + 3 concurrent 1.7B prefill loops + burn kernel | 29.4 W | — |
| CPU stress + 1.7B prefill loop + 1.7B decode loop | 33.0 W | 84.5 °C |
| CPU stress + 4B prefill loop + 4B decode loop | 32.6 W | 82.9 °C |
| 4B GPU load first, then CPU cores staged one at a time (20s hold) | 34.2 W | 80.7 °C |
| 4B GPU load first, then CPU cores staged one at a time (90s hold) | 34.3 W | 87.3 °C |
| CPU load first (all 8 cores), then 2 concurrent prefill + 2 decode streams slammed on | crashed (SIGABRT), not comparable | — |
| CPU load first (all 8 cores), then single prefill + decode stream slammed on | 32.5 W | 74.3 °C |
| Staged GPU-first loading, **with external fan** supplementing stock cooling | 34.07 W | 76.2 °C |

Every combination attempted — from a pure compute kernel to a 4B-parameter
model under combined CPU and tensor-core load — converged on the same
**29–33 W band**, regardless of how much heavier the compute got. Notably,
adding *more* concurrent GPU work (three prefill streams instead of one)
made peak power slightly *worse* (29.4 W vs. 32.8–33.0 W for a single
stream), and moving from a 1.7B to a 4B model changed nothing (32.6 W either
way) — this ceiling behaves like a board-level power limit rather than a
workload-scaling one.

This is not unique to this unit. An NVIDIA developer-forums thread on the
same class of hardware (Orin NX 16GB devkit, Super Mode) reports the
identical shape of result: full 8-core CPU load at 1984 MHz plus 99% GPU
utilization peaked at 32–33 W, matching this study almost exactly. An NVIDIA
staff response in that thread states directly: *"You cannot push over 40W.
That will trigger throttling and even give you lower power consumption,"*
and recommends staggering per-core CPU load rather than saturating all cores
simultaneously — which, if accurate, would also explain why this study's
three-concurrent-stream attempt performed *worse* than a single stream: an
all-at-once load pattern may itself be the reason 40W isn't reached, not
insufficient workload weight. A separate response in the same thread notes
that some Orin NX carrier boards lack the high-voltage rail and thermal
design needed for genuine 40W sustained operation even when JetPack reports
Super Mode as active — a carrier-board hardware constraint this study cannot
distinguish from a firmware-level one without access to NVIDIA's internal
power-delivery documentation for the J401 carrier specifically.
[Source: NVIDIA Developer Forums, "Orin NX 16GB on DevKit Super Mode power
loading can't up to 40W"](https://forums.developer.nvidia.com/t/orin-nx-16gb-on-devkit-super-mode-power-loading-cant-up-to-40w/324604)

**A follow-up test applied the forum thread's own recommendation directly:
start GPU load first, then add CPU cores one at a time rather than all at
once.** GPU inference load (concurrent 4B prefill and decode loops) was
started and allowed to stabilize (12s) before CPU cores were added
individually via separate single-core `stress` processes, staggered 6-8s
apart, holding all 8 cores + GPU load together afterward. This produced a
real, repeatable improvement — **34.2 W** in a first run (20s hold at full
load) and **34.3 W** in a second, longer run (90s hold) — a genuine gain
over the ~33 W ceiling from simultaneous-load tests, but still 5.7-5.8 W
short of 40 W. Junction temperature reached **87.25 °C** during the extended
hold, noticeably hotter than any prior test in this study and approaching
territory where thermal throttling becomes a plausible explanation in its
own right for why power did not climb further — consistent with the forum
thread's own claim that pushing harder triggers throttling rather than
higher sustained draw.

**Reversing the order made it worse, not better.** Two further tests loaded
CPU first (all 8 `stress` cores at once, stabilized for 10s) and then added
GPU load on top, testing whether ordering itself — not just gradualness —
mattered. A first attempt slamming two concurrent prefill and two concurrent
decode streams onto the pre-loaded CPU crashed four of the four `llm_bench`
processes outright (`SIGABRT`) — running multiple concurrent contexts
against the same engine file is not reliably stable, a caveat for anyone
attempting a similar multi-stream test. A corrected retry with a single
prefill and single decode stream avoided the crash but only reached
**32.5 W** — lower than the GPU-first staged approach's 34.2-34.3 W, and
close to the ~33 W ceiling from the naive simultaneous-load tests earlier in
this section. Abrupt loading underperforms gradual staging regardless of
which resource (CPU or GPU) is added second; the direction of the ordering
matters less than whether the added load is introduced gradually or all at
once.

The staged GPU-first, CPU-cores-added-one-at-a-time technique is the best
result obtained across eight workload/ordering combinations tested in this
study (17.7-34.3 W total range), but did not close the remaining gap to the
nameplate figure.

**A final test isolated whether thermal throttling explains the ceiling, and
the answer is no.** The reComputer J4012's stock cooling was supplemented
with an external fan blowing directly on the enclosure, and the winning
staged-loading test (§7, GPU load first, CPU cores added one at a time, 90s
hold) was repeated identically under this added cooling. Peak junction
temperature dropped substantially — **76.2 °C vs. 87.25 °C** for the
otherwise-identical no-external-fan run, an 11 °C reduction — but peak power
draw did not move: **34.07 W**, statistically indistinguishable from the
34.2-34.3 W measured without external cooling. If the ~34 W plateau were a
thermal-throttling response, removing 11 °C of headroom should have let the
board draw measurably more power before hitting its trigger; it did not.
This rules out thermal throttling as this study's explanation for the power
ceiling and points instead toward a **board-level power-delivery limit**
(VRM or input-rail current cap specific to the reComputer J4012's design) as
the more likely cause — distinct from, and better supported by evidence
than, the thermal-throttling explanation offered in the NVIDIA forum thread
cited earlier in this section, which was not itself tested under controlled
cooling. This finding is specific to the J4012 enclosure measured here; it
does not establish whether the same ceiling exists on a bare module or a
different carrier's power-delivery design.

Readers should treat 34.3 W as this board's practical power ceiling under
the workloads and staging technique available to this study, not 40 W — the
mode name describes an allowed cap under `MAXN_SUPER`, not a draw this study
was able to demonstrate or reliably reach, even applying NVIDIA's own
documented guidance for approaching it, and not a draw limited by heat
under this study's cooling conditions.

## Evidence

Raw logs for every step in this paper are in `data/`, prefixed `orinnx-20260810-` (all files except the final external-fan test, which is `orinnx-20260811-`, run the following day):
- `strip-dryrun.log`, `strip-removal.log`, `strip-autoremove.log` — package removal
- `system-state-after.log` — post-tuning `nvpmodel`/`jetson_clocks`/systemd snapshot
- `engine-build.log` — corrected TensorRT engine build (with the `EDGELLM_PLUGIN_PATH` fix)
- `qwen3-1.7b-output.json` — inference correctness check output
- `prefill-decode10-bench.log` — 10-iteration prefill/decode benchmark
- `decode500-bench.log`, `tegrastats-decode500.csv` — sustained 500-iteration decode run with concurrent power/thermal capture
- `tegrastats-stress-precorrection.csv` — combined CPU+GPU stress before the `nvfancontrol` restart fix (fan not engaged, 74.5 °C peak)
- `tegrastats-stress-postfanfix.csv` — same stress test after the fix (fan confirmed engaging, 71.75 °C peak)
- `tegrastats-40w-attempt.csv` — CPU stress + single 1.7B prefill loop + burn kernel (32.6-32.8W)
- `tegrastats-40w-multistream.csv` — CPU stress + 3 concurrent 1.7B prefill loops + burn kernel (29.4W, worse than single-stream)
- `tegrastats-40w-prefilldecode.csv` — CPU stress + 1.7B prefill loop + 1.7B decode loop (33.0W)
- `tegrastats-40w-4b.csv` — CPU stress + 4B prefill loop + 4B decode loop (32.6W)
- `tegrastats-40w-staged.csv` — GPU load first, then CPU cores staged one at a time, 20s hold (34.2W)
- `tegrastats-40w-staged-extended.csv` — same staging technique, 90s hold (34.3W, 87.3°C)
- `tegrastats-40w-reverse-order.csv` — CPU loaded first, single GPU stream slammed on top afterward (32.5W, worse than staged)
- `tegrastats-40w-external-fan.csv` — staged loading technique repeated with an external fan supplementing stock cooling (34.07W at 76.2°C — power unchanged despite 11°C cooler, ruling out thermal throttling)
