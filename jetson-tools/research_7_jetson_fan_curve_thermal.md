# Fan Curve Tuning on Jetson Orin: The Thermal-Margin Encoding Trap

**Platform:** Seeed Studio reComputer J3011 — NVIDIA Jetson Orin Nano 8GB (module P3767-0003) on a J401 carrier, not an NVIDIA developer kit
**Software stack:** JetPack 7.x, L4T R39.2.0, CUDA 13.2, nvfancontrol (stock)
**Date:** August 2026

> **Platform identification note.** The device tree `model` string on this unit
> reads `NVIDIA Jetson Orin NX Engineering Reference Developer Kit Super`, which
> is incorrect. The hardware identifiers agree it is an Orin Nano 8GB:
>
> ```
> TNSPEC      3767-301-0003-F.1-1-0-recomputer-orin-j401-
> compatible  nvidia,p3768-0000+p3767-0003-super / nvidia,p3767-0003
> 7.4 GiB RAM · 6x Cortex-A78AE · 8 SM · 128-bit LPDDR5
> ```
>
> Two traps for anyone identifying a Jetson from software. The DT `model` string
> is free text set by the BSP and can name the wrong module entirely — use
> `/etc/nv_boot_control.conf` TNSPEC or the `compatible` property. And the fan
> config resolves to `nvfancontrol_p3767_0000.conf` on a `p3767-0003` module,
> because NVIDIA ships one config for the whole P3767 family; the filename is
> not a SKU identifier.
>
> The carrier matters for every temperature reported here. The reComputer J3011
> pairs the module with a J401 carrier and its own heatsink, fan and airflow
> path, so absolute temperatures will not transfer to an NVIDIA P3768 devkit or
> any other enclosure. All measurements here were taken with the stock case
> closed. The encoding, methods and failure modes do transfer.

## Abstract

NVIDIA's `nvfancontrol` fan curve configuration on Jetson Orin encodes its temperature column as **thermal margin below the throttle limit**, not absolute temperature, whenever `TMARGIN` is enabled. The encoding is undocumented in the config file itself and produces a curve that reads as inverted. An operator who edits the file assuming degrees Celsius will set thresholds that are wrong by `limit − intended`, in the direction opposite to intent. This paper documents the encoding, gives three independent methods to verify it on a live system, and reports measured thermal results from a retuned curve under combined CPU+GPU load at both 25 W and MAXN_SUPER power modes, including a 30-minute maximum-configuration soak. Two further results concern measurement rather than cooling: with closed-loop RPM control the fan's declared ceiling is a config value rather than a hardware limit, and the module's power envelope is reachable only by a load that exercises tensor cores and DRAM, not by SM occupancy alone. A secondary finding concerns `nvpmodel` mode switches that report success and silently revert without a reboot, and the resulting requirement that thermal benchmarks assert on *achieved* clocks rather than *requested* power mode.

## Key Finding #1: The TEMP Column Is Margin, Not Temperature

The stock `quiet` profile (from `nvfancontrol_p3767_0000.conf`, which serves the
whole P3767 family) reads as follows:

```
<FAN 1>
	TMARGIN ENABLED
	FAN_PROFILE quiet {
		#TEMP 	HYST	PWM	RPM
		0	0	255	6000
		10	0	255	6000
		11	0	187	4000
		31	0	187	4000
		70	0 	0	0
		105	0 	0 	0
	}
	THERMAL_GROUP 0 {
		GROUP_MAX_TEMP 105
		...
	}
```

Read as Celsius, this says: full fan at 0–10 °C, fan **off** at 70–105 °C. That is not a plausible cooling policy.

The `TMARGIN ENABLED` directive changes the semantics of column 1 to **degrees of margin remaining below `GROUP_MAX_TEMP`**. The conversion is:

```
T_junction = GROUP_MAX_TEMP − TEMP_column
```

With `GROUP_MAX_TEMP 105`, the stock curve is therefore:

| TEMP (margin) | Actual Tj | Fan |
|---|---|---|
| 0 | 105 °C | 6000 RPM |
| 10 | 95 °C | 6000 RPM |
| 11 | 94 °C | 4000 RPM |
| 31 | 74 °C | 4000 RPM |
| 70 | 35 °C | 0 RPM |
| 105 | 0 °C | 0 RPM |

**The practical consequence:** an operator wanting the fan to engage at 60 °C who writes `60` into the file has actually set the threshold to 45 °C — the fan will behave *more* aggressively than intended at low temperatures and the operator's mental model of the curve is inverted for every subsequent edit. The correct value is `105 − 60 = 45`.

## Key Finding #2: Three Independent Verifications

The encoding is inferable from a running system without documentation. All three should agree before editing:

**1. Kernel trip points mirror the profile breakpoints.** The `tj-thermal` zone exposes trips that are the Celsius complements of the profile's margin values:

```bash
$ cat /sys/devices/virtual/thermal/thermal_zone8/trip_point_*_temp
35000    # → margin 70
74000    # → margin 31
95000    # → margin 10
104500   # → critical
```

Every trip corresponds to a breakpoint in the `quiet` profile under the margin reading, and to nothing under the Celsius reading.

**2. Reductio on the Celsius reading.** A curve specifying zero fan at 105 °C — above the 104.5 °C critical trip — cannot be the intended policy.

**3. Live interpolation check.** This is the decisive test. Compute the RPM the curve predicts under each hypothesis and compare against the tachometer:

```bash
Tj = 52.5 °C  →  margin 52.5
# interpolate between (31, 4000) and (70, 0):
#   4000 × (70 − 52.5) / (70 − 31) = 1795 RPM
$ cat /sys/class/hwmon/hwmon3/rpm
1756
```

A caution for anyone reproducing this: at `Tj = GROUP_MAX_TEMP / 2` (52.5 °C here) the margin and Celsius values are numerically identical, and the test is degenerate. Sample at any other temperature.

## Key Finding #3: The Stock Curve Is Deliberately Permissive

Under the margin reading, the stock `quiet` profile holds the fan **completely off until Tj reaches 35 °C**, and does not exceed 4000 RPM until Tj passes 94 °C. On a desk-idle unit at 52.5 °C the fan runs at ~1756 RPM.

This is a reasonable acoustic default, but it means the SoC routinely operates in the 50–75 °C band with minimal active cooling. For workloads where sustained-clock stability matters — anything being benchmarked — this permissiveness introduces thermal variance that is invisible unless explicitly measured.

## Key Finding #4: Measured Results From a Retuned Curve

The `quiet` profile was replaced with an aggressive curve reaching full 6000 RPM by Tj 57 °C:

```
	FAN_PROFILE quiet {
		#TEMP 	HYST	PWM	RPM
		0	0	255	6000
		48	0	255	6000     ← Tj 57 °C
		55	0	187	4000     ← Tj 50 °C
		65	0	120	2500     ← Tj 40 °C
		105	0	0	0
	}
```

Measured under combined load — 6 CPU workers (`stress --cpu 6`) plus a saturating GPU FMA kernel across all 8 SMs — with 5 s sampling and a 300 s load phase, at both available power ceilings. Clocks were asserted as *achieved* at harness start (see Finding #5).

**25 W mode** (CPU capped 1344 MHz, GPU 918 MHz):

| Phase | n | Peak Tj | Mean Tj | Peak fan | Mean VDD_IN |
|---|---|---|---|---|---|
| Idle | 12 | 45 °C | 45.0 °C | 3266 RPM | 4.5 W |
| Load | 60 | **57 °C** | 54.2 °C | 5666 RPM | 12.8 W |
| Cooldown | 24 | 55 °C | 50.2 °C | 5666 RPM | 4.8 W |

**MAXN_SUPER** (CPU 1728 MHz, GPU 1020 MHz, uncapped):

| Phase | n | Peak Tj | Mean Tj | Peak fan | Mean VDD_IN |
|---|---|---|---|---|---|
| Idle | 12 | 47 °C | 46.6 °C | 3560 RPM | 4.6 W |
| Load | 60 | **60 °C** | 57.8 °C | **6003 RPM** | 15.3 W |
| Cooldown | 24 | 60 °C | 52.2 °C | 5990 RPM | 4.9 W |

Observations:

- **The 60 °C target holds at 25 W with 3 °C to spare, and at MAXN_SUPER with none.** At full clocks Tj reached exactly 60 °C at t=240 s and plateaued there for the remaining 115 s, with the GPU zone touching 61 °C. The target is met at the boundary, not within it.
- **The fan curve saturated.** From t≈155 s the closed-loop controller was requesting its maximum 6000 RPM target and measured 5929–6003 RPM for the rest of the run. Thermal equilibrium at full load is therefore set by the fan's ceiling, not by the curve's shape — no further tuning of the profile can improve this operating point.
- **PWM headroom remained at saturation.** The controller met its 6000 RPM target at PWM 225/255 (88%). Because `FAN_CONTROL close_loop` targets RPM rather than PWM, the binding constraint is the curve's declared RPM ceiling, not the PWM range. Raising the target above that ceiling confirms the fan reaches 6258–6296 RPM at PWM 255 — see Finding #6.
- **No thermal throttling occurred in either mode.** CPU held a flat 1728 MHz across all 60 MAXN_SUPER load samples and GPU held 1020 MHz. The thermal solution sustains full clocks indefinitely at ~60 °C; the 95 °C throttle trip was never approached.
- **Power scaled far less than clocks.** MAXN_SUPER drew 15.3 W mean against 12.8 W at 25 W mode — a 20% increase for a 29% CPU clock and 11% GPU clock increase. Note the "25 W" mode measured 12.8 W at the wall-adjacent VDD_IN rail under a synthetic worst case; the mode names are configuration labels, not observed consumption.
- **Idle cost is real and permanent.** Idle Tj fell from 52.5 °C to 45–47 °C, but idle fan speed rose from ~1756 to ~3266–3560 RPM. That is a continuous acoustic cost bought for thermal margin with no reliability benefit — the throttle trip is 95 °C and critical is 104.5 °C.

**Interpretation:** the retuned curve is sufficient at 25 W and marginal at MAXN_SUPER. At full clocks the system is in fan-saturated equilibrium precisely at the target, so any increase in ambient temperature, dust loading, enclosure restriction, or load duration will breach 60 °C. A 60 °C ceiling at MAXN_SUPER should be treated as unattained rather than achieved; the honest options are accepting ~60–65 °C, or capping power at 25 W.

## Key Finding #5: `nvpmodel` Mode Switches Can Fail Silently

The first attempt at the `MAXN_SUPER` experiment (mode 2, uncapped clocks) produced a complete, plausible dataset that described the wrong power mode. The harness issued `nvpmodel -m 2 -f`, logged mode `2` into every CSV row, and ran to completion. The switch did not take:

```bash
# Requested: MAXN_SUPER (CPU MAX_FREQ -1, i.e. uncapped)
# Observed throughout the run:
cpu0_mhz = 1344    # the 25 W cap, not the 1728 MHz the silicon supports
gpu_mhz  = 918     # the 25 W cap, not 1020 MHz
# After the run:
$ nvpmodel -q
NV Power Mode: 25W
```

The command returned without error and the daemon reverted to mode 1. Had the harness trusted its own `nvpmodel` column, the resulting data would have been published as MAXN_SUPER results while actually describing 25 W behavior — an error invisible in the output, and one that would have understated peak Tj by 3 °C while entirely concealing that the fan saturates at full clocks.

**Root cause: MAXN_SUPER requires a reboot to apply.** Re-issuing the same command followed by a restart produced the intended state:

```bash
$ sudo nvpmodel -m 2 && sudo reboot
# after reboot:
$ nvpmodel -q | head -1
NV Power Mode: MAXN_SUPER
$ cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
1728000     # was 1344000
```

The failure mode is not that the switch is rejected — it is that `nvpmodel` accepts the request, exits zero, and the setting is discarded before taking effect. No warning is emitted on the path taken here.

**Methodological rule:** thermal and performance benchmarks on Jetson must record **achieved** state, not requested state. Log `scaling_cur_freq` and the GPU `cur_freq` on every sample, and assert against `scaling_max_freq` at harness start:

```bash
CPUMAX=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)
[[ $CPUMAX -ge 1600000 ]] || { echo "mode did not apply"; exit 1; }
```

This generalizes beyond fan tuning: any Jetson benchmark comparing power modes is vulnerable to silently measuring the same mode twice. The assertion must gate data collection — a harness that logs the anomaly but proceeds still produces a file someone will later analyze. The revised harness exits non-zero without writing a CSV when the achieved clocks contradict the requested mode.

## Key Finding #6: The Declared Fan Ceiling Is Not the Hardware Ceiling

The stock profile declares 6000 RPM as its maximum, and `FAN_CONTROL close_loop`
chases RPM rather than PWM. During the MAXN_SUPER run the controller met that
6000 RPM target at **PWM 225/255** — it stopped pushing because it had reached
the number in the config, not because the fan had reached its limit.

Setting an intentionally unreachable target (8000 RPM) forces PWM to saturate:

```bash
FAN_PROFILE quiet {
	#TEMP 	HYST	PWM	RPM
	0	0	255	8000
	105	0	255	8000
}
```

Measured result: **6258–6296 RPM at PWM 255/255.** The stock profile leaves
roughly 4–5% of available airflow unused. This is small in absolute terms, but
it is free, and it matters precisely in the fan-saturated regime where nothing
else in the curve can help.

The general point: with closed-loop RPM control, the declared ceiling is a
policy choice in a config file, not a hardware property. Any conclusion of the
form "the fan is maxed out" should be checked against PWM, not RPM.

## Key Finding #7: Sustained Maximum-Load Behavior

Full configuration — MAXN_SUPER, `jetson_clocks` (DVFS disabled, clocks pinned
at min=max), fan pinned flat out, and a multi-engine GPU load (concurrent
tensor-core HMMA, DRAM streaming, FP32 FMA) alongside 6 CPU workers — run for
30 minutes at 10 s sampling:

| Metric | Value |
|---|---|
| Load samples | 177 |
| Peak Tj | **67 °C** |
| Mean Tj | 64.6 °C |
| Fan | 6296 RPM peak, PWM 255 throughout |
| Mean VDD_IN | **20.0 W** |
| CPU clock | 1728–1728 MHz (pinned) |
| Throttled samples | **0** |
| Idle baseline | 45.3 °C, 6.6 W |

Convergence, by quarter of the load phase:

| Quarter | Window | Drift | Mean Tj |
|---|---|---|---|
| 1 | 120–558 s | +1.858 °C/min | 60.4 °C |
| 2 | 568–1005 s | +0.056 °C/min | 65.6 °C |
| 3 | 1016–1453 s | −0.049 °C/min | 65.9 °C |
| 4 | 1463–1911 s | +0.191 °C/min | 66.4 °C |

**Equilibrium is reached at roughly 10 minutes, near 65–66 °C, and holds.** The
system sustains pinned maximum clocks indefinitely at 28 °C below the 95 °C
throttle trip, with no throttled samples in 30 minutes. On this hardware the
thermal solution is not the constraint at any available power mode.

This was measured **inside the stock closed aluminium enclosure**, not on an
open bench, which makes it the configuration the product actually ships in
rather than a best case. The J3011's passive case and carrier fan together
absorb a 20 W sustained load with 28 °C of margin.

The residual movement in quarters 3–4 is approximately 1 °C of wander, not a
trend — see the methodology note below before reading it as one.

## Key Finding #8: Power Draw Is a Property of the Load, Not the Board

Reaching the board's actual power envelope required rewriting the load, not
changing any setting. Three loads, same hardware and same power mode:

| Load | VDD_IN | CPU_GPU_CV rail | SOC rail |
|---|---|---|---|
| Idle | 5.5 W | 1.3 W | 1.5 W |
| FP32 FMA only (GPU) | 11.7 W | 7.5 W | **1.4 W** |
| Tensor + DRAM + FP32 (GPU) | 16.2 W | 4.2 W | **5.1 W** |
| Above + 6 CPU workers + `jetson_clocks` | **20.0 W** | 7.8 W | 5.2 W |

The diagnostic is the SOC rail. Under a register-resident FP32 kernel it does
not move at all (1.5 → 1.4 W), proving the kernel generates zero DRAM traffic
despite fully occupying the SMs. Adding a 512 MB streaming working set — larger
than L2, so every pass reaches DRAM — moves that rail to 5.1 W and raises total
draw by 4.5 W even as GPU *compute* power falls.

**Implication for thermal testing:** a GPU load that saturates SM occupancy can
still leave a third of the module's power envelope untouched. Thermal headroom
measured with an FP32 microbenchmark will be optimistic against real inference
workloads, which stream weights from DRAM and use tensor cores. Per-rail
instrumentation, not just total wattage, is what reveals the gap.

## Methodology Note: Log Millidegrees, Not Degrees

The initial harness logged Tj truncated to whole degrees. At that resolution a
single 1 °C rounding step across a 7-minute regression window presents as
≈0.2 °C/min of apparent trend — enough to move a 30-minute soak from "converged"
to "still drifting" on quantization noise alone.

The quarter-by-quarter table above shows the failure clearly: quarters 2 and 3
are flat (+0.056, −0.049 °C/min) while quarter 4 reads +0.191 °C/min, from a
total movement of about 0.5 °C. The kernel exposes millidegrees at
`thermal_zone*/temp`; regress on that and reserve the rounded value for display.

This applies to any convergence test where the effect size is comparable to the
logging resolution — a category that includes most thermal soaks, since the
question is precisely whether a small slope is real.

## Reproduction

```bash
# 1. Identify the real config (the /etc path is a symlink; editing it
#    with sed -i would replace the symlink with a regular file)
readlink -f /etc/nvfancontrol.conf
# → /etc/nvpower/nvfancontrol/nvfancontrol_p3767_0000.conf

# 2. Convert intended Celsius thresholds to margin: TEMP = 105 − Tj

# 3. Apply, clearing the daemon's cached state
sudo systemctl stop nvfancontrol
sudo rm -f /var/lib/nvfancontrol/status   # new curve ignored without this
sudo systemctl start nvfancontrol

# 4. Verify direction of response — fan RPM must RISE with temperature.
#    If it falls, the margin interpretation was applied backwards.
watch -n2 'echo "Tj $(($(cat /sys/devices/virtual/thermal/thermal_zone8/temp)/1000))C \
  RPM $(cat /sys/class/hwmon/hwmon3/rpm)"'
```

Note that `/etc/nvpower/nvfancontrol/*.conf` is a stock NVIDIA file and may be overwritten by a JetPack or OTA update. Retain a timestamped backup as the record of the customization.

## Limitations

- **Single unit, single ambient.** No ambient temperature control and no external airflow; results are from one reComputer J3011 in its stock closed aluminium enclosure at room temperature. Absolute temperatures will shift with ambient and with any change to the cooling assembly.
- **Load duration is short relative to the question asked.** The MAXN_SUPER run reached fan saturation at t≈155 s and sat at 60 °C for the final 115 s. Temperature was flat but the system was at its cooling ceiling, so a longer run may drift upward. A 300 s phase is sufficient to locate the equilibrium and insufficient to prove it is stable over hours.
- **Synthetic load.** `stress` plus an FMA kernel is a thermal worst case, not a representative inference workload. Real TensorRT pipelines have duty cycles that produce lower sustained temperatures, and do not load CPU and GPU simultaneously at full occupancy.
- **Ambient uncontrolled.** The +0.5 °C movement in the final quarter of the 30-minute soak is within the range a warming room would produce, and no ambient sensor was logged. Distinguishing residual device drift from room drift requires an external probe.
- **30 minutes is not 30 days.** The soak establishes that the thermal design converges. It says nothing about dust accumulation, fan bearing wear, or seasonal ambient swings, all of which move the equilibrium over a deployment lifetime.
- **Margin encoding verified on this module/BSP only.** Other Jetson modules ship different `nvfancontrol_*.conf` files and may not enable `TMARGIN`. Check for the directive before assuming the conversion applies.
- Comparisons against previously published figures from this platform are invalid unless fan curve and `nvpmodel` mode both match, since both affect sustained-clock behavior.

## Tooling

The scripts used here are published separately as
[**jetson-tools**](https://github.com/Hi5808/jetson-tools) — runtime platform
detection, a fan-curve editor that speaks in degrees Celsius, and the soak
harness with the achieved-clock gate and millidegree drift regression described
above.

## Raw Data

Sample-level CSVs (5 s interval; Tj, CPU/GPU/SoC temperatures, fan RPM and PWM, CPU and GPU clocks, VDD_IN power from the on-board INA3221):

- `data/thermal-20260803-25W-combined-load.csv` — 25 W mode, 96 samples
- `data/thermal-20260803-MAXN_SUPER-combined-load.csv` — MAXN_SUPER, 96 samples
- `data/thermal-20260803-MAXED-30min-soak.csv` — MAXN_SUPER + `jetson_clocks` + fan pinned + multi-engine load, 30 min, 213 samples

The 25 W file retains a `nvpmodel_requested` column recording the value that was requested but not applied; the MAXN_SUPER file records `nvpmodel_achieved` alongside `cpu_max_mhz`, verified before collection. The column-name difference between the two files is deliberate and preserves the distinction described in Finding #5.
