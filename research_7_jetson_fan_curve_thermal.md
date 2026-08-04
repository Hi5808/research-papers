# Fan Curve Tuning on Jetson Orin: The Thermal-Margin Encoding Trap

**Platform:** NVIDIA Jetson Orin Nano 8GB (P3767-0000)
**Software stack:** JetPack 7.x, L4T R39.2.0, CUDA 13.2, nvfancontrol (stock)
**Date:** August 2026

## Abstract

NVIDIA's `nvfancontrol` fan curve configuration on Jetson Orin encodes its temperature column as **thermal margin below the throttle limit**, not absolute temperature, whenever `TMARGIN` is enabled. The encoding is undocumented in the config file itself and produces a curve that reads as inverted. An operator who edits the file assuming degrees Celsius will set thresholds that are wrong by `limit − intended`, in the direction opposite to intent. This paper documents the encoding, gives three independent methods to verify it on a live system, and reports measured thermal results from a retuned curve under combined CPU+GPU load at both 25 W and MAXN_SUPER power modes. A secondary finding concerns `nvpmodel` mode switches that report success and silently revert without a reboot, and the resulting requirement that thermal benchmarks assert on *achieved* clocks rather than *requested* power mode.

## Key Finding #1: The TEMP Column Is Margin, Not Temperature

The stock `quiet` profile on a P3767-0000 reads as follows:

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
- **PWM headroom remained at saturation.** The controller met its 6000 RPM target at PWM 225/255 (88%). Because `FAN_CONTROL close_loop` targets RPM rather than PWM, the binding constraint is the curve's declared RPM ceiling, not the PWM range. Whether this fan can be driven meaningfully above 6000 RPM is untested and would need the profile's RPM column raised to find out.
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

- **Single unit, single ambient.** No ambient temperature control; results are from one P3767-0000 in open air. Absolute temperatures will shift with enclosure and room conditions.
- **Load duration is short relative to the question asked.** The MAXN_SUPER run reached fan saturation at t≈155 s and sat at 60 °C for the final 115 s. Temperature was flat but the system was at its cooling ceiling, so a longer run may drift upward. A 300 s phase is sufficient to locate the equilibrium and insufficient to prove it is stable over hours.
- **Synthetic load.** `stress` plus an FMA kernel is a thermal worst case, not a representative inference workload. Real TensorRT pipelines have duty cycles that produce lower sustained temperatures, and do not load CPU and GPU simultaneously at full occupancy.
- **Fan maximum assumed, not measured.** The 6000 RPM ceiling is the value declared in the stock profile. Whether the hardware exceeds it was not tested (PWM was at 225/255 when the RPM target was met).
- **Margin encoding verified on P3767-0000 only.** Other Jetson modules ship different `nvfancontrol_*.conf` files and may not enable `TMARGIN`. Check for the directive before assuming the conversion applies.
- Comparisons against previously published figures from this platform are invalid unless fan curve and `nvpmodel` mode both match, since both affect sustained-clock behavior.

## Raw Data

Sample-level CSVs (5 s interval; Tj, CPU/GPU/SoC temperatures, fan RPM and PWM, CPU and GPU clocks, VDD_IN power from the on-board INA3221):

- `data/thermal-20260803-25W-combined-load.csv` — 25 W mode, 96 samples
- `data/thermal-20260803-MAXN_SUPER-combined-load.csv` — MAXN_SUPER, 96 samples

The 25 W file retains a `nvpmodel_requested` column recording the value that was requested but not applied; the MAXN_SUPER file records `nvpmodel_achieved` alongside `cpu_max_mhz`, verified before collection. The column-name difference between the two files is deliberate and preserves the distinction described in Finding #5.
