# Jetson Thermal Toolkit

Portable fan-curve and thermal-soak tooling for NVIDIA Jetson. Everything is
detected at runtime — sysfs indices, config paths, thermal limits and the fan
curve's temperature encoding all differ between Jetson modules and between
boots, so nothing here is hardcoded to one board.

Companion to [*Fan Curve Tuning on Jetson Orin: The Thermal-Margin Encoding
Trap*](../research_7_jetson_fan_curve_thermal.md).

## The problem this exists to solve

NVIDIA's `nvfancontrol` config encodes its temperature column as **thermal
margin below the limit**, not degrees Celsius, whenever `TMARGIN ENABLED` is
present. The stock curve therefore looks inverted:

```
FAN_PROFILE quiet {
	#TEMP 	HYST	PWM	RPM
	0	0	255	6000     <- reads as "full fan at 0 C"
	70	0 	0	0        <- reads as "fan off at 70 C"
}
```

It isn't inverted; column 1 is `limit − Tj`. Someone who wants the fan at full
speed by 60 °C and writes `60` has actually set 45 °C, and every subsequent
edit compounds the error. `jetson-fan-curve.sh` always speaks in degrees C and
converts internally.

## Contents

| File | Purpose |
|---|---|
| `jetson_thermal_lib.sh` | Runtime detection of thermal zones, fan, INA3221, GPU devfreq, fan config, thermal limit, TMARGIN. Source it from your own scripts. |
| `jetson-fan-curve.sh` | Inspect and set the fan curve, in degrees C. |
| `jetson-soak.sh` | Sustained load soak with CSV logging and thermal-runaway detection. |
| `gpu_burn.cu` | Multi-engine GPU load: tensor cores, DRAM streaming, FP32. |
| `build-gpu-burn.sh` | Builds `gpu_burn` for the local compute capability. |

## Quick start

```bash
git clone <this repo> && cd research-papers/tools
./build-gpu-burn.sh              # needs the JetPack CUDA toolkit

./jetson-fan-curve.sh --show     # what is my board actually doing?
sudo ./jetson-fan-curve.sh --target 60
./jetson-soak.sh --load 600
```

## `jetson-fan-curve.sh`

```bash
./jetson-fan-curve.sh --show            # current curve, translated to degrees C
sudo ./jetson-fan-curve.sh --target 65  # full fan by Tj 65 C, tapering below
sudo ./jetson-fan-curve.sh --max        # pin flat out (see note below)
sudo ./jetson-fan-curve.sh --restore    # revert the newest backup
```

Every write is backed up alongside the config first. `--profile cool` targets
the other stock profile; `JT_RPM_MAX=7000` overrides the detected fan ceiling.

**After any change, verify the direction of response.** Fan RPM must *rise* as
Tj rises. If it falls, the encoding on your board is the opposite of what was
detected — restore immediately.

**On `--max`:** it sets an RPM target above the declared ceiling. Because
`FAN_CONTROL close_loop` chases RPM rather than PWM, an unreachable target
drives PWM to 255 and reveals the fan's true maximum. On an Orin Nano Super
class dev kit this produced 6258-6296 RPM against a declared ceiling of 6000 -
the stock profile leaves roughly 4-5% of airflow unused. It also means "the fan
is maxed" should be judged from PWM, not RPM: the controller stops at the
config's number, not the hardware's. Expect continuous full-speed
noise; this is a test mode, not a daily driver.

## `jetson-soak.sh`

```bash
./jetson-soak.sh --load 1800                    # 30 min soak
./jetson-soak.sh --load 600 --no-gpu            # CPU only
./jetson-soak.sh --load 900 --expect-cpu-mhz 1700
```

Runs unprivileged. Writes a CSV of Tj, CPU/GPU temperatures, fan RPM and PWM,
CPU and GPU clocks, and three INA3221 power rails at a fixed interval, then
reports peak, mean, throttle-sample count, and a least-squares temperature
slope over the final third of the load phase:

```
  peak Tj      : 60 C
  final-third drift: +0.04 C/min
  VERDICT: CONVERGED - thermally stable at this load
```

The drift figure is the point of the tool. A peak temperature alone doesn't
distinguish "hot but stable" from "still climbing when the test ended," and
those have opposite implications for a 24/7 deployment.

The regression runs on **millidegrees**, not the rounded `tj_c` column. This
matters more than it sounds: at whole-degree resolution a single rounding step
across a 7-minute window reads as ~0.2 C/min of trend, which is enough to flip
a converged 30-minute soak to "still drifting" on quantization noise alone.
Both columns are logged; use `tj_mc` for any analysis.

Note that the verdict thresholds are heuristics, and ambient drift is not
separated from device drift — no room sensor is involved. A slow positive slope
may be your workspace warming up rather than the board failing to converge.

### `--expect-cpu-mhz` — read this before trusting any power-mode comparison

`nvpmodel -m <n>` can return success, log as applied, and silently revert;
MAXN modes generally require a **reboot** to take effect. A benchmark that
records the *requested* mode will happily produce a complete dataset labelled
MAXN_SUPER that actually describes the previous power cap.

`--expect-cpu-mhz` gates data collection on the achieved `scaling_max_freq`
and exits non-zero **without writing a CSV** if it doesn't match. Use it in any
automated comparison between power modes.

## Interpreting results

- **Throttle trips are high.** On Orin the active trip is ~95 °C and critical
  ~104.5 °C. Temperatures in the 60–80 °C range under load are normal and carry
  no reliability penalty. Tuning for a low ceiling buys acoustics, not lifespan.
- **A fan curve cannot beat physics.** Once the fan saturates, equilibrium is
  set by airflow and ambient, not by the profile. Further curve tuning past
  that point does nothing; the remaining levers are the `nvpmodel` power cap,
  ambient temperature, and the enclosure.
- **Synthetic load is a worst case.** `gpu_burn` runs tensor cores, DRAM and
  FP32 concurrently, which few real workloads do. Inference pipelines with
  duty cycles run cooler.
- **Idle cost is permanent.** An aggressive curve that drops idle temperature
  by 7 °C may double idle fan speed. That noise is paid continuously.

## Compatibility

Developed and verified on an Orin Nano Super class dev kit (P3767 carrier,
JetPack 7 / L4T R39.2, CUDA 13.2). The detection layer is written against
generic Jetson sysfs and should work on Orin NX/AGX and Xavier; the tensor-core
path in `gpu_burn.cu` falls back to FP32 below `sm_70` for Nano/TX2 class
hardware. It has not been tested on those boards — `--show` is read-only and is
the safe way to check what a new board reports before changing anything.

Boards without `TMARGIN` are handled: curves are then written in absolute
degrees, ascending. Passive boards with no fan are detected and reported rather
than crashed on.

## Caveats

`/etc/nvfancontrol.conf` is a **symlink** to a board-specific file under
`/etc/nvpower/nvfancontrol/`. Editing the symlink path with `sed -i` replaces
the symlink with a regular file; these scripts resolve it with `readlink -f`
first. The target is a stock NVIDIA file and may be overwritten by a JetPack or
OTA update — the timestamped backups are your record of the customization.

`nvfancontrol` caches state in `/var/lib/nvfancontrol/status`. A curve change
is unreliable without removing it and restarting the service; the scripts do
this for you.

These tools drive hardware to its thermal limits deliberately. The SoC's own
protection (throttle and critical trips) remains active throughout and is not
modified — but run soaks where you can hear and see the machine.
