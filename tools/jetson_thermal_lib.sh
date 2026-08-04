#!/usr/bin/env bash
# jetson_thermal_lib.sh - runtime detection of Jetson thermal/fan/power interfaces.
#
# Source this from other scripts. Nothing here is board-specific: sysfs indices,
# config paths and the thermal limit are all discovered at runtime, because they
# differ across Jetson modules (Orin Nano/NX/AGX, Xavier, ...) and across boots.
#
# Exports:
#   JT_TJ_ZONE      thermal zone path for junction temperature
#   JT_CPU_ZONE     thermal zone path for CPU (may be empty)
#   JT_GPU_ZONE     thermal zone path for GPU (may be empty)
#   JT_FAN_PWM      sysfs pwm file (may be empty on passive boards)
#   JT_FAN_RPM      sysfs tachometer file (may be empty)
#   JT_INA_DIR      INA3221 hwmon dir (may be empty)
#   JT_GPU_DEVFREQ  GPU devfreq dir (may be empty)
#   JT_CPUFREQ      cpu0 cpufreq dir
#   JT_FAN_CONF     resolved nvfancontrol config path (may be empty)
#   JT_MAX_TEMP     GROUP_MAX_TEMP from config (default 105)
#   JT_TMARGIN      1 if TMARGIN enabled (TEMP column is margin), else 0

jt_hwmon_by_name() {   # $1 = hwmon name to match
	local h n
	for h in /sys/class/hwmon/hwmon*; do
		[[ -r $h/name ]] || continue
		n=$(<"$h/name")
		[[ $n == "$1" ]] && { echo "$h"; return 0; }
	done
	return 1
}

jt_zone_by_type() {    # $1 = substring of thermal zone type
	local z t
	for z in /sys/devices/virtual/thermal/thermal_zone* /sys/class/thermal/thermal_zone*; do
		[[ -r $z/type ]] || continue
		t=$(<"$z/type")
		if [[ $t == *"$1"* ]]; then
			# Skip zones that expose no reading (some are stubs).
			[[ -n $(cat "$z/temp" 2>/dev/null) ]] || continue
			echo "$z"; return 0
		fi
	done
	return 1
}

jt_detect() {
	# --- thermal zones. tj is the SoC-wide junction temp; prefer it. ---
	JT_TJ_ZONE=$(jt_zone_by_type "tj-thermal" || jt_zone_by_type "Tj" || true)
	JT_CPU_ZONE=$(jt_zone_by_type "cpu-thermal" || jt_zone_by_type "CPU-therm" || true)
	JT_GPU_ZONE=$(jt_zone_by_type "gpu-thermal" || jt_zone_by_type "GPU-therm" || true)
	# Fallback: hottest readable zone.
	if [[ -z ${JT_TJ_ZONE:-} ]]; then
		local z best=-1 bt
		for z in /sys/devices/virtual/thermal/thermal_zone*; do
			bt=$(cat "$z/temp" 2>/dev/null) || continue
			[[ -n $bt ]] || continue
			(( bt > best )) && { best=$bt; JT_TJ_ZONE=$z; }
		done
	fi

	# --- fan. pwmfan drives it, pwm_tach reads it back; some boards merge them. ---
	local fanpwm fantach
	fanpwm=$(jt_hwmon_by_name pwmfan || true)
	fantach=$(jt_hwmon_by_name pwm_tach || true)
	JT_FAN_PWM=""
	[[ -n $fanpwm && -e $fanpwm/pwm1 ]] && JT_FAN_PWM=$fanpwm/pwm1
	[[ -z $JT_FAN_PWM && -n $fantach && -e $fantach/pwm1 ]] && JT_FAN_PWM=$fantach/pwm1
	JT_FAN_RPM=""
	for c in "$fantach/rpm" "$fanpwm/rpm" "$fantach/fan1_input" "$fanpwm/fan1_input"; do
		[[ -n $c && -e $c ]] && { JT_FAN_RPM=$c; break; }
	done

	# --- power monitor ---
	JT_INA_DIR=$(jt_hwmon_by_name ina3221 || true)

	# --- gpu devfreq (path differs by SoC: 17000000.gpu, 57000000.gpu, ...) ---
	JT_GPU_DEVFREQ=""
	for d in /sys/class/devfreq/*gpu* /sys/devices/platform/*.gpu/devfreq_dev; do
		[[ -d $d && -r $d/cur_freq ]] && { JT_GPU_DEVFREQ=$d; break; }
	done

	JT_CPUFREQ=/sys/devices/system/cpu/cpu0/cpufreq

	# --- nvfancontrol config (a symlink to a board-specific file) ---
	JT_FAN_CONF=""
	if [[ -e /etc/nvfancontrol.conf ]]; then
		JT_FAN_CONF=$(readlink -f /etc/nvfancontrol.conf)
	fi

	# --- thermal limit and TEMP-column semantics ---
	JT_MAX_TEMP=105
	JT_TMARGIN=0
	if [[ -n $JT_FAN_CONF && -r $JT_FAN_CONF ]]; then
		local gmt
		gmt=$(grep -oP 'GROUP_MAX_TEMP\s+\K[0-9]+' "$JT_FAN_CONF" 2>/dev/null | head -1)
		[[ -n $gmt ]] && JT_MAX_TEMP=$gmt
		grep -qE '^\s*TMARGIN\s+ENABLED' "$JT_FAN_CONF" 2>/dev/null && JT_TMARGIN=1
	fi

	export JT_TJ_ZONE JT_CPU_ZONE JT_GPU_ZONE JT_FAN_PWM JT_FAN_RPM \
	       JT_INA_DIR JT_GPU_DEVFREQ JT_CPUFREQ JT_FAN_CONF JT_MAX_TEMP JT_TMARGIN
}

# --- readers. All return 0 rather than failing, so sampling never aborts. -----
jt_read()      { cat "$1" 2>/dev/null || echo 0; }
jt_temp_c()    { echo $(( $(jt_read "${1:-$JT_TJ_ZONE}/temp") / 1000 )); }
jt_fan_rpm()   { [[ -n $JT_FAN_RPM ]] && jt_read "$JT_FAN_RPM" || echo 0; }
jt_fan_pwm()   { [[ -n $JT_FAN_PWM ]] && jt_read "$JT_FAN_PWM" || echo 0; }
jt_cpu_mhz()   { echo $(( $(jt_read "$JT_CPUFREQ/scaling_cur_freq") / 1000 )); }
jt_gpu_mhz()   { [[ -n $JT_GPU_DEVFREQ ]] && echo $(( $(jt_read "$JT_GPU_DEVFREQ/cur_freq") / 1000000 )) || echo 0; }

# Rail power in mW. $1 = channel index (1=VDD_IN on most carriers).
jt_power_mw() {
	local ch=${1:-1}
	[[ -n $JT_INA_DIR ]] || { echo 0; return; }
	local mv ma
	mv=$(jt_read "$JT_INA_DIR/in${ch}_input")
	ma=$(jt_read "$JT_INA_DIR/curr${ch}_input")
	echo $(( mv * ma / 1000 ))
}

# Label of an INA channel, e.g. VDD_IN / VDD_CPU_GPU_CV / VDD_SOC.
jt_power_label() {
	[[ -n $JT_INA_DIR ]] || { echo "ch$1"; return; }
	jt_read "$JT_INA_DIR/in${1}_label"
}

# Convert an intended junction temperature to the value the fan curve wants.
# This is the core portability trap: with TMARGIN the column is margin, not degC.
jt_temp_to_curve() {   # $1 = intended Tj in degC
	if [[ $JT_TMARGIN -eq 1 ]]; then
		echo $(( JT_MAX_TEMP - $1 ))
	else
		echo "$1"
	fi
}

jt_print_detected() {
	echo "Detected platform interfaces:"
	echo "  model        : $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
	echo "  L4T          : $(head -1 /etc/nv_tegra_release 2>/dev/null | grep -oP 'R\d+.*REVISION: [0-9.]+' || echo unknown)"
	echo "  Tj zone      : ${JT_TJ_ZONE:-none}"
	echo "  fan pwm/rpm  : ${JT_FAN_PWM:-none} / ${JT_FAN_RPM:-none}"
	echo "  INA3221      : ${JT_INA_DIR:-none}"
	echo "  GPU devfreq  : ${JT_GPU_DEVFREQ:-none}"
	echo "  fan config   : ${JT_FAN_CONF:-none}"
	echo "  thermal limit: ${JT_MAX_TEMP} C"
	if [[ $JT_TMARGIN -eq 1 ]]; then
		echo "  TEMP column  : MARGIN below ${JT_MAX_TEMP} C  (curve value = ${JT_MAX_TEMP} - Tj)"
	else
		echo "  TEMP column  : absolute degrees C"
	fi
}
