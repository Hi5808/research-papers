#!/usr/bin/env bash
# Wraps a command with power/thermal telemetry + OOM-kill detection, for
# audio.cpp smoke tests on Jetson Orin. Jetson's unified memory means CUDA
# allocation failures often don't surface as a CUDA error at all — the
# kernel OOM killer silently kills the process instead — so watch dmesg,
# don't just trust a clean exit code.
#
# Usage: run_with_telemetry.sh <tag> <outdir> -- <cmd...>
# Paths below match the INA3221/soctherm-oc sysfs nodes already validated on
# this board family (see orinnx spec_telemetry.sh); adjust hwmon index if a
# board enumerates them differently (check `cat /sys/class/hwmon/hwmon*/name`).
set -uo pipefail

TAG="$1"; OUTDIR="$2"; shift 2
[ "$1" = "--" ] && shift

mkdir -p "${OUTDIR}"
POWER_LOG="${OUTDIR}/${TAG}_power.csv"
DMESG_LOG="${OUTDIR}/${TAG}_dmesg_oom.log"
SUMMARY="${OUTDIR}/${TAG}_summary.txt"

CURR_NODE="$(find /sys/devices/platform/bus@0/*/i2c-*/*-0040/hwmon/hwmon*/curr1_input 2>/dev/null | head -1)"
VOLT_NODE="$(find /sys/devices/platform/bus@0/*/i2c-*/*-0040/hwmon/hwmon*/in1_input 2>/dev/null | head -1)"

echo "ts_ms,curr_mA,volt_mV" > "${POWER_LOG}"

# Baseline dmesg position so we only report OOM events from during this run.
DMESG_START_LINES="$(dmesg | wc -l)"

# Background power sampler at 100ms, only if the sysfs nodes were found.
SAMPLER_PID=""
if [ -n "${CURR_NODE}" ] && [ -n "${VOLT_NODE}" ]; then
    (
        while true; do
            ts=$(date +%s%3N)
            curr=$(cat "${CURR_NODE}" 2>/dev/null || echo "")
            volt=$(cat "${VOLT_NODE}" 2>/dev/null || echo "")
            [ -n "${curr}" ] && echo "${ts},${curr},${volt}" >> "${POWER_LOG}"
            sleep 0.1
        done
    ) &
    SAMPLER_PID=$!
else
    echo "WARNING: INA3221 curr1_input/in1_input not found — power telemetry skipped" | tee -a "${SUMMARY}"
fi

START_TS=$(date +%s.%N)
"$@"
EXIT_CODE=$?
END_TS=$(date +%s.%N)

[ -n "${SAMPLER_PID}" ] && kill "${SAMPLER_PID}" 2>/dev/null
wait "${SAMPLER_PID}" 2>/dev/null

# Check for OOM-kill events that occurred during the run.
dmesg | tail -n +"$((DMESG_START_LINES + 1))" | grep -i "oom\|killed process\|out of memory" > "${DMESG_LOG}" || true

{
    echo "tag: ${TAG}"
    echo "command: $*"
    echo "exit_code: ${EXIT_CODE}"
    echo "wall_time_s: $(echo "${END_TS} - ${START_TS}" | bc)"
    if [ -s "${DMESG_LOG}" ]; then
        echo "OOM_KILL_DETECTED: YES — see ${DMESG_LOG}"
        echo "  (a clean-looking CUDA exit code does NOT rule this out on unified memory — always check this file)"
    else
        echo "OOM_KILL_DETECTED: no"
    fi
    if [ -s "${POWER_LOG}" ] && [ "$(wc -l < "${POWER_LOG}")" -gt 1 ]; then
        awk -F, 'NR>1{c+=$2; if($2>max)max=$2; n++} END{if(n>0) printf "avg_curr_mA: %.1f\npeak_curr_mA: %.1f\nsamples: %d\n", c/n, max, n}' "${POWER_LOG}"
    fi
} | tee "${SUMMARY}"

exit "${EXIT_CODE}"
