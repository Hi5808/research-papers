#!/usr/bin/env bash
# jetson-soak.sh - sustained thermal soak with runaway detection for Jetson.
#
# Runs unprivileged. Records ACHIEVED clocks, not the requested power mode:
# nvpmodel can report success and silently revert (it may need a reboot), which
# otherwise yields a plausible dataset describing the wrong mode.
#
#   ./jetson-soak.sh                       # 10 min load, default phases
#   ./jetson-soak.sh --load 1800           # 30 min soak
#   ./jetson-soak.sh --load 600 --no-gpu   # CPU only
#
set -uo pipefail
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=jetson_thermal_lib.sh
source "$DIR/jetson_thermal_lib.sh"

BASELINE=60; LOAD=600; COOL=120; INTERVAL=5
USE_GPU=1; USE_CPU=1; OUT=""; EXPECT_CPU_MHZ=""

while [[ $# -gt 0 ]]; do
	case $1 in
		--load) LOAD=$2; shift ;;
		--baseline) BASELINE=$2; shift ;;
		--cooldown) COOL=$2; shift ;;
		--interval) INTERVAL=$2; shift ;;
		--out) OUT=$2; shift ;;
		--no-gpu) USE_GPU=0 ;;
		--no-cpu) USE_CPU=0 ;;
		--expect-cpu-mhz) EXPECT_CPU_MHZ=$2; shift ;;
		-h|--help) sed -n '2,14p' "$0"; exit 0 ;;
		*) echo "unknown arg: $1" >&2; exit 1 ;;
	esac
	shift
done

jt_detect
jt_print_detected

CPUMAX=$(( $(jt_read "$JT_CPUFREQ/scaling_max_freq") / 1000 ))
GPUMAX=0
[[ -n $JT_GPU_DEVFREQ ]] && GPUMAX=$(( $(jt_read "$JT_GPU_DEVFREQ/max_freq") / 1000000 ))
MODE=$(nvpmodel -q 2>/dev/null | sed -n 2p | tr -d ' ')
MODENAME=$(nvpmodel -q 2>/dev/null | head -1)
echo "  power mode   : ${MODENAME:-unknown} (id=${MODE:-?})"
echo "  cpu_max      : ${CPUMAX} MHz"
echo "  gpu_max      : ${GPUMAX} MHz"
echo

# Gate on achieved clocks so we never write data labelled with a mode that
# did not apply.
if [[ -n $EXPECT_CPU_MHZ ]]; then
	if (( CPUMAX < EXPECT_CPU_MHZ )); then
		echo "ABORT: cpu_max ${CPUMAX} MHz < expected ${EXPECT_CPU_MHZ} MHz." >&2
		echo "       The power mode did not apply (MAXN often needs a reboot)." >&2
		echo "       No data written." >&2
		exit 2
	fi
	echo "Clock assertion passed (${CPUMAX} >= ${EXPECT_CPU_MHZ} MHz)."
	echo
fi

# --- workload discovery ------------------------------------------------------
NPROC=$(nproc)
CPU_CMD=""
if (( USE_CPU )); then
	if command -v stress-ng >/dev/null; then CPU_CMD="stress-ng --cpu $NPROC --cpu-method matrixprod"
	elif command -v stress >/dev/null;    then CPU_CMD="stress --cpu $NPROC"
	else echo "note: no stress/stress-ng found; using shell CPU burners"; fi
fi
GPU_BIN=""
if (( USE_GPU )); then
	for c in "$DIR/gpu_burn" "$HOME/gpu_burn" "$(command -v gpu_burn 2>/dev/null)"; do
		[[ -n $c && -x $c ]] && { GPU_BIN=$c; break; }
	done
	[[ -z $GPU_BIN ]] && echo "note: gpu_burn not found (build tools/gpu_burn.cu); GPU will stay idle"
fi

OUT=${OUT:-"$PWD/soak-$(date +%Y%m%d-%H%M%S).csv"}
echo "t_s,phase,nvpmodel,cpu_max_mhz,tj_c,cpu_c,gpu_c,fan_rpm,fan_pwm,cpu_mhz,gpu_mhz,vdd_in_mw,cpu_gpu_mw,soc_mw" > "$OUT"

PEAK=0; THROTTLED=0
sample() {
	local tj cpu gpu rpm pwm cf gf
	tj=$(jt_temp_c)
	cpu=$([[ -n $JT_CPU_ZONE ]] && jt_temp_c "$JT_CPU_ZONE" || echo 0)
	gpu=$([[ -n $JT_GPU_ZONE ]] && jt_temp_c "$JT_GPU_ZONE" || echo 0)
	rpm=$(jt_fan_rpm); pwm=$(jt_fan_pwm)
	cf=$(jt_cpu_mhz); gf=$(jt_gpu_mhz)
	(( tj > PEAK )) && PEAK=$tj
	# Under load, a CPU clock materially below max implies throttling or DVFS idle.
	[[ $2 == load ]] && (( cf < CPUMAX * 85 / 100 )) && THROTTLED=$((THROTTLED+1))
	echo "$1,$2,${MODE:-?},$CPUMAX,$tj,$cpu,$gpu,$rpm,$pwm,$cf,$gf,$(jt_power_mw 1),$(jt_power_mw 2),$(jt_power_mw 3)" >> "$OUT"
	printf "  %5ss %-4s Tj %2sC  fan %-5s PWM %-4s cpu %4sMHz gpu %4sMHz  %5sm W\n" \
		"$1" "$2" "$tj" "$rpm" "$pwm" "$cf" "$gf" "$(jt_power_mw 1)"
}

run_phase() {  # $1=label $2=duration
	local t0=$SECONDS
	while (( SECONDS - t0 < $2 )); do
		sample $((ELAPSED + SECONDS - t0)) "$1"
		sleep "$INTERVAL"
	done
	ELAPSED=$((ELAPSED + $2))
}

ELAPSED=0
echo "=== 1/3 baseline (${BASELINE}s) ==="
run_phase idle "$BASELINE"

echo
echo "=== 2/3 load (${LOAD}s) ==="
PIDS=()
if [[ -n $CPU_CMD ]]; then
	$CPU_CMD --timeout $((LOAD+15)) >/dev/null 2>&1 & PIDS+=($!)
elif (( USE_CPU )); then
	for ((i=0;i<NPROC;i++)); do (timeout $((LOAD+15)) bash -c 'while :; do :; done') & PIDS+=($!); done
fi
[[ -n $GPU_BIN ]] && { "$GPU_BIN" $((LOAD+15)) >/dev/null 2>&1 & PIDS+=($!); }
sleep 2
run_phase load "$LOAD"
for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
wait 2>/dev/null

echo
echo "=== 3/3 cooldown (${COOL}s) ==="
run_phase cool "$COOL"

# --- analysis: is the temperature converged, or still climbing? --------------
echo
echo "=== summary ==="
awk -F, -v thr="$THROTTLED" '
	NR>1 && $2=="load" {
		n++; s+=$5; if($5>mx)mx=$5; if($8>rmx)rmx=$8; p+=$12;
		if(cmin==0||$10<cmin)cmin=$10; if($10>cmx)cmx=$10;
		t[n]=$1; y[n]=$5
	}
	END{
		if(n==0){print "  no load samples"; exit}
		printf "  load samples : %d\n  peak Tj      : %d C\n  mean Tj      : %.1f C\n", n, mx, s/n
		printf "  peak fan     : %d RPM\n  mean VDD_IN  : %.1f W\n  cpu clk      : %d-%d MHz\n", rmx, p/n/1000, cmin, cmx
		printf "  throttled    : %s\n", (thr>0 ? thr " samples below 85% of max clock" : "no")
		# Least-squares slope over the final third of the load phase.
		st=int(n*2/3); if(st<2) st=2
		for(i=st;i<=n;i++){k++; sx+=t[i]; sy+=y[i]; sxy+=t[i]*y[i]; sxx+=t[i]*t[i]}
		if(k>2 && (k*sxx-sx*sx)!=0){
			slope=(k*sxy-sx*sy)/(k*sxx-sx*sx)*60
			printf "  final-third drift: %+.2f C/min\n", slope
			if(slope>0.5)      print "  VERDICT: STILL CLIMBING - not converged, extend the soak"
			else if(slope>0.1) print "  VERDICT: slow drift - marginal, longer soak advised"
			else               print "  VERDICT: CONVERGED - thermally stable at this load"
		}
	}' "$OUT"
echo
echo "CSV: $OUT"
