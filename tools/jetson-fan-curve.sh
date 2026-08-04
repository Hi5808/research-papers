#!/usr/bin/env bash
# jetson-fan-curve.sh - inspect and set the nvfancontrol fan curve on Jetson.
#
# Handles the TMARGIN encoding: when enabled, the config's TEMP column is
# thermal MARGIN below the limit, not degrees C. Editing it as if it were
# degrees sets thresholds inverted. This tool always speaks in degrees C and
# converts internally.
#
#   sudo ./jetson-fan-curve.sh --show
#   sudo ./jetson-fan-curve.sh --target 60      # full fan by Tj 60 C
#   sudo ./jetson-fan-curve.sh --max            # pin flat out
#   sudo ./jetson-fan-curve.sh --restore        # revert newest backup
#
set -uo pipefail
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=jetson_thermal_lib.sh
source "$DIR/jetson_thermal_lib.sh"

PROFILE=${JT_PROFILE:-quiet}    # override with JT_PROFILE=cool
RPM_MAX=${JT_RPM_MAX:-}         # override autodetected max RPM
ACTION=""; TARGET=""

while [[ $# -gt 0 ]]; do
	case $1 in
		--show) ACTION=show ;;
		--max) ACTION=max ;;
		--restore) ACTION=restore ;;
		--target) ACTION=target; TARGET=${2:-}; shift ;;
		--profile) PROFILE=${2:-quiet}; shift ;;
		-h|--help) sed -n '2,16p' "$0"; exit 0 ;;
		*) echo "unknown arg: $1" >&2; exit 1 ;;
	esac
	shift
done
[[ -n $ACTION ]] || { sed -n '2,16p' "$0"; exit 1; }

jt_detect
[[ -n $JT_FAN_CONF ]] || { echo "ERROR: no nvfancontrol config found. Passive board?" >&2; exit 1; }

# The fan's declared ceiling = highest RPM in the config. Read it from the
# OLDEST backup when one exists, not the live file: --max deliberately writes an
# unreachable target, so detecting from the live config would inherit that
# inflated value and compound on every subsequent run.
if [[ -z $RPM_MAX ]]; then
	RPM_SRC=$(ls -1tr "$JT_FAN_CONF".bak.* 2>/dev/null | head -1)
	RPM_SRC=${RPM_SRC:-$JT_FAN_CONF}
	RPM_MAX=$(grep -oP '^\s*\d+\s+\d+\s+\d+\s+\K\d+' "$RPM_SRC" 2>/dev/null | sort -n | tail -1)
	RPM_MAX=${RPM_MAX:-6000}
	[[ $RPM_SRC != "$JT_FAN_CONF" ]] && echo "fan ceiling ${RPM_MAX} RPM (from stock backup $(basename "$RPM_SRC"))"
fi

show_curve() {
	echo
	jt_print_detected
	echo
	echo "Profile '$PROFILE' as stored, with Tj translation:"
	printf "  %-10s %-8s %-6s %-6s\n" "TEMPcol" "-> Tj" "PWM" "RPM"
	awk -v prof="$PROFILE" -v max="$JT_MAX_TEMP" -v tm="$JT_TMARGIN" '
		$0 ~ "FAN_PROFILE[ \t]+"prof"[ \t]*\\{" {inb=1; next}
		inb && /\}/ {inb=0}
		inb && /^[ \t]*[0-9]/ {
			tj = (tm==1) ? max - $1 : $1
			printf "  %-10s %-8s %-6s %-6s\n", $1, tj " C", $3, $4
		}' "$JT_FAN_CONF"
	echo
	echo "Live: Tj $(jt_temp_c) C   fan $(jt_fan_rpm) RPM at PWM $(jt_fan_pwm)/255"
}

if [[ $ACTION == show ]]; then show_curve; exit 0; fi

[[ $EUID -eq 0 ]] || { echo "ERROR: run as root (writes $JT_FAN_CONF)" >&2; exit 1; }

if [[ $ACTION == restore ]]; then
	BAK=$(ls -1t "$JT_FAN_CONF".bak.* 2>/dev/null | head -1)
	[[ -n $BAK ]] || { echo "ERROR: no backup found" >&2; exit 1; }
	cp -a "$BAK" "$JT_FAN_CONF"
	echo "restored from $BAK"
else
	cp -a "$JT_FAN_CONF" "$JT_FAN_CONF.bak.$(date +%F-%H%M%S)"

	if [[ $ACTION == max ]]; then
		# Target above the ceiling: with close_loop control PWM saturates at 255,
		# which reveals the fan's true maximum rather than the declared one.
		OVER=$(( RPM_MAX * 4 / 3 ))
		ROWS=$(printf "\t\t0\t0\t255\t%s\n\t\t%s\t0\t255\t%s\n" "$OVER" "$JT_MAX_TEMP" "$OVER")
		echo "pinning fan to maximum (target ${OVER} RPM, above the ${RPM_MAX} ceiling)"
	else
		[[ $TARGET =~ ^[0-9]+$ ]] || { echo "ERROR: --target needs a temperature in C" >&2; exit 1; }
		(( TARGET > 0 && TARGET < JT_MAX_TEMP )) || { echo "ERROR: target must be 1..$((JT_MAX_TEMP-1))" >&2; exit 1; }
		# Ramp: full speed by TARGET, stepping down over the 25 C below it.
		c_full=$(jt_temp_to_curve "$TARGET")
		c_mid=$(jt_temp_to_curve $((TARGET - 7)))
		c_low=$(jt_temp_to_curve $((TARGET - 17)))
		r_mid=$(( RPM_MAX * 2 / 3 )); r_low=$(( RPM_MAX * 5 / 12 ))
		p_mid=$(( 255 * 2 / 3 ));     p_low=$(( 255 * 5 / 12 ))
		if [[ $JT_TMARGIN -eq 1 ]]; then   # margin column ascends as temp falls
			ROWS=$(printf "\t\t0\t0\t255\t%s\n\t\t%s\t0\t255\t%s\n\t\t%s\t0\t%s\t%s\n\t\t%s\t0\t%s\t%s\n\t\t%s\t0\t0\t0\n" \
				"$RPM_MAX" "$c_full" "$RPM_MAX" "$c_mid" "$p_mid" "$r_mid" "$c_low" "$p_low" "$r_low" "$JT_MAX_TEMP")
		else
			ROWS=$(printf "\t\t0\t0\t0\t0\n\t\t%s\t0\t%s\t%s\n\t\t%s\t0\t%s\t%s\n\t\t%s\t0\t255\t%s\n\t\t%s\t0\t255\t%s\n" \
				"$c_low" "$p_low" "$r_low" "$c_mid" "$p_mid" "$r_mid" "$c_full" "$RPM_MAX" "$JT_MAX_TEMP" "$RPM_MAX")
		fi
		echo "building curve: full ${RPM_MAX} RPM by Tj ${TARGET} C (column value $c_full)"
	fi

	ROWS="$ROWS" PROFILE="$PROFILE" python3 - "$JT_FAN_CONF" <<'PYEOF'
import os, re, sys
path = sys.argv[1]
src = open(path).read()
rows = "\t\t#TEMP \tHYST\tPWM\tRPM\n" + os.environ["ROWS"].rstrip("\n") + "\n"
pat = re.compile(r"(FAN_PROFILE\s+" + re.escape(os.environ["PROFILE"]) + r"\s*\{\n)(.*?)(\n?\t*\})", re.DOTALL)
m = pat.search(src)
if not m:
    sys.exit("ERROR: profile block not found in config")
open(path, "w").write(src[:m.start()] + m.group(1) + rows + "\t}" + src[m.end():])
PYEOF
	[[ $? -eq 0 ]] || exit 1
fi

# nvfancontrol caches state; without clearing it the new curve may be ignored.
systemctl stop nvfancontrol 2>/dev/null
rm -f /var/lib/nvfancontrol/status
systemctl start nvfancontrol 2>/dev/null
sleep 5
show_curve
echo
echo "Verify the DIRECTION of response: fan RPM must RISE as Tj rises."
echo "If it falls, the TMARGIN interpretation is inverted on this board -"
echo "restore with: sudo $0 --restore"
