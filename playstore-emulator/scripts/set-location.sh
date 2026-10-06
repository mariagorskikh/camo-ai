#!/usr/bin/env bash
# Move the phone's GPS. Usage (inside or outside the container):
#   docker exec playstore-emulator /opt/scripts/set-location.sh 37.7749 -122.4194
set -euo pipefail
LAT="${1:?latitude}"; LON="${2:?longitude}"
adb -s emulator-5554 emu geo fix "$LON" "$LAT"
echo "GPS set to lat=$LAT lon=$LON"
