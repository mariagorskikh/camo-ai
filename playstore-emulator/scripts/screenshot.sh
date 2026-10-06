#!/usr/bin/env bash
# Save a PNG of the phone screen. Usage:
#   docker exec playstore-emulator /opt/scripts/screenshot.sh > shot.png
set -euo pipefail
adb -s emulator-5554 exec-out screencap -p
