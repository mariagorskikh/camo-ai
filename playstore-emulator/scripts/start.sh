#!/usr/bin/env bash
# Container entrypoint: virtual X screen -> VNC -> noVNC, then the emulator.
set -euo pipefail

: "${VNC_PASSWORD:?VNC_PASSWORD must be set}"
AVD_NAME="${AVD_NAME:-playstore}"
DEVICE_PROFILE="${DEVICE_PROFILE:-pixel_6}"      # 1080x2400, portrait phone
SCREEN_WIDTH="${SCREEN_WIDTH:-1000}"
SCREEN_HEIGHT="${SCREEN_HEIGHT:-1900}"
EMULATOR_RAM_MB="${EMULATOR_RAM_MB:-4096}"
GEO_LAT="${GEO_LAT:-40.7580}"
GEO_LON="${GEO_LON:--73.9855}"
EMULATOR_EXTRA_ARGS="${EMULATOR_EXTRA_ARGS:-}"
export DISPLAY=:0
export ANDROID_AVD_HOME="${ANDROID_AVD_HOME:-/data/avd}"
export ANDROID_EMULATOR_HOME="${ANDROID_EMULATOR_HOME:-/data/.android}"
export ANDROID_USER_HOME="${ANDROID_EMULATOR_HOME}"

log() { echo "[start] $*"; }

mkdir -p "$ANDROID_AVD_HOME" "$ANDROID_EMULATOR_HOME"

# ---------------------------------------------------------------- X + VNC ----
log "starting Xvfb ${SCREEN_WIDTH}x${SCREEN_HEIGHT}"
Xvfb :0 -screen 0 "${SCREEN_WIDTH}x${SCREEN_HEIGHT}x24" -nolisten tcp -ac +extension RANDR &
for _ in $(seq 1 50); do [ -e /tmp/.X11-unix/X0 ] && break; sleep 0.2; done
xset s off -dpms 2>/dev/null || true

mkdir -p /root/.vnc
x11vnc -storepasswd "$VNC_PASSWORD" /root/.vnc/passwd >/dev/null 2>&1
log "starting x11vnc on :5900 (password protected)"
x11vnc -display :0 -rfbport 5900 -rfbauth /root/.vnc/passwd -forever -shared \
       -noxdamage -repeat -quiet -bg -o /tmp/x11vnc.log

log "starting noVNC on http://0.0.0.0:6080/vnc.html"
websockify --daemon --web /usr/share/novnc 6080 localhost:5900 >/tmp/websockify.log 2>&1

# -------------------------------------------------------------------- AVD ----
"$(dirname "$0")/create-avd.sh" "$AVD_NAME" "$DEVICE_PROFILE" "$EMULATOR_RAM_MB"

# ------------------------------------------------------------- acceleration --
if [ -e /dev/kvm ] && [ -r /dev/kvm ] && [ -w /dev/kvm ]; then
  ACCEL="-accel on"
  log "KVM available: hardware acceleration on"
else
  ACCEL="-accel off"
  log "WARNING: /dev/kvm not available. Running in pure software emulation;"
  log "         boot can take 15+ minutes and the phone will be very slow."
  log "         Run the container with --device /dev/kvm on a host that has KVM."
fi

# --------------------------------------------------------------- emulator ----
log "booting AVD '$AVD_NAME' (Play Store image: ${SYSTEM_IMAGE:-?})"
# shellcheck disable=SC2086
emulator -avd "$AVD_NAME" $ACCEL \
  -gpu swiftshader_indirect \
  -no-metrics -no-snapshot -no-boot-anim -no-audio \
  -camera-back none -camera-front none \
  -netdelay none -netspeed full \
  -port 5554 \
  $EMULATOR_EXTRA_ARGS >/tmp/emulator.log 2>&1 &
EMU_PID=$!

# Fit the phone window to the virtual screen (the emulator keeps the aspect
# ratio itself; the 60 px strip on the right is its toolbar).
(
  for _ in $(seq 1 120); do
    WID=$(xdotool search --name '^Android Emulator' 2>/dev/null | head -1)
    [ -n "$WID" ] && break
    sleep 1
  done
  if [ -n "${WID:-}" ]; then
    xdotool windowmove "$WID" 0 0
    xdotool windowsize "$WID" $((SCREEN_WIDTH - 60)) "$SCREEN_HEIGHT"
    log "phone window fitted to ${SCREEN_WIDTH}x${SCREEN_HEIGHT}"
  fi
) &

# Optional: let "adb connect <host>:5555" reach the phone from outside.
(sleep 20; adb -s emulator-5554 tcpip 5555 >/dev/null 2>&1 || true) &

# Post-boot setup runs in the background so a slow boot never blocks VNC.
(
  adb wait-for-device >/dev/null 2>&1
  until [ "$(adb -s emulator-5554 shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = "1" ]; do
    sleep 5
  done
  log "Android booted"
  adb -s emulator-5554 shell settings put system screen_off_timeout 2147483647 >/dev/null 2>&1 || true
  adb -s emulator-5554 shell svc power stayon true >/dev/null 2>&1 || true
  adb -s emulator-5554 emu geo fix "$GEO_LON" "$GEO_LAT" >/dev/null 2>&1 || true
  log "GPS set to lat=$GEO_LAT lon=$GEO_LON"
  log "READY -> open http://<host>:6080/vnc.html and use the VNC password"
) &

trap 'log "stopping"; adb -s emulator-5554 emu kill >/dev/null 2>&1 || true; kill $EMU_PID 2>/dev/null || true' TERM INT
tail -F /tmp/emulator.log 2>/dev/null &
wait $EMU_PID
