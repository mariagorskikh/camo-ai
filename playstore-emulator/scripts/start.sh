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
# Platforms such as Railway tell the app which port to serve HTTP on via PORT.
NOVNC_PORT="${PORT:-6080}"
export DISPLAY=:0
export ANDROID_AVD_HOME="${ANDROID_AVD_HOME:-/data/avd}"
export ANDROID_EMULATOR_HOME="${ANDROID_EMULATOR_HOME:-/data/.android}"
export ANDROID_USER_HOME="${ANDROID_EMULATOR_HOME}"

log() { echo "[start] $*"; }

mkdir -p "$ANDROID_AVD_HOME" "$ANDROID_EMULATOR_HOME"

# Without KVM, default to a lighter phone (smaller screen, less RAM) unless
# the caller chose explicitly. Set PHONE_PROFILE=full to force the Pixel 6 size.
if [ ! -e /dev/kvm ] && [ "${PHONE_PROFILE:-auto}" != "full" ]; then
  export LCD_WIDTH="${LCD_WIDTH:-540}" LCD_HEIGHT="${LCD_HEIGHT:-1200}" LCD_DENSITY="${LCD_DENSITY:-240}"
  EMULATOR_RAM_MB="${EMULATOR_RAM_MB_SOFTWARE:-2560}"
  log "no KVM: using the light phone profile ${LCD_WIDTH}x${LCD_HEIGHT}@${LCD_DENSITY}dpi, ${EMULATOR_RAM_MB} MB RAM"
fi

# ---------------------------------------------------------------- X + VNC ----
log "starting Xvfb ${SCREEN_WIDTH}x${SCREEN_HEIGHT}"
# -s 0 / -dpms: never blank the virtual screen.
Xvfb :0 -screen 0 "${SCREEN_WIDTH}x${SCREEN_HEIGHT}x24" -nolisten tcp -ac +extension RANDR -s 0 -dpms &
# Wait until the X server actually answers (the socket appears a bit earlier).
for _ in $(seq 1 100); do xset q >/dev/null 2>&1 && break; sleep 0.2; done
xset q >/dev/null 2>&1 || { log "ERROR: Xvfb did not come up"; exit 1; }
xset s off -dpms

# The VNC protocol only checks the first 8 characters of a password. The real
# protection is keeping port 6080 off the public internet (see README).
if [ "${#VNC_PASSWORD}" -gt 8 ]; then
  log "note: VNC authentication only uses the first 8 characters of VNC_PASSWORD"
fi
mkdir -p /root/.vnc
x11vnc -storepasswd "$VNC_PASSWORD" /root/.vnc/passwd >/dev/null 2>&1
log "starting x11vnc on :5900 (password protected)"
x11vnc -display :0 -rfbport 5900 -rfbauth /root/.vnc/passwd -forever -shared \
       -noxdamage -repeat -quiet -bg -o /tmp/x11vnc.log

# websockify stays on localhost; the public port is served by wsproxy.py,
# which restores WebSocket upgrade headers that HTTP/2 edges (Railway) strip.
websockify --daemon --web /usr/share/novnc 127.0.0.1:6081 localhost:5900 >/tmp/websockify.log 2>&1
# WebSocket-free viewer at /phone?p=<password> for edges that break WebSockets.
python3 "$(dirname "$0")/webview.py" 6082 "$VNC_PASSWORD" "$SCREEN_WIDTH" "$SCREEN_HEIGHT" &
log "starting noVNC on http://0.0.0.0:${NOVNC_PORT}/vnc.html (and /phone?p=... viewer)"
python3 "$(dirname "$0")/wsproxy.py" "$NOVNC_PORT" 6081 6082 &

# -------------------------------------------------------------------- AVD ----
"$(dirname "$0")/create-avd.sh" "$AVD_NAME" "$DEVICE_PROFILE" "$EMULATOR_RAM_MB"

# ------------------------------------------------------------- acceleration --
if [ -e /dev/kvm ] && [ -r /dev/kvm ] && [ -w /dev/kvm ]; then
  ACCEL="-accel on"
  log "KVM available: hardware acceleration on"
else
  ACCEL="-accel off"
  # Multi-threaded TCG (one host thread per vCPU) sounds attractive but in
  # testing the guest never finished booting with it; opt in with
  # EMULATOR_EXTRA_ARGS="-qemu -accel tcg,thread=multi" if you want to try.
  QEMU_TAIL=""
  log "WARNING: /dev/kvm not available. Running in pure software emulation;"
  log "         boot takes 15-25 minutes and the phone is slow."
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
  $EMULATOR_EXTRA_ARGS ${QEMU_TAIL:-} >/tmp/emulator.log 2>&1 &
EMU_PID=$!

# Fit the phone window to the virtual screen (the emulator keeps the aspect
# ratio itself; the 60 px strip on the right is its toolbar).
(
  for _ in $(seq 1 120); do
    WID=$(xdotool search --name '^Android Emulator' 2>/dev/null | head -1 || true)
    [ -n "$WID" ] && break
    sleep 1
  done
  if [ -n "${WID:-}" ]; then
    # The emulator re-applies its own geometry shortly after mapping the
    # window, so keep asking until the height actually sticks.
    for _ in $(seq 1 30); do
      xdotool windowmove "$WID" 0 0 || true
      xdotool windowsize "$WID" $((SCREEN_WIDTH - 60)) "$SCREEN_HEIGHT" || true
      sleep 2
      H=$(xdotool getwindowgeometry --shell "$WID" 2>/dev/null | sed -n 's/^HEIGHT=//p')
      if [ "${H:-0}" -ge $((SCREEN_HEIGHT - 50)) ]; then
        log "phone window fitted to ${SCREEN_WIDTH}x${SCREEN_HEIGHT} (height $H)"
        break
      fi
    done
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
  # Animations cost real frames in software rendering; turn them off.
  for k in window_animation_scale transition_animation_scale animator_duration_scale; do
    adb -s emulator-5554 shell settings put global "$k" 0 >/dev/null 2>&1 || true
  done
  if [ ! -e /dev/kvm ] && [ "${KEEP_GOOGLE_APPS:-0}" != "1" ]; then
    # Without KVM every background Google app competes for the emulated CPU.
    # Disable the heavy ones that are not needed to install and use apps
    # (Play Store, Play Services, Chrome and the keyboard stay). KEEP_GOOGLE_APPS=1 skips this.
    for p in com.google.android.googlequicksearchbox com.google.android.apps.photos \
             com.google.android.youtube com.google.android.gm com.google.android.apps.messaging \
             com.google.android.apps.wellbeing com.google.android.tts \
             com.google.android.apps.youtube.music com.google.android.videos com.google.android.apps.maps; do
      adb -s emulator-5554 shell pm disable-user --user 0 "$p" >/dev/null 2>&1 || true
    done
    log "software mode: disabled background Google apps to free CPU"
  fi
  adb -s emulator-5554 shell svc power stayon true >/dev/null 2>&1 || true
  adb -s emulator-5554 emu geo fix "$GEO_LON" "$GEO_LAT" >/dev/null 2>&1 || true
  log "GPS set to lat=$GEO_LAT lon=$GEO_LON"
  log "READY -> open http://<host>:${NOVNC_PORT}/vnc.html and use the VNC password"
) &

# Graceful stop: ask Android to shut down and give it time to flush the
# persisted AVD before falling back to signals.
shutdown() {
  log "stopping: asking the emulator to shut down"
  # adb can block when the guest is not fully up, so bound the request.
  timeout 15 adb -s emulator-5554 emu kill >/dev/null 2>&1 || true
  for _ in $(seq 1 60); do
    kill -0 "$EMU_PID" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$EMU_PID" 2>/dev/null; then
    log "emulator still running after 60s, sending TERM"
    kill "$EMU_PID" 2>/dev/null || true
    for _ in $(seq 1 15); do kill -0 "$EMU_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$EMU_PID" 2>/dev/null || true
  fi
}
trap shutdown TERM INT
tail -F /tmp/emulator.log 2>/dev/null &
wait $EMU_PID || true
wait $EMU_PID 2>/dev/null || true
# Stop the helper subshells (boot watcher, log tail) left behind.
for j in $(jobs -p); do kill "$j" 2>/dev/null || true; done
log "emulator exited"
