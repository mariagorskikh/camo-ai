#!/usr/bin/env bash
# Create the portrait Play Store AVD if it does not exist yet.
#   create-avd.sh <name> <device profile> <ram MB>
set -euo pipefail
NAME="${1:-playstore}"
PROFILE="${2:-pixel_6}"
RAM_MB="${3:-4096}"
: "${SYSTEM_IMAGE:?SYSTEM_IMAGE must be set (e.g. system-images;android-34;google_apis_playstore;x86_64)}"
export ANDROID_AVD_HOME="${ANDROID_AVD_HOME:-/data/avd}"
mkdir -p "$ANDROID_AVD_HOME"

if [ -f "$ANDROID_AVD_HOME/$NAME.avd/config.ini" ]; then
  echo "[avd] '$NAME' already exists, keeping its data"
  exit 0
fi

echo "[avd] creating '$NAME' from $SYSTEM_IMAGE ($PROFILE)"
echo no | avdmanager --silent create avd --name "$NAME" --package "$SYSTEM_IMAGE" --device "$PROFILE"

CFG="$ANDROID_AVD_HOME/$NAME.avd/config.ini"
set_cfg() {  # set_cfg key value  (replace or append)
  if grep -q "^$1=" "$CFG"; then sed -i "s|^$1=.*|$1=$2|" "$CFG"; else echo "$1=$2" >> "$CFG"; fi
}
set_cfg hw.initialOrientation portrait
set_cfg hw.keyboard yes
set_cfg hw.gpu.enabled yes
set_cfg hw.gpu.mode swiftshader_indirect
set_cfg hw.ramSize "$RAM_MB"
set_cfg hw.cpu.ncore 4
set_cfg vm.heapSize 576
set_cfg disk.dataPartition.size 8G
set_cfg hw.audioInput no
set_cfg hw.audioOutput no
set_cfg hw.camera.back none
set_cfg hw.camera.front none
set_cfg hw.gps yes
set_cfg PlayStore.enabled true
set_cfg showDeviceFrame no
set_cfg skin.dynamic yes
echo "[avd] created: $CFG"
