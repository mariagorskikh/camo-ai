# Android emulator with the Google Play Store, portrait phone, reachable from a
# browser through noVNC. Needs /dev/kvm on the host for usable speed.
#
#   docker build -t playstore-emulator .
#   docker run --device /dev/kvm -p 6080:6080 -e VNC_PASSWORD=change-me \
#              -v avd-data:/data playstore-emulator
#
# The Google Play system image is downloaded from Google at build time under
# the Android SDK licence; keep the built image private.

FROM ubuntu:24.04

ARG API_LEVEL=34
ARG IMG_TYPE=google_apis_playstore
ARG ABI=x86_64
ARG CMDLINE_TOOLS_BUILD=13114758

ENV DEBIAN_FRONTEND=noninteractive \
    ANDROID_SDK_ROOT=/opt/android-sdk \
    ANDROID_HOME=/opt/android-sdk \
    ANDROID_AVD_HOME=/data/avd \
    SYSTEM_IMAGE="system-images;android-${API_LEVEL};${IMG_TYPE};${ABI}" \
    PATH="/opt/android-sdk/cmdline-tools/latest/bin:/opt/android-sdk/platform-tools:/opt/android-sdk/emulator:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl unzip openjdk-17-jre-headless \
        xvfb x11vnc novnc websockify xdotool \
        libpulse0 libnss3 libasound2t64 libfontconfig1 libdbus-1-3 libglib2.0-0 \
        libgl1 libegl1 libgl1-mesa-dri libx11-6 libx11-xcb1 libxcb1 libxext6 \
        libxcomposite1 libxcursor1 libxdamage1 libxfixes3 libxi6 libxrandr2 \
        libxrender1 libxtst6 libxkbcommon0 libxkbcommon-x11-0 libxkbfile1 \
        libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-shm0 \
        libxcb-sync1 libxcb-xfixes0 libxcb-xinerama0 libxcb-xinput0 libxcb-xkb1 \
        libfreetype6 libnspr4 libsm6 libice6 libbz2-1.0 libtinfo6 \
    && rm -rf /var/lib/apt/lists/*

# Android command-line tools, emulator, platform-tools and the Play Store image.
RUN mkdir -p ${ANDROID_SDK_ROOT}/cmdline-tools \
    && curl -fsSL -o /tmp/ct.zip \
         https://dl.google.com/android/repository/commandlinetools-linux-${CMDLINE_TOOLS_BUILD}_latest.zip \
    && unzip -q /tmp/ct.zip -d /tmp/ct \
    && mv /tmp/ct/cmdline-tools ${ANDROID_SDK_ROOT}/cmdline-tools/latest \
    && rm -rf /tmp/ct /tmp/ct.zip \
    && yes | sdkmanager --sdk_root=${ANDROID_SDK_ROOT} --licenses > /dev/null \
    && sdkmanager --sdk_root=${ANDROID_SDK_ROOT} \
         "platform-tools" "emulator" "${SYSTEM_IMAGE}" \
    && rm -rf ${ANDROID_SDK_ROOT}/.downloadIntermediates

COPY scripts/ /opt/scripts/
RUN chmod +x /opt/scripts/*.sh

# Virtual-device data (Google login, installed apps) lives here. Mount a volume.
VOLUME /data

# 6080 = noVNC in the browser, 5900 = raw VNC, 5555 = adb (optional)
EXPOSE 6080 5900 5555

ENTRYPOINT ["/opt/scripts/start.sh"]
