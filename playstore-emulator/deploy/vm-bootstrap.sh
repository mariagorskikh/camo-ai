#!/usr/bin/env bash
# Run on a fresh Ubuntu 22.04/24.04 VM that has /dev/kvm (nested virtualisation
# or bare metal). Installs Docker, builds the emulator image and starts it.
#
#   sudo VNC_PASSWORD='something-long' bash vm-bootstrap.sh
#
# Optional env: GEO_LAT / GEO_LON (starting GPS), REPO_URL (git source if this
# script is run on its own rather than from a copied checkout).
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/mariagorskikh/camo-ai.git}"
VNC_PASSWORD="${VNC_PASSWORD:-}"
GEO_LAT="${GEO_LAT:-40.7580}"
GEO_LON="${GEO_LON:--73.9855}"

if [ ! -e /dev/kvm ]; then
  echo "ERROR: /dev/kvm is missing. Use a VM with nested virtualisation enabled" >&2
  echo "       (GCP: --enable-nested-virtualization, Azure Dv3+/Ev3+, Hetzner Cloud, AWS *.metal)." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[bootstrap] installing Docker"
  curl -fsSL https://get.docker.com | sh
fi
systemctl enable --now docker

# Locate the project: next to this script, or clone it.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$HERE/Dockerfile" ]; then
  PROJECT="$HERE"
else
  apt-get install -y -qq git
  git clone --depth 1 "$REPO_URL" /opt/camo-ai
  PROJECT=/opt/camo-ai/playstore-emulator
fi
cd "$PROJECT"

if [ -z "$VNC_PASSWORD" ]; then
  # 8 characters: VNC authentication ignores anything beyond that anyway.
  VNC_PASSWORD="$(head -c 48 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | cut -c1-8)"
  echo "[bootstrap] generated VNC password: $VNC_PASSWORD"
fi
cat > .env <<ENV
VNC_PASSWORD=$VNC_PASSWORD
GEO_LAT=$GEO_LAT
GEO_LON=$GEO_LON
NOVNC_BIND=127.0.0.1
ENV
chmod 600 .env

# When run through sudo, hand the checkout (and .env) back to the real user and
# let them run docker without sudo (takes effect on their next login).
if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ]; then
  chown -R "$SUDO_USER" "$PROJECT"
  usermod -aG docker "$SUDO_USER" || true
fi

echo "[bootstrap] building image (downloads ~2 GB from Google, takes a few minutes)"
docker compose build
echo "[bootstrap] starting emulator"
docker compose up -d

cat <<MSG

Done. The phone boots in about a minute. Port 6080 listens on this VM's
localhost only, so connect with an SSH tunnel from your laptop:
    ssh -L 6080:localhost:6080 <user>@<this-vm>
and open  http://localhost:6080/vnc.html
VNC password: $VNC_PASSWORD
Logs: docker compose logs -f   (prefix with sudo until you log in again)
MSG
