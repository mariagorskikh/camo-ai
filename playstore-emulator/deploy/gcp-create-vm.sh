#!/usr/bin/env bash
# Create a Google Cloud VM with nested virtualisation, copy this project to it,
# build and start the emulator, and print how to connect.
#
#   gcloud auth login && gcloud config set project <your-project>
#   VNC_PASSWORD='something-long' ./deploy/gcp-create-vm.sh
#
# n2-standard-4 (4 vCPU, 16 GB) costs roughly $0.20/hour. Delete the VM when
# you are done:  gcloud compute instances delete android-playstore --zone $ZONE
set -euo pipefail

NAME="${NAME:-android-playstore}"
ZONE="${ZONE:-us-central1-a}"
MACHINE="${MACHINE:-n2-standard-4}"
VNC_PASSWORD="${VNC_PASSWORD:-$(head -c 48 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | cut -c1-8)}"
GEO_LAT="${GEO_LAT:-40.7580}"
GEO_LON="${GEO_LON:--73.9855}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[gcp] creating $NAME ($MACHINE, nested virtualisation) in $ZONE"
gcloud compute instances create "$NAME" \
  --zone "$ZONE" --machine-type "$MACHINE" \
  --enable-nested-virtualization \
  --min-cpu-platform "Intel Cascade Lake" \
  --image-family ubuntu-2404-lts-amd64 --image-project ubuntu-os-cloud \
  --boot-disk-size 60GB --boot-disk-type pd-balanced

# From here on a failure would leave a billing VM behind: delete it unless
# KEEP_VM_ON_FAILURE=1.
cleanup_on_failure() {
  local rc=$?
  trap - EXIT
  if [ "$rc" -ne 0 ]; then
    if [ "${KEEP_VM_ON_FAILURE:-0}" = "1" ]; then
      echo "[gcp] FAILED (exit $rc). VM $NAME kept for debugging; delete it with:" >&2
      echo "      gcloud compute instances delete $NAME --zone $ZONE" >&2
    else
      echo "[gcp] FAILED (exit $rc). Deleting VM $NAME so it does not keep billing." >&2
      gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet || true
    fi
  fi
  exit "$rc"
}
trap cleanup_on_failure EXIT

echo "[gcp] waiting for SSH"
for _ in $(seq 1 30); do
  gcloud compute ssh "$NAME" --zone "$ZONE" --quiet --command 'true' 2>/dev/null && break
  sleep 10
done

echo "[gcp] copying project"
gcloud compute scp --zone "$ZONE" --recurse "$PROJECT_DIR" "$NAME:~/playstore-emulator" --quiet

echo "[gcp] bootstrapping (Docker install + image build, ~5-10 min)"
# Values are shell-quoted with printf %q so quotes or metacharacters in the
# password or coordinates cannot alter the remote command.
REMOTE_CMD="$(printf 'sudo VNC_PASSWORD=%q GEO_LAT=%q GEO_LON=%q bash ~/playstore-emulator/deploy/vm-bootstrap.sh' \
  "$VNC_PASSWORD" "$GEO_LAT" "$GEO_LON")"
gcloud compute ssh "$NAME" --zone "$ZONE" --quiet --command "$REMOTE_CMD"

cat <<MSG

Connect with an SSH tunnel (nothing is exposed publicly):
  gcloud compute ssh $NAME --zone $ZONE -- -L 6080:localhost:6080
then open  http://localhost:6080/vnc.html   (VNC password: $VNC_PASSWORD)

Stop paying:  gcloud compute instances delete $NAME --zone $ZONE
MSG
