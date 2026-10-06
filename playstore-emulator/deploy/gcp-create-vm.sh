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
VNC_PASSWORD="${VNC_PASSWORD:-$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 16)}"
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

echo "[gcp] waiting for SSH"
for _ in $(seq 1 30); do
  gcloud compute ssh "$NAME" --zone "$ZONE" --quiet --command 'true' 2>/dev/null && break
  sleep 10
done

echo "[gcp] copying project"
gcloud compute scp --zone "$ZONE" --recurse "$PROJECT_DIR" "$NAME:~/playstore-emulator" --quiet

echo "[gcp] bootstrapping (Docker install + image build, ~5-10 min)"
gcloud compute ssh "$NAME" --zone "$ZONE" --quiet --command \
  "sudo VNC_PASSWORD='$VNC_PASSWORD' GEO_LAT='$GEO_LAT' GEO_LON='$GEO_LON' bash ~/playstore-emulator/deploy/vm-bootstrap.sh"

cat <<MSG

Connect with an SSH tunnel (nothing is exposed publicly):
  gcloud compute ssh $NAME --zone $ZONE -- -L 6080:localhost:6080
then open  http://localhost:6080/vnc.html   (VNC password: $VNC_PASSWORD)

Stop paying:  gcloud compute instances delete $NAME --zone $ZONE
MSG
