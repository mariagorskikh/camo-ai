# Play Store Android phone in your browser (for Uber Eats)

A portrait Android 14 phone with the real Google Play Store, running in Docker
and shown in your browser through noVNC. Sign in with your Google account,
install Uber Eats from the Play Store, and order.

```
browser ──http://host:6080──> noVNC ──> x11vnc ──> Xvfb ──> Android emulator (Pixel 6, portrait)
```

## What you need

A Linux host with **KVM** (`/dev/kvm`). Without it the phone still runs but in
pure software emulation, which takes 15+ minutes to boot and is painful to use.
Hosts that work:

| Host | Notes |
|---|---|
| Google Cloud VM | `deploy/gcp-create-vm.sh` does everything (needs `gcloud`). Uses `--enable-nested-virtualization`. |
| Hetzner Cloud, Azure Dv3+/Ev3+, AWS `*.metal` | Create an Ubuntu 24.04 VM, copy this folder, run `deploy/vm-bootstrap.sh`. |
| Your own Linux PC / Intel Mac with Linux | `docker compose up` works directly. |
| Railway, Render, Fly, DigitalOcean, plain Docker Desktop on macOS | No KVM. Don't bother. |

Recommended size: 4 vCPU, 16 GB RAM, 60 GB disk.

## Quick start (any Ubuntu VM with KVM)

```bash
# on the VM
sudo VNC_PASSWORD='pick-a-long-password' bash playstore-emulator/deploy/vm-bootstrap.sh
```

Then from your laptop:

```bash
ssh -L 6080:localhost:6080 <user>@<vm-ip>
```

and open <http://localhost:6080/vnc.html>. Enter the VNC password. The phone
appears in portrait. Keyboard and mouse work as touch and typing.

### Google Cloud in one command

```bash
gcloud auth login && gcloud config set project <your-project>
VNC_PASSWORD='pick-a-long-password' ./deploy/gcp-create-vm.sh
```

It creates an `n2-standard-4` VM (about $0.20/hour), builds the image, starts
the phone, and prints the tunnel command. Delete the VM when done:

```bash
gcloud compute instances delete android-playstore --zone us-central1-a
```

## Ordering Uber Eats

1. Open <http://localhost:6080/vnc.html> and wait for the Android home screen
   (first boot takes about a minute with KVM).
2. Open **Play Store**, sign in with your Google account. If Google asks for
   2-step verification, approve it on your real phone as usual.
3. Search **Uber Eats**, install, open, sign in to Uber.
4. Delivery address: type it manually, or set GPS first so "current location"
   is right:
   ```bash
   docker exec playstore-emulator /opt/scripts/set-location.sh 37.7749 -122.4194
   ```
   or change `GEO_LAT`/`GEO_LON` in `.env` and restart.
5. Pay with a card saved in Uber or typed in the app. Google Pay generally does
   not work on emulators, so add a card inside Uber Eats instead.

Your Google login, installed apps and Uber session persist across restarts in
the `avd-data` Docker volume.

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `VNC_PASSWORD` | required | Password asked by the browser viewer. |
| `GEO_LAT`, `GEO_LON` | `40.7580`, `-73.9855` | GPS position set after boot. |
| `SCREEN_WIDTH`, `SCREEN_HEIGHT` | `1000`, `1900` | Virtual monitor the phone is drawn on. Keep it taller than wide. |
| `EMULATOR_RAM_MB` | `4096` | RAM given to Android. |

Build-time arguments (`docker build --build-arg ...`): `API_LEVEL` (default
`34`, Android 14; `35` and `36` also have Play Store images), `ABI` (`x86_64`).

## Useful commands

```bash
docker compose logs -f                                   # boot progress, "READY" line
docker exec playstore-emulator /opt/scripts/screenshot.sh > shot.png
docker exec playstore-emulator adb shell input text 'hello'
adb connect <vm-ip>:5555                                 # if you publish port 5555
```

The emulator's own toolbar (right side of the phone) has rotate, volume,
and the "..." extended controls for location, cellular, and battery.

## Security

Everything that can order food with your cards is behind the VNC password only.
Keep port 6080 closed on the firewall and use the SSH tunnel. If you must open
it, put it behind HTTPS with a reverse proxy.

## Caveats

* Uber's app usually runs fine on Play Store emulator images, but Uber can
  change its device-integrity checks at any time. If the app refuses to start,
  try `API_LEVEL=33` or `35`.
* No audio and no camera are wired up (not needed to order).
* The Google Play system image is downloaded under Google's SDK licence at
  build time. Do not push the built image to a public registry.
