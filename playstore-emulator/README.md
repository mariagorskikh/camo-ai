# Play Store Android phone in your browser (for Uber Eats)

A portrait Android 14 phone with the real Google Play Store, running in Docker
and shown in your browser through noVNC. Sign in with your Google account,
install Uber Eats from the Play Store, and order.

```
browser ──http://host:6080──> noVNC ──> x11vnc ──> Xvfb ──> Android emulator (Pixel 6, portrait)
```

## What you need

A Linux host with **KVM** (`/dev/kvm`). The compose file requires it. Without
KVM the phone can still run in pure software emulation (`docker run` line
below), but it takes about 30 minutes to boot and is painful to use.
Hosts that work:

| Host | Notes |
|---|---|
| Google Cloud VM | `deploy/gcp-create-vm.sh` does everything (needs `gcloud`). Uses `--enable-nested-virtualization`. |
| Hetzner Cloud, Azure Dv3+/Ev3+, AWS `*.metal` | Create an Ubuntu 24.04 VM, copy this folder, run `deploy/vm-bootstrap.sh`. |
| Your own Linux PC / Intel Mac with Linux | `cp .env.example .env`, edit the password, `docker compose up`. |
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

Port 6080 is published on the VM's localhost only, so the tunnel is required.
After the bootstrap, run Docker commands with `sudo` until you log in again
(the bootstrap adds your user to the `docker` group).

### Plain Docker, including software-only mode

```bash
cp .env.example .env            # set VNC_PASSWORD
docker build -t playstore-emulator .
docker run -d --name playstore-emulator --device /dev/kvm \
  -p 127.0.0.1:6080:6080 --env-file .env -v avd-data:/data --shm-size 2g \
  playstore-emulator
```

Drop `--device /dev/kvm` on a host without KVM to get the slow software mode.

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

## Two ways to see the phone

| URL | Transport | Use when |
|---|---|---|
| `/vnc.html` | noVNC over WebSocket | Direct access or SSH tunnel. Smoothest. |
| `/phone?p=<VNC_PASSWORD>` | Plain HTTP polling (JPEG frames + taps) | Behind an edge that breaks WebSockets over HTTP/2, such as Railway's public domains. Works in any browser. |

The `/phone` page draws the screen at a few frames per second; click to tap,
drag to swipe, use the buttons for Back/Home/Recents and the text box to type.

## Railway

`railway up` from this directory works as is (`railway.json` selects the
Dockerfile). Set `VNC_PASSWORD` as a service variable, attach a volume at
`/data`, and open `https://<service-domain>/phone?p=<VNC_PASSWORD>`. Railway
has no KVM, so the phone runs in software mode: expect a 30-minute first boot
and a slow UI. Logins persist on the volume.

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `VNC_PASSWORD` | required | Password asked by the browser viewer. VNC only checks the first 8 characters. |
| `NOVNC_BIND` | `127.0.0.1` | Host address port 6080 is published on. Leave it unless you have TLS in front. |
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

Everything that can order food with your cards is behind the VNC password
only, and the VNC protocol checks just the first 8 characters of it over a
plain, unencrypted connection. That is why port 6080 is bound to `127.0.0.1`
by default: the SSH tunnel provides the real authentication and encryption.
If you need remote access without SSH, put a TLS-terminating reverse proxy
with its own login (for example Caddy with basic auth) in front and only then
set `NOVNC_BIND=0.0.0.0`.

## Caveats

* Uber's app usually runs fine on Play Store emulator images, but Uber can
  change its device-integrity checks at any time. If the app refuses to start,
  try `API_LEVEL=33` or `35`.
* No audio and no camera are wired up (not needed to order).
* The Google Play system image is downloaded under Google's SDK licence at
  build time. Do not push the built image to a public registry.
