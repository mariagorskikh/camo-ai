#!/usr/bin/env python3
"""Plain-HTTP phone viewer (no WebSockets) for edges that break them.

Serves a page that polls JPEG frames of the X screen and sends taps,
swipes, keys and text, all as ordinary HTTP requests, so it works through
HTTP/2-only edges such as Railway's. Input is injected with xdotool into
the emulator window on DISPLAY.

    webview.py <listen_port> <password> <screen_width> <screen_height>

The password must be sent as ?p=<password> on every request.
"""
import base64, hmac, json, os, subprocess, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]); PASSWORD = sys.argv[2]
SCREEN_W = int(sys.argv[3]); SCREEN_H = int(sys.argv[4])
DISPLAY = os.environ.get("DISPLAY", ":0")
PHONE_W = SCREEN_W - 60          # the emulator window is fitted to this (toolbar on the right)
SCALE = 0.5                      # frames are sent at half resolution

_frame = {"jpg": b"", "t": 0.0}
_lock = threading.Lock()

def grab():
    """Capture the phone area of the X screen as a JPEG."""
    cmd = ["ffmpeg", "-loglevel", "error", "-f", "x11grab", "-video_size", f"{PHONE_W}x{SCREEN_H}",
           "-i", f"{DISPLAY}+0,0", "-frames:v", "1", "-vf", f"scale=iw*{SCALE}:ih*{SCALE}",
           "-q:v", "6", "-f", "image2pipe", "-vcodec", "mjpeg", "-"]
    return subprocess.run(cmd, capture_output=True, timeout=10).stdout

def grabber():
    while True:
        try:
            jpg = grab()
            if jpg:
                with _lock:
                    _frame["jpg"] = jpg; _frame["t"] = time.time()
        except Exception as e:  # noqa: BLE001
            print("[webview] grab error", e, flush=True)
        time.sleep(0.25)

def xdo(*args):
    subprocess.run(["xdotool", *args], env={**os.environ, "DISPLAY": DISPLAY}, timeout=10)

def to_screen(x, y):
    return int(x / SCALE), int(y / SCALE)

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phone</title><style>
html,body{margin:0;background:#111;color:#ddd;font:14px system-ui;height:100%}
#wrap{display:flex;flex-direction:column;align-items:center;gap:8px;padding:8px}
#scr{max-height:calc(100vh - 70px);max-width:100vw;touch-action:none;cursor:crosshair;background:#000}
#bar{display:flex;gap:6px;flex-wrap:wrap;justify-content:center}
button{background:#333;color:#eee;border:1px solid #555;border-radius:6px;padding:6px 10px}
input{background:#222;color:#eee;border:1px solid #555;border-radius:6px;padding:6px;width:220px}
#st{font-size:12px;color:#888}
</style></head><body><div id="wrap">
<img id="scr" alt="phone screen">
<div id="bar">
<button data-k="Escape">Back</button><button data-k="super">Home</button><button data-k="alt+Tab">Recents</button>
<input id="txt" placeholder="type text, press Enter"><button id="send">Send text</button>
<button data-k="Return">Enter</button><button data-k="BackSpace">Backspace</button>
</div><div id="st">connecting…</div></div>
<script>
const P = new URLSearchParams(location.search).get('p') || '';
const BASE = location.pathname.replace(/\\/$/, '');
const q = (path, body) => fetch(BASE + path + '?p=' + encodeURIComponent(P), body ? {method:'POST', body: JSON.stringify(body)} : {}).then(r => r.ok ? r : Promise.reject(r.status));
const img = document.getElementById('scr'), st = document.getElementById('st');
let busy = false, frames = 0, t0 = Date.now();
async function poll(){ if (busy) return; busy = true; try { const r = await q('/frame'); const b = await r.blob(); img.src = URL.createObjectURL(b); frames++; st.textContent = `live · ${(frames/((Date.now()-t0)/1000)).toFixed(1)} fps`; } catch(e){ st.textContent = 'error ' + e; } busy = false; }
setInterval(poll, 350); poll();
function pos(ev){ const r = img.getBoundingClientRect(); return [Math.round((ev.clientX - r.left) * img.naturalWidth / r.width), Math.round((ev.clientY - r.top) * img.naturalHeight / r.height)]; }
let down = null;
img.addEventListener('pointerdown', ev => { down = [pos(ev), Date.now()]; ev.preventDefault(); });
img.addEventListener('pointerup', ev => { if (!down) return; const [p0, t] = down; down = null; const p1 = pos(ev);
  const dist = Math.hypot(p1[0]-p0[0], p1[1]-p0[1]);
  if (dist < 8) q('/tap', {x:p0[0], y:p0[1]}); else q('/swipe', {x0:p0[0], y0:p0[1], x1:p1[0], y1:p1[1], ms: Math.min(1500, Date.now()-t)}); });
document.querySelectorAll('button[data-k]').forEach(b => b.onclick = () => q('/key', {k: b.dataset.k}));
const txt = document.getElementById('txt'); const sendTxt = () => { if (txt.value) { q('/text', {t: txt.value}); txt.value=''; } };
document.getElementById('send').onclick = sendTxt; txt.addEventListener('keydown', ev => { if (ev.key === 'Enter') { sendTxt(); } });
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass
    def _auth(self):
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        ok = hmac.compare_digest(qs.get("p", [""])[0], PASSWORD)
        if not ok:
            self.send_response(401); self.send_header("Content-Type", "text/plain"); self.end_headers()
            self.wfile.write(b"add ?p=<password> to the URL")
        return ok
    def _send(self, code, ctype, body):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    @staticmethod
    def _strip(path):
        return path[len("/phone"):] if path.startswith("/phone") else path
    def do_GET(self):
        path = self._strip(urllib.parse.urlparse(self.path).path)
        if path == "/healthz":
            return self._send(200, "text/plain", b"ok")
        if not self._auth(): return
        if path == "/frame":
            with _lock: jpg = _frame["jpg"]
            return self._send(200, "image/jpeg", jpg) if jpg else self._send(503, "text/plain", b"no frame yet")
        return self._send(200, "text/html; charset=utf-8", PAGE.encode())
    def do_POST(self):
        if not self._auth(): return
        path = self._strip(urllib.parse.urlparse(self.path).path)
        n = int(self.headers.get("Content-Length", "0") or 0)
        try: body = json.loads(self.rfile.read(n) or b"{}")
        except Exception: body = {}
        try:
            if path == "/tap":
                x, y = to_screen(float(body["x"]), float(body["y"]))
                xdo("mousemove", str(x), str(y), "click", "1")
            elif path == "/swipe":
                x0, y0 = to_screen(float(body["x0"]), float(body["y0"])); x1, y1 = to_screen(float(body["x1"]), float(body["y1"]))
                steps = 8; ms = max(100, min(1500, int(body.get("ms", 300))))
                xdo("mousemove", str(x0), str(y0), "mousedown", "1")
                for i in range(1, steps + 1):
                    xdo("mousemove", str(int(x0 + (x1 - x0) * i / steps)), str(int(y0 + (y1 - y0) * i / steps)))
                    time.sleep(ms / 1000 / steps)
                xdo("mouseup", "1")
            elif path == "/key":
                k = str(body.get("k", ""))[:32]
                if k == "super": subprocess.run(["adb", "-s", "emulator-5554", "shell", "input", "keyevent", "KEYCODE_HOME"], timeout=20)
                elif k == "alt+Tab": subprocess.run(["adb", "-s", "emulator-5554", "shell", "input", "keyevent", "KEYCODE_APP_SWITCH"], timeout=20)
                elif k == "Escape": subprocess.run(["adb", "-s", "emulator-5554", "shell", "input", "keyevent", "KEYCODE_BACK"], timeout=20)
                else: xdo("key", "--clearmodifiers", k)
            elif path == "/text":
                xdo("type", "--delay", "40", str(body.get("t", ""))[:500])
            else:
                return self._send(404, "text/plain", b"unknown")
        except Exception as e:  # noqa: BLE001
            return self._send(500, "text/plain", str(e).encode())
        return self._send(200, "application/json", b'{"ok":true}')

threading.Thread(target=grabber, daemon=True).start()
print(f"[webview] listening on {PORT}", flush=True)
ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
