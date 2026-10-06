#!/usr/bin/env python3
"""Tiny front proxy for noVNC behind HTTP/2-terminating edges (Railway).

Browsers open WebSockets over HTTP/2 (RFC 8441). Some edges translate that
to HTTP/1.1 without the classic `Upgrade: websocket` / `Connection: Upgrade`
headers, which websockify requires. This proxy listens on the public port,
restores those headers when it sees a WebSocket key, and relays bytes in
both directions. Plain HTTP requests are forwarded untouched.

    wsproxy.py <listen_port> <websockify_port>
"""
import socket, sys, threading

LISTEN = int(sys.argv[1]); UPSTREAM = int(sys.argv[2])

def log(*a):
    print("[wsproxy]", *a, flush=True)

def pump(src, dst):
    try:
        while True:
            b = src.recv(65536)
            if not b: break
            dst.sendall(b)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try: s.shutdown(socket.SHUT_RDWR)
            except OSError: pass

def handle(client):
    try:
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = client.recv(65536)
            if not chunk: return
            buf += chunk
            if len(buf) > 65536: return
        head, rest = buf.split(b"\r\n\r\n", 1)
        lines = head.split(b"\r\n")
        request_line, headers = lines[0], lines[1:]
        names = {h.split(b":", 1)[0].strip().lower() for h in headers if b":" in h}
        if b"sec-websocket-key" in names:
            headers = [h for h in headers if h.split(b":", 1)[0].strip().lower() not in (b"upgrade", b"connection")]
            headers += [b"Upgrade: websocket", b"Connection: Upgrade"]
            log("websocket handshake", request_line.decode(errors="replace"))
        head = b"\r\n".join([request_line] + headers)
        up = socket.create_connection(("127.0.0.1", UPSTREAM))
        up.sendall(head + b"\r\n\r\n" + rest)
        t = threading.Thread(target=pump, args=(up, client), daemon=True); t.start()
        pump(client, up)
    except Exception as e:  # noqa: BLE001
        log("error", e)
    finally:
        try: client.close()
        except OSError: pass

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("0.0.0.0", LISTEN)); srv.listen(64)
log(f"listening on {LISTEN}, forwarding to websockify on {UPSTREAM}")
while True:
    c, _ = srv.accept()
    threading.Thread(target=handle, args=(c,), daemon=True).start()
