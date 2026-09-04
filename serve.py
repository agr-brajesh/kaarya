"""Zero-dependency server. Same routes as app.py, stdlib only.

Why this exists: on demo day pip is the thing most likely to fail you. This
runs on a bare Python 3.10+ install with numpy as the only import, so you
always have a working UI. Run: python serve.py  ->  http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import api_core

HOST = os.environ.get("KAARYA_HOST", "127.0.0.1")
PORT = int(os.environ.get("KAARYA_PORT", "8000"))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, default=str).encode(), "application/json")

    def log_message(self, fmt, *args):          # quieter console during a demo
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/":
                self._send(200, api_core.index_html().encode(), "text/html; charset=utf-8")
            elif path == "/api/health":
                self._json(api_core.health())
            elif path.startswith("/api/jd/"):
                self._json(api_core.do_jd(path.rsplit("/", 1)[1]))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:                  # never leave the demo hanging
            self._json({"error": type(e).__name__, "detail": str(e)}, 500)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if path == "/api/match":
                self._json(api_core.do_match(body))
            elif path == "/api/audit":
                self._json(api_core.do_audit(body))
            elif path == "/api/promote":
                self._json(api_core.do_promote(body, body.get("substring", "")))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": type(e).__name__, "detail": str(e)}, 500)


if __name__ == "__main__":
    print(f"kaarya (stdlib server) on http://{HOST}:{PORT}  embedder="
          f"{api_core.health()['embedder']}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
