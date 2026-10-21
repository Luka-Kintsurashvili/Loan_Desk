"""Minimal stdlib HTTP server: serves the static chat UI and the /api/chat endpoint.

No web framework - this app is small enough that http.server is simpler and has one
fewer third-party dependency to audit.
"""
from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import agent, batch, config

logger = logging.getLogger(__name__)
STATIC_DIR = config.HERE / "static"


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body, ctype: str = "application/json") -> None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, (STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path not in ("/api/chat", "/api/batch"):
            return self._send(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length") or 0)
        if length > config.MAX_BODY_BYTES:
            return self._send(413, {"error": "request body too large"})
        try:
            body = json.loads(self.rfile.read(length))
            if self.path == "/api/batch":
                self._send(200, {"results": batch.score_csv(body.get("csv", ""))})
            else:
                self._send(200, agent.chat(body))
        except Exception as e:
            logger.exception("%s request failed", self.path)  # server-side detail; the client gets a short message
            self._send(500, {"error": str(e)})

    def log_message(self, *args) -> None:
        pass  # silence http.server's default per-request stderr logging


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the HTTP server and block forever."""
    ThreadingHTTPServer((host, port), Handler).serve_forever()
