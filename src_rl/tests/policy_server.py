"""Minimal stdlib HTTP server exposing a policy via the Battlesnake API.

Used by the differential-test recorder and local CLI games — no Flask needed.
Run: python -m src_rl.tests.policy_server <port> <policy>
where <policy> is one of: legacy, legacy_heuristic.
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict


def resolve_policy(name: str) -> Callable[[Dict], str]:
    from src_rl.baselines import legacy_policy

    if name == "legacy":
        return legacy_policy.choose_move
    if name == "legacy_heuristic":
        return legacy_policy.choose_move_heuristic
    raise ValueError(f"unknown policy: {name}")


def make_handler(choose_move: Callable[[Dict], str]):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, payload: Dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - http.server API
            self._send({"apiversion": "1", "author": "src_rl-test", "color": "#888888"})

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            if self.path == "/move":
                game_state = json.loads(raw)
                self._send({"move": choose_move(game_state)})
            else:  # /start, /end
                self._send({})

        def log_message(self, *args):  # silence request logging
            pass

    return Handler


def serve_in_thread(port: int, choose_move: Callable[[Dict], str]) -> ThreadingHTTPServer:
    """Start a policy server on a daemon thread; caller shuts it down."""
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(choose_move))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    port, policy = int(sys.argv[1]), sys.argv[2]
    ThreadingHTTPServer(("127.0.0.1", port), make_handler(resolve_policy(policy))).serve_forever()
