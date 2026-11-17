"""A tiny OpenAI-compatible chat-completions server that imitates the pruning agent.

Used by the tests (and handy for a smoke run without API keys):

    python tests/mock_controller.py --port 8765

It reads the unit table from the prompt and proposes +0.10 on the six safest
units (or converts four units in 2:4 mode).  ``fail_first`` makes it return
HTTP 503 for the first N requests to exercise retries/failover.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class State:
    def __init__(self, fail_first: int = 0, garbage_first: int = 0):
        self.fail_first = fail_first
        self.garbage_first = garbage_first
        self.calls = 0
        self.lock = threading.Lock()


def make_handler(state: State):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            with state.lock:
                state.calls += 1
                k = state.calls
            if k <= state.fail_first:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b'{"error":"overloaded"}')
                return
            user = body["messages"][-1]["content"]
            if k <= state.fail_first + state.garbage_first:
                content = "I think we should prune some layers."  # unparseable -> repair path
            else:
                names = re.findall(r"^(L\d+\.\w+) \| [\d.]+ \| \S+ \| \S+ \| ([\d.]+)$", user, re.M)
                nm = "2:4" in body["messages"][0]["content"] or "Semi-structured" in body["messages"][0]["content"]
                picks = [n for n, s in names if float(s) < (0.01 if nm else 0.8)][: 4 if nm else 6]
                content = json.dumps({
                    "reasoning": f"mock agent: prune the safest units ({len(picks)})",
                    "stop_pruning": False,
                    "layer_decisions": [{"layer": n, "additional_sparsity": 0.1} for n in picks],
                })
            out = {"choices": [{"message": {"role": "assistant", "content": content}}],
                   "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": len(content) // 4}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return H


def serve(port: int, fail_first: int = 0, garbage_first: int = 0):
    st = State(fail_first, garbage_first)
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(st))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    srv, _ = serve(a.port)
    print(f"mock controller on http://127.0.0.1:{a.port}/v1 (Ctrl+C to stop)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
