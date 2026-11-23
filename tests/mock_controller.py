"""A tiny OpenAI-compatible chat-completions server that imitates the pruning agent.

Used by the tests (and handy for a smoke run without API keys):

    python tests/mock_controller.py --port 8765

It reads the unit table from the prompt and proposes +0.10 on the six safest
units (or converts four units in 2:4 mode).  ``fail_first`` makes it return
HTTP 503 for the first N requests to exercise retries/failover.

``script`` replaces that rule with fixed proposals keyed by the iteration number
in the prompt ("Iteration N"), so every arm sees the same proposal at the same
iteration:  ``python tests/mock_controller.py --port 8765 --script live``.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Live-gate smoke script for the tiny structured Qwen3 (4 MLP units of 256 channels).  With the
# live-smoke data (dense model 100 % correct on the pruning-time pool) a proposal's loss is the share of
# answers it changes (measured: tests/live_smoke.py); gate tolerance 0.15, budget 360 of 640 questions:
#   1 tiny   L1 +6.25 %        changes 3-6 % per category              -> ACCEPT (new parent)
#   2 heavy  every MLP +87.5 %  changes ~50 % of "Other"                -> REJECT, roll back to parent 1
#   3 medium L0 +31.25 %       changes 10-14.4 %, all just under 15 %  -> INCONCLUSIVE, roll back to parent 1
#   4 small  L1 +12.5 %        built from parent 1 again
LIVE_SCRIPT = {
    1: [{"layer": "L1.mlp", "additional_sparsity": 0.0625}],
    2: [{"layer": f"L{i}.mlp", "additional_sparsity": 0.875} for i in range(4)],
    3: [{"layer": "L0.mlp", "additional_sparsity": 0.3125}],
    4: [{"layer": "L1.mlp", "additional_sparsity": 0.125}],
}


class State:
    def __init__(self, fail_first: int = 0, garbage_first: int = 0, script: dict | None = None):
        self.fail_first = fail_first
        self.garbage_first = garbage_first
        self.script = script
        self.calls = 0
        self.lock = threading.Lock()


def make_handler(state: State):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):  # not an Ollama server: /api/tags, /api/ps ... are unknown
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if not self.path.endswith("/chat/completions"):
                self.send_response(404)
                self.end_headers()
                return
            with state.lock:
                state.calls += 1
                k = state.calls
            if k <= state.fail_first:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b'{"error":"overloaded"}')
                return
            user = body["messages"][-1]["content"]
            system = body["messages"][0]["content"]
            bad = str(body.get("model", "")).endswith("-bad")  # fixed-state smoke: invalid choice / abstention
            if k <= state.fail_first + state.garbage_first:
                content = "I think we should prune some layers."  # unparseable -> repair path
            elif '"action_id"' in system:  # shortlist condition: pick the first listed action (or an unknown id)
                ids = re.findall(r"^(A\d+):", user, re.M)
                content = json.dumps({"reasoning": "mock shortlist choice", "action_id": "A99" if bad else (ids[0] if ids else None)})
            elif bad and "Fixed step size" in system:
                content = json.dumps({"reasoning": "mock abstains", "stop_pruning": True, "layer_decisions": []})
            elif state.script is not None:
                it = int(re.search(r"^Iteration (\d+)", user, re.M).group(1))
                decs = state.script.get(it, [])
                content = json.dumps({"reasoning": f"scripted proposal for iteration {it}",
                                      "stop_pruning": not decs, "layer_decisions": decs})
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


def serve(port: int, fail_first: int = 0, garbage_first: int = 0, script: dict | None = None):
    st = State(fail_first, garbage_first, script)
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(st))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--script", help="'live' (built-in live-gate smoke script) or a JSON file {iteration: [decisions]}")
    a = ap.parse_args()
    script = None
    if a.script == "live":
        script = LIVE_SCRIPT
    elif a.script:
        with open(a.script, encoding="utf-8") as f:
            script = {int(k): v for k, v in json.load(f).items()}
    srv, _ = serve(a.port, script=script)
    print(f"mock controller on http://127.0.0.1:{a.port}/v1 (Ctrl+C to stop)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
