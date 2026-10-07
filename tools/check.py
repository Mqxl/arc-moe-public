#!/usr/bin/env python3
"""Ask a running Strata server the same short question the way different clients do, and say which ways come back
as text and which as garbage (the "!!!!" of a sampler that found nothing: token 0 is "!").

    python3 tools/check.py [--url http://127.0.0.1:8080] [--api-key KEY] [--max-tokens 48]

The rows differ in one thing at a time: greedy or sampled, thinking off or on, streaming, penalties, top_k / top_p,
tools - and the last ones send a ~10K-token prompt (the batched prompt path, as an agent harness's first request
does). Exit code 3 when any row is broken. Only the standard library is used.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import AGENT_SYSTEM, AGENT_TOOLS, broken, call, long_code, reply_text  # noqa: E402

QUESTION = "Привет! Ты какая модель? Ответь одним предложением."
OFF = {"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}
ON = {"chat_template_kwargs": None, "reasoning_effort": None}
ROWS = [
    ("greedy, thinking off", {"temperature": 0, **OFF}),
    ("greedy, thinking on", {"temperature": 0, **ON}),
    ("greedy, stream", {"temperature": 0, "stream": True, **OFF}),
    ("t=0.7, thinking off", {"temperature": 0.7, **OFF}),
    ("t=0.7, thinking on", {"temperature": 0.7, **ON}),
    ("t=0.7, stream", {"temperature": 0.7, "stream": True, **OFF}),
    ("t=0.7 top_p=0.9 top_k=20", {"temperature": 0.7, "top_p": 0.9, "top_k": 20, **OFF}),
    ("t=1.0 top_k=0 (all 64)", {"temperature": 1.0, "top_k": 0, **OFF}),
    ("t=0.7 + repetition 1.1", {"temperature": 0.7, "repetition_penalty": 1.1, **OFF}),
    ("t=0.7 + presence/frequency", {"temperature": 0.7, "presence_penalty": 0.5, "frequency_penalty": 0.3, **OFF}),
    ("greedy + repetition 1.1", {"temperature": 0, "repetition_penalty": 1.1, **OFF}),
    ("agent: tools, stream, t=0.7", {"temperature": 0.7, "top_p": 0.95, "stream": True, "tools": AGENT_TOOLS, **ON,
                                     "messages": [{"role": "system", "content": AGENT_SYSTEM},
                                                  {"role": "user", "content": QUESTION}]}),
]
LONG = long_code(36000) + "\n\nIn two sentences: what do these files do?"
LONG_ROWS = [
    ("10K prompt, greedy", {"temperature": 0, **OFF, "messages": [{"role": "user", "content": LONG}]}),
    ("10K agent: tools, stream, t=0.7", {"temperature": 0.7, "top_p": 0.95, "stream": True, "tools": AGENT_TOOLS, **ON,
                                         "messages": [{"role": "system", "content": AGENT_SYSTEM},
                                                      {"role": "user", "content": LONG}]}),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--no-long", action="store_true", help="leave out the ~10K-token prompts")
    a = ap.parse_args()
    try:
        with urllib.request.urlopen(a.url.rstrip("/") + "/health", timeout=5) as r:
            r.read()
    except OSError as e:
        print(f"the server at {a.url} does not answer ({e}): ./arc start", file=sys.stderr)
        return 1
    bad = 0
    rows = ROWS + ([] if a.no_long else LONG_ROWS)
    for name, extra in rows:
        try:
            text = reply_text(call(a.url, a.api_key, QUESTION, a.max_tokens, dict(extra)))
            why = broken(text)
        except Exception as e:   # an HTTP error is a result too: the row names it
            text, why = "", f"request failed: {e}"
        bad += why is not None
        mark = "BROKEN" if why else "ok    "
        print(f"{mark} {name:30s} {(why + ': ') if why else ''}{' '.join(text.split())[:90]!r}", flush=True)
    print(f"\n{bad} of {len(rows)} broken" if bad else f"\nall {len(rows)} ok")
    return 3 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
