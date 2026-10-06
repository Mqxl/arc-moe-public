#!/usr/bin/env python3
"""Measure a running Strata server the way the community reports do: the server's own `timings` per request
(prompt_per_second, predicted_per_second), one request at a time, thinking off, greedy.

    python3 tools/bench.py [--url http://127.0.0.1:8080] [--runs 3] [--max-tokens 256] [--long]

Prompts: a short code question, a ~2K-token code prompt (--long) and a short Russian prose question - decode speed
with speculative decoding depends on the text (code drafts well, prose less), so the table keeps them apart.
Only the standard library is used.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

SHORT_CODE = "Write a Python function that returns the n-th Fibonacci number, with a docstring and a test."
SHORT_PROSE = "Объясни простыми словами, как работает смесь экспертов (MoE) в языковых моделях. Пять предложений."


def long_code() -> str:
    """~2K tokens of real code from this repository, then a task - the shape of the 2x B60 report's 2K prompt."""
    here = Path(__file__).resolve().parent.parent
    src = ""
    for p in sorted((here / "third_party" / "strata" / "tools").glob("*.py")):
        try:
            src += f"# file: {p.name}\n" + p.read_text(encoding="utf-8", errors="replace") + "\n"
        except OSError:
            continue
        if len(src) > 7000:
            break
    src = src[:7000] or "def f(x):\n    return x\n" * 200
    return src + "\n\nUsing the style of the code above, write a binary search tree class with insert, find and delete."


def call(url: str, key: str | None, prompt: str, max_tokens: int) -> dict:
    body = {"model": "strata", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": False, "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none"}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        res = json.load(r)
    res["_wall"] = time.time() - t
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--long", action="store_true", help="also the ~2K-token code prompt")
    ap.add_argument("--json", help="write the raw rows here")
    a = ap.parse_args()

    try:
        with urllib.request.urlopen(a.url.rstrip("/") + "/health", timeout=5) as r:
            r.read()
    except OSError as e:
        print(f"the server at {a.url} does not answer ({e}): ./arc start", file=sys.stderr)
        return 1

    cases = [("short code", SHORT_CODE), ("short prose (ru)", SHORT_PROSE)]
    if a.long:
        cases.append(("2K code", long_code()))
    rows = []
    print(f"{'prompt':18s} {'run':>3s} {'in':>6s} {'out':>5s} {'prompt tok/s':>13s} {'decode tok/s':>13s} {'wall s':>7s}")
    for name, prompt in cases:
        for i in range(a.runs):
            try:
                res = call(a.url, a.api_key, prompt + ("" if i == 0 else f" (#{i})"), a.max_tokens)
            except OSError as e:
                print(f"{name:18s} {i + 1:3d}  failed: {e}")
                continue
            t = res.get("timings") or {}
            u = res.get("usage") or {}
            row = {"case": name, "run": i + 1, "prompt_tokens": t.get("prompt_n", u.get("prompt_tokens")),
                   "tokens": t.get("predicted_n", u.get("completion_tokens")),
                   "prompt_tps": t.get("prompt_per_second"), "decode_tps": t.get("predicted_per_second"),
                   "wall": res["_wall"]}
            rows.append(row)
            f = lambda v: f"{v:13.1f}" if isinstance(v, (int, float)) else f"{'-':>13s}"
            print(f"{name:18s} {i + 1:3d} {str(row['prompt_tokens']):>6s} {str(row['tokens']):>5s} "
                  f"{f(row['prompt_tps'])} {f(row['decode_tps'])} {row['wall']:7.1f}")
    print()
    for name, _ in cases:
        d = [r["decode_tps"] for r in rows if r["case"] == name and isinstance(r["decode_tps"], (int, float))]
        if d:
            print(f"{name:18s} decode: median {statistics.median(d):.1f} tok/s (min {min(d):.1f}, max {max(d):.1f})")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
