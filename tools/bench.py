#!/usr/bin/env python3
"""Measure a running Strata server the way the community reports do: the server's own `timings` per request
(prompt_per_second, predicted_per_second), one request at a time, thinking off, greedy - and every reply checked for
text (a broken one, e.g. the "!!!!" of NaN logits, is flagged and the exit code is 3). `sampled` and `agent` send what
OpenAI-style clients and agent harnesses do: temperature / top_p / top_k / penalties, a system prompt, tools, streaming.

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


def long_code(chars: int = 7000) -> str:
    """~2K tokens of real code from this repository (chars=7000), then a task - the shape of the 2x B60 report's 2K
    prompt. chars=28000: the ~8K-token prompt (several prompt chunks, the shape of an agent's first request)."""
    here = Path(__file__).resolve().parent.parent
    src = ""
    for p in sorted((here / "third_party" / "strata" / "tools").glob("*.py")):
        try:
            src += f"# file: {p.name}\n" + p.read_text(encoding="utf-8", errors="replace") + "\n"
        except OSError:
            continue
        if len(src) > chars:
            break
    src = src[:chars] or "def f(x):\n    return x\n" * (chars // 25)
    return src + "\n\nUsing the style of the code above, write a binary search tree class with insert, find and delete."


# what an agent harness / OpenRouter sends: a system prompt, tools, sampling and penalties, streaming, thinking left on
AGENT_SYSTEM = ("You are a coding agent working in the user's repository. Use the tools when you need to look at files. "
                "Answer concisely.")
AGENT_TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file of the repository.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "the file's path"}},
                   "required": ["path"]}}}]
SAMPLED = {"temperature": 0.7, "top_p": 0.95, "top_k": 40, "repetition_penalty": 1.05, "presence_penalty": 0.1}


def call(url: str, key: str | None, prompt: str, max_tokens: int, extra: dict | None = None) -> dict:
    body = {"model": "strata", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": False, "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none"}
    if extra:
        body.update(extra)
        for k in [k for k, v in body.items() if v is None]:
            del body[k]
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        if not body.get("stream"):
            res = json.load(r)
        else:   # SSE: the text and tool calls from the deltas, timings/usage from whichever chunk carries them
            res, text, calls = {}, "", {}
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                ch = json.loads(line[5:])
                for k in ("timings", "usage"):
                    if ch.get(k):
                        res[k] = ch[k]
                for c in ch.get("choices") or []:
                    d = c.get("delta") or {}
                    text += (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                    for tc in d.get("tool_calls") or []:
                        index = tc.get("index", 0)
                        acc = calls.setdefault(index, {"function": {"name": "", "arguments": ""}})
                        for field in ("id", "type"):
                            if tc.get(field): acc[field] = tc[field]
                        fn = tc.get("function") or {}
                        for field in ("name", "arguments"):
                            acc["function"][field] += fn.get(field) or ""
            res["choices"] = [{"message": {"content": text, "tool_calls": [calls[i] for i in sorted(calls)]}}]
    res["_wall"] = time.time() - t
    return res


def reply_text(res: dict) -> str:
    m = ((res.get("choices") or [{}])[0].get("message") or {})
    t = (m.get("content") or "") + (m.get("reasoning_content") or m.get("reasoning") or "")
    if m.get("tool_calls"):
        t += " [tool call]"
    return t


def broken(text: str) -> str | None:
    """Why a reply is not text: empty, or one character over and over (the '!!!!' of NaN logits: token 0 is '!')."""
    s = "".join(text.split())
    if not s:
        return "empty reply"
    top = max(set(s), key=s.count)
    if len(s) >= 16 and s.count(top) / len(s) > 0.8:
        return f"{s.count(top)} of {len(s)} characters are {top!r}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key")
    ap.add_argument("--spec-min-p", type=float, help="per-request draft confidence threshold (strata_tune)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--long", action="store_true", help="also the ~2K-token code prompt")
    ap.add_argument("--json", help="write the raw rows here")
    ap.add_argument("--cases", help="comma list of short, prose, prose-en, prose-zh, prose-ru-long, 1k, 1.5k, 2k, 8k, 10k, sampled, agent "
                                    "(default: short,prose, and 2k with --long)")
    ap.add_argument("--show", action="store_true", help="print every reply's start")
    a = ap.parse_args()
    if a.spec_min_p is not None and not 0 <= a.spec_min_p <= 1:
        ap.error("--spec-min-p must be between 0 and 1")

    try:
        with urllib.request.urlopen(a.url.rstrip("/") + "/health", timeout=5) as r:
            r.read()
    except OSError as e:
        print(f"the server at {a.url} does not answer ({e}): ./arc start", file=sys.stderr)
        return 1

    known = {"short": ("short code", lambda: SHORT_CODE, None), "prose": ("short prose (ru)", lambda: SHORT_PROSE, None),
             "prose-en": ("short prose (en)", lambda: "Explain how a mixture of experts language model works, with concrete examples. Write at least ten sentences in English.", None),
             "prose-zh": ("short prose (zh)", lambda: "用中文解释混合专家语言模型的工作原理，并给出具体例子。请写至少十句话。", None),
             "prose-ru-long": ("long prose (ru)", lambda: "Объясни устройство языковой модели со смесью экспертов, с конкретными примерами. Напиши не менее десяти предложений по-русски.", None),
             "1k": ("0.7K code", lambda: long_code(2400), None),
             "1.5k": ("1.5K code", lambda: long_code(5000), None),
             "2k": ("2K code", long_code, None), "8k": ("8K code", lambda: long_code(28000), None),
             "10k": ("10K code", lambda: long_code(36000), None),
             # temperature / top_p / top_k / penalties: the sampled path every OpenAI-style client takes
             "sampled": ("sampled code", lambda: SHORT_CODE, dict(SAMPLED)),
             # a harness's request: system prompt, tools, streaming, thinking on, sampled
             "agent": ("agent (stream)", lambda: "Look at README.md and tell me what this project does.",
                       {**SAMPLED, "stream": True, "tools": AGENT_TOOLS, "chat_template_kwargs": None,
                        "reasoning_effort": None, "stream_options": {"include_usage": True}})}
    want = a.cases.split(",") if a.cases else ["short", "prose"] + (["2k"] if a.long else [])
    bad = [c for c in want if c not in known]
    if bad:
        print(f"unknown case(s) {bad}: pick from {sorted(known)}", file=sys.stderr)
        return 2
    cases = [(known[c][0], known[c][1](), known[c][2]) for c in want]
    if a.spec_min_p is not None:
        cases = [(name, prompt, {**(extra or {}), "strata_tune": {"spec_min_p": a.spec_min_p}}) for name, prompt, extra in cases]
    rows = []
    print(f"{'prompt':18s} {'run':>3s} {'in':>6s} {'out':>5s} {'prompt tok/s':>13s} {'decode tok/s':>13s} {'wall s':>7s}")
    n_broken = 0
    n_errors = 0
    for name, prompt, extra in cases:
        for i in range(a.runs):
            if extra and extra.get("tools"):
                extra = {**extra, "messages": [{"role": "system", "content": AGENT_SYSTEM},
                                               {"role": "user", "content": prompt + ("" if i == 0 else f" (#{i})")}]}
            try:
                res = call(a.url, a.api_key, prompt + ("" if i == 0 else f" (#{i})"), a.max_tokens, extra)
            except OSError as e:
                n_errors += 1
                detail = str(e)
                # Keep useful engine errors without exposing authentication responses or the key.
                if isinstance(e, urllib.error.HTTPError) and e.code not in (401, 403):
                    body = e.read(2048).decode("utf-8", errors="replace")
                    try:
                        error = json.loads(body).get("error", {})
                        message = error.get("message", "") if isinstance(error, dict) else str(error)
                    except (ValueError, AttributeError):
                        message = "unparseable error response (body omitted)"
                    if a.api_key:
                        message = message.replace(a.api_key, "[REDACTED]")
                    if message:
                        detail += ": " + message[:1000]
                if a.api_key:
                    detail = detail.replace(a.api_key, "[REDACTED]")
                rows.append({"case": name, "run": i + 1, "prompt_tokens": None, "tokens": None,
                             "prompt_tps": None, "decode_tps": None, "wall": None, "error": detail})
                print(f"{name:18s} {i + 1:3d}  failed: {detail}")
                continue
            t = res.get("timings") or {}
            u = res.get("usage") or {}
            row = {"case": name, "run": i + 1, "prompt_tokens": t.get("prompt_n", u.get("prompt_tokens")),
                   "tokens": t.get("predicted_n", u.get("completion_tokens")),
                   "prompt_tps": t.get("prompt_per_second"), "decode_tps": t.get("predicted_per_second"),
                   "wall": res["_wall"]}
            rows.append(row)
            f = lambda v: f"{v:13.1f}" if isinstance(v, (int, float)) else f"{'-':>13s}"
            text = reply_text(res)
            why = broken(text)
            n_broken += why is not None
            print(f"{name:18s} {i + 1:3d} {str(row['prompt_tokens']):>6s} {str(row['tokens']):>5s} "
                  f"{f(row['prompt_tps'])} {f(row['decode_tps'])} {row['wall']:7.1f}"
                  + (f"  BROKEN REPLY: {why}: {text[:80]!r}" if why else ""))
            if a.show and not why:
                print(f"{'':18s}     {' '.join(text.split())[:120]!r}")
    print()
    for name, _, _ in cases:
        d = [r["decode_tps"] for r in rows if r["case"] == name and isinstance(r["decode_tps"], (int, float))]
        if d:
            print(f"{name:18s} decode: median {statistics.median(d):.1f} tok/s (min {min(d):.1f}, max {max(d):.1f})")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    if n_errors:
        print(f"\n{n_errors} API request(s) failed; this run is incomplete")
    if n_broken:
        print(f"\n{n_broken} broken repl{'y' if n_broken == 1 else 'ies'} (see BROKEN REPLY above): "
              f"restart with STRATA_DBG_NAN=1 and send the engine log")
        return 3
    return 3 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
