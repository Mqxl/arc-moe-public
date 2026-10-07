#!/usr/bin/env python3
"""Summarize a ./arc profile file: the eager verify windows' stage clocks averaged per GPU stage (the last stage
prints the head), and the prompt path's phase lines kept as they are. Appends to the file given."""
import collections
import re
import sys

path = sys.argv[1]
lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
st = [l for l in lines if l.startswith("verify stages")]
out = []
for name, sel in (("first GPU (no head)", [l for l in st if " head " not in l]), ("last GPU (head)", [l for l in st if " head " in l])):
    if not sel:
        continue
    acc = collections.defaultdict(float)
    for l in sel:
        body = l.split(":", 1)[1]
        for part in body.split("|"):
            kind = "QSA" if "QSA layers" in part else "GDN" if "GDN layers" in part else None
            if "total" in part:
                acc["total ms/window"] += float(re.search(r"total ([0-9.]+)", part).group(1))
                continue
            if kind is None:
                continue
            for m in re.finditer(r"([A-Za-z0-9+/()\- ]+?) ([0-9]+\.[0-9]+)", part.split(":", 1)[1]):
                acc[f"{kind} {m.group(1).strip()}"] += float(m.group(2))
    out.append(f"verify stages averaged, {name}, {len(sel)} windows (ms per window, summed over the GPU's layers):")
    for k, v in sorted(acc.items(), key=lambda x: -x[1]):
        out.append(f"  {k:32s} {v / len(sel):8.2f}")
# one line per run: its bench rows (case: prompt tok/s / decode tok/s)
runs, cur = [], None
for l in lines:
    if l.startswith("================ ") and not l.startswith("================ summary"):
        cur = [l[17:].split(":")[0].strip(), l.split(":", 1)[1].strip() if ":" in l else "", []]
        runs.append(cur)
    elif cur is not None:
        m = re.match(r"(short code|short prose \(ru\)|0.7K code|2K code|8K code|10K code|sampled code|agent \(stream\))\s+\d+\s+\d+\s+\d+\s+([0-9.]+)\s+([0-9.]+)", l)
        if m:
            cur[2].append(f"{m.group(1).replace('short ', '').replace(' (ru)', '')} {m.group(2)}/{m.group(3)}")
if runs:
    out.append("runs (case prompt tok/s / decode tok/s):")
    for name, what, rows in runs:
        out.append(f"  {name:12s} {' | '.join(rows) or '(no rows)':70s} {what[:70]}")
with open(path, "a", encoding="utf-8") as f:
    f.write("\n================ summary\n" + "\n".join(out) + "\n")
