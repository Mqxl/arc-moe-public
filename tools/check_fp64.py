#!/usr/bin/env python3
"""Find FP64 (double) in Strata's SYCL device code.

Intel Arc A-series (Alchemist, DG2: A770 / A750 / A580 / A380) has no FP64 hardware. An AOT build for
`acm-g10` stops with "Double type is not supported on this platform", and a JIT build fails when the kernel is
first launched (or, with the driver's emulation forced on, runs the double math in software, 10-100x slower).

This compiles every SYCL translation unit of a configured build to SPIR-V (`-fsycl-device-only`), disassembles
it with oneAPI's llvm-spirv and lists each device function that still computes in double. No GPU is needed: it
is the check to run after every upstream merge.

    cmake -S third_party/strata/sycl -B build-ir -G Ninja -DCMAKE_CXX_COMPILER=icpx \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DSTRATA_SYCL_NO_FP64=ON
    python3 tools/check_fp64.py build-ir            # exit 1 when anything is left

The `-j N` option runs N compilers at once (default: the number of CPUs).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LLVM_SPIRV = None   # found in main()
# Opcodes that are not computations: their operand 2 is not a result type.
NOT_VALUES = {"Name", "MemberName", "Decorate", "MemberDecorate", "EntryPoint", "ExecutionMode", "Capability",
              "Extension", "ExtInstImport", "MemoryModel", "Source", "Store", "Return", "ReturnValue", "Branch",
              "BranchConditional", "LoopMerge", "SelectionMerge", "ControlBarrier", "MemoryBarrier", "Label",
              "FunctionEnd", "Switch", "Unreachable", "CopyMemory", "CopyMemorySized", "LifetimeStart",
              "LifetimeStop", "DecorationGroup", "GroupDecorate"}


def device_ir(entry: dict, out: Path) -> tuple[str, str | None]:
    """Compile one translation unit to device IR. Returns (file, error)."""
    args = shlex.split(entry["command"]) if "command" in entry else list(entry["arguments"])
    keep, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a in ("-o", "-MF", "-MT", "-MQ"):
            skip = True
            continue
        if a in ("-c", "-MD", "-MMD") or a.startswith("-fsycl-targets") or a.startswith("-Xsycl-target"):
            continue
        keep.append(a)
    spv = out.with_suffix(".spv")
    keep += ["-fsycl-device-only", "-fsycl-device-obj=spirv", "-c", "-o", str(spv)]
    r = subprocess.run(keep, cwd=entry["directory"], capture_output=True, text=True)
    if r.returncode:
        return entry["file"], r.stderr[-2000:]
    if not spv.exists():                                 # a host-only file: no device code at all
        return entry["file"], None
    r = subprocess.run([LLVM_SPIRV, "-to-text", str(spv), "-o", str(out)], capture_output=True, text=True)
    return entry["file"], (r.stderr[-2000:] if r.returncode else None)


def scan(ir: Path) -> dict[str, int]:
    """device function -> number of SPIR-V instructions whose result is a double (or a vector/array of them)."""
    if not ir.exists():
        return {}
    lines = [l.split() for l in ir.read_text(errors="replace").splitlines()]
    f64: set[str] = set()
    names: dict[str, str] = {}
    for t in lines:
        if len(t) < 3:
            continue
        op = t[1]
        if op == "TypeFloat" and t[3:4] == ["64"]:
            f64.add(t[2])
        elif op in ("TypeVector", "TypeArray", "TypeRuntimeArray") and t[3] in f64:
            f64.add(t[2])
        elif op == "Name":
            names[t[2]] = t[3].strip('"')
        elif op == "EntryPoint":
            names[t[3]] = t[4].strip('"')
    hits: dict[str, int] = {}
    fn = None
    for t in lines:
        if len(t) < 2:
            continue
        op = t[1]
        if op == "Function":
            fn = names.get(t[3], "%" + t[3])
        elif op == "FunctionEnd":
            fn = None
        elif fn and op not in NOT_VALUES and len(t) > 2 and t[2] in f64:
            hits[fn] = hits.get(fn, 0) + 1
    return hits


def demangle(names: list[str]) -> dict[str, str]:
    tool = shutil.which("llvm-cxxfilt") or shutil.which("c++filt")
    if not tool or not names:
        return {n: n for n in names}
    r = subprocess.run([tool], input="\n".join(names), capture_output=True, text=True)
    out = r.stdout.splitlines()
    return dict(zip(names, out)) if len(out) == len(names) else {n: n for n in names}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("build", help="a build directory configured with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
    ap.add_argument("-j", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--filter", default="", help="only sources whose path contains this text")
    a = ap.parse_args()
    global LLVM_SPIRV
    icpx = shutil.which("icpx")
    cand = [shutil.which("llvm-spirv")] + ([str(Path(icpx).resolve().parent / "compiler" / "llvm-spirv")] if icpx else [])
    LLVM_SPIRV = next((c for c in cand if c and Path(c).exists()), None)
    if not LLVM_SPIRV:
        sys.exit("llvm-spirv not found: source /opt/intel/oneapi/setvars.sh first")

    db = json.loads((Path(a.build) / "compile_commands.json").read_text())
    entries = [e for e in db if "-fsycl" in e.get("command", " ".join(e.get("arguments", [])))
               and "/_deps/" not in e["file"] and "/ggml/" not in e["file"] and a.filter in e["file"]]
    tmp = Path(tempfile.mkdtemp(prefix="fp64-"))
    bad = 0
    with cf.ThreadPoolExecutor(a.j) as pool:
        futs = {pool.submit(device_ir, e, tmp / f"{i}.ll"): (e, tmp / f"{i}.ll") for i, e in enumerate(entries)}
        for f in cf.as_completed(futs):
            src, err = f.result()
            ir = futs[f][1]
            name = os.path.relpath(src)
            if err:
                print(f"?? {name}: device compile failed\n{err}")
                bad += 1
                continue
            hits = scan(ir)
            if hits:
                bad += 1
                pretty = demangle(list(hits))
                print(f"FP64 {name}", flush=True)
                for fn, n in sorted(hits.items(), key=lambda kv: -kv[1]):
                    print(f"     {n:4d}  {pretty[fn][:160]}")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(entries)} translation units scanned, {bad} with FP64 or errors")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
