"""Counted, non-repeating context fixtures and Linux DRM memory observation."""
from __future__ import annotations

import base64
import hashlib
import json
import random
import re
import struct
import threading
import urllib.request
import zlib
from pathlib import Path


def mixed_source(chars: int, seed: str) -> str:
    rng = random.Random(seed)
    records = []
    size = 0
    i = 0
    while size < chars:
        ident = rng.getrandbits(64)
        value = rng.randrange(1000, 1000000)
        digest = hashlib.sha256(f"{seed}:{i}:{ident}".encode()).hexdigest()[:20]
        record = (
            f"\n# module {i}: record_{ident:x}; audit={digest}\n"
            f"def transform_{i}(items, limit={value}):\n"
            f"    return [(key, val * {i + 3}) for key, val in items if val < limit]\n"
            f"Observation {i}: batch {ident} contains {value} records. Preserve stable ordering; "
            f"validate missing fields before updating revision {digest}.\n"
            f"Запись {i}: измерение {value}, идентификатор {ident:x}. "
            "Проверка входных данных предшествует сохранению результата.\n"
        )
        records.append(record)
        size += len(record)
        i += 1
    return "".join(records)[:chars]


def count_input(url: str, key: str | None, text: str) -> int:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    body = {"model": "strata", "messages": [{"role": "user", "content": text}], "max_tokens": 128}
    req = urllib.request.Request(url.rstrip("/") + "/v1/messages/count_tokens",
                                 json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=120) as response:
        return int(json.load(response)["input_tokens"])


def counted_context(url: str, key: str | None, target: int, seed: str) -> tuple[str, int]:
    """Count with the serving tokenizer. Actual completion usage remains authoritative."""
    prefix = f"Context benchmark {seed}. Read the independent modules and notes below.\n"
    suffix = "\nWrite a Python binary search tree with insert, find and delete. Explain its invariants."
    source = mixed_source(target * 5, seed)
    lo, hi = 0, len(source)
    best = None
    for _ in range(18):
        mid = (lo + hi) // 2
        text = prefix + source[:mid] + suffix
        n = count_input(url, key, text)
        if n <= target:
            best = (text, n)
            if target - n <= 16:
                break
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None or not target - 64 <= best[1] <= target:
        raise RuntimeError("Could not prepare a counted context within 64 tokens of target")
    return best


def red_image_url(seed: str = 'baseline') -> str:
    """Dominant red fixture with unique corner pixels to avoid the vision hash cache."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    pixels = bytearray((b"\0" + b"\xff\0\0" * 64) * 64)
    digest = hashlib.sha256(seed.encode()).digest()
    for i in range(16):
        pixels[1 + i * 3 + 1] = digest[i] & 15
        pixels[1 + i * 3 + 2] = digest[16 + i] & 15
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


class VramPeak:
    """Sample engine clients' resident/allocated local memory, not the i915 free-size estimate.

    Counts each (PCI BDF, DRM client id) once despite duplicated file descriptors.
    These are observed request peaks, with a finite sampling interval, not exact maxima.
    Missing permissions/counters produce empty maps, never fabricated zero measurements.
    """
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak = {}
        self.allocated = {}
        self.samples = 0
        self.stop = threading.Event()
        self.thread = None
        self.pids = []

    def _sample(self):
        clients = {}
        for pid in self.pids:
            try:
                paths = list((Path('/proc') / pid / 'fdinfo').iterdir())
            except OSError:
                continue
            for path in paths:
                try:
                    data = dict(line.split(':', 1) for line in path.read_text().splitlines() if ':' in line)
                except OSError:
                    continue
                bdf = data.get('drm-pdev', '').strip()
                client = data.get('drm-client-id', '').strip()
                if not bdf or not client:
                    continue
                def memory(prefix):
                    total = 0
                    found = False
                    for name, value in data.items():
                        if not re.fullmatch(prefix + r'local\d+', name):
                            continue
                        fields = value.split()
                        if len(fields) == 2 and fields[1] in ('KiB', 'kB', 'MiB', 'B'):
                            total += int(fields[0]) * {'KiB': 1024, 'kB': 1024, 'MiB': 1048576, 'B': 1}[fields[1]]
                            found = True
                    return total if found else None
                clients[(bdf, client)] = (memory('drm-resident-'), memory('drm-total-'))
        resident, allocated = {}, {}
        for (bdf, _), (used, held) in clients.items():
            if used is not None:
                resident[bdf] = resident.get(bdf, 0) + used
            if held is not None:
                allocated[bdf] = allocated.get(bdf, 0) + held
        for bdf, used in resident.items():
            self.peak[bdf] = max(self.peak.get(bdf, 0), used)
        for bdf, held in allocated.items():
            self.allocated[bdf] = max(self.allocated.get(bdf, 0), held)
        self.samples += bool(clients)

    def _run(self):
        while not self.stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        for path in Path('/proc').glob('[0-9]*'):
            try:
                first = (path / 'cmdline').read_bytes().split(b'\0')[0]
                if first.endswith(b'/strata'):
                    self.pids.append(path.name)
            except OSError:
                continue
        self._sample()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self._sample()

    def result(self):
        return {'vram_peak_mib': {k: round(v / 1048576, 2) for k, v in self.peak.items()},
                'vram_peak_allocated_mib': {k: round(v / 1048576, 2) for k, v in self.allocated.items()},
                'vram_source': 'engine DRM fdinfo resident-local; sampled',
                'vram_samples': self.samples, 'vram_sample_interval_s': self.interval}
