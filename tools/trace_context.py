#!/usr/bin/env python3
"""Mark short/64K request intervals in a unitrace session.

Use --chrome-kernel-logging --chrome-call-logging from process start and
--always-on here for captured graphs. Pausing while graphs are recorded can
leave their replay without events in some collectors. For ordinary launches,
--start-paused --session NAME permits dynamic control. This tool does not
control server lifetime. Stop the engine gracefully afterward to flush traces.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path
from bench import call, SHORT_CODE, broken, reply_text
from bench_context import counted_context


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', required=True)
    ap.add_argument('--api-key')
    ap.add_argument('--unitrace', required=True)
    ap.add_argument('--session', default='arccontext')
    ap.add_argument('--always-on', action='store_true', help='collector is already active; do not pause recorded graphs')
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    rows = []
    def control(action):
        if a.always_on:
            return
        subprocess.run([a.unitrace, '--' + action, a.session], check=True, capture_output=True)
    def request(label, text, collect):
        if collect:
            control('resume')
        try:
            r = call(a.url, a.api_key, text, 32, {'stream': True, 'stream_options': {'include_usage': True}})
        finally:
            if collect:
                control('pause')
        why = broken(reply_text(r))
        if why:
            raise RuntimeError(label + ': ' + why)
        row = {'case': label, 'started_epoch_us': r['_started_epoch_us'], 'ended_epoch_us': r['_ended_epoch_us'],
               'ttft_s': r['_ttft'], 'timings': r.get('timings'), 'usage': r.get('usage'),
               'reply_preview': reply_text(r)[:200]}
        if collect:
            rows.append(row)
            Path(a.output).write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    control('pause')
    request('short warm-up, collection paused', SHORT_CODE, False)
    request('short verify', SHORT_CODE, True)
    prompt, counted = counted_context(a.url, a.api_key, 64000, 'trace/64k')
    print('Counted 64K trace input:', counted, flush=True)
    # Full prefill is collected so a steady-state owner chunk and a post-prefill
    # verify body can be selected without perturbing GPU scheduling with callbacks.
    request('64K prefill and verify', prompt, True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
