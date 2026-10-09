#!/usr/bin/env python3
"""Summarize selected request intervals from unitrace Chrome logs without loading flow events.

Category times are unions, and can overlap across queues/categories. Idle means
no recorded GPU operation; it is not proof of a hardware sleep or PCIe limit.
Copy directions are explicit. Inter-GPU attribution requires known hand-off sizes.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def union(intervals):
    total = 0
    end = None
    for a, b in sorted(intervals):
        total += max(0, b - max(a, end if end is not None else a))
        end = max(end if end is not None else b, b)
    return total


def classify(name, handoff_bytes=()):
    if 'MemoryCopy' in name:
        size = re.search(r'\[(\d+)\]', name)
        if size and int(size.group(1)) in handoff_bytes:
            return 'handoff_copy'
        return 'dma_' + next((x for x in ('P2P', 'D2D', 'D2M', 'M2D', 'M2M', 'H2D', 'D2H', 'H2H') if '(' + x + ')' in name), 'other')
    if 'fetch_blobs' in name:
        return 'expert_pcie_kernel'
    if 'copy_from_mapped' in name or 'copy_i32_from_mapped' in name:
        return 'mapped_copy_unattributed'
    if any(x in name for x in ('attn_chunk', 'attn_reduce', 'attn_batch', 'qsa_prompt_attn', 'attention')):
        return 'attention'
    if any(x in name for x in ('launch_gu', 'launch_down', 'native_expert', 'launch_split', 'grouped_iq')):
        return 'moe_experts'
    if any(x in name for x in ('qsa_index', 'indexer', 'topk', 'index_score')):
        return 'attention_indexer'
    if any(x in name for x in ('wait_flag', 'plan_wait')):
        return 'device_flag_wait'
    return 'other_compute' if not name.startswith('zeCommandList') else 'other_gpu_op'


def analyze(path, requests, handoff_bytes=()):
    events = []
    devices = {}
    host = []
    for line in Path(path).open():
        try:
            e = json.loads(line.rstrip(',\n'))
        except ValueError:
            continue
        if e.get('ph') == 'M' and e.get('name') == 'process_name':
            n = e.get('args', {}).get('name', '')
            if n.startswith('DEVICE'):
                devices[e['pid']] = n
        if e.get('ph') != 'X':
            continue
        item = (e['ts'], e['ts'] + e['dur'], e.get('pid'), e.get('tid'), e.get('name', ''))
        if e.get('cat') == 'gpu_op':
            events.append(item)
        elif any(x in e.get('name', '') for x in ('HostSynchronize', 'Synchronize', 'WaitForEvents')):
            host.append(item)
    events.sort()
    if not events:
        raise ValueError('Trace contains no GPU operations; repeat collection before drawing conclusions')
    rows = []
    for req in requests:
        lo, hi = req['started_epoch_us'], req['ended_epoch_us']
        scoped = [e for e in events if e[0] < hi and e[1] > lo]
        # The first productive token follows the first decode window. Select a
        # subsequent window, excluding short-read prompt windows from the trace.
        decode_begin = lo + (req.get('ttft_s') or 0) * 1e6
        plans = [e for e in scoped if 'resident_plan_kernel' in e[4]]
        samples = [e for e in scoped if 'sampler_greedy_kernel' in e[4]]
        selected = []
        # Pick one complete verifier body; the 48 plan launches are the 48 routed layers.
        previous_sample_end = lo
        for sample in samples:
            window = [e for e in plans if e[0] >= previous_sample_end and e[1] <= sample[0]]
            previous_sample_end = sample[1]
            if len(window) == 48 and window[0][0] >= decode_begin:
                selected.append(('verify_window_after_first_token', window[0][0], sample[1]))
                break
        # Stable owner-side chunk interval, with peer pipeline activity visible in the same interval.
        broadcasts = [e for e in scoped if 'gr_broadcast' in e[4]]
        if len(broadcasts) >= 3:
            n = len(broadcasts) - 3
            selected.append(('prefill_pipeline_interval', broadcasts[n][0], broadcasts[n + 1][0]))
        for label, begin, end in selected:
            span = end - begin
            detail = {'request': req['case'], 'interval': label, 'start_us': begin, 'end_us': end,
                      'span_ms': span / 1000, 'devices': {},
                      'gpu_union_ms': union([(max(a, begin), min(b, end)) for a,b,_,_,_ in scoped
                                             if a < end and b > begin]) / 1000,
                      'host_sync_union_ms': union([(max(a, begin), min(b, end)) for a,b,_,_,_ in host
                                                  if a < end and b > begin]) / 1000}
            for pid, name in devices.items():
                groups = defaultdict(list)
                copies = defaultdict(float)
                named = defaultdict(list)
                allops = []
                for a, b, p, _, n in scoped:
                    if p != pid or a >= end or b <= begin:
                        continue
                    item = (max(a, begin), min(b, end))
                    groups[classify(n, handoff_bytes)].append(item)
                    named[n].append(item)
                    allops.append(item)
                    if 'MemoryCopy' in n:
                        m = re.search(r'\[(\d+)\]', n)
                        if m:
                            copies[str(int(m.group(1)))] += (item[1] - item[0]) / 1000
                busy = union(allops)
                detail['devices'][name] = {
                    'busy_ms': busy / 1000, 'unrecorded_idle_pct': 100 * (1 - busy / span),
                    'categories': {k: {'union_ms': union(v) / 1000, 'pct_of_span': 100 * union(v) / span}
                                   for k, v in groups.items()}, 'copy_bytes_duration_ms': dict(copies),
                    'top_gpu_ops_ms': {n: v / 1000 for n,v in
                                       sorted(((n, union(v)) for n,v in named.items()), key=lambda x:x[1], reverse=True)[:12]}}
            rows.append(detail)
    if not rows:
        raise ValueError('No complete intervals matched the request markers')
    return {'trace': str(path), 'intervals': rows, 'gpu_events': len(events),
            'note': 'Category unions overlap; no residual time is automatically assigned to host sync. '
                    'Prefill interval includes concurrent neighboring peer chunks.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('trace')
    ap.add_argument('--requests', required=True, help='bench-like rows with epoch timestamps')
    ap.add_argument('--output', required=True)
    ap.add_argument('--handoff-bytes', type=int, nargs='*', default=[],
                    help='verified model hand-off sizes; only matching copy operations are attributed')
    a = ap.parse_args()
    out = analyze(a.trace, json.loads(Path(a.requests).read_text()), a.handoff_bytes)
    Path(a.output).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
