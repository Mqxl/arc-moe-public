#!/usr/bin/env python3
"""Validate complete benchmark JSON and compare every latency/throughput row."""
import argparse
import json
import math
import statistics
from pathlib import Path

LABELS = {'short': 'short code', 'prose': 'short prose (ru)', '10k': '10K code',
          '32k': '32K mixed code', '64k': '64K mixed code', '120k': '120K mixed code',
          'agent': 'agent (stream)', 'vision': 'vision + 2K'}
METRICS = ('prompt_tps', 'decode_tps', 'wall', 'ttft_s')


def load(path, cases, runs):
    if runs < 1 or not cases or len(set(cases.split(','))) != len(cases.split(',')):
        raise ValueError('positive runs and distinct cases are required')
    rows = json.loads(Path(path).read_text())
    labels = [LABELS[c] for c in cases.split(',')]
    if not isinstance(rows, list) or len(rows) != len(labels) * runs:
        raise ValueError('incomplete benchmark row count')
    expected = {(label, run) for label in labels for run in range(1, runs + 1)}
    seen = set()
    for r in rows:
        if not isinstance(r, dict):
            raise ValueError('benchmark row is not an object')
        if isinstance(r.get('run'), bool) or not isinstance(r.get('run'), int):
            raise ValueError('invalid run index')
        identity = (r.get('case'), r.get('run'))
        if identity not in expected or identity in seen:
            raise ValueError('unexpected or duplicate benchmark row')
        seen.add(identity)
        if r.get('error') or r.get('broken') or not r.get('reply_chars', 0) > 0 or not r.get('tokens', 0) > 0:
            raise ValueError(f'{identity}: failed, empty or broken reply')
        for key in ('reply_chars', 'tokens'):
            if isinstance(r.get(key), bool) or not isinstance(r.get(key), int):
                raise ValueError(f'{identity}: invalid {key}')
        for key in METRICS:
            v = r.get(key)
            if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f'{identity}: invalid {key}')
    return {label: {key: statistics.median(r[key] for r in rows if r['case'] == label)
                    for key in METRICS} for label in labels}


def compare(base, candidate, target, gain_required=.02, tolerance=.02, agent_tolerance=.02):
    if base.keys() != candidate.keys():
        raise ValueError('different benchmark cases')
    regressions = []
    for case in base:
        for key in METRICS:
            old, new = base[case][key], candidate[case][key]
            limit = agent_tolerance if case == LABELS['agent'] else tolerance
            change = new / old - 1
            loss = change if key in ('wall', 'ttft_s') else -change
            if loss > limit + 1e-9:
                regressions.append({'case': case, 'metric': key, 'before': old, 'after': new,
                                    'loss_percent': 100 * loss})
    decode_gain = candidate[LABELS['short']]['decode_tps'] / base[LABELS['short']]['decode_tps'] - 1
    prompt_gain = candidate[LABELS['10k']]['prompt_tps'] / base[LABELS['10k']]['prompt_tps'] - 1
    gain = decode_gain if target == 'decode' else prompt_gain if target == 'prompt' else max(decode_gain, prompt_gain)
    return {'keep': not regressions and gain + 1e-9 >= gain_required,
            'gain': gain, 'regressions': regressions}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('validate', 'summary', 'compare'))
    ap.add_argument('baseline')
    ap.add_argument('candidate', nargs='?')
    ap.add_argument('--cases', default='short,prose,10k,64k,agent,vision')
    ap.add_argument('--runs', type=int, default=2)
    ap.add_argument('--baseline-cases', help='validate a larger baseline before selecting comparison cases')
    ap.add_argument('--target', choices=('decode', 'prompt', 'any'), default='decode')
    ap.add_argument('--gain', type=float, default=.02)
    ap.add_argument('--agent-tolerance', type=float, default=.02)
    a = ap.parse_args()
    try:
        base = load(a.baseline, a.baseline_cases or a.cases, a.runs)
        if a.baseline_cases:
            base = {LABELS[c]: base[LABELS[c]] for c in a.cases.split(',')}
        if a.command == 'validate':
            print('valid')
        elif a.command == 'summary':
            print(json.dumps(base, sort_keys=True))
        else:
            if not a.candidate:
                raise ValueError('candidate JSON is required')
            result = compare(base, load(a.candidate, a.cases, a.runs), a.target,
                             a.gain, agent_tolerance=a.agent_tolerance)
            print(('keep' if result['keep'] else 'drop') + f" {result['gain']:.6f} " + json.dumps(result['regressions']))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as e:
        print('invalid: ' + str(e))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
