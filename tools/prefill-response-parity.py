#!/usr/bin/env python3
"""Save a fixed 10K greedy continuation and top-20 log probabilities for prompt-path A/B checks."""
import argparse, hashlib, importlib.util, json, os
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--url', required=True)
p.add_argument('--api-key')
p.add_argument('--output', required=True)
p.add_argument('--max-tokens', type=int, default=32)
a = p.parse_args()
spec = importlib.util.spec_from_file_location('arc_bench', Path(__file__).with_name('bench.py'))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
prompt = 'Pipeline parity prompt; use the following source as context.\n' + bench.long_code(35000)
result = bench.call(a.url, a.api_key or os.environ.get('STRATA_API_KEY'), prompt, a.max_tokens,
                    {'logprobs': True, 'top_logprobs': 20})
assert result.get('choices', [{}])[0].get('logprobs', {}).get('content'), 'API did not return requested log probabilities'
reason = bench.broken(bench.reply_text(result))
if reason: raise SystemExit(reason)
result['_arc_parity'] = {'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'max_tokens': a.max_tokens}
Path(a.output).write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
print('Saved fixed greedy/logprob comparison: ' + a.output)
