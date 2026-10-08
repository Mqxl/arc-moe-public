#!/usr/bin/env python3
"""Compare greedy continuations and normalized logits from two controlled prompt modes."""
import json, sys
from pathlib import Path
left,right=(json.loads(Path(x).read_text()) for x in sys.argv[1:3])
if '_arc_parity' in left and '_arc_parity' in right:
 assert left['_arc_parity'] == right['_arc_parity'], 'Prompt hash or token limit differs'
a,b=left['choices'][0],right['choices'][0]
assert a['message']['content']==b['message']['content'], 'Greedy continuations differ'
x,y=a['logprobs']['content'],b['logprobs']['content']
assert len(x)==len(y), 'Token counts differ'
delta=0.0
for l,r in zip(x,y):
 assert l.get('bytes',l['token'])==r.get('bytes',r['token']), 'Greedy tokens differ'
 delta=max(delta,abs(l['logprob']-r['logprob']))
 lp={tuple(v.get('bytes',v['token'].encode())):v['logprob'] for v in l['top_logprobs']}
 rp={tuple(v.get('bytes',v['token'].encode())):v['logprob'] for v in r['top_logprobs']}
 assert lp.keys()==rp.keys(), 'Top-20 token sets differ'
 delta=max(delta,max((abs(lp[k]-rp[k]) for k in lp),default=0))
print(f'greedy parity: {len(x)} tokens identical; top-20 normalized-logit max delta {delta:.9g}')
assert delta <= 1e-4, f'Normalized logits differ by {delta}'
