#!/usr/bin/env python3
"""Stream a unitrace Chrome timeline and summarize verifier-body GPU intervals.

Each body spans the first resident-plan kernel through the main head sampler
for `--layers` routed layers. Prefix work is excluded. Includes trace overhead;
normal untraced profile runs remain the source of wall-clock throughput.
"""
import json,bisect,re,statistics,sys
from collections import defaultdict
import argparse
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('timeline',nargs='+')
ap.add_argument('--layers',type=int,default=48)
opts=ap.parse_args()
if opts.layers<1:ap.error('--layers must be positive')
for path in opts.timeline:
 events=[];plans=[];samples=[];stats=defaultdict(float)
 for line in open(path):
  try:e=json.loads(line.rstrip(',\n'))
  except ValueError:continue
  if e.get('ph')!='X' or e.get('cat')!='gpu_op':continue
  n=e['name'];a=e['ts'];b=a+e['dur'];kernel=not n.startswith('zeCommandList')
  events.append((a,b,kernel,n))
  if 'resident_plan_kernel' in n:plans.append((a,b))
  if 'sampler_greedy_kernel' in n:samples.append((a,b,n))
 events.sort();plans.sort();samples.sort();starts=[e[0] for e in events];st=[e[0] for e in samples]
 def union(xs):
  if not xs:return 0
  xs.sort();total=0;a,b=xs[0]
  for c,d in xs[1:]:
   if c>b:total+=b-a;a,b=c,d
   else:b=max(b,d)
  return total+b-a
 rows=[]
 for i in range(0,len(plans)-opts.layers+1,opts.layers):
  begin=plans[i][0];last=plans[i+opts.layers-1][1];j=bisect.bisect_left(st,last)
  if j>=len(samples):continue
  b=samples[j][1]
  k0=bisect.bisect_left(starts,begin);k1=bisect.bisect_right(starts,b)
  k=[];allops=[];sumk=0
  for c,d,isk,n in events[k0:k1]:
   c=max(c,begin);d=min(d,b)
   if d<=c:continue
   allops.append((c,d))
   if isk:k.append((c,d));sumk+=d-c
  m=re.search(r'\[SIMD\d+ \{(\d+);',samples[j][2]);T=int(m.group(1)) if m else 0
  rows.append(dict(T=T,body_ms=(b-begin)/1000,kernel_sum_ms=sumk/1000,kernel_union_ms=union(k)/1000,all_ops_union_ms=union(allops)/1000))
 out={'file':path,'plans':len(plans),'windows':len(rows),'rows':rows}
 for label,selected in [('all',rows)]+[(f'T={t}',[r for r in rows if r['T']==t]) for t in sorted({r['T'] for r in rows})]:
  if selected:out[label]={k:round(statistics.mean(r[k] for r in selected),3) for k in ['body_ms','kernel_sum_ms','kernel_union_ms','all_ops_union_ms']};out[label]['n']=len(selected)
 if not rows:raise SystemExit(f'No complete resident-plan/head windows found in {path}')
 print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
 with open(path+'.summary.json','w') as f:json.dump(out,f,indent=2)
