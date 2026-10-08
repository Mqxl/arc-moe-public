"""Gate further screening on a measurable long-prompt gain without a clear 2K loss."""
import json, sys
from collections import defaultdict
from statistics import mean
try:
    def rates(path):
        d=defaultdict(list)
        for row in json.load(open(path)):
            rate=row.get('prompt_tps')
            if rate is not None and rate>0: d[row['case']].append(rate)
        return {k:mean(v) for k,v in d.items() if len(v)>=2}
    base, candidate = map(rates,sys.argv[1:3])
    gain=candidate['10K code']>=1.03*base['10K code'] and candidate['2K code']>=0.98*base['2K code']
except (OSError,ValueError,KeyError,TypeError):
    gain=False
sys.exit(0 if gain else 1)
