"""Skip a single-GPU screening arm only after a clear dual-GPU regression."""
import json, sys
from collections import defaultdict
from statistics import mean
try:
    def rates(path):
        d=defaultdict(list)
        for row in json.load(open(path)):
            rate=row.get('prompt_tps')
            if rate is not None and rate > 0: d[row['case']].append(rate)
        return {k:mean(v) for k,v in d.items() if len(v)>=2}
    base, candidate = map(rates, sys.argv[1:3])
    reject=bool(base) and base.keys()==candidate.keys() and all(candidate[k]<0.95*base[k] for k in base)
except (OSError,ValueError,KeyError,TypeError):
    reject=False
sys.exit(0 if reject else 1)
