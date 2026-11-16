"""Unmodified broad aggregation and source-group bootstrap."""
import statistics
import numpy as np
from collections import defaultdict

def mean(values):
    return statistics.mean(values) if values else None

def equal_groups(rows,field):
    groups=defaultdict(list)
    for r in rows:groups[r['group_id']].append(float(r[field]))
    return {'n_groups':len(groups),'estimate':mean([mean(v) for v in groups.values()])}

def paired_summary(rows,left,right,reps=5000,seed=31):
    deltas=[dict(r,delta=float(r[right])-float(r[left])) for r in rows]
    groups=defaultdict(list)
    for r in deltas:groups[r['group_id']].append(r['delta'])
    gm=np.array([mean(groups[g]) for g in sorted(groups)])
    rng=np.random.default_rng(seed)
    draws=gm[rng.integers(0,len(gm),size=(reps,len(gm)))].mean(axis=1)
    return {'n_pairs':len(rows),'n_groups':len(groups),'left':mean([r[left] for r in rows]),'right':mean([r[right] for r in rows]),
            'question_difference':mean([r['delta'] for r in deltas]),'equal_group_difference':float(gm.mean()),
            'equal_group_interval':[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],
            'wins':sum(r['delta']>0 for r in deltas),'ties':sum(r['delta']==0 for r in deltas),'losses':sum(r['delta']<0 for r in deltas),
            'bootstrap_replicates':reps,'bootstrap_seed':seed,'interpretation':'Descriptive source-group bootstrap, models fixed; not population/model significance ranking.'}
