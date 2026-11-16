"""Unmodified scoring/statistical functions from the frozen evaluation."""
import math, random
from collections import defaultdict
ATOM_LABELS=("covered","contradicted","absent","unclear")
FAILURES={"api_failure","refusal","malformed_json","schema_invalid","truncation","abstention","insufficient_evidence"}

def indexed(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('duplicate identity: ' + key)
    return result

def logic_paths(logic):
    """DNF alternatives: omission of an unchosen equivalent path is not an error."""
    if isinstance(logic,str): return [frozenset([logic])]
    args = logic['args']
    if not args or logic['op'] not in ('all_of','any_of'): raise ValueError('invalid required atom logic')
    if logic['op']=='any_of':
        return [p for a in args for p in logic_paths(a)]
    result = [frozenset()]
    for a in args: result = [p|q for p in result for q in logic_paths(a)]
    return list(set(result))

def rule_metrics(annotation, labels):
    atoms = indexed(annotation['atoms'],'atom_id')
    required = set(annotation['required_atom_ids'])
    if len(required) != len(annotation['required_atom_ids']): raise ValueError('duplicate required atom')
    if set(labels) != required or any(x not in ATOM_LABELS for x in labels.values()):
        raise ValueError('required atom labels incomplete')
    paths = logic_paths(annotation['omission_scoring']['required_atom_logic'])
    if set().union(*paths) != required: raise ValueError('required logic/atom mismatch')
    if any(atoms[x]['supplied_in_question'] or atoms[x]['is_supplementary'] for x in required):
        raise ValueError('given/supplementary omission penalty')
    # Choose one best legal alternative, then report every metric on that SAME path.
    def coverage(path): return sum(labels[x]=='covered' for x in path)/len(path)
    best = sorted(paths,key=lambda p:(-coverage(p),tuple(sorted(p))))[0]
    def role_mean(field):
        subset = [x for x in best if atoms[x][field]]
        return sum(labels[x]=='covered' for x in subset)/len(subset) if subset else None
    return {'answer_value_coverage':role_mean('is_answer_value'),
            'qualification_coverage':role_mean('is_qualification'),
            'required_atom_coverage':coverage(best),
            'full_required_rule_completeness':float(any(all(labels[x]=='covered' for x in p) for p in paths)),
            'selected_required_path':sorted(best),
            'contradicted_required_atoms':sum(x=='contradicted' for x in labels.values())}

def aggregate_fixed(planned, observed, metric, group_field=None):
    """Missing evaluations remain missing; failures must be explicit zeros upstream."""
    indexed(planned,'request_id'); obs=indexed(observed,'request_id')
    if not set(obs) <= {x['request_id'] for x in planned}: raise ValueError('unplanned result')
    buckets=defaultdict(list); missing=0
    for p in planned:
        v=obs.get(p['request_id'],{}).get(metric)
        if v is not None and (type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1):
            raise ValueError('invalid metric value')
        missing += v is None
        buckets[p[group_field] if group_field else p['request_id']].append(v)
    if not buckets: return {'n_planned':0,'n_scored':0,'n_groups':0,'estimate':None,'lower_bound':None,'upper_bound':None}
    lower=sum(sum(v or 0 for v in vs)/len(vs) for vs in buckets.values())/len(buckets)
    upper=sum(sum(1 if v is None else v for v in vs)/len(vs) for vs in buckets.values())/len(buckets)
    return {'n_planned':len(planned),'n_scored':len(planned)-missing,'n_groups':len(buckets),
            'estimate':lower if not missing else None,'lower_bound':lower,'upper_bound':upper}

def paired_cluster_bootstrap(rows, left, right, group_field, seed=20261105, replicates=2000):
    """Paired question-weighted difference, resampling entire source/fact groups."""
    indexed(rows,'canonical_question_id')
    groups=defaultdict(list)
    for r in rows:
        if r[left] is None or r[right] is None: raise ValueError('incomplete paired outcomes')
        groups[r[group_field]].append(r[right]-r[left])
    keys=sorted(groups); rng=random.Random(seed); draws=[]
    if len(keys)<2: return {'interval':None,'n_clusters':len(keys),'interpretation':'insufficient clusters'}
    for _ in range(replicates):
        vals=[v for k in rng.choices(keys,k=len(keys)) for v in groups[k]]
        draws.append(sum(vals)/len(vals))
    draws.sort()
    return {'interval':[draws[int(.025*replicates)],draws[min(replicates-1,int(.975*replicates))]],
            'n_clusters':len(keys),'replicates':replicates,'seed':seed,
            'interpretation':'descriptive, potentially unstable operational-group bootstrap; not population inference'}
