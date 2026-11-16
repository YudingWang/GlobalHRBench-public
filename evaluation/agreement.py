"""Unmodified agreement calculation; review slots are not person identities."""
from collections import Counter

def agreement(pairs):
    """Pre-adjudication exact agreement, marginals, directional confusion, kappa."""
    n=len(pairs)
    if not n: return {'n':0,'raw_exact_agreement':None,'cohens_kappa':None}
    left=Counter(a for a,b in pairs);right=Counter(b for a,b in pairs)
    observed=sum(a==b for a,b in pairs)/n
    expected=sum(left[x]*right[x] for x in set(left)|set(right))/n**2
    kappa=(observed-expected)/(1-expected) if expected<1 else None
    return {'n':n,'raw_exact_agreement':observed,'reviewer_1_marginals':dict(left),'reviewer_2_marginals':dict(right),
            'cohens_kappa':kappa,'kappa_note':'undefined for degenerate marginals; descriptive with clustered units',
            'directional_confusion':[{'reviewer_1':a,'reviewer_2':b,'count':v} for (a,b),v in sorted(Counter(pairs).items())]}
