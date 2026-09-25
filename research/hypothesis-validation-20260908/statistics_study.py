"""Explicit paired statistics, retaining missing/ambiguous task denominators."""
from collections import Counter
import math
import numpy as np
from scipy.stats import binomtest
from study import SEED


def event_summary(labels):
    counts=Counter(labels)
    if not set(counts)<= {'confirmed','none','ambiguous','unassessable'}:
        raise ValueError(counts)
    n=len(labels)
    k=counts['confirmed']
    z=1.959963984540054
    if n:
        p=k/n
        center=(p+z*z/(2*n))/(1+z*z/n)
        half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        ci=[max(0.,center-half),min(1.,center+half)]
    else:
        p=None
        ci=[None,None]
    decidable=k+counts['none']
    return dict(n=n,**{x:counts[x] for x in ['confirmed','none','ambiguous','unassessable']},
                rate_all=p,rate_decidable=k/decidable if decidable else None,
                wilson95=ci,binomial_p_against_5pct=binomtest(k,n,.05,alternative='greater').pvalue if n else None)


def paired_binary(a,b):
    if len(a)!=len(b) or not a:
        raise ValueError('Nonempty aligned pairs required')
    a=np.asarray(a,dtype=int)
    b=np.asarray(b,dtype=int)
    if not np.all(np.isin(a,[0,1])) or not np.all(np.isin(b,[0,1])):
        raise ValueError('Binary observations required')
    a_only=int(((a==1)&(b==0)).sum())
    b_only=int(((b==1)&(a==0)).sum())
    n=len(a)
    rng=np.random.default_rng(SEED)
    draws=rng.integers(0,n,size=(5000,n))
    delta=(a-b)[draws].mean(axis=1)
    return dict(n=n,a_only=a_only,b_only=b_only,both=int(((a==1)&(b==1)).sum()),
                neither=int(((a==0)&(b==0)).sum()),difference=float((a-b).mean()),
                difference_ci95=np.quantile(delta,[.025,.975]).tolist(),
                bootstrap_degenerate=bool(np.all((a-b)==(a-b)[0])),
                uncertainty_note='An empirical bootstrap interval with no observed paired variation does not establish population equivalence or exclude unseen discordant outcomes.' if np.all((a-b)==(a-b)[0]) else None,
                mcnemar_exact_p=binomtest(a_only,a_only+b_only,.5).pvalue if a_only+b_only else 1.)


def holm(pvalues):
    order=sorted(range(len(pvalues)),key=lambda i:pvalues[i])
    adjusted=[0.]*len(pvalues)
    largest=0.
    for rank,i in enumerate(order):
        largest=max(largest,min(1.,pvalues[i]*(len(order)-rank)))
        adjusted[i]=largest
    return adjusted
