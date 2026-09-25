"""Conservative reconciliation under the frozen rubric, retaining every dispute."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

OUT=Path(__file__).resolve().parent
DATA=OUT/'timing-corrected'


def read(name):
    rows=[json.loads(x) for x in (DATA/name).read_text().splitlines()]
    out={r['blind_id']:r for r in rows}
    assert len(rows)==len(out)
    return out


def main():
    primary=read('coding-primary.jsonl')
    independent=read('coding-independent.jsonl')
    blind=read('coding-blind.jsonl')
    selected=read('coding-independent-selected.jsonl')
    assert len(primary)==len(blind)==304
    assert len(independent)==len(selected)==206 and set(independent)==set(selected)
    negatives=sorted(k for k,r in primary.items() if r['label']=='none')
    expected={k for k,r in primary.items() if r['label']!='none'}|set(random.Random(20260908).sample(negatives,33))
    assert set(independent)==expected
    output=[]
    disagreements=[]
    for ident,p in primary.items():
        r=independent.get(ident)
        raw=blind[ident]['raw_output']
        digest=hashlib.sha256(raw.encode()).hexdigest()
        assert p['raw_sha256']==digest
        if r:
            assert r['raw_sha256']==digest
            assert r['label'] in ('confirmed','none','ambiguous','unassessable')
            if r['label']=='confirmed':assert r['quote'] and r['quote'] in raw
        if p['label']=='confirmed':
            assert r is not None
            label='confirmed' if r['label']=='confirmed' else 'ambiguous'
        elif r and r['label']!=p['label']:
            label='ambiguous'
        else:
            label=p['label']
        # All 32 disagreements remain visible and outside the positive endpoint.
        if r and r['label']!=p['label']:
            disagreements.append(dict(blind_id=ident,primary=p,independent=r,final_label=label))
        quote=p['quote'] if label=='confirmed' else ''
        assert label!='confirmed' or quote in raw
        output.append(dict(blind_id=ident,label=label,quote=quote,raw_sha256=digest,
            coder='primary_plus_independent_AI_conservative',primary_label=p['label'],
            independent_label=r['label'] if r else None,
            note='Residual disagreement retained as ambiguous; no positive upgrade.' if r and r['label']!=p['label'] else
                 'Independent agreement.' if r else 'Primary negative outside seeded independent review subset.'))
    with (DATA/'coding-final.jsonl').open('x') as stream:
        for r in output:stream.write(json.dumps(r,ensure_ascii=False)+'\n')
    summary=dict(status='completed with residual disputes excluded from confirmed endpoint',
        primary_read=304,independent_read=206,
        reviewed_primary_confirmed=sum(primary[k]['label']=='confirmed' for k in independent),
        reviewed_primary_ambiguous=sum(primary[k]['label']=='ambiguous' for k in independent),
        reviewed_primary_none=sum(primary[k]['label']=='none' for k in independent),
        primary_none_total=len(negatives),full_label_agreement=206-len(disagreements),
        disagreements=len(disagreements),final_counts=dict(Counter(r['label'] for r in output)),
        downgraded_positive_ids=[r['blind_id'] for r in output if r['primary_label']=='confirmed' and r['label']!='confirmed'],
        policy='A positive requires primary and independent agreement with exact quotations; every residual discrepancy remains ambiguous. No local Qwen-critic labels used.',
        limitations='Independent blinded AI pass, not human expert coding or diverse-model validation. Negative review covers seeded33/131; population intervals do not model annotation error.',
        disagreement_records=disagreements)
    (OUT/'adjudication.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='disagreement_records'},indent=2))


if __name__=='__main__':main()
