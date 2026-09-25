"""Blind a fixed subset of all tasks; never select outputs by their content."""
import hashlib
import json
from pathlib import Path
import random
from study import SEED,build_cases

OUT=Path(__file__).resolve().parent


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    cases=[(model,c['case_id']) for model in ('dar','qwen') for c in build_cases(manifest['videos'])
           if c['condition'] in ('static_middle','gray') and c['prompt_kind']!='factual']
    random.Random(SEED+101).shuffle(cases)
    assert len(cases)==304
    key={f'B{i+1:03d}':dict(model=model,case_id=case_id) for i,(model,case_id) in enumerate(cases)}
    found={}
    for p in sorted(OUT.glob('*-shard*.jsonl')):
        for line in p.read_text().splitlines():
            r=json.loads(line)
            k=(r['model'],r['case_id'])
            assert k not in found
            found[k]=r
    rows=[]
    for blind_id,k in key.items():
        r=found.get((k['model'],k['case_id']))
        if r is None:
            continue
        rows.append(dict(blind_id=blind_id,input_kind='uniform_gray' if r['condition']=='gray' else 'identical_repeated_frame',
                         duration=r['video_duration'],raw_output=r['raw_output'],
                         raw_sha256=hashlib.sha256(r['raw_output'].encode()).hexdigest()))
    (OUT/'coding-key.json').write_text(json.dumps(key,indent=2)+'\n')
    for part in [0,1]:
        subset=[r for r in rows if (int(r['blind_id'][1:])-1)%2==part]
        (OUT/f'coding-blind-part{part}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in subset))
    (OUT/'coding-blind.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(json.dumps(dict(available=len(rows),expected=304,part0=sum((int(r['blind_id'][1:])-1)%2==0 for r in rows))))


if __name__=='__main__':
    main()
