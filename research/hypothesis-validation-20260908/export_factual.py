"""Blinded full factual-answer panel, independent from event-rate coding."""
import hashlib
import json
from pathlib import Path
import random
from study import SEED,build_cases
OUT=Path(__file__).resolve().parent/'timing-corrected'

def main():
    m=json.loads((OUT/'manifest.json').read_text())
    cases=[(model,c['case_id']) for model in ('dar','qwen') for c in build_cases(m['videos']) if c['prompt_kind']=='factual']
    random.Random(SEED+201).shuffle(cases)
    key={f'F{i+1:03d}':dict(model=model,case_id=case) for i,(model,case) in enumerate(cases)}
    found={}
    for path in OUT.glob('*-shard*.jsonl'):
        for line in path.read_text().splitlines():
            r=json.loads(line)
            found[(r['model'],r['case_id'])]=r
    rows=[]
    for blind_id,k in key.items():
        r=found.get((k['model'],k['case_id']))
        if r is not None:
            rows.append(dict(blind_id=blind_id,kind=r['kind'],expected_change=r['kind']=='step',
                raw_output=r['raw_output'],raw_sha256=hashlib.sha256(r['raw_output'].encode()).hexdigest()))
    (OUT/'factual-key.json').write_text(json.dumps(key,indent=2)+'\n')
    (OUT/'factual-blind.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print('Factual available',len(rows),'of176')

if __name__=='__main__':main()
