"""Record explicit researcher coding decisions without selecting unseen outputs."""
import argparse
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent/'timing-corrected'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('decisions',help='JSON array of [blind_id,label,exact_quote,note]')
    args=ap.parse_args()
    source={r['blind_id']:r for r in map(json.loads,(OUT/'coding-blind.jsonl').read_text().splitlines())}
    path=OUT/'coding-primary.jsonl'
    existing={r['blind_id']:r for r in map(json.loads,path.read_text().splitlines())} if path.exists() else {}
    for blind_id,label,quote,note in json.loads(args.decisions):
        assert blind_id in source
        assert label in ('confirmed','none','ambiguous','unassessable')
        assert not quote or quote in source[blind_id]['raw_output'],(blind_id,quote)
        assert label!='confirmed' or quote
        assert blind_id not in existing,blind_id
        existing[blind_id]=dict(blind_id=blind_id,label=label,quote=quote,note=note,
                               raw_sha256=source[blind_id]['raw_sha256'],coder='primary_researcher_AI')
    path.write_text(''.join(json.dumps(existing[k],ensure_ascii=False)+'\n' for k in sorted(existing)))
    print('Coded',len(existing),'of304')


if __name__=='__main__':
    main()
