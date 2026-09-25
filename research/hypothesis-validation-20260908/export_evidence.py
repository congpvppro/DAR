"""Export every confirmed event and four illustrative, fully auditable examples."""
import csv
import json
from pathlib import Path
from analyze import read_outputs

OUT=Path(__file__).resolve().parent
DATA=OUT/'timing-corrected'


def main():
    labels=[json.loads(x) for x in (DATA/'coding-final.jsonl').read_text().splitlines()]
    key=json.loads((DATA/'coding-key.json').read_text())
    rows=read_outputs(DATA)
    confirmed=[]
    examples=[]
    for label in labels:
        if label['label']!='confirmed':continue
        ident=key[label['blind_id']]
        row=rows[ident['model'],ident['case_id']]
        confirmed.append(dict(blind_id=label['blind_id'],model=ident['model'],case_id=ident['case_id'],
            kind=row['kind'],duration=row['video_duration'],quote=label['quote'],
            input_sha256=row['input_sha256'],raw_sha256=label['raw_sha256']))
        if label['blind_id'] in ('B014','B021','B042','B074'):
            examples.append(dict(**confirmed[-1],raw_output=row['raw_output'],input_shape=row['input_shape'],
                mm_processor_kwargs=row['mm_processor_kwargs'],known_invariant='All16frames exactly identical; gray also all pixels/channels127.5.',
                selection='Illustrative cases selected after reading; rates use all coded tasks.'))
    assert len(confirmed)==124 and len(examples)==4
    with (OUT/'confirmed-events.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(confirmed[0]))
        writer.writeheader()
        writer.writerows(confirmed)
    (OUT/'evidence-cases.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(confirmed_rows=len(confirmed),illustrative_cases=len(examples))))


if __name__=='__main__':main()
