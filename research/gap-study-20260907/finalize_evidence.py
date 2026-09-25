"""Export compact case studies and fingerprint the experiment inputs and sources."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    if Path('/tmp/dar-arxiv-metadata.xml').exists():
        shutil.copyfile('/tmp/dar-arxiv-metadata.xml',OUT/'arxiv-target.atom')
    records=[json.loads(x) for p in sorted(OUT.glob('dar-shard[01].jsonl')) for x in p.read_text().splitlines()]
    cases=[]
    for vid,interpretation in [
        ('07227','The second phase asserts that small white squares emerge. The entire submitted tensor is uniform gray.'),
        ('56770','The second/third phases assert moving gray blocks and an ending title card. The entire submitted tensor is uniform gray.')]:
        row=next(r for r in records if r['video_id']==vid and r['condition']=='gray')
        cases.append(dict(video_id=vid,condition='gray',input_shape=row['input_shape'],input_sha256=row['input_sha256'],
                          segments=row['segments'],interpretation=interpretation,
                          scope='Qualitative selected counterexample; not a population hallucination-rate estimate or expected emotion label.'))
    (OUT/'evidence-cases.json').write_text(json.dumps(cases,indent=2,ensure_ascii=False)+'\n')
    sources=[ROOT/'DAR/test.py',ROOT/'DAR/tools/run_affectgpt_dar.py',
             ROOT/'DAR/qwen-vl-utils/src/qwen_vl_utils/vision_process.py',
             ROOT/'AffectGPT-src/AffectGPT/my_affectgpt/models/affectgpt.py',
             ROOT/'AffectGPT-src/AffectGPT/train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz.yaml',
             ROOT/'research/sources/DAR-2607.10238v1.pdf',
             ROOT/'data/DAR-R1-annotations/train.jsonl',ROOT/'data/DAR-R1-annotations/test.jsonl']
    for model in ['DAR-R1','Qwen2.5-VL-3B-Instruct']:
        sources.extend(sorted((ROOT/'model'/model).glob('*.safetensors')))
        sources.extend(sorted((ROOT/'model'/model).glob('*config.json')))
    sources.append(ROOT/'model/AffectGPT/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz_20250408110/checkpoint_000030_loss_0.751.pth')
    commits={name:subprocess.check_output(['git','-C',str(ROOT/name),'rev-parse','HEAD'],text=True).strip() for name in ['DAR','AffectGPT-src']}
    fingerprints=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p)) for p in sources]
    report=dict(date='2026-09-07',commits=commits,input_fingerprints=fingerprints,
                reproduction_scope='Hashes fingerprint artifacts now; do not reconstruct or assert the historical revision of prior baseline runs.',
                affectgpt_dependencies='The adapter checkpoint and code/config are fingerprinted; language/CLIP/audio backbone cache weights are not all independently hashed in this study.')
    (OUT/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(commits=commits,fingerprinted_files=len(fingerprints),cases=len(cases))))


if __name__=='__main__':
    main()
