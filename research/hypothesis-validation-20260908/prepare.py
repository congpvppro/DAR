"""Select new intervention IDs and freeze machine-verifiable synthetic stimuli."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import time
import av
import numpy as np
from study import SEED,DURATIONS,build_cases,CONSERVATIVE,FACTUAL

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent


def main():
    target=OUT/'manifest.json'
    if target.exists():
        raise FileExistsError('Frozen manifest already exists; refusing to resample')
    old=json.loads((ROOT/'research/gap-study-20260907/manifest.json').read_text())
    excluded={r['video_id'] for r in old['videos']}
    source=ROOT/'data/DAR-R1-annotations/test.jsonl'
    rows=[json.loads(x) for x in source.read_text().splitlines() if x.strip()]
    eligible=[r for r in rows if 0<float(r['video_duration'])<=30 and Path(r['video']).stem not in excluded]
    eligible.sort(key=lambda r:r['video'])
    random.Random(SEED).shuffle(eligible)
    selected=[]
    rejected=[]
    for row in eligible:
        vid=Path(row['video']).stem
        path=ROOT/'data/vce_dataset/videos'/f'{vid}.mp4'
        with av.open(str(path)) as container:
            count=container.streams.video[0].frames
        if count<16:
            rejected.append(vid)
            continue
        selected.append(dict(video_id=vid,kind='natural',video_path=str(path),
                             video_duration=float(row['video_duration']),encoded_frames=count,
                             gt_segments=json.loads(row['conversations'][1]['value'])['segments']))
        if len(selected)==64:
            break
    assert len(selected)==64
    (OUT/'stimuli').mkdir(exist_ok=True)
    for duration in DURATIONS:
        path=OUT/'stimuli'/f'canvas-{duration:02d}.mp4'
        with av.open(str(path),'w') as container:
            stream=container.add_stream('libx264',rate=Fraction(16,duration))
            stream.width=336
            stream.height=336
            stream.pix_fmt='yuv420p'
            for _ in range(16):
                frame=av.VideoFrame.from_ndarray(np.full((336,336,3),128,dtype=np.uint8),format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(path)) as container:
            assert container.streams.video[0].frames==16
        for kind in ('gray','step'):
            selected.append(dict(video_id=f'{kind}-{duration:02d}',kind=kind,video_path=str(path),
                                 video_duration=float(duration),encoded_frames=16))
    manifest=dict(seed=SEED,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  excluded_prior_ids=sorted(excluded),eligible_before_frame_filter=len(eligible),
                  rejected_short_frame_ids=rejected,source_test_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  protocol_sha256=hashlib.sha256((OUT/'protocol.md').read_bytes()).hexdigest(),
                  prompts=dict(conservative_append=CONSERVATIVE,factual=FACTUAL),videos=selected)
    cases=build_cases(selected)
    assert len(cases)==368
    manifest['cases_per_model']=len(cases)
    manifest['expected_generations']=2*len(cases)
    target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('videos','prompts','excluded_prior_ids')},indent=2))


if __name__=='__main__':
    main()
