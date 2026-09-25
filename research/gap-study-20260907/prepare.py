"""Freeze sample using annotations and video metadata only, never model outputs."""
import hashlib
import json
import random
from pathlib import Path
import av

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
seed = 20260907
rng = random.Random(seed)
rows = [json.loads(x) for x in (ROOT/'data/DAR-R1-annotations/test.jsonl').read_text().splitlines() if x.strip()]
groups = {'single': [], 'multi': []}
for row in rows:
    if not 0 < row['video_duration'] <= 30:
        continue
    n = len(json.loads(row['conversations'][1]['value'])['segments'])
    groups['single' if n == 1 else 'multi'].append(row)
chosen = []
for group, count in [('single',16), ('multi',32)]:
    pool = sorted(groups[group], key=lambda x: x['video'])
    rng.shuffle(pool)
    accepted = 0
    for row in pool:
        vid = Path(row['video']).stem
        path = ROOT/'data/vce_dataset/videos'/f'{vid}.mp4'
        with av.open(str(path)) as container:
            frames = container.streams.video[0].frames
        if frames < 16:
            continue
        chosen.append(dict(video_id=vid, video_path=str(path), video_duration=row['video_duration'],
                           gt_segments=json.loads(row['conversations'][1]['value'])['segments'],
                           stratum=group, encoded_frames=frames))
        accepted += 1
        if accepted == count:
            break
    assert accepted == count
rng.shuffle(chosen)
manifest = dict(seed=seed, sample_size=len(chosen), eligible_pools_before_frame_filter={k:len(v) for k,v in groups.items()},
                test_sha256=hashlib.sha256((ROOT/'data/DAR-R1-annotations/test.jsonl').read_bytes()).hexdigest(),
                conditions=['original','reverse','static_middle','gray'], videos=chosen)
path = OUT/'manifest.json'
if path.exists():
    assert json.loads(path.read_text()) == manifest, 'Existing sample differs; refusing overwrite'
else:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k != 'videos'},indent=2))
