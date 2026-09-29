"""Build DAR + whole-video graph SFT with the exact teacher keyframes."""
import argparse
import copy
import json
from pathlib import Path
import sys

from common import compact, digest, dump, file_hash, read_jsonl, require
from graph import validate
from prompts import STUDENT

LEGACY = Path(__file__).resolve().parents[1] / 'dar_stsg'
sys.path.insert(0, str(LEGACY))
from core import sft_rows as legacy_rows


def sft_rows(row, evidence, arm='stsg'):
    answer = legacy_rows(row, {}, 'baseline')[0]
    graph = copy.deepcopy(evidence['graph'])
    prompt = STUDENT
    if arm == 'stsg_no_links':
        graph['motion_links'], graph['reference_links'], graph['event_links'] = [], [], []
        prompt += '\nFor this ablation leave motion_links, reference_links and event_links empty.'
    frames = evidence['sampling']['frames']
    prompt += '\nScene intervals: ' + compact(evidence['sampling']['scenes']) + '\n'
    prompt += '\n'.join(f"{f['id']}, {f['scene']}, t={f['time']:.4f}s: <image>" for f in frames)
    aux = dict(id=row['video_id'] + ':' + arm, images=[f['path'] for f in frames], messages=[
        dict(role='user', content=prompt), dict(role='assistant', content=compact(graph))])
    return [answer, aux]


def build(split, evidence_path, output, arm):
    rows = read_jsonl(Path(split) / 'train.jsonl')
    evidence = read_jsonl(evidence_path)
    by_id = {e['video_id']: e for e in evidence}
    require(len(by_id) == len(evidence) and set(by_id) == {r['video_id'] for r in rows},
            'Evidence must cover exactly all frozen train IDs, including failures')
    require(len({e['config_sha256'] for e in evidence}) == 1, 'Mixed teacher runs')
    accepted, rejected, dataset = [], [], []
    for row in rows:
        e = by_id[row['video_id']]
        try:
            require('error' not in e, e.get('error', 'Teacher error'))
            identity = {k: row[k] for k in ('video_id', 'video_path', 'video_duration')}
            require(e['input_sha256'] == digest(identity), 'Teacher input mismatch')
            validate(e['graph'], row['video_duration'])
            require(e['sampling']['video_sha256'] == file_hash(row['video_path']), 'Source video changed')
            expected = [(f['id'], f['scene'], f['time']) for f in e['graph']['frames']]
            sampled = [(f['id'], f['scene'], f['time']) for f in e['sampling']['frames']]
            require(expected == sampled, 'Student/teacher frames or timestamps differ')
            intervals = [{k: s[k] for k in ('id', 'start', 'end')} for s in e['graph']['scenes']]
            require(intervals == e['sampling']['scenes'], 'Student/teacher scene intervals differ')
            for frame in e['sampling']['frames']:
                require(file_hash(frame['path']) == frame['sha256'], 'Teacher keyframe changed')
            dataset.extend(sft_rows(row, e, arm))
            accepted.append(row['video_id'])
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append(dict(video_id=row['video_id'], error=str(exc)))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'build.json', dict(accepted=len(accepted), ids=accepted, rejected=rejected,
         source_count=len(rows), arm=arm, evidence_sha256=digest(evidence),
         dataset_sha256=digest(dataset), note='Graph target only during training; images exactly match teacher selection.'))
    require(accepted, 'No accepted graph; inspect teacher calls and build.json')
    (out / (arm + '.jsonl')).write_text(''.join(compact(r) + '\n' for r in dataset), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('split', 'evidence', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--arm', choices=('stsg', 'stsg_no_links'), default='stsg')
    args = parser.parse_args()
    build(args.split, args.evidence, args.output, args.arm)
