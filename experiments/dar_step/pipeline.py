"""Run scene -> FSG -> temporal merge -> cross-clip bridge, on train inputs only."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time

from common import compact, digest, dump, file_hash, parse, read_jsonl, require
from frames import select_frames
from graph import assemble, fsg, pair_result, scoped, validate, verified_graph
import prompts


class Calls:
    def __init__(self, teacher, folder):
        self.teacher, self.folder, self.count = teacher, Path(folder), 0

    def __call__(self, frames, prompt, stage, max_tokens=900):
        from PIL import Image
        pictures = []
        for frame in frames:
            with Image.open(frame['path']) as image:
                pictures.append(image.convert('RGB'))
        start = time.monotonic()
        output = self.teacher.generate(pictures, prompt, max_tokens=max_tokens)
        self.count += 1
        dump(self.folder / f'{self.count:04d}-{stage}.json', dict(
            stage=stage, frame_ids=[f['id'] for f in frames],
            image_sha256=[f['sha256'] for f in frames], prompt=prompt,
            seconds=time.monotonic() - start, **output))
        require(not output['hit_token_limit'], 'Teacher output hit token limit; inspect saved call')
        return parse(output['raw'])


def create_graph(sampling, ask):
    parsed, local_pairs, bridges, descriptions = {}, [], [], {}
    for frame in sampling['frames']:
        graph = fsg(ask([frame], prompts.PARSE, 'parse'))
        verdict = ask([frame], prompts.verify(graph), 'verify', max_tokens=500)
        parsed[frame['id']] = scoped(verified_graph(graph, verdict), frame['id'])
    scene_frames = {s['id']: [f for f in sampling['frames'] if f['scene'] == s['id']]
                    for s in sampling['scenes']}
    for scene in sampling['scenes']:
        frames = scene_frames[scene['id']]
        for a, b in zip(frames, frames[1:]):
            left, right = parsed[a['id']], parsed[b['id']]
            if left['objects'] and right['objects']:
                result = ask([a, b], prompts.pair(left, right), 'merge', max_tokens=500)
                local_pairs.append(pair_result(result, left, right))
        event_frames = frames if len(frames) <= 4 else [frames[i * (len(frames)-1) // 3] for i in range(4)]
        result = ask(event_frames, prompts.event_prompt(event_frames), 'event', max_tokens=160)
        from common import bounded_text
        require(set(result) == {'description'}, 'Invalid scene event response')
        bounded_text(result['description'])
        verdict = ask(event_frames, prompts.VISIBLE + 'Is every claim in this candidate scene description '
                      'supported by these images? Return exactly {"supported":true} or '
                      '{"supported":false}. Candidate: ' + compact(result), 'verify-event', max_tokens=40)
        require(set(verdict) == {'supported'} and type(verdict['supported']) is bool, 'Invalid event verdict')
        descriptions[scene['id']] = result['description'] if verdict['supported'] else ''
    # Build local identities first. One representative observation per local entity;
    # group them by frame to avoid querying the same image pair repeatedly.
    local = assemble(sampling, parsed, local_pairs, [], descriptions)
    representative, representatives_by_scene = {}, {s['id']: {} for s in sampling['scenes']}
    frame_by_id = {f['id']: f for f in sampling['frames']}
    for frame in local['frames']:
        for obj in frame['objects']:
            if obj['entity'] not in representative:
                representative[obj['entity']] = obj['id']
                representatives_by_scene[frame['scene']].setdefault(frame['id'], set()).add(obj['id'])
    for earlier, later in itertools.combinations(sampling['scenes'], 2):
        for aid, a_objects in representatives_by_scene[earlier['id']].items():
            for bid, b_objects in representatives_by_scene[later['id']].items():
                def subset(fid, ids):
                    return dict(objects=[o for o in parsed[fid]['objects'] if o['id'] in ids])
                left, right = subset(aid, a_objects), subset(bid, b_objects)
                result = ask([frame_by_id[aid], frame_by_id[bid]],
                             prompts.pair(left, right, bridge=True), 'bridge', max_tokens=300)
                bridges.append(pair_result(result, left, right, bridge=True))
    graph = assemble(sampling, parsed, local_pairs, bridges, descriptions)
    return graph, dict(frame_graphs=parsed, local_pairs=local_pairs, bridges=bridges,
                       event_frame_ids={sid: [f['id'] for f in (fs if len(fs) <= 4 else [fs[i*(len(fs)-1)//3] for i in range(4)])]
                                        for sid, fs in scene_frames.items()},
                       clip_graphs=[dict(scene=s, entities=[e for e in graph['entities'] if e['scene'] == s['id']],
                                        frames=[f for f in graph['frames'] if f['scene'] == s['id']],
                                        motion_links=[m for m in graph['motion_links']
                                                      if any(m[0] == o['id'] for f in graph['frames']
                                                             if f['scene'] == s['id'] for o in f['objects'])])
                                    for s in graph['scenes']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('input', 'output', 'model'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--threshold', type=float, default=27.)
    parser.add_argument('--min-scene-seconds', type=float, default=.5)
    parser.add_argument('--max-frames', type=int, default=24)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    require(rows and len({r['video_id'] for r in rows}) == len(rows), 'Empty/duplicate teacher IDs')
    require(all(set(r) == {'video_id', 'video_path', 'video_duration'} for r in rows),
            'Teacher input must be label-free: ID, path, duration only')
    require(1 <= args.max_frames <= 64, 'max_frames must be 1..64')
    # One fresh directory per run. Resume is intentionally not implicit.
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    import torch
    import transformers
    from teacher_vila import Vila, VILA_REVISION
    torch.manual_seed(args.seed)
    config = dict(vars(args), input_sha256=digest(rows), vila_revision=VILA_REVISION,
                  torch=torch.__version__, transformers=transformers.__version__,
                  source_sha256={p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')},
                  model_configs={str(p.relative_to(args.model)): file_hash(p)
                                 for p in Path(args.model).rglob('*.json')},
                  sampling='CFR; all frames; Hecate HSV/edge histograms; global k-means; subshot stillness',
                  verification='one visual verification pass; no self-consistency sampling')
    dump(out / 'run.json', config)
    teacher = Vila(args.model)
    with (out / 'evidence.jsonl').open('x', encoding='utf-8') as stream:
        for row in rows:
            folder = out / digest(row['video_id'])[:16]
            folder.mkdir()
            entry = dict(video_id=row['video_id'], input_sha256=digest(row), config_sha256=digest(config))
            calls = Calls(teacher, folder / 'calls')
            try:
                sampling = select_frames(row, folder / 'frames',
                    threshold=args.threshold, min_scene_seconds=args.min_scene_seconds,
                    max_frames=args.max_frames, seed=args.seed)
                dump(folder / 'sampling.json', sampling)
                graph, intermediate = create_graph(sampling, calls)
                validate(graph, row['video_duration'])
                dump(folder / 'intermediate.json', intermediate)
                dump(folder / 'graph.json', graph)
                entry.update(graph=graph, sampling=sampling)
            except (ValueError, KeyError, TypeError) as exc:
                entry['error'] = f'{type(exc).__name__}: {exc}'
                dump(folder / 'failure.json', entry)
            # Decoder/CUDA/infrastructure failures propagate rather than becoming data rejections.
            entry['calls'] = calls.count
            stream.write(compact(entry) + '\n')
            stream.flush()
            print(row['video_id'], 'error=', entry.get('error'), 'calls=', calls.count, flush=True)


if __name__ == '__main__':
    main()
