"""Freeze pilot split, then create matched SFT arms from train-only evidence."""
import argparse
import json
from pathlib import Path

from core import ARMS, annotation, digest, read_jsonl, sft_rows, validate_graph, write_jsonl


def split(args):
    rows = [annotation(r, args.video_root) for r in read_jsonl(args.train)]
    ids = [r['video_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate source video IDs; split at video level first')
    heldout = {annotation(r)['video_id'] for r in read_jsonl(args.test)}
    if set(ids) & heldout:
        raise ValueError('Train/test video overlap')
    rows.sort(key=lambda r: digest([args.seed, r['video_id']]))
    train_size = len(rows) - args.dev_size if args.train_size == -1 else args.train_size
    if train_size < 1 or args.dev_size < 1 or len(rows) < train_size + args.dev_size:
        raise ValueError('Insufficient rows or invalid requested sizes')
    dev, train = rows[:args.dev_size], rows[args.dev_size:args.dev_size + train_size]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / 'train.jsonl', train)
    write_jsonl(out / 'dev.jsonl', dev)
    # The teacher receives no labels, GT event boundaries or previous model predictions.
    write_jsonl(out / 'teacher_inputs.jsonl', [{k: r[k] for k in ('video_id', 'video_path', 'video_duration')} for r in train])
    (out / 'split.json').write_text(json.dumps(dict(seed=args.seed, train=len(train), dev=len(dev),
        train_source_sha256=digest(read_jsonl(args.train)), test_source_sha256=digest(read_jsonl(args.test)),
        train_ids=[r['video_id'] for r in train], dev_ids=[r['video_id'] for r in dev]), indent=2), encoding='utf-8')


def build(args):
    arms = getattr(args, 'arms', ARMS)
    need_graph = any(a in ('stsg', 'stsg_no_links') for a in arms)
    need_caption = 'caption' in arms
    rows = read_jsonl(Path(args.split) / 'train.jsonl')
    evidence = read_jsonl(args.evidence)
    by_id = {r['video_id']: r for r in evidence}
    if len(by_id) != len(evidence):
        raise ValueError('Duplicate evidence; do not select the best retry')
    if set(by_id) != {r['video_id'] for r in rows}:
        raise ValueError('Evidence must cover exactly the frozen train IDs, including failures')
    accepted, rejected = [], []
    for row in rows:
        e = by_id[row['video_id']]
        try:
            identity = {k: row[k] for k in ('video_id', 'video_path', 'video_duration')}
            if e.get('input_sha256') != digest(identity):
                raise ValueError('Evidence input identity mismatch')
            if need_graph:
                validate_graph(e['stsg'], row['video_duration'])
            if need_caption and (set(e['caption']) != {'caption'} or not isinstance(e['caption']['caption'], str) or not e['caption']['caption'].strip()):
                raise ValueError('Invalid caption')
            accepted.append(row)
        except (KeyError, ValueError, TypeError) as exc:
            rejected.append({'video_id': row['video_id'], 'error': str(exc)})
    if not accepted:
        raise ValueError('No common valid evidence; inspect teacher outputs')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    for arm in arms:
        write_jsonl(out / (arm + '.jsonl'), [s for r in accepted for s in sft_rows(r, by_id[r['video_id']], arm)])
    (out / 'build.json').write_text(json.dumps(dict(accepted=len(accepted), rejected=rejected,
        arms=list(arms), source_count=len(rows), evidence_sha256=digest(evidence), ids=[r['video_id'] for r in accepted],
        note='Same accepted video IDs and two rows per video in every arm. Schema validation is not factual verification.'), indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('split')
    for flag in ('train', 'test', 'video-root', 'output'):
        p.add_argument('--' + flag, required=True)
    p.add_argument('--train-size', type=int, default=256)
    p.add_argument('--dev-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=20260923)
    p.set_defaults(run=split)
    p = sub.add_parser('build')
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    for flag in ('split', 'evidence', 'output'):
        p.add_argument('--' + flag, required=True)
    p.set_defaults(run=build)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
