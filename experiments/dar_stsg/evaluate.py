"""Paired, video-level evaluation. Missing/invalid outputs count as failures."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from core import digest, number, official_functions, read_jsonl


OFFICIAL = official_functions()


def valid_segments(obj, duration):
    if not isinstance(obj, dict) or set(obj) != {'segments'}:
        return []
    segs = obj['segments']
    if not isinstance(segs, list) or not segs:
        return []
    end, emotion = 0., None
    try:
        for s in segs:
            start, stop = number(s['start_time']), number(s['end_time'])
            if abs(start - end) > .051 or stop <= start or stop > duration + .051:
                return []
            if any(abs(x - round(x, 1)) > 1e-6 for x in (start, stop)):
                return []
            if s['emotion'] not in OFFICIAL['EMOTION_CANDIDATES'] or s['emotion'] == emotion:
                return []
            if not isinstance(s['reason'], str) or not s['reason'].strip():
                return []
            end, emotion = stop, s['emotion']
        return segs if abs(end - duration) <= .051 else []
    except (ValueError, TypeError, KeyError):
        return []


def stats(pred, gt):
    matched = 0
    if pred and gt:
        eligible = np.array([[OFFICIAL['calculate_iou'](a, b) >= .5 and a['emotion'] == b['emotion'] for b in gt] for a in pred])
        i, j = linear_sum_assignment(eligible.astype(int), maximize=True)
        matched = int(eligible[i, j].sum())
    return [matched, len(pred), len(gt), int(bool(pred)), int(len(pred) == len(gt))]


def aggregate(array):
    tp, p, g, valid, count = array.sum(axis=0)
    return dict(joint_f1_at_05=float(2 * tp / (p + g)) if p + g else 0.,
                coverage=float(valid / len(array)), count_accuracy=float(count / len(array)),
                predicted_segments=int(p), gt_segments=int(g))


def evaluate(manifest, predictions):
    by_id = {r['video_id']: r for r in predictions}
    if len(by_id) != len(predictions) or set(by_id) - {r['video_id'] for r in manifest}:
        raise ValueError('Duplicate or out-of-manifest predictions')
    arrays, official_results = [], {}
    for row in manifest:
        prediction = by_id.get(row['video_id'], {})
        identity = {k: row[k] for k in ('video_id', 'video_path', 'video_duration')}
        if prediction.get('input_sha256') is not None and prediction['input_sha256'] != digest(identity):
            raise ValueError('Prediction is from a different video input manifest')
        obj = prediction.get('dar')
        pred = valid_segments(obj, row['video_duration'])
        gt = row['target']['segments']
        arrays.append(stats(pred, gt))
        # A separate, explicitly labelled compatibility view uses repo repairs.
        try:
            repaired = OFFICIAL['validate_and_fix_segments'](copy.deepcopy(obj['segments']), row['video_duration']) if obj else []
            result = OFFICIAL['evaluate_single_video'](repaired, gt)
            if not all(np.isfinite(v) for v in result['ious']):
                raise ValueError('Non-finite repaired times')
        except (ValueError, TypeError, KeyError, AttributeError):
            result = OFFICIAL['evaluate_single_video']([], gt)
        official_results[row['video_id']] = result
    array = np.asarray(arrays)
    return array, dict(strict=aggregate(array), official_repaired=OFFICIAL['compute_overall_metrics'](official_results),
                       missing_videos=len(manifest) - len(by_id))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--predictions', nargs='+', required=True, help='name=path pairs')
    p.add_argument('--reference', default='caption')
    p.add_argument('--output', required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    args = p.parse_args()
    manifest = read_jsonl(args.manifest)
    if not manifest or len({r['video_id'] for r in manifest}) != len(manifest):
        raise ValueError('Empty or duplicate manifest')
    arrays, summary, frames = {}, {}, {}
    for item in args.predictions:
        name, path = item.split('=', 1)
        if name in arrays:
            raise ValueError('Repeated arm name')
        predictions = read_jsonl(path)
        for row in predictions:
            h = row.get('calls', {}).get('dar', {}).get('frame_sha256')
            if h is not None and row['video_id'] in frames and frames[row['video_id']] != h:
                raise ValueError('Mismatched sampled frames across arms')
            if h is not None:
                frames[row['video_id']] = h
        arrays[name], summary[name] = evaluate(manifest, predictions)
    if args.reference not in arrays or args.bootstrap < 1:
        raise ValueError('Missing reference or invalid bootstrap count')
    contrasts = {}
    for name, array in arrays.items():
        if name == args.reference:
            continue
        rng = np.random.default_rng(20260923)
        diffs = []
        for _ in range(args.bootstrap):
            idx = rng.integers(0, len(manifest), len(manifest))
            diffs.append(aggregate(array[idx])['joint_f1_at_05'] - aggregate(arrays[args.reference][idx])['joint_f1_at_05'])
        contrasts[name + '_minus_' + args.reference] = dict(
            delta=summary[name]['strict']['joint_f1_at_05'] - summary[args.reference]['strict']['joint_f1_at_05'],
            paired_video_bootstrap_95ci=np.quantile(diffs, [.025, .975]).tolist())
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(dict(videos=len(manifest), arms=summary, contrasts=contrasts,
                       note='Fractions, not percentages. CI conditions on training seed and selected videos; not training variance.'), stream, indent=2)


if __name__ == '__main__':
    main()
