"""Transparent exploratory metrics; values are fractions, never implicit percent."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def transform_frames(frames, condition):
    if condition == 'original':
        return frames.copy()
    if condition == 'reverse':
        return frames[::-1].copy()
    if condition == 'static_middle':
        return np.repeat(frames[len(frames)//2:len(frames)//2+1], len(frames), axis=0)
    if condition == 'gray':
        return np.full_like(frames, 127.5)
    raise ValueError(condition)


def iou(p, g):
    lo = max(float(p['start_time']), float(g['start_time']))
    hi = min(float(p['end_time']), float(g['end_time']))
    union = max(float(p['end_time']), float(g['end_time'])) - min(float(p['start_time']), float(g['start_time']))
    return max(0, hi-lo)/union if union > 0 else 0.


def label(p):
    return p.get('emotion', '').strip()


def video_stats(pred, gt):
    pairs = list(zip(pred, gt))
    overlaps = [iou(p, g) for p, g in pairs]
    correct = [label(p) == label(g) for p, g in pairs]
    qualified = [x >= .5 for x in overlaps]
    tp = 0
    temporal_tp = 0
    if pred and gt:
        mat = np.array([[iou(p, g) for g in gt] for p in pred])
        eligible = mat >= .5
        r, c = linear_sum_assignment(eligible.astype(float), maximize=True)
        temporal_tp = int(eligible[r, c].sum())
        emotion_ok = np.array([[label(p) == label(g) for g in gt] for p in pred])
        eligible = eligible & emotion_ok
        r, c = linear_sum_assignment(eligible.astype(float), maximize=True)
        tp = int(eligible[r, c].sum())
    return dict(valid=int(bool(pred)), count_correct=int(len(pred)==len(gt)), pred=len(pred),
                gt=len(gt), pairs=len(pairs), iou_sum=sum(overlaps),
                label_correct=sum(correct), qualified=sum(qualified),
                joint=sum(a and b for a, b in zip(correct, qualified)), tp=tp,
                temporal_tp=temporal_tp, gt_count=len(gt))


def aggregate(rows):
    keys = ['valid','count_correct','pred','gt','pairs','iou_sum','label_correct','qualified','joint','tp','temporal_tp']
    s = {k: sum(r[k] for r in rows) for k in keys}
    def div(a, b):
        return a/b if b else 0.
    return dict(videos=len(rows), valid_count=s['valid'], coverage=div(s['valid'],len(rows)),
                gt_segments=s['gt'], predicted_segments=s['pred'], compared_pairs=s['pairs'],
                qualified_pairs=s['qualified'], joint_correct=s['joint'], matched_correct=s['tp'],
                segment_count_accuracy_all=div(s['count_correct'],len(rows)),
                segment_count_accuracy_valid=div(s['count_correct'],s['valid']),
                index_miou=div(s['iou_sum'],s['pairs']),
                unconditioned_emotion_accuracy=div(s['label_correct'],s['pairs']),
                conditional_emotion_accuracy=div(s['joint'],s['qualified']),
                strict_compared_joint_accuracy=div(s['joint'],s['pairs']),
                fixed_gt_index_joint_recall=div(s['joint'],s['gt']),
                fixed_gt_index_iou=div(s['iou_sum'],s['gt']),
                matched_emotion_precision=div(s['tp'],s['pred']),
                matched_emotion_recall=div(s['tp'],s['gt']),
                matched_emotion_f1=div(2*s['tp'],s['pred']+s['gt']),
                matched_temporal_recall=div(s['temporal_tp'],s['gt']))
