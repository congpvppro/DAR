"""Describe DAR-R1 errors against full-test labels; no new inference or video audit."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def overlap(a, b):
    intersection = max(0, min(a['end_time'], b['end_time']) - max(a['start_time'], b['start_time']))
    union = max(a['end_time'], b['end_time']) - min(a['start_time'], b['start_time'])
    return intersection / union if union > 0 else 0


def main():
    gt_path = ROOT / 'data/DAR-R1-annotations/test.jsonl'
    gt = {Path(r['video']).stem: json.loads(r['conversations'][1]['value'])['segments'] for r in rows(gt_path)}
    base = ROOT / 'outputs/paper-repro-gpu01'
    paths = sorted((base / 'strict-4096').glob('predictions_shard_*.jsonl'))
    paths += sorted((base / 'retry-8192').glob('predictions_retry_shard_*.jsonl'))
    pred = {}
    for path in paths:
        for r in rows(path):
            if r.get('segments'):
                pred[r['video_id']] = r['segments']
    assert set(gt) == set(pred) and len(gt) == 1441
    support = Counter(s['emotion'].strip() for g in gt.values() for s in g)
    predicted = Counter(s['emotion'].strip() for p in pred.values() for s in p)
    counts = Counter()
    matrix = Counter()
    pairs = []
    for vid, g in gt.items():
        p = pred[vid]
        counts['under' if len(p) < len(g) else 'over' if len(p) > len(g) else 'equal'] += 1
        matrix[len(g), len(p)] += 1
        for index, (a, b) in enumerate(zip(p, g)):
            pairs.append(dict(video_id=vid, segment_index=index, gt=b['emotion'].strip(),
                              pred=a['emotion'].strip(), iou=overlap(a, b),
                              gt_interval=[b['start_time'], b['end_time']],
                              pred_interval=[a['start_time'], a['end_time']]))
    panels = {}
    for threshold in [.5, .8]:
        selected = [p for p in pairs if p['iou'] >= threshold]
        confusion = Counter((p['gt'], p['pred']) for p in selected)
        qualified = Counter(p['gt'] for p in selected)
        examples = defaultdict(list)
        for p in sorted(selected, key=lambda p: (-p['iou'], p['video_id'], p['segment_index'])):
            if p['gt'] != p['pred']:
                examples[p['gt'], p['pred']].append(p)
        panels[str(threshold)] = dict(
            qualified_pairs=len(selected), correct=sum(p['gt'] == p['pred'] for p in selected),
            per_class=[dict(emotion=e, gt_segments=support[e], predicted_segments=predicted[e],
                            qualified=qualified[e], correct=confusion[e, e]) for e in sorted(support)],
            confusion=[dict(gt=g, pred=p, count=n, qualified_gt=qualified[g], examples=examples[g, p][:3])
                       for (g, p), n in sorted(confusion.items(), key=lambda x: (-x[1], x[0]))])
    summary = dict(
        scope='DAR-R1 full benchmark; last nonempty historical prediction per ID, including retries',
        method='Same-index pairing as existing evaluator, then IoU threshold. Counts are segments, not videos. Label disagreement with benchmark is not independent proof of human emotion or visual hallucination.',
        videos=len(gt), gt_segments=sum(support.values()), predicted_segments=sum(predicted.values()),
        compared_pairs=len(pairs), gt_not_compared=sum(support.values()) - len(pairs),
        segmentation_counts=dict(counts),
        segment_count_matrix=[dict(gt_count=g, predicted_count=p, videos=n) for (g, p), n in sorted(matrix.items())],
        panels=panels,
        sources=[dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [gt_path] + paths])
    previous = json.loads((OUT / 'analysis.json').read_text())['full_test']['dar']
    assert len(pairs) == previous['compared_pairs'] == 3005
    assert panels['0.5']['qualified_pairs'] == previous['qualified_pairs'] == 1550
    assert panels['0.5']['correct'] == previous['joint_correct'] == 483
    assert counts['equal'] == 570 and sum(counts.values()) == 1441
    (OUT / 'full-error-breakdown.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = ['# DAR-R1: lỗi trên toàn bộ benchmark', '',
             'Phân tích output full-test có sẵn, bao gồm retry lịch sử; không chạy inference mới. '
             'So với nhãn benchmark, không phải audit hình ảnh của 1.441 video. '
             'Ghép đoạn cùng chỉ số rồi giữ IoU ≥ 0,5; số nhầm là số đoạn, không phải số video.', '',
             '1.441 video; 3.713 đoạn GT; 3.349 đoạn dự đoán. 3.005 cặp được so: '
             '1.550 cặp IoU ≥ 0,5, trong đó 483 đúng nhãn và 1.067 sai nhãn. '
             '1.455 cặp không đạt IoU; 708 đoạn GT không có cặp cùng chỉ số.', '',
             f'Chia thiếu: {counts["under"]}/1441; chia thừa: {counts["over"]}/1441; đúng số đoạn: {counts["equal"]}/1441.', '',
             '| GT → dự đoán | Số nhầm / số cặp đạt IoU của nhãn GT | ID ví dụ (IoU cao nhất) |',
             '|---|---:|---|']
    mistakes = [r for r in panels['0.5']['confusion'] if r['gt'] != r['pred']]
    for r in mistakes[:15]:
        ids = ', '.join(e['video_id'] for e in r['examples'])
        lines.append(f'| {r["gt"]} → {r["pred"]} | {r["count"]}/{r["qualified_gt"]} | {ids} |')
    lines += ['', '| Nhãn GT | Tổng đoạn GT | Dự đoán nhãn này | Đúng / cặp đạt IoU | Tỷ lệ đúng có điều kiện |', '|---|---:|---:|---:|---:|']
    for r in sorted(panels['0.5']['per_class'], key=lambda r: r['correct'] / max(r['qualified'], 1)):
        lines.append(f'| {r["emotion"]} | {r["gt_segments"]} | {r["predicted_segments"]} | {r["correct"]}/{r["qualified"]} | {100*r["correct"]/max(r["qualified"],1):.2f}% |')
    lines += ['', 'Kiểm tra IoU ≥ 0,8: 597 cặp, 193 đúng và 404 sai nhãn (67,67% sai). '
              'Lệch nhãn còn xuất hiện khi thời gian khớp chặt, nhưng đây vẫn là phép chấm theo index và nhãn benchmark.', '',
              'Các nhóm nhãn ít mẫu cần thận trọng: Craving chỉ có 5 cặp đạt IoU. '
              'Chưa thể kết luận model nhầm vật thể hoặc bịa sự kiện trên full benchmark từ confusion cảm xúc. '
              'Nguồn, hash, toàn bộ confusion, thời gian và ID ví dụ: [full-error-breakdown.json](full-error-breakdown.json).']
    (OUT / 'full-error-breakdown.vi.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(verified=True, videos=len(gt), counts=dict(counts),
                          qualified=1550, correct=483, wrong=1067), ensure_ascii=False))


if __name__ == '__main__':
    main()
