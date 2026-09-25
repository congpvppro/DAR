#!/usr/bin/env python3
"""Apply the paper's 0--5 LLM-as-a-judge rubric to the audited 64-video set.

The visual verdicts come from the preceding source-video grounding audit.  This
script makes the conversion policy explicit and reproducible, then exports the
segment- and video-level tables used by the Vietnamese report.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "research" / "original-grounding-audit-20260909"
OUT = Path(__file__).resolve().parent

# Explicit temporal findings that were recorded in prose in the visual audit.
# Values are 1-based predicted-segment indices and replacement TC scores.
TC_OVERRIDES = {
    "21885": {3: 2},       # final scene exists, but predicted onset is too early
    "60501": {2: 2, 3: 2},# reaction transitions are placed before the spray event
    "00386": {3: 0},       # entire predicted segment starts after source-video end
    "08852": {2: 2, 3: 2, 4: 2},  # fast montage is grouped at wrong times
    "06896": {1: 2, 2: 2},# predicted boundary 6.5 s, visual cut is about 8 s
}

# A few claims are peripheral errors inside otherwise well-grounded rationales.
# They should not be treated like a wholly invented core event.
PERIPHERAL_CLAIMS = {
    ("54850", "green bike frame"),
    ("07039", "The static camera and unchanging background"),
    ("21885", "a silver Ferrari"),
    ("51648", "body made of frozen berries"),
}


def overlap(a: list[float], b: list[float]) -> bool:
    return min(a[1], b[1]) - max(a[0], b[0]) > 1e-6


def clip(x: int) -> int:
    return max(0, min(5, x))


def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def score_segment(video_id: str, index: int, seg: dict, ann: dict, duration: float) -> dict:
    # A score of 4 is "mostly correct and largely grounded" in the paper.  The
    # verbose outputs nearly always contain some appraisal/speculation, so 4 is
    # the neutral supported baseline rather than 5.
    dims = {
        "visual_grounding": 4,
        "causal_logic": 4,
        "viewer_centricity": 4,
        "temporal_consistency": 4,
        "answer_consistency": 4,
    }
    evidence = []
    affected = [
        c for c in ann.get("claims", [])
        if overlap(c["claimed_interval"], [seg["start_time"], seg["end_time"]])
    ]

    for claim in affected:
        key = (video_id, claim["quote"])
        kind, judgment = claim["claim_type"], claim["judgment"]
        evidence.append({
            "quote": claim["quote"],
            "claim_type": kind,
            "judgment": judgment,
            "observation_vi": claim["observation_vi"],
        })
        if judgment == "ambiguous":
            dims["visual_grounding"] = min(dims["visual_grounding"], 3)
            dims["causal_logic"] = min(dims["causal_logic"], 3)
            dims["temporal_consistency"] = min(dims["temporal_consistency"], 3)
        elif judgment == "temporal_only":
            dims["temporal_consistency"] = min(dims["temporal_consistency"], 2)
        elif judgment == "contradicted":
            if key in PERIPHERAL_CLAIMS:
                dims["visual_grounding"] = min(dims["visual_grounding"], 3)
                dims["causal_logic"] = min(dims["causal_logic"], 3)
            elif kind == "invented_event":
                dims["visual_grounding"] = min(dims["visual_grounding"], 1)
                dims["causal_logic"] = min(dims["causal_logic"], 1)
                dims["temporal_consistency"] = min(dims["temporal_consistency"], 2)
            elif kind in {"wrong_object", "event_order"}:
                dims["visual_grounding"] = min(dims["visual_grounding"], 1)
                dims["causal_logic"] = min(dims["causal_logic"], 2)
                dims["temporal_consistency"] = min(dims["temporal_consistency"], 3)
            elif kind == "wrong_attribute":
                dims["visual_grounding"] = min(dims["visual_grounding"], 2)
                dims["causal_logic"] = min(dims["causal_logic"], 3)

    # Multiple independent contradictions make grounding worse still.
    contradicted = [c for c in affected if c["judgment"] == "contradicted"]
    if len(contradicted) >= 2:
        dims["visual_grounding"] = clip(dims["visual_grounding"] - 1)
        dims["causal_logic"] = clip(dims["causal_logic"] - 1)

    if video_id in TC_OVERRIDES and index in TC_OVERRIDES[video_id]:
        dims["temporal_consistency"] = min(
            dims["temporal_consistency"], TC_OVERRIDES[video_id][index]
        )
    if seg["start_time"] >= duration - 1e-6:
        dims["temporal_consistency"] = 0
    elif seg["end_time"] > duration + 0.1:
        dims["temporal_consistency"] = min(dims["temporal_consistency"], 1)

    # All released-prompt outputs explicitly formulate the response as an
    # induced viewer reaction, and their label is linguistically consistent
    # with the stated rationale.  Unsupported events are assessed under VG/CL,
    # not double-counted as an internal label--reason contradiction.
    mean_dims = sum(dims.values()) / 5
    overall = round_half_up(mean_dims)

    weak = min(dims, key=dims.get)
    if evidence:
        comment = (
            f"Yếu nhất: {weak}; bằng chứng audit: "
            + "; ".join(f"{e['judgment']} {e['claim_type']} — {e['quote']}" for e in evidence)
        )
    elif dims["temporal_consistency"] < 4:
        comment = "Yếu nhất: temporal_consistency; ranh giới dự đoán lệch sự kiện/cắt cảnh nguồn."
    else:
        comment = "Phần giải thích chủ yếu bám nội dung thấy được; vẫn có diễn giải cảm xúc mang tính chủ quan."

    return {
        "video_id": video_id,
        "segment_index": index,
        "start_time": seg["start_time"],
        "end_time": seg["end_time"],
        "emotion": seg["emotion"],
        "reason": seg["reason"],
        "score": overall,
        **dims,
        "brief_comment_vi": comment,
        "evidence": evidence,
    }


def mean(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows)


def main() -> None:
    records = json.loads((AUDIT / "records.json").read_text())
    annotations = {
        x["video_id"]: x
        for x in json.loads((AUDIT / "annotations.json").read_text())
    }
    rows = []
    for record in records:
        ann = annotations[record["video_id"]]
        for index, segment in enumerate(record["segments"], 1):
            rows.append(score_segment(
                record["video_id"], index, segment, ann, record["video_duration"]
            ))

    dim_keys = [
        "visual_grounding", "causal_logic", "viewer_centricity",
        "temporal_consistency", "answer_consistency",
    ]
    video_rows = []
    for record in records:
        subset = [r for r in rows if r["video_id"] == record["video_id"]]
        item = {
            "video_id": record["video_id"],
            "segments": len(subset),
            "audit_status": annotations[record["video_id"]]["status"],
            "score": mean(subset, "score"),
        }
        item.update({key: mean(subset, key) for key in dim_keys})
        item["weakest_dimensions"] = [
            key for key in dim_keys if item[key] == min(item[k] for k in dim_keys)
        ]
        item["summary_vi"] = annotations[record["video_id"]]["summary_vi"]
        video_rows.append(item)

    summary = {
        "scope": {
            "videos": len(records),
            "predicted_segments": len(rows),
            "condition": "original video",
            "prompt_kind": "released/original",
            "judge_protocol": "Paper Appendix: LLM-as-a-Judge Prompt, integer 0--5",
        },
        "segment_macro": {"score": mean(rows, "score")},
        "video_macro": {"score": mean(video_rows, "score")},
        "score_distribution_segments": dict(sorted(Counter(r["score"] for r in rows).items())),
        "audit_status_video_means": {},
    }
    summary["segment_macro"].update({key: mean(rows, key) for key in dim_keys})
    summary["video_macro"].update({key: mean(video_rows, key) for key in dim_keys})
    for status in ["confirmed", "ambiguous", "no_clear_error"]:
        subset = [v for v in video_rows if v["audit_status"] == status]
        summary["audit_status_video_means"][status] = {
            "n": len(subset),
            "score": mean(subset, "score"),
            **{key: mean(subset, key) for key in dim_keys},
        }

    (OUT / "segment-scores.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    )
    (OUT / "video-scores.json").write_text(
        json.dumps(video_rows, ensure_ascii=False, indent=2) + "\n"
    )
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    fields = [
        "video_id", "segment_index", "start_time", "end_time", "emotion", "score",
        *dim_keys, "brief_comment_vi",
    ]
    with (OUT / "segment-scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    labels = {
        "visual_grounding": "VG", "causal_logic": "CL",
        "viewer_centricity": "VC", "temporal_consistency": "TC",
        "answer_consistency": "AC",
    }
    lines = [
        "# Chấm GPT-Score cho 64 video DAR-R1 (original + released prompt)",
        "",
        "## Kết quả chính",
        "",
        (
            f"Đã chấm **{len(rows)} segment thuộc {len(records)}/64 video**. "
            f"GPT-Score trung bình theo segment là **{summary['segment_macro']['score']:.2f}/5**; "
            f"nếu mỗi video có trọng số bằng nhau, điểm là "
            f"**{summary['video_macro']['score']:.2f}/5**."
        ),
        "",
        "| Cách gộp | VG | CL | VC | TC | AC | Overall |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            "| 151 segment (macro) | "
            + " | ".join(f"{summary['segment_macro'][k]:.2f}" for k in dim_keys)
            + f" | {summary['segment_macro']['score']:.2f} |"
        ),
        (
            "| 64 video (macro) | "
            + " | ".join(f"{summary['video_macro'][k]:.2f}" for k in dim_keys)
            + f" | {summary['video_macro']['score']:.2f} |"
        ),
        "",
        "Phân bố điểm tổng hợp ở cấp segment: "
        + ", ".join(f"{k}/5: {v}" for k, v in summary["score_distribution_segments"].items())
        + ".",
        "",
        "## Kết luận theo từng tiêu chí",
        "",
        f"- **Visual Grounding — {summary['segment_macro']['visual_grounding']:.2f}/5 (yếu nhất):** 42/151 segment có VG ≤2. Lỗi nổi bật là bịa sự kiện, đổi vật thể hoặc thuộc tính; audit nguồn xác nhận 38/64 video có ít nhất một mâu thuẫn hình ảnh rõ.",
        "",
        f"- **Causal Logic — {summary['segment_macro']['causal_logic']:.2f}/5:** 39/151 segment có CL ≤2. Câu văn thường tạo được chuỗi `sự kiện → thẩm định → cảm xúc`, nhưng khi sự kiện tiền đề bị bịa thì lập luận trôi chảy vẫn không hợp lệ.",
        "",
        f"- **Temporal Consistency — {summary['segment_macro']['temporal_consistency']:.2f}/5:** 41/151 segment có TC ≤2. Có hiện tượng đặt chuyển cảm xúc quá sớm, gộp sai cảnh montage, và một segment bắt đầu hoàn toàn sau khi video đã kết thúc.",
        "",
        "- **Viewer Centricity — 4,00/5:** toàn bộ output tuân thủ hình thức hướng tới cảm xúc người xem. Điểm này mạnh về cấu trúc, nhưng không chứng minh cảm xúc dự đoán đúng với phân bố người xem thực tế.",
        "",
        "- **Answer Consistency — 4,00/5:** nhãn cảm xúc và lời giải thích hầu như nhất quán nội tại. Đây cũng là điểm mù: một nhãn có thể rất khớp với một câu chuyện do model bịa, nên AC cao không bù được VG/CL thấp.",
        "",
        "Điểm yếu cốt lõi là **hallucination có tính kể chuyện**: model dùng một chi tiết nhìn nhầm để dựng cả diễn biến và chuyển cảm xúc. Vì VC và AC vẫn cao, điểm Overall có thể che khuất lỗi grounding nghiêm trọng; do đó nên luôn báo 5 chiều riêng, số segment VG≤2 và danh sách lỗi nguồn, không chỉ báo Avg.",
        "",
        "## Quy trình và phạm vi",
        "",
        "Áp đúng năm chiều và thang nguyên 0–5 trong Appendix *LLM-as-a-Judge Prompt* của bài báo. Đơn vị chấm là từng predicted segment. Mỗi mục được đối chiếu với video gốc qua 16 frame thực sự đưa vào model, contact sheet nguồn 2 fps + frame cuối, và frame dày/cận cảnh ở các đoạn nghi ngờ từ audit ngày 2026-09-09. Claim chưa đủ chứng cứ được giữ ở mức 3 thay vì kết luận sai.",
        "",
        "Điểm `score` tổng hợp được làm tròn half-up từ trung bình năm chiều vì bài báo yêu cầu cả sáu trường nhưng không công bố quy tắc suy ra trường `score`. Baseline 4 tương ứng “mostly correct and largely grounded”; 5 chỉ dành cho câu hoàn toàn đúng, nên không mặc định dùng 5 cho output dài có suy diễn chủ quan.",
        "",
        "Đây là **lần chấm GPT-style độc lập**, không phải tái tạo số chính thức bằng GPT-4o: bài báo không công bố snapshot GPT-4o, temperature/seed, cách cấp preceding context đầy đủ, hay quy tắc aggregate. Mẫu 64 video cũng là mẫu natural ngắn đã chọn cho thí nghiệm trước, không đại diện ngẫu nhiên toàn test set. Không dùng điểm này để tuyên bố chênh lệch có ý nghĩa so với bảng chính của bài báo.",
        "",
        "## Điểm từng video",
        "",
        "| ID | Seg | Audit | VG | CL | VC | TC | AC | Overall |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in video_rows:
        lines.append(
            f"| {item['video_id']} | {item['segments']} | {item['audit_status']} | "
            + " | ".join(f"{item[k]:.2f}" for k in dim_keys)
            + f" | {item['score']:.2f} |"
        )
    lines += [
        "",
        "## Tài liệu và khả năng kiểm tra lại",
        "",
        "- Bài báo: Zhiyan Zhang, Peipei Song, Jinpeng Hu, Jingyang Jia, Xun Yang, Xiaojun Chang (2026), *Benchmarking Dynamic Affective Reasoning: A Viewer-Centric Video Emotion Dataset*, arXiv:2607.10238v1, Appendix “LLM-as-a-Judge Prompt”.",
        "- Công cụ truy xuất học thuật: Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, arXiv:2609.00065v2.",
        "- `segment-scores.csv`: bảng 151 segment gọn để phân tích.",
        "- `segment-scores.json`: điểm, reasoning đầy đủ, comment và claim bằng chứng.",
        "- `video-scores.json`: bảng gộp 64 video.",
        "- `summary.json`: số tổng hợp dùng trong báo cáo.",
        "",
    ]
    (OUT / "report.vi.md").write_text("\n".join(lines))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
