#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_solution(item: Dict[str, Any]) -> str:
    conversations = item.get("conversations", [])
    for turn in reversed(conversations):
        if turn.get("from") == "gpt":
            value = turn.get("value", "")
            parsed = json.loads(value)
            segments = parsed.get("segments") if isinstance(parsed, dict) else None
            if not isinstance(segments, list) or not segments:
                raise ValueError("solution must contain a non-empty segments list")
            for segment_index, segment in enumerate(segments):
                if not isinstance(segment, dict):
                    raise ValueError(f"solution segment {segment_index} must be an object")
                for key in ("start_time", "end_time", "emotion", "reason"):
                    if key not in segment:
                        raise ValueError(f"solution segment {segment_index} missing {key}")
            return value
    raise ValueError("missing gpt solution turn")


def resolve_video(item: Dict[str, Any], index: int, video_root: Optional[str]) -> str:
    video = item.get("video") or item.get("video_path")
    if not video:
        raise ValueError(f"item {index}: missing video path")
    if video_root:
        video = os.path.join(video_root, os.path.basename(video))
    return os.path.abspath(video)


def convert_item(
    item: Dict[str, Any],
    index: int,
    system_prompt: str,
    video_root: Optional[str] = None,
) -> Dict[str, Any]:
    video = resolve_video(item, index, video_root)
    solution = extract_solution(item)

    duration = item.get("video_duration")
    if duration is None:
        solution_obj = json.loads(solution)
        segments = solution_obj.get("segments") or []
        duration = segments[-1]["end_time"] if segments else "unknown"
    duration_text = f"{float(duration):.1f}s" if isinstance(duration, (int, float)) else f"{duration}"

    return {
        "id": item.get("id") or f"dar-grpo-{index:06d}",
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "<video>\n"
                    f"Analyze this SILENT video (duration: {duration_text}).\n"
                    "Segment by when the viewer's dominant emotion changes; "
                    "choose an allowed emotion for each segment and explain why "
                    "based on the visuals."
                ),
            },
        ],
        "videos": [video],
        "solution": solution,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/path/to/DAR/train.jsonl",
        help="DAR camera-ready SFT JSONL.",
    )
    parser.add_argument(
        "--output",
        default="/path/to/DAR/train_qwen25vl_ms_grpo.jsonl",
        help="Output JSONL consumed by ms-swift GRPO.",
    )
    parser.add_argument(
        "--prompt",
        default=str(Path(__file__).resolve().parents[1] / "prompt.txt"),
        help="System prompt text file.",
    )
    parser.add_argument(
        "--video-root",
        default=None,
        help="Directory holding the videos; when set, each path is rebuilt as "
        "<video-root>/<basename from the annotation>.",
    )
    parser.add_argument(
        "--check-videos",
        action="store_true",
        help="Fail or skip when a resolved video path does not exist.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort on the first invalid record instead of skipping it.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    prompt_path = Path(args.prompt)
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as f:
        for index, item in enumerate(read_jsonl(input_path)):
            if args.max_samples is not None and written >= args.max_samples:
                break
            try:
                converted = convert_item(item, index, system_prompt, args.video_root)
                if args.check_videos and not os.path.isfile(converted["videos"][0]):
                    raise ValueError(f"item {index}: video not found: {converted['videos'][0]}")
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                if args.strict:
                    raise
                skipped += 1
                print(f"skip: {exc}")
                continue
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} GRPO examples to {output_path} (skipped {skipped})")


if __name__ == "__main__":
    main()
