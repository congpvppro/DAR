"""PySceneDetect scenes and source-derived Hecate keyframe selection."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from common import file_hash, require


def detect_scenes(path, duration, threshold=27.0, min_scene_seconds=.5):
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    video = open_video(str(path), backend='opencv')
    fps = float(video.frame_rate)
    require(fps > 0 and math.isfinite(fps), 'Invalid video fps')
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold,
                                        min_scene_len=max(1, round(min_scene_seconds * fps))))
    decoded = manager.detect_scenes(video=video, show_progress=False)
    require(decoded > 0, 'Video contains no decodable frames')
    scenes = manager.get_scene_list(start_in_scene=True)
    bounds = [(a.get_frames(), b.get_frames()) for a, b in scenes] or [(0, decoded)]
    require(bounds[0][0] == 0 and bounds[-1][1] == decoded, 'Incomplete scene coverage')
    # Frame-index timestamps assume CFR. Do not stretch the video to annotation length.
    measured = decoded / fps
    require(abs(measured - duration) <= max(.25, 2 / fps),
            f'Video/annotation duration mismatch: {measured:.3f} vs {duration}')
    return fps, bounds


def select_frames(row, out, *, threshold=27., min_scene_seconds=.5,
                  max_frames=24, seed=1234):
    """All frames -> Hecate histograms -> global k-means -> still subshots."""
    import cv2
    from PIL import Image
    from hecate_keyframes import (HECATE_REVISION, histogram_features,
                                  stillness_costs, cluster_labels, select_subshots)
    duration = row['video_duration']
    fps, bounds = detect_scenes(row['video_path'], duration, threshold, min_scene_seconds)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(row['video_path']))
    pictures = []
    try:
        for index in range(bounds[-1][1]):
            ok, bgr = capture.read()
            require(ok, f'Cannot decode frame {index}')
            actual = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            require(abs(actual - index / fps) < max(.1, 2 / fps),
                    'Nonuniform/unsupported timestamps; normalize video to CFR before this pilot')
            height, width = bgr.shape[:2]
            scale = min(1., 160 / max(height, width))
            if scale < 1:
                bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            pictures.append(bgr)
        features = np.stack([histogram_features(p) for p in pictures])
        labels, k = cluster_labels(features, len(bounds), seed)
        costs = stillness_costs(pictures, bounds)
        subshots = select_subshots(labels, costs, bounds)
        require(len(subshots) <= max_frames,
                f'Keyframe budget {max_frames} exceeded ({len(subshots)} subshots); increase budget and context together')
        scenes = [dict(id=f'c{i}', start=round(a / fps, 4), end=round(min(duration, b / fps), 4))
                  for i, (a, b) in enumerate(bounds)]
        frames = []
        # Save original-resolution images, not the 160px analysis images.
        for subshot in subshots:
            index = subshot['index']
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = capture.read()
            require(ok, f'Cannot decode selected frame {index}')
            path = out / f'f{index:06d}.png'
            Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(path)
            frames.append(dict(id=f'f{len(frames)}', scene=f"c{subshot['scene']}", index=index,
                               time=round(index / fps, 4), path=str(path.resolve()), sha256=file_hash(path)))
    finally:
        capture.release()
    return dict(scenes=scenes, frames=frames, fps=fps, video_sha256=file_hash(row['video_path']),
                selection=dict(method='hecate_histogram_subshot_stillness', revision=HECATE_REVISION,
                               feature_dim=2000, candidate_count=len(pictures), clusters=k, seed=seed,
                               subshots=subshots))
