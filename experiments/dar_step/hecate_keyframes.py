"""Python/OpenCV port of Hecate's histogram/subshot keyframe selection.

Derived from Yahoo Hecate, Copyright 2016 Yahoo Inc., Apache-2.0.
Upstream: 63d1c91eea847ac45545520274b78fdad1df32e0
See HECATE_NOTICE.md and HECATE_LICENSE for attribution and modifications.
"""
from __future__ import annotations

import numpy as np

from common import require

HECATE_REVISION = '63d1c91eea847ac45545520274b78fdad1df32e0'


def histogram_features(bgr):
    """hist_opencv.hpp: two levels, x-major patches, HSV128 + edge8/8.

    Match source defaults (2000 dimensions), including separate L2 histogram
    normalization, HSV range [0,256), patch-local Scharr and magnitude clipping
    by histogram range. Input is the resized uint8 BGR frame.
    """
    import cv2
    require(bgr.dtype == np.uint8 and bgr.ndim == 3 and bgr.shape[2] == 3,
            'Expected uint8 BGR frame')
    height, width = bgr.shape[:2]
    require(min(height, width) >= 2, 'Frame too small for spatial pyramid')
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    color, edges = [], []

    def hist(a, bins, upper):
        values = cv2.calcHist([a], [0], None, [bins], [0, upper])
        return cv2.normalize(values, None).ravel()

    for side in (1, 2):
        w, h = width // side, height // side
        for x in range(side):
            for y in range(side):
                area = np.s_[y*h:(y+1)*h, x*w:(x+1)*w]
                hsv = cv2.cvtColor(bgr[area], cv2.COLOR_BGR2HSV)
                color.extend(hist(channel, 128, 256) for channel in cv2.split(hsv))
                patch = gray[area]
                gx = cv2.Scharr(patch, cv2.CV_32F, 1, 0)
                gy = cv2.Scharr(patch, cv2.CV_32F, 0, 1)
                # Compute angles in float64 like upstream atan2(double,double).
                angle = np.degrees(np.arctan2(gy.astype(np.float64), gx.astype(np.float64)))
                angle = np.mod(angle, 180).astype(np.float32)
                edges.extend((hist(angle, 8, 180), hist(cv2.magnitude(gx, gy), 8, 256)))
    return np.concatenate(color + edges).astype(np.float32)


def stillness_costs(pictures, bounds):
    """Two-neighbor L2 difference from video_parser.cpp, with safe endpoints.

    Float subtraction corrects upstream uint8 saturation. Compare only original
    adjacent frames within each PySceneDetect scene; endpoints use one neighbor.
    Lower cost means more still. No optical flow or semantic motion estimator.
    """
    costs = np.zeros(len(pictures), dtype=np.float64)
    for start, end in bounds:
        differences = []
        for i in range(start + 1, end):
            a, b = pictures[i - 1], pictures[i]
            differences.append(float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)))
                               / (a.shape[0] * a.shape[1]))
        if differences:
            costs[start], costs[end - 1] = differences[0], differences[-1]
            for i in range(start + 1, end - 1):
                costs[i] = (differences[i - start - 1] + differences[i - start]) / 2
    return costs


def cluster_labels(features, scene_count, seed=1234):
    """Hecate global k-means++: K=min(N//2, number of shots), guarded at 1."""
    import cv2
    x = np.ascontiguousarray(features, dtype=np.float32)
    require(x.ndim == 2 and len(x) > 0 and np.isfinite(x).all(), 'Invalid histogram features')
    require(scene_count >= 1, 'Need at least one scene')
    k = max(1, min(len(x) // 2, scene_count))
    # Avoid arbitrary label fragmentation when all/some feature rows coincide.
    k = min(k, len(np.unique(x, axis=0)))
    if k == 1:
        return np.zeros(len(x), dtype=np.int32), k
    cv2.setRNGSeed(int(seed))
    _, labels, _ = cv2.kmeans(x, k, None,
        (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 1000, .0001),
        1, cv2.KMEANS_PP_CENTERS)
    return labels.ravel(), k


def select_subshots(labels, costs, bounds):
    """One minimum-cost frame per contiguous label run, bounded by scenes.

    Half-open intervals fix Hecate's inclusion of the next run's first frame.
    A repeated cluster at a later time remains a distinct subshot.
    """
    labels, costs = np.asarray(labels), np.asarray(costs)
    require(len(labels) == len(costs) and np.isfinite(costs).all(), 'Invalid subshot inputs')
    result = []
    for scene, (start, end) in enumerate(bounds):
        require(0 <= start < end <= len(labels), 'Invalid scene bounds')
        first = start
        for stop in range(start + 1, end + 1):
            if stop == end or labels[stop] != labels[first]:
                pick = first + int(np.argmin(costs[first:stop]))
                result.append(dict(scene=scene, start=first, end=stop,
                                   cluster=int(labels[first]), index=pick,
                                   stillness_cost=float(costs[pick])))
                first = stop
    return result
