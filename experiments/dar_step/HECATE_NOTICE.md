# Hecate-derived keyframe selector

`hecate_keyframes.py` is a Python/OpenCV adaptation of Yahoo Hecate:
Copyright 2016 Yahoo Inc.; original developer Yale Song.
Licensed under Apache License 2.0; full text in `HECATE_LICENSE`.

Pinned upstream revision: `63d1c91eea847ac45545520274b78fdad1df32e0`.

Source mapping:

- `histogram_features`: [hist_opencv.hpp](https://github.com/yahoo/hecate/blob/63d1c91eea847ac45545520274b78fdad1df32e0/include/hecate/hist_opencv.hpp),
  `calc_pyr_color_hist`, `calc_pyr_edge_hist`; defaults from `video_parser.hpp`.
- `stillness_costs`: [video_parser.cpp](https://github.com/yahoo/hecate/blob/63d1c91eea847ac45545520274b78fdad1df32e0/src/hecate/video_parser.cpp), `filter_transition`.
- `cluster_labels`, `select_subshots`: same file, `filter_redundant_and_obtain_subshots`;
  k-means settings from `include/hecate/gapstat.hpp::perform_kmeans`.
- Local original-source snapshot: `research/thumbnail-hecate-20260929/hecate/`.

Modifications made 2026-09-29:

- Port C++ to Python/OpenCV. Keep source histogram defaults: 128 bins for each
  HSV channel, 8 each for edge orientation/magnitude, five pyramid regions,
  2000 dimensions (paper describes 2220). Analyze all decoded frames at max
  side 160px; save selected images at original resolution.
- PySceneDetect supplies scene boundaries. Retain every frame as a candidate
  within these scenes. Hecate's quality/transition rejection, long-shot
  post-processing and final thumbnail ranking are outside this replacement.
- Global k-means uses K=min(N//2, number of scenes), at least one; cap to unique
  descriptors and fix the OpenCV seed. No SigLIP clustering, fixed per-scene K,
  endpoint injection, gap-statistic search, or final thumbnail ranking.
- Use float subtraction for two-neighbor stillness rather than saturating uint8
  subtraction. Use one neighbor at scene endpoints (zero only for a singleton),
  and never compare across a cut.
- Use half-open contiguous-label runs: fix the original loop's inclusion of the
  next label's first frame. Ties choose the earliest frame within the subshot.
- Preserve one selected frame per subshot. Reject overflow of the video budget;
  do not silently discard subshots/scenes. A single-scene video has K=1 and may
  have just one selected frame; it then has no within-scene motion pair.
- VILA parses every selected frame and merges all adjacent selected frames.
  Scene descriptions use at most four evenly spaced selected frames to respect
  the native teacher limit. Their IDs are logged in intermediate.json and calls.

This replaces only keyframe selection in the STEP-inspired pipeline. It is not
a complete replication of Hecate or the paper's thumbnail system. CPU tests do
not establish graph quality or bit-for-bit equivalence to the C++ binary.
