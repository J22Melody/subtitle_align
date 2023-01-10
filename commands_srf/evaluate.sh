python misc/evaluate_sub_alignment.py \
--gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles' \
--pred_path_root '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_audio_aligned' \
--test_videos_txt 'data/srf_align_test_1.txt' \

python misc/evaluate_sub_alignment.py \
--gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles' \
--pred_path_root '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_median_aligned' \
--test_videos_txt 'data/srf_align_test_1.txt' \

python misc/evaluate_sub_alignment.py \
--gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles' \
--pred_path_root '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_audio_aligned' \
--test_videos_txt 'data/srf_align_test.txt' \

python misc/evaluate_sub_alignment.py \
--gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles' \
--pred_path_root '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_median_aligned' \
--test_videos_txt 'data/srf_align_test.txt' \

# 1it [00:00,  7.55it/s]
# total  39656 subs 252
# Computed over 39656 frames, 252 sentences - Frame-level accuracy: 48.37 F1@0.10: 71.23 F1@0.25: 59.96 F1@0.50: 31.79
# 1it [00:00,  7.64it/s]
# total  39656 subs 252
# Computed over 39656 frames, 252 sentences - Frame-level accuracy: 63.70 F1@0.10: 81.53 F1@0.25: 76.71 F1@0.50: 60.24
# 29it [00:03,  7.72it/s]
# total  1160059 subs 7056
# Computed over 1160059 frames, 7056 sentences - Frame-level accuracy: 59.69 F1@0.10: 84.16 F1@0.25: 74.31 F1@0.50: 47.96
# 29it [00:03,  7.71it/s]
# total  1160059 subs 7056
# Computed over 1160059 frames, 7056 sentences - Frame-level accuracy: 73.24 F1@0.10: 92.14 F1@0.25: 88.10 F1@0.50: 74.05
