python misc/evaluate_sub_alignment.py \
--gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles' \
--pred_path_root '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_audio_aligned' \
--test_videos_txt 'data/srf_align_test.txt' \

python misc/evaluate_sub_alignment.py \
--gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles' \
--pred_path_root '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_mean_aligned' \
--test_videos_txt 'data/srf_align_test.txt' \

python misc/evaluate_sub_alignment.py \
--gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles' \
--pred_path_root '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_median_aligned' \
--test_videos_txt 'data/srf_align_test.txt' \

# python misc/evaluate_sub_alignment.py \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles' \
# --pred_path_root './inference_output_srf/subtitles' \
# --test_videos_txt 'data/srf_align_test.txt' \

# python misc/evaluate_sub_alignment.py \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles' \
# --pred_path_root './inference_output_srf/subtitles_postprocessing' \
# --test_videos_txt 'data/srf_align_test.txt' \

# data: all
# on audio/median aligned 

# total  1160059 subs 7056
# Computed over 1160059 frames, 7056 sentences - Frame-level accuracy: 59.69 F1@0.10: 84.16 F1@0.25: 74.31 F1@0.50: 47.96

# total  1160059 subs 7056
# Computed over 1160059 frames, 7056 sentences - Frame-level accuracy: 73.24 F1@0.10: 92.14 F1@0.25: 88.10 F1@0.50: 74.05

# on predictions from the bobsl checkpoint

# before DTW output
# total  1160059 subs 7056
# Computed over 1160059 frames, 7056 sentences - Frame-level accuracy: 66.92 F1@0.10: 87.41 F1@0.25: 80.91 F1@0.50: 63.15

# after DTW output
# total  1160059 subs 7056
# Computed over 1160059 frames, 7056 sentences - Frame-level accuracy: 70.55 F1@0.10: 86.71 F1@0.25: 81.19 F1@0.50: 66.40

# on predictions from the bobsl checkpoint (bert-multilingual)

# before DTW output
# total  1160059 subs 7056
# Computed over 1160059 frames, 7056 sentences - Frame-level accuracy: 68.58 F1@0.10: 89.71 F1@0.25: 82.49 F1@0.50: 60.63

# after DTW output
# total  1160059 subs 7056
# Computed over 1160059 frames, 7056 sentences - Frame-level accuracy: 72.92 F1@0.10: 89.55 F1@0.25: 84.33 F1@0.50: 69.39

# data: test
# on audio/median aligned 

# total  147952 subs 863
# Computed over 147952 frames, 863 sentences - Frame-level accuracy: 64.06 F1@0.10: 84.81 F1@0.25: 75.70 F1@0.50: 46.85
# total  147952 subs 863
# Computed over 147952 frames, 863 sentences - Frame-level accuracy: 76.94 F1@0.10: 93.47 F1@0.25: 88.58 F1@0.50: 72.38
# total  147952 subs 863
# Computed over 147952 frames, 863 sentences - Frame-level accuracy: 77.66 F1@0.10: 93.36 F1@0.25: 89.74 F1@0.50: 74.83

# on predictions from the bobsl checkpoint (zero-shot)

# before DTW output
# total  147952 subs 863
# Computed over 147952 frames, 863 sentences - Frame-level accuracy: 73.55 F1@0.10: 90.29 F1@0.25: 84.68 F1@0.50: 61.17

# after DTW output
# total  147952 subs 863
# Computed over 147952 frames, 863 sentences - Frame-level accuracy: 77.50 F1@0.10: 89.92 F1@0.25: 85.17 F1@0.50: 69.52

# on predictions from the bobsl checkpoint (fine-tuned) 

# before DTW output
# total  147952 subs 863
# Computed over 147952 frames, 863 sentences - Frame-level accuracy: 77.69 F1@0.10: 92.53 F1@0.25: 89.15 F1@0.50: 73.98

# after DTW output
# total  147952 subs 863
# Computed over 147952 frames, 863 sentences - Frame-level accuracy: 78.75 F1@0.10: 91.66 F1@0.25: 87.37 F1@0.50: 74.39
