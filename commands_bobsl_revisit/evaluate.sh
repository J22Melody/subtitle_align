# python misc/evaluate_sub_alignment.py \
# --gt_sub_path '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/subtitles/manually-aligned/' \
# --pred_path_root '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/subtitles/audio-aligned-heuristic-correction/' \
# --test_videos_txt 'data/bobsl_align_train.txt' \

# python misc/evaluate_sub_alignment.py \
# --gt_sub_path '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/subtitles/manually-aligned/' \
# --pred_path_root '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/subtitles/audio-aligned-heuristic-correction/' \
# --test_videos_txt 'data/bobsl_align_test.txt' \

# 16it [00:11,  1.35it/s]
# total  1173319 subs 9168
# Mean and median start offset: 2.72 / 2.58
# Mean and median end offset: 2.96 / 2.77
# Computed over 1173319 frames, 9168 sentences - Frame-level accuracy: 46.13 F1@0.10: 49.58 F1@0.25: 35.20 F1@0.50: 14.00

# 35it [00:23,  1.50it/s]
# total  2642663 subs 20338
# Mean and median start offset: 2.91 / 2.91
# Mean and median end offset: 3.37 / 3.36
# Computed over 2642663 frames, 20338 sentences - Frame-level accuracy: 40.15 F1@0.10: 46.38 F1@0.25: 33.47 F1@0.50: 14.11

python misc/evaluate_sub_alignment.py \
--gt_sub_path '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/subtitles/manually-aligned/' \
--pred_path_root '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/subtitles/audio-aligned-heuristic-correction/' \
--test_videos_txt 'data/bobsl_align_test.txt' \
--pr_subs_delta_bias_start 2.7 \
--pr_subs_delta_bias_end 2.7 \

python misc/evaluate_sub_alignment.py \
--gt_sub_path '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/subtitles/manually-aligned/' \
--pred_path_root '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/subtitles/audio-aligned-heuristic-correction/' \
--test_videos_txt 'data/bobsl_align_test.txt' \
--pr_subs_delta_bias_start 2.72 \
--pr_subs_delta_bias_end 2.96 \

python misc/evaluate_sub_alignment.py \
--gt_sub_path '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/subtitles/manually-aligned/' \
--pred_path_root '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/subtitles/audio-aligned-heuristic-correction/' \
--test_videos_txt 'data/bobsl_align_test.txt' \
--pr_subs_delta_bias_start 2.58 \
--pr_subs_delta_bias_end 2.77 \

# 35it [00:20,  1.74it/s]
# total  2642663 subs 20338
# Mean and median start offset: 0.21 / 0.21
# Mean and median end offset: 0.67 / 0.66
# Computed over 2642663 frames, 20338 sentences - Frame-level accuracy: 62.41 F1@0.10: 72.78 F1@0.25: 64.10 F1@0.50: 44.61

# 35it [00:19,  1.76it/s]
# total  2642663 subs 20338
# Mean and median start offset: 0.19 / 0.19
# Mean and median end offset: 0.41 / 0.40
# Computed over 2642663 frames, 20338 sentences - Frame-level accuracy: 62.96 F1@0.10: 73.26 F1@0.25: 64.48 F1@0.50: 45.38

# 35it [00:19,  1.78it/s]
# total  2642663 subs 20338
# Mean and median start offset: 0.33 / 0.33
# Mean and median end offset: 0.60 / 0.59
# Computed over 2642663 frames, 20338 sentences - Frame-level accuracy: 62.61 F1@0.10: 72.96 F1@0.25: 64.34 F1@0.50: 44.88