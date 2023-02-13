# python misc/evaluate_sub_alignment.py \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles' \
# --pred_path_root '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_audio_aligned' \
# --test_videos_txt 'data/srf_align_train.txt' \

# python misc/evaluate_sub_alignment.py \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles' \
# --pred_path_root '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_audio_aligned' \
# --test_videos_txt 'data/srf_align_test.txt' \

# 21it [00:06,  3.27it/s]
# total  836056 subs 5175
# Mean and median start offset: 0.65 / 1.06
# Mean and median end offset: 1.73 / 1.75
# Computed over 836056 frames, 5175 sentences - Frame-level accuracy: 59.15 F1@0.10: 84.32 F1@0.25: 74.23 F1@0.50: 47.83

# 4it [00:00,  4.42it/s]
# total  147952 subs 863
# Mean and median start offset: 0.72 / 1.06
# Mean and median end offset: 1.45 / 1.75
# Computed over 147952 frames, 863 sentences - Frame-level accuracy: 64.06 F1@0.10: 84.81 F1@0.25: 75.70 F1@0.50: 46.85

# python misc/evaluate_sub_alignment.py \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles' \
# --pred_path_root '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_mean_aligned' \
# --test_videos_txt 'data/srf_align_test.txt' \

# python misc/evaluate_sub_alignment.py \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles' \
# --pred_path_root '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_median_aligned' \
# --test_videos_txt 'data/srf_align_test.txt' \

# mean
# python misc/evaluate_sub_alignment.py \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles' \
# --pred_path_root '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_audio_aligned' \
# --test_videos_txt 'data/srf_align_test.txt' \
# --pr_subs_delta_bias_start 0.65 \
# --pr_subs_delta_bias_end 1.73 \

# median
# python misc/evaluate_sub_alignment.py \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles' \
# --pred_path_root '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_audio_aligned' \
# --test_videos_txt 'data/srf_align_test.txt' \
# --pr_subs_delta_bias_start 1.06 \
# --pr_subs_delta_bias_end 1.75 \

# 4it [00:00,  4.68it/s]
# total  147952 subs 863
# Mean and median start offset: 0.07 / 0.41
# Mean and median end offset: -0.28 / 0.02
# Computed over 147952 frames, 863 sentences - Frame-level accuracy: 76.75 F1@0.10: 93.13 F1@0.25: 87.66 F1@0.50: 71.25

# 4it [00:00,  4.78it/s]
# total  147952 subs 863
# Mean and median start offset: -0.34 / 0.00
# Mean and median end offset: -0.30 / -0.00
# Computed over 147952 frames, 863 sentences - Frame-level accuracy: 77.66 F1@0.10: 93.36 F1@0.25: 89.74 F1@0.50: 74.83

# python misc/evaluate_sub_alignment.py \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles' \
# --pred_path_root './inference_output_srf/subtitles' \
# --test_videos_txt 'data/srf_align_test.txt' \

# python misc/evaluate_sub_alignment.py \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles' \
# --pred_path_root './inference_output_srf/subtitles_postprocessing' \
# --test_videos_txt 'data/srf_align_test.txt' \

# on predictions from the bobsl checkpoint (zero-shot)

# before DTW output
# total  147952 subs 863
# Computed over 147952 frames, 863 sentences - Frame-level accuracy: 73.55 F1@0.10: 90.29 F1@0.25: 84.68 F1@0.50: 61.17

# after DTW output
# total  147952 subs 863
# Computed over 147952 frames, 863 sentences - Frame-level accuracy: 77.50 F1@0.10: 89.92 F1@0.25: 85.17 F1@0.50: 69.52

# on predictions from the bobsl checkpoint (bert-multilingual)

# before DTW output
# total  147952 subs 863
# Mean and median start offset: -0.43 / -0.04 
# Mean and median end offset: -0.10 / 0.14 
# Computed over 147952 frames, 863 sentences - Frame-level accuracy: 72.53 F1@0.10: 89.42 F1@0.25: 82.29 F1@0.50: 66.86

# after DTW output
# total  147952 subs 863
# Mean and median start offset: -0.40 / 0.00 
# Mean and median end offset: -0.20 / 0.12 
# Computed over 147952 frames, 863 sentences - Frame-level accuracy: 75.88 F1@0.10: 87.72 F1@0.25: 82.73 F1@0.50: 69.76

# on predictions from the bobsl checkpoint (bert-multilingual + fine-tuned) 

# before DTW output
# total  147952 subs 863
# Computed over 147952 frames, 863 sentences - Frame-level accuracy: 77.69 F1@0.10: 92.53 F1@0.25: 89.15 F1@0.50: 73.98

# after DTW output
# total  147952 subs 863
# Computed over 147952 frames, 863 sentences - Frame-level accuracy: 78.75 F1@0.10: 91.66 F1@0.25: 87.37 F1@0.50: 74.39

# on predictions from the bobsl checkpoint (bert-multilingual fine-tuned + fine-tuned) 

# before DTW output
# total  147952 subs 863
# Mean and median start offset: -0.41 / 0.00 
# Mean and median end offset: -0.36 / -0.07 
# Computed over 147952 frames, 863 sentences - Frame-level accuracy: 76.74 F1@0.10: 92.43 F1@0.25: 87.89 F1@0.50: 72.88

# after DTW output
# total  147952 subs 863
# Mean and median start offset: -0.45 / 0.00 
# Mean and median end offset: -0.43 / -0.08 
# Computed over 147952 frames, 863 sentences - Frame-level accuracy: 78.64 F1@0.10: 90.50 F1@0.25: 86.79 F1@0.50: 75.20

# on predictions from the bobsl checkpoint (bert-multilingual fine-tuned + fine-tuned + segmentation) 

# before DTW output
# total  147952 subs 863
# Mean and median start offset: -0.29 / 0.06 
# Mean and median end offset: -0.48 / -0.13 
# Computed over 147952 frames, 863 sentences - Frame-level accuracy: 77.68 F1@0.10: 92.48 F1@0.25: 89.45 F1@0.50: 74.64

# after DTW output
# total  147952 subs 863
# Mean and median start offset: -0.38 / 0.04 
# Mean and median end offset: -0.45 / -0.04 
# Computed over 147952 frames, 863 sentences - Frame-level accuracy: 78.67 F1@0.10: 92.47 F1@0.25: 87.95 F1@0.50: 74.97
