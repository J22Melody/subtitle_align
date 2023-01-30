# python main.py \
# --features_path '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/features/i3d_c2281_16f_m8_-15_4_d0.8_-3_22/' \
# --gt_sub_path '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/subtitles/manually-aligned/' \
# --pr_sub_path '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/subtitles/audio-aligned-heuristic-correction/' \
# --test_videos_txt 'data/bobsl_align_test_1.txt' \
# --gpu_id 0 \
# --n_workers 32 \
# --batch_size 1 \
# --pr_subs_delta_bias 2.7 \
# --fixed_feat_len 20 \
# --centre_window \
# --test_only \
# --save_vtt True \
# --save_probs True \
# --dtw_postpro True \
# --resume '/shares/volk.cl.uzh/zifjia/subtitle_align/inference_output/finetune_subtitles/checkpoints/model_0000264841.pt' \

python main.py \
--features_path '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/features/i3d_c2281_16f_m8_-15_4_d0.8_-3_22/' \
--gt_sub_path '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/subtitles/manually-aligned/' \
--pr_sub_path '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/subtitles/audio-aligned-heuristic-correction/' \
--test_videos_txt 'data/bobsl_align_test.txt' \
--gpu_id 0 \
--n_workers 32 \
--batch_size 1 \
--pr_subs_delta_bias 2.7 \
--fixed_feat_len 20 \
--centre_window \
--test_only \
--save_vtt True \
--save_probs True \
--dtw_postpro True \
--resume '/shares/volk.cl.uzh/zifjia/subtitle_align/checkpoints_subtitle_align/finetune_subtitles/checkpoints/model_0000264041.pt' \


# before DTW output
# Computed over 2642663 frames, 20338 sentences - Frame-level accuracy: 65.09 F1@0.10: 71.47 F1@0.25: 63.61 F1@0.50: 46.41
# after DTW output
# Computed over 2642663 frames, 20338 sentences - Frame-level accuracy: 70.89 F1@0.10: 74.08 F1@0.25: 66.78 F1@0.50: 53.22

### 2.7s shift baseline
# frame_accuracy: 62.40
# f1_10: 72.77
# f1_25: 64.08
# f1_50: 44.60

