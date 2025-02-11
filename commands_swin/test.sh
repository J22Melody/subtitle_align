python main.py \
--features_path '/users/zifan/BOBSL/derivatives/video_features/swin_v2/lmdb-feats_vswin_t-bs256_float16' \
--videos_path '/users/zifan/BOBSL/derivatives/original_videos' \
--gt_sub_path '/users/zifan/BOBSL/v1.4/manual_annotations/signing_aligned_subtitles' \
--pr_sub_path '/users/zifan/BOBSL/v1.4/automatic_annotations/signing_aligned_subtitles/audio_aligned_heuristic_correction' \
--gpu_id 0 \
--feature_dim 768 \
--load_features_from_lmdb True \
--n_workers 8 \
--batch_size 1 \
--pr_subs_delta_bias 2.7 \
--fixed_feat_len 20 \
--centre_window \
--test_only \
--save_vtt True \
--save_probs True \
--dtw_postpro True \
--save_path '/scratch/shared/beegfs/zifan/checkpoints/subtitle_align_swin/finetune_subtitles/' \
--resume '/scratch/shared/beegfs/zifan/checkpoints/subtitle_align_swin/finetune_subtitles/checkpoints/model_0000122814.pt' \


# Computed over 2642663 frames, 20338 sentences - Frame-level accuracy: 70.89 F1@0.10: 74.08 F1@0.25: 66.78 F1@0.50: 53.22

### 2.7s shift baseline
# frame_accuracy: 62.40
# f1_10: 72.77
# f1_25: 64.08
# f1_50: 44.60

