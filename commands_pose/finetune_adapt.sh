python main.py \
--features_path '/users/zifan/BOBSL/derivatives/video_features/mediapipe' \
--gt_sub_path '/users/zifan/BOBSL/v1.4/manual_annotations/signing_aligned_subtitles' \
--pr_sub_path '/users/zifan/BOBSL/v1.4/automatic_annotations/signing_aligned_subtitles/audio_aligned_heuristic_correction' \
--gpu_id 0 \
--feature_dim 1024 \
--feature_dim_adapt 534 \
--load_features_from_pose True \
--input_features_stride 4 \
--batch_size 256 \
--n_workers 8 \
--pr_subs_delta_bias 2.7 \
--fixed_feat_len 20 \
--jitter_location \
--jitter_abs \
--jitter_loc_quantity 2. \
--load_words False \
--load_subtitles True \
--lr 1e-6 \
--save_path '/scratch/shared/beegfs/zifan/checkpoints/subtitle_align_pose/finetune_subtitles_adapt/' \
--n_epochs 100 \
--concatenate_prior True \
--min_sent_len_filter 0.5 \
--max_sent_len_filter 20 \
--shuffle_words_subs 0.5 \
--drop_words_subs 0.15 \
--resume '/scratch/shared/beegfs/zifan/checkpoints/subtitle_align/train_coarse_subtitles/checkpoints/model_0000250341.pt' \
# --train_videos_txt 'data/bobsl_align_train.txt' \
# --val_videos_txt 'data/bobsl_align_test.txt' \
# --test_videos_txt 'data/bobsl_test_254.txt' \
