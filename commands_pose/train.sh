python main.py \
--features_path '/users/zifan/BOBSL/derivatives/video_features/mediapipe' \
--videos_path '/users/zifan/BOBSL/derivatives/original_videos' \
--gt_sub_path '/users/zifan/BOBSL/v1.4/automatic_annotations/signing_aligned_subtitles/audio_aligned_heuristic_correction' \
--pr_sub_path '/users/zifan/BOBSL/v1.4/automatic_annotations/signing_aligned_subtitles/audio_aligned_heuristic_correction' \
--gpu_id 0 \
--feature_dim 534 \
--load_features_from_pose True \
--input_features_stride 2 \
--batch_size 512 \
--n_workers 8 \
--pr_subs_delta_bias 2.7 \
--gt_subs_delta_bias 2.7 \
--fixed_feat_len 20 \
--jitter_location \
--jitter_abs \
--jitter_loc_quantity 3. \
--load_words False \
--load_subtitles True \
--lr 5e-6 \
--save_path '/scratch/shared/beegfs/zifan/checkpoints/subtitle_align_pose/train_coarse_subtitles/' \
--n_epochs 100 \
--concatenate_prior True \
--min_sent_len_filter 0.5 \
--max_sent_len_filter 20 \
--shuffle_words_subs 0.5 \
--drop_words_subs 0.15 \
--train_videos_txt 'data/bobsl_train_1658.txt' \
--val_videos_txt 'data/bobsl_val_32.txt' \
--test_videos_txt 'data/bobsl_test_250.txt' \
--resume '/scratch/shared/beegfs/zifan/checkpoints/subtitle_align_pose/train_coarse_subtitles/checkpoints/model_0000078994.pt' \
# --resume '/scratch/shared/beegfs/zifan/checkpoints/subtitle_align_pose/word_pretrain/checkpoints/model_0000075328.pt' \
# --train_videos_txt 'data/bobsl_val_32.txt' \
# --val_videos_txt 'data/bobsl_val_32.txt' \
# --test_videos_txt 'data/bobsl_val_32.txt' \
