python main.py \
--features_path '/users/zifan/BOBSL/derivatives/video_features/mediapipe' \
--videos_path '/users/zifan/BOBSL/derivatives/original_videos' \
--spottings_path '/users/zifan/BOBSL/derivatives/annotationsMDAPEN.pkl' \
--gpu_id 0 \
--feature_dim 249 \
--load_features_from_pose True \
--load_features_from_pose_reduce True \
--input_features_stride 2 \
--batch_size 1024 \
--n_workers 8 \
--pr_subs_delta_bias 0 \
--fixed_feat_len 20 \
--jitter_location \
--jitter_abs \
--jitter_loc_quantity 10. \
--load_words True \
--load_subtitles False \
--lr 1e-5 \
--centre_window \
--save_path '/scratch/shared/beegfs/zifan/checkpoints/subtitle_align_pose/word_pretrain_reduce/' \
--pos_weight 19. \
--n_epochs 100 \
--shuffle_getitem True \
--concatenate_prior True \
--train_videos_txt 'data/bobsl_train_1658.txt' \
--val_videos_txt 'data/bobsl_val_32.txt' \
--test_videos_txt 'data/bobsl_test_250.txt' \
# --train_videos_txt 'data/bobsl_align_test_1.txt' \
# --val_videos_txt 'data/bobsl_align_test_1.txt' \
# --test_videos_txt 'data/bobsl_align_test_1.txt' \