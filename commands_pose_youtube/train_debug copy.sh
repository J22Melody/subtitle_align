python main.py \
--features_path '/scratch/shared/beegfs/zifan/Youtube-SL-25-BSL/mediapipe' \
--videos_path '/scratch/shared/beegfs/zifan/Youtube-SL-25-BSL/videos' \
--gt_sub_path '/scratch/shared/beegfs/zifan/Youtube-SL-25-BSL/subtitles' \
--pr_sub_path '/scratch/shared/beegfs/zifan/Youtube-SL-25-BSL/subtitles' \
--gpu_id 0 \
--feature_dim 255 \
--load_features_from_pose True \
--load_features_from_pose_reduce True \
--input_features_stride 2 \
--batch_size 256 \
--n_workers 8 \
--lr 1e-6 \
--d_model 512 \
--n_dec_layers 6 \
--n_heads 8 \
--transformer_dropout 0.1 \
--pr_subs_delta_bias 0 \
--gt_subs_delta_bias 0 \
--fixed_feat_len 20 \
--load_words False \
--load_subtitles True \
--save_path '/scratch/shared/beegfs/zifan/checkpoints/subtitle_align_pose_youtube/train_coarse_subtitles_debug' \
--n_epochs 100 \
--concatenate_prior True \
--min_sent_len_filter 0.5 \
--max_sent_len_filter 20 \
--shuffle_words_subs 0.5 \
--drop_words_subs 0.15 \
--train_videos_txt 'data/youtube_bsl_1.txt' \
--val_videos_txt 'data/youtube_bsl_1.txt' \
--resume '/scratch/shared/beegfs/zifan/checkpoints/subtitle_align/word_pretrain/checkpoints/model_0000191709.pt' \
# --debug
# --train_videos_txt 'data/youtube_bsl_train.txt' \
# --val_videos_txt 'data/youtube_bsl_val.txt' \
# --jitter_location \
# --jitter_abs \
# --jitter_loc_quantity 3.5 \
# --jitter_width_secs 1.5 \
