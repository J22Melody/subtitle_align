python main.py \
--features_path '/scratch/shared/beegfs/zifan/Youtube-SL-25-BSL/mediapipe' \
--videos_path '/scratch/shared/beegfs/zifan/Youtube-SL-25-BSL/videos' \
--gt_sub_path '/scratch/shared/beegfs/zifan/Youtube-SL-25-BSL/subtitles' \
--pr_sub_path '/scratch/shared/beegfs/zifan/Youtube-SL-25-BSL/subtitles' \
--gpu_id 0 \
--feature_dim 1024 \
--feature_dim_adapt 534 \
--load_features_from_pose True \
--batch_size 1024 \
--n_workers 8 \
--pr_subs_delta_bias 0 \
--gt_subs_delta_bias 0 \
--fixed_feat_len 20 \
--load_words False \
--load_subtitles True \
--lr 5e-6 \
--save_path '/scratch/shared/beegfs/zifan/checkpoints/subtitle_align_pose_youtube/train_coarse_subtitles_dumb' \
--n_epochs 100 \
--concatenate_prior True \
--min_sent_len_filter 0.5 \
--max_sent_len_filter 20 \
--shuffle_words_subs 0.5 \
--drop_words_subs 0.15 \
--resume '/scratch/shared/beegfs/zifan/checkpoints/subtitle_align/word_pretrain/checkpoints/model_0000191709.pt' \
--train_videos_txt 'data/youtube_bsl_train.txt' \
--val_videos_txt 'data/youtube_bsl_val.txt' \
# --train_videos_txt 'data/youtube_bsl_1.txt' \
# --val_videos_txt 'data/youtube_bsl_1.txt' \
# --debug
