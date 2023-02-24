# original
python main.py \
--features_path '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/features/i3d_c2281_16f_m8_-15_4_d0.8_-3_22/' \
--gt_sub_path '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/subtitles/manually-aligned/' \
--pr_sub_path '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/subtitles/audio-aligned-heuristic-correction/' \
--gpu_id 0 \
--batch_size 64 \
--n_workers 32 \
--pr_subs_delta_bias 2.7 \
--fixed_feat_len 20 \
--jitter_location \
--jitter_abs \
--jitter_loc_quantity 2. \
--load_words False \
--load_subtitles True \
--lr 1e-6 \
--save_path 'inference_output/' \
--train_videos_txt 'data/bobsl_align_train.txt' \
--val_videos_txt 'data/bobsl_align_val.txt' \
--test_videos_txt 'data/bobsl_align_test.txt' \
--n_epochs 100 \
--concatenate_prior True \
--min_sent_len_filter 0.5 \
--max_sent_len_filter 20 \
--shuffle_words_subs 0.5 \
--drop_words_subs 0.15 \
--resume 'checkpoints_subtitle_align/train_coarse_subtitles/checkpoints/model_0000250341.pt' \

# seg
# python main.py \
# --features_path '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/features/i3d_c2281_16f_m8_-15_4_d0.8_-3_22/' \
# --segmentation_path '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/videos/' \
# --gt_sub_path '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/subtitles/manually-aligned/' \
# --pr_sub_path '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/subtitles/audio-aligned-heuristic-correction/' \
# --gpu_id 0 \
# --batch_size 64 \
# --n_workers 32 \
# --pr_subs_delta_bias 2.7 \
# --fixed_feat_len 20 \
# --jitter_location \
# --jitter_abs \
# --jitter_loc_quantity 2. \
# --load_words False \
# --load_subtitles True \
# --load_segmentation True \
# --lr 1e-6 \
# --save_path 'inference_output_seg/' \
# --train_videos_txt 'data/bobsl_align_train.txt' \
# --val_videos_txt 'data/bobsl_align_val.txt' \
# --test_videos_txt 'data/bobsl_align_test.txt' \
# --n_epochs 100 \
# --concatenate_prior True \
# --concatenate_segmentation True \
# --min_sent_len_filter 0.5 \
# --max_sent_len_filter 20 \
# --shuffle_words_subs 0.5 \
# --drop_words_subs 0.15 \
# --resume 'checkpoints_subtitle_align/train_coarse_subtitles/checkpoints/model_0000250341.pt' \

# jitter gt
# python main.py \
# --features_path '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/features/i3d_c2281_16f_m8_-15_4_d0.8_-3_22/' \
# --gt_sub_path '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/subtitles/manually-aligned/' \
# --pr_sub_path '/shares/volk.cl.uzh/zifjia/bobsl/bobsl/subtitles/audio-aligned-heuristic-correction/' \
# --gpu_id 0 \
# --batch_size 64 \
# --n_workers 32 \
# --pr_subs_delta_bias 2.7 \
# --fixed_feat_len 20 \
# --jitter_location \
# --jitter_abs \
# --jitter_loc_quantity 2. \
# --jitter_towards_gt \
# --load_words False \
# --load_subtitles True \
# --lr 1e-6 \
# --save_path 'inference_output_jitter_gt/' \
# --train_videos_txt 'data/bobsl_align_train.txt' \
# --val_videos_txt 'data/bobsl_align_val.txt' \
# --test_videos_txt 'data/bobsl_align_test.txt' \
# --n_epochs 100 \
# --concatenate_prior True \
# --min_sent_len_filter 0.5 \
# --max_sent_len_filter 20 \
# --shuffle_words_subs 0.5 \
# --drop_words_subs 0.15 \
# --resume 'checkpoints_subtitle_align/train_coarse_subtitles/checkpoints/model_0000250341.pt' \
