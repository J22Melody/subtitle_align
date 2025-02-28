# audio_aligned + median + SAT (finetune)
# python main.py \
# --features_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/videos_i3d/' \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles/' \
# --pr_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_median_aligned/' \
# --gpu_id 0 \
# --batch_size 64 \
# --n_workers 32 \
# --pr_subs_delta_bias 0 \
# --fixed_feat_len 20 \
# --jitter_location \
# --jitter_abs \
# --jitter_loc_quantity 2 \
# --load_words False \
# --load_subtitles True \
# --lr 1e-6 \
# --save_path 'inference_output_srf/' \
# --train_videos_txt 'data/srf_align_train.txt' \
# --val_videos_txt 'data/srf_align_val.txt' \
# --test_videos_txt 'data/srf_align_test.txt' \
# --n_epochs 100 \
# --concatenate_prior True \
# --min_sent_len_filter 0.5 \
# --max_sent_len_filter 20 \
# --shuffle_words_subs 0.5 \
# --drop_words_subs 0.15 \
# --resume '/shares/volk.cl.uzh/zifjia/subtitle_align/checkpoints_subtitle_align/finetune_subtitles/checkpoints/model_0000264041.pt' \
# --bert_model 'bert-base-multilingual-cased' \
# --finetune_bert False \

# audio_aligned + median + SAT (finetune)
# jitter_loc_quantity 1
# python main.py \
# --features_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/videos_i3d/' \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles/' \
# --pr_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_median_aligned/' \
# --gpu_id 0 \
# --batch_size 64 \
# --n_workers 32 \
# --pr_subs_delta_bias 0 \
# --fixed_feat_len 20 \
# --jitter_location \
# --jitter_abs \
# --jitter_loc_quantity 1 \
# --load_words False \
# --load_subtitles True \
# --lr 1e-6 \
# --save_path 'inference_output_srf_1/' \
# --train_videos_txt 'data/srf_align_train.txt' \
# --val_videos_txt 'data/srf_align_val.txt' \
# --test_videos_txt 'data/srf_align_test.txt' \
# --n_epochs 100 \
# --concatenate_prior True \
# --min_sent_len_filter 0.5 \
# --max_sent_len_filter 20 \
# --shuffle_words_subs 0.5 \
# --drop_words_subs 0.15 \
# --resume '/shares/volk.cl.uzh/zifjia/subtitle_align/checkpoints_subtitle_align/finetune_subtitles/checkpoints/model_0000264041.pt' \
# --bert_model 'bert-base-multilingual-cased' \
# --finetune_bert False \

# audio_aligned + median + SAT (finetune)
# jitter_loc_quantity 0.5
# python main.py \
# --features_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/videos_i3d/' \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles/' \
# --pr_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_median_aligned/' \
# --gpu_id 0 \
# --batch_size 64 \
# --n_workers 32 \
# --pr_subs_delta_bias 0 \
# --fixed_feat_len 20 \
# --jitter_location \
# --jitter_abs \
# --jitter_loc_quantity 0.5 \
# --load_words False \
# --load_subtitles True \
# --lr 1e-6 \
# --save_path 'inference_output_srf_0.5/' \
# --train_videos_txt 'data/srf_align_train.txt' \
# --val_videos_txt 'data/srf_align_val.txt' \
# --test_videos_txt 'data/srf_align_test.txt' \
# --n_epochs 100 \
# --concatenate_prior True \
# --min_sent_len_filter 0.5 \
# --max_sent_len_filter 20 \
# --shuffle_words_subs 0.5 \
# --drop_words_subs 0.15 \
# --resume '/shares/volk.cl.uzh/zifjia/subtitle_align/checkpoints_subtitle_align/finetune_subtitles/checkpoints/model_0000264041.pt' \
# --bert_model 'bert-base-multilingual-cased' \
# --finetune_bert False \

# audio_aligned + median + SAT (finetune)
# jitter_loc_quantity 0.1
# python main.py \
# --features_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/videos_i3d/' \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles/' \
# --pr_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_median_aligned/' \
# --gpu_id 0 \
# --batch_size 64 \
# --n_workers 32 \
# --pr_subs_delta_bias 0 \
# --fixed_feat_len 20 \
# --jitter_location \
# --jitter_abs \
# --jitter_loc_quantity 0.1 \
# --load_words False \
# --load_subtitles True \
# --lr 1e-6 \
# --save_path 'inference_output_srf_0.1/' \
# --train_videos_txt 'data/srf_align_train.txt' \
# --val_videos_txt 'data/srf_align_val.txt' \
# --test_videos_txt 'data/srf_align_test.txt' \
# --n_epochs 100 \
# --concatenate_prior True \
# --min_sent_len_filter 0.5 \
# --max_sent_len_filter 20 \
# --shuffle_words_subs 0.5 \
# --drop_words_subs 0.15 \
# --resume '/shares/volk.cl.uzh/zifjia/subtitle_align/checkpoints_subtitle_align/finetune_subtitles/checkpoints/model_0000264041.pt' \
# --bert_model 'bert-base-multilingual-cased' \
# --finetune_bert False \

# audio_aligned + median + SAT (finetune)
# jitter_loc_quantity 0.5
# lr 1e-5
# python main.py \
# --features_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/videos_i3d/' \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles/' \
# --pr_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_median_aligned/' \
# --gpu_id 0 \
# --batch_size 64 \
# --n_workers 32 \
# --pr_subs_delta_bias 0 \
# --fixed_feat_len 20 \
# --jitter_location \
# --jitter_abs \
# --jitter_loc_quantity 0.5 \
# --load_words False \
# --load_subtitles True \
# --lr 1e-5 \
# --save_path 'inference_output_srf_lr_5/' \
# --train_videos_txt 'data/srf_align_train.txt' \
# --val_videos_txt 'data/srf_align_val.txt' \
# --test_videos_txt 'data/srf_align_test.txt' \
# --n_epochs 100 \
# --concatenate_prior True \
# --min_sent_len_filter 0.5 \
# --max_sent_len_filter 20 \
# --shuffle_words_subs 0.5 \
# --drop_words_subs 0.15 \
# --resume '/shares/volk.cl.uzh/zifjia/subtitle_align/checkpoints_subtitle_align/finetune_subtitles/checkpoints/model_0000264041.pt' \
# --bert_model 'bert-base-multilingual-cased' \
# --finetune_bert False \

# audio_aligned + median + SAT (finetune)
# jitter_loc_quantity 0.5
# lr 1e-5
# lr_reduce True
# python main.py \
# --features_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/videos_i3d/' \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles/' \
# --pr_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_median_aligned/' \
# --gpu_id 0 \
# --batch_size 64 \
# --n_workers 32 \
# --pr_subs_delta_bias 0 \
# --fixed_feat_len 20 \
# --jitter_location \
# --jitter_abs \
# --jitter_loc_quantity 0.5 \
# --load_words False \
# --load_subtitles True \
# --lr 1e-5 \
# --lr_reduce True \
# --save_path 'inference_output_srf_lr_5_reduce/' \
# --train_videos_txt 'data/srf_align_train.txt' \
# --val_videos_txt 'data/srf_align_val.txt' \
# --test_videos_txt 'data/srf_align_test.txt' \
# --n_epochs 100 \
# --concatenate_prior True \
# --min_sent_len_filter 0.5 \
# --max_sent_len_filter 20 \
# --shuffle_words_subs 0.5 \
# --drop_words_subs 0.15 \
# --resume '/shares/volk.cl.uzh/zifjia/subtitle_align/checkpoints_subtitle_align/finetune_subtitles/checkpoints/model_0000264041.pt' \
# --bert_model 'bert-base-multilingual-cased' \
# --finetune_bert False \

# audio_aligned + median + SAT (finetune + segmentation)
# python main.py \
# --features_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/videos_i3d/' \
# --segmentation_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/videos_256/' \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles/' \
# --pr_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_median_aligned/' \
# --gpu_id 0 \
# --batch_size 64 \
# --n_workers 32 \
# --pr_subs_delta_bias 0 \
# --fixed_feat_len 20 \
# --jitter_location \
# --jitter_abs \
# --jitter_loc_quantity 2. \
# --load_words False \
# --load_subtitles True \
# --load_segmentation True \
# --lr 1e-6 \
# --save_path 'inference_output_srf/' \
# --train_videos_txt 'data/srf_align_train.txt' \
# --val_videos_txt 'data/srf_align_val.txt' \
# --test_videos_txt 'data/srf_align_test.txt' \
# --n_epochs 100 \
# --concatenate_prior True \
# --concatenate_segmentation True \
# --min_sent_len_filter 0.5 \
# --max_sent_len_filter 20 \
# --shuffle_words_subs 0.5 \
# --drop_words_subs 0.15 \
# --resume '/shares/volk.cl.uzh/zifjia/subtitle_align/checkpoints_subtitle_align/finetune_subtitles/checkpoints/model_0000264041.pt' \
# --bert_model 'bert-base-multilingual-cased' \
# --finetune_bert False \
