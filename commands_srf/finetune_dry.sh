python main.py \
--features_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/videos_i3d/' \
--gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles/' \
--pr_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_median_aligned/' \
--gpu_id 0 \
--batch_size 4 \
--n_workers 32 \
--pr_subs_delta_bias 0 \
--fixed_feat_len 20 \
--load_words False \
--load_subtitles True \
--lr 1e-5 \
--save_path 'inference_output_srf_dry/finetune_subtitles' \
--train_videos_txt 'data/srf_align_test_1.txt' \
--val_videos_txt 'data/srf_align_test_1.txt' \
--test_videos_txt 'data/srf_align_test_1.txt' \
--n_epochs 100 \
--concatenate_prior True \
--min_sent_len_filter 0.5 \
--max_sent_len_filter 20 \
--shuffle_words_subs 0.5 \
--drop_words_subs 0.15 \
--resume '/shares/volk.cl.uzh/zifjia/subtitle_align/checkpoints_subtitle_align/finetune_subtitles/checkpoints/model_0000264041.pt' \
--bert_model 'bert-base-multilingual-cased' \

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
# --jitter_loc_quantity 2 \
# --load_words False \
# --load_subtitles True \
# --load_segmentation True \
# --lr 1e-5 \
# --lr_reduce True \
# --save_path 'inference_output_srf_dry/' \
# --train_videos_txt 'data/srf_align_test_1.txt' \
# --val_videos_txt 'data/srf_align_test_1.txt' \
# --test_videos_txt 'data/srf_align_test_1.txt' \
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