#### their checkpoint ####

# python main.py \
# --features_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/videos_i3d/' \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles/' \
# --pr_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_median_aligned/' \
# --test_videos_txt 'data/srf_align_test.txt' \
# --gpu_id 0 \
# --n_workers 32 \
# --batch_size 1 \
# --pr_subs_delta_bias 0 \
# --fixed_feat_len 20 \
# --centre_window \
# --test_only \
# --save_vtt True \
# --save_probs True \
# --dtw_postpro True \
# --save_path 'inference_output_srf/' \
# --save_probs_folder 'inference_output_srf/probabilities' \
# --save_subs_folder 'inference_output_srf/subtitles' \
# --save_postpro_subs_folder 'inference_output_srf/subtitles_postprocessing' \
# --resume '/shares/volk.cl.uzh/zifjia/subtitle_align/checkpoints_subtitle_align/finetune_subtitles/checkpoints/model_0000264041.pt' \

#### our checkpoint ####

# python main.py \
# --features_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/videos_i3d/' \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles/' \
# --pr_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_median_aligned/' \
# --test_videos_txt 'data/srf_align_test.txt' \
# --gpu_id 0 \
# --n_workers 32 \
# --batch_size 1 \
# --pr_subs_delta_bias 0 \
# --fixed_feat_len 20 \
# --centre_window \
# --test_only \
# --save_vtt True \
# --save_probs True \
# --dtw_postpro True \
# --save_path 'inference_output_srf/' \
# --resume '/shares/volk.cl.uzh/zifjia/subtitle_align/inference_output_srf/checkpoints/model_0000269097.pt' \
# --bert_model 'bert-base-multilingual-cased' \

# python main.py \
# --features_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/videos_i3d/' \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles/' \
# --pr_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_median_aligned/' \
# --test_videos_txt 'data/srf_align_test.txt' \
# --gpu_id 0 \
# --n_workers 32 \
# --batch_size 1 \
# --pr_subs_delta_bias 0 \
# --fixed_feat_len 20 \
# --centre_window \
# --test_only \
# --save_vtt True \
# --save_probs True \
# --dtw_postpro True \
# --save_path 'inference_output_srf_1/' \
# --resume '/shares/volk.cl.uzh/zifjia/subtitle_align/inference_output_srf_1/checkpoints/model_0000269097.pt' \
# --bert_model 'bert-base-multilingual-cased' \

# python main.py \
# --features_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/videos_i3d/' \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles/' \
# --pr_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_median_aligned/' \
# --test_videos_txt 'data/srf_align_test.txt' \
# --gpu_id 0 \
# --n_workers 32 \
# --batch_size 1 \
# --pr_subs_delta_bias 0 \
# --fixed_feat_len 20 \
# --centre_window \
# --test_only \
# --save_vtt True \
# --save_probs True \
# --dtw_postpro True \
# --save_path 'inference_output_srf_0.5/' \
# --resume '/shares/volk.cl.uzh/zifjia/subtitle_align/inference_output_srf_0.5/checkpoints/model_0000269097.pt' \
# --bert_model 'bert-base-multilingual-cased' \

# python main.py \
# --features_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/videos_i3d/' \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles/' \
# --pr_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_median_aligned/' \
# --test_videos_txt 'data/srf_align_test.txt' \
# --gpu_id 0 \
# --n_workers 32 \
# --batch_size 1 \
# --pr_subs_delta_bias 0 \
# --fixed_feat_len 20 \
# --centre_window \
# --test_only \
# --save_vtt True \
# --save_probs True \
# --dtw_postpro True \
# --save_path 'inference_output_srf_0.1/' \
# --resume '/shares/volk.cl.uzh/zifjia/subtitle_align/inference_output_srf_0.1/checkpoints/model_0000269097.pt' \
# --bert_model 'bert-base-multilingual-cased' \

# python main.py \
# --features_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/videos_i3d/' \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles/' \
# --pr_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_median_aligned/' \
# --test_videos_txt 'data/srf_align_test.txt' \
# --gpu_id 0 \
# --n_workers 32 \
# --batch_size 1 \
# --pr_subs_delta_bias 0 \
# --fixed_feat_len 20 \
# --centre_window \
# --test_only \
# --save_vtt True \
# --save_probs True \
# --dtw_postpro True \
# --save_path 'inference_output_srf_lr_5/' \
# --resume '/shares/volk.cl.uzh/zifjia/subtitle_align/inference_output_srf_lr_5/checkpoints/model_0000269097.pt' \
# --bert_model 'bert-base-multilingual-cased' \

# python main.py \
# --features_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/videos_i3d/' \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles/' \
# --pr_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_median_aligned/' \
# --test_videos_txt 'data/srf_align_test.txt' \
# --gpu_id 0 \
# --n_workers 32 \
# --batch_size 1 \
# --pr_subs_delta_bias 0 \
# --fixed_feat_len 20 \
# --centre_window \
# --test_only \
# --save_vtt True \
# --save_probs True \
# --dtw_postpro True \
# --save_path 'inference_output_srf_lr_5_reduce/' \
# --resume '/shares/volk.cl.uzh/zifjia/subtitle_align/inference_output_srf_lr_5_reduce/checkpoints/model_0000269097.pt' \
# --bert_model 'bert-base-multilingual-cased' \

# python main.py \
# --features_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/videos_i3d/' \
# --segmentation_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/videos_256/' \
# --gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles/' \
# --pr_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_median_aligned/' \
# --test_videos_txt 'data/srf_align_test.txt' \
# --load_segmentation True \
# --concatenate_segmentation True \
# --gpu_id 0 \
# --n_workers 32 \
# --batch_size 1 \
# --pr_subs_delta_bias 0 \
# --fixed_feat_len 20 \
# --centre_window \
# --test_only \
# --save_vtt True \
# --save_probs True \
# --dtw_postpro True \
# --save_path 'inference_output_srf/' \
# --save_probs_folder 'inference_output_srf/probabilities' \
# --save_subs_folder 'inference_output_srf/subtitles' \
# --save_postpro_subs_folder 'inference_output_srf/subtitles_postprocessing' \
# --resume '/shares/volk.cl.uzh/zifjia/subtitle_align/inference_output_srf/checkpoints/model_0000271941.pt' \
# --bert_model 'bert-base-multilingual-cased' \
