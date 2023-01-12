python main.py \
--features_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/videos_i3d/' \
--gt_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles/' \
--pr_sub_path '/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/subtitles_median_aligned/' \
--test_videos_txt 'data/srf_align_test.txt' \
--gpu_id 0 \
--n_workers 32 \
--batch_size 1 \
--pr_subs_delta_bias 0 \
--fixed_feat_len 20 \
--centre_window \
--test_only \
--save_vtt True \
--save_probs True \
--dtw_postpro True \
--save_path 'inference_output_srf/' \
--save_probs_folder 'inference_output_srf/probabilities' \
--save_subs_folder 'inference_output_srf/subtitles' \
--save_postpro_subs_folder 'inference_output_srf/subtitles_postprocessing' \
--resume '/shares/volk.cl.uzh/zifjia/subtitle_align/checkpoints_subtitle_align/finetune_subtitles/checkpoints/model_0000264041.pt' \

# before DTW output
# Computed over 1160059 frames, 7056 sentences - Frame-level accuracy: 66.92 F1@0.10: 87.41 F1@0.25: 80.91 F1@0.50: 63.15
