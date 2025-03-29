#!/usr/bin/env python3
import os
import sys
import random
import argparse
import numpy as np
import re
import csv
import glob
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import subprocess
from itertools import product

# Import subtitle utility functions and ELAN helpers from utils.py
from utils import (
    shift_cues,
    get_subtitle_cues,
    reconstruct_vtt,
    get_sign_segments_from_eaf,
    write_updated_eaf,
    extract_f1_score,
)

# Add the ../misc directory to sys.path to import the evaluation function.
current_dir = os.path.dirname(os.path.abspath(__file__))
misc_dir = os.path.join(current_dir, "../misc")
if misc_dir not in sys.path:
    sys.path.append(misc_dir)
from evaluate_sub_alignment import eval_subtitle_alignment

from align_dp import dp_align_subtitles_to_signs

def get_cslr_signs(video_id, cslr_dir):
    """
    Recursively search for a CSV file named <video_id>.csv under cslr_dir
    and return sign annotations.
    """
    pattern = os.path.join(cslr_dir, '**', f"{video_id}.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return []
    csv_file = files[0]
    signs = []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            approx_gloss = row.get("approx gloss sequence", "")
            english_sentence = row.get("english sentence", "").strip()
            matches = re.findall(r'(\S+)\[(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\]', approx_gloss)
            for word, start_str, end_str in matches:
                start = float(start_str)
                end = float(end_str)
                mid = (start + end) / 2
                signs.append({'start': start, 'end': end, 'mid': mid, 'text': word, 'subtitle': english_sentence})
    return sorted(signs, key=lambda s: s['start'])

def merge_signs(elan_signs, cslr_signs):
    """
    Merge CSLR signs into the base ELAN sign list.
    For each CSLR sign, remove overlapping ELAN segments and insert the CSLR sign.
    Even if there is no overlap, the CSLR sign is still appended.
    """
    elan_signs = sorted(elan_signs, key=lambda s: s['start'])
    cslr_signs = sorted(cslr_signs, key=lambda s: s['start'])
    merged = elan_signs[:]
    
    for cs in cslr_signs:
        new_merged = []
        for seg in merged:
            # Remove ELAN segments that overlap with the current CSLR sign
            if seg['end'] > cs['start'] and seg['start'] < cs['end']:
                continue
            new_merged.append(seg)
        # Always append the current CSLR sign
        new_merged.append(cs)
        merged = sorted(new_merged, key=lambda s: s['start'])
        
    return merged

# New function: Filter cues to those that overlap with any of the CSLR signs.
def filter_cues_by_cslr(cues, cslr_signs):
    """
    Return a filtered list of cues that have temporal overlap with any of the CSLR signs.
    A cue overlaps if its end time is greater than a sign's start time and its start time is less than the sign's end time.
    """
    filtered_cues = []
    for cue in cues:
        for sign in cslr_signs:
            if cue['end'] > sign['start'] and cue['start'] < sign['end']:
                filtered_cues.append(cue)
                break
    return filtered_cues

def process_video(video_id, args, dp_duration_penalty_weight, dp_gap_penalty_weight,
                  dp_window_size, dp_max_gap, similarity_weight, output_dir, save_elan,
                  seg_model, seg_sign_b, seg_sign_o,
                  pr_subs_delta_bias_start, pr_subs_delta_bias_end,
                  post_subs_delta_bias_start, post_subs_delta_bias_end):
    print(f"Processing video: {video_id}")
    # Derive a cleaned segmentation model name: remove "model_" prefix and ".pth" suffix.
    seg_model_name = seg_model
    if seg_model_name.startswith("model_"):
        seg_model_name = seg_model_name[len("model_"):]
    if seg_model_name.endswith(".pth"):
        seg_model_name = seg_model_name[:-4]
    # Build the subdirectory path.
    seg_subdir = os.path.join(args.segmentation_dir, f"{seg_model_name}_{seg_sign_b}_{seg_sign_o}")
    segmentation_file = os.path.join(seg_subdir, f"{video_id}.eaf")
    
    if os.path.exists(segmentation_file):
        elan_signs = get_sign_segments_from_eaf(segmentation_file)  
    else:
        raise FileNotFoundError(f"Segmentation {segmentation_file} does not exist!")
    signs = elan_signs

    # Initialize cslr_signs to ensure it's defined.
    cslr_signs = []
    if args.cslr:
        cslr_signs = get_cslr_signs(video_id, args.cslr_dir)
        if cslr_signs:
            signs = merge_signs(elan_signs, cslr_signs)
    
    subtitle_file = os.path.join(args.pr_sub_path, f"{video_id}.vtt")
    if not os.path.exists(subtitle_file):
        print(f"Subtitle file for video {video_id} not found. Skipping.")
        return
    try:
        with open(subtitle_file, "r", encoding="utf-8") as fin:
            vtt_content = fin.read()
    except Exception:
        return
    header_lines, cues = get_subtitle_cues(vtt_content)
    if not cues:
        return

    # Apply pre-alignment bias on cues.
    cues = shift_cues(cues, pr_subs_delta_bias_start, pr_subs_delta_bias_end)

    gt_subtitle_file = os.path.join(args.gt_sub_path, f"{video_id}.vtt")
    gt_cues = []
    if os.path.exists(gt_subtitle_file):
        try:
            with open(gt_subtitle_file, "r", encoding="utf-8") as fgt:
                gt_vtt_content = fgt.read()
            _, gt_cues = get_subtitle_cues(gt_vtt_content)
        except Exception:
            pass

    subtitle_embedding = None
    segmentation_embedding = None
    if args.similarity_measure == "sign_clip_embedding":
        subtitle_emb_file = os.path.join(args.subtitle_embedding_dir, f"{video_id}.npy")
        segmentation_emb_file = os.path.join(args.segmentation_embedding_dir, f"{video_id}.npy")
        if os.path.exists(subtitle_emb_file) and os.path.exists(segmentation_emb_file):
            subtitle_embedding = np.load(subtitle_emb_file)
            segmentation_embedding = np.load(segmentation_emb_file)
        else:
            print(f"Embedding files for video {video_id} not found. Skipping video.")
            return

    if args.debug:
        debug_sec = 30
        cues_ = [cue for cue in cues if cue['start'] < debug_sec]
        gt_cues_ = [cue for cue in gt_cues if cue['start'] < debug_sec]
        signs_ = [seg for seg in signs if seg['start'] < debug_sec]
    else:
        cues_, gt_cues_, signs_ = cues, gt_cues, signs

    dp_align_subtitles_to_signs(cues_, signs_, gt_cues=gt_cues_,
       duration_penalty_weight=dp_duration_penalty_weight,
       gap_penalty_weight=dp_gap_penalty_weight,
       window_size=dp_window_size,
       max_gap=dp_max_gap,
       similarity_weight=similarity_weight,
       similarity_measure=args.similarity_measure,
       subtitle_embedding=subtitle_embedding,
       segmentation_embedding=segmentation_embedding,
       visualize_similarity=args.visualize_similarity)
    
    # Apply post-alignment bias on the cues.
    cues = shift_cues(cues, post_subs_delta_bias_start, post_subs_delta_bias_end)
    
    # If --cslr_partial_eval is set, filter cues to only those overlapping with CSLR signs.
    if args.cslr_partial_eval and cslr_signs:
        cues = filter_cues_by_cslr(cues, cslr_signs)
    
    updated_vtt = reconstruct_vtt(header_lines, cues)
    output_vtt = os.path.join(output_dir, f"{video_id}.vtt")
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(output_vtt, "w", encoding="utf-8") as fout:
            fout.write(updated_vtt)
    except Exception:
        pass

    if save_elan and os.path.exists(segmentation_file):
        write_updated_eaf(segmentation_file, cues, video_id, signs)

def process_all_videos(video_ids, args, dp_dpw, dp_gpw, dp_ws, dp_mg, similarity_weight, output_dir, save_elan,
                       seg_model, seg_sign_b, seg_sign_o,
                       pr_subs_start, pr_subs_end,
                       post_subs_start, post_subs_end):
    # If live_segmentation is set, run segmentation.py as a subprocess.
    if args.live_segmentation:
        seg_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "segmentation.py")
        cmd = f"python {seg_script} --video_ids {args.video_ids} --save_dir {args.segmentation_dir} --num_workers {args.num_workers}"
        if args.overwrite:
            cmd += " --overwrite"
        print("Running live segmentation subprocess...")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print("Error in live segmentation subprocess, aborting alignment.")
            return

    func = partial(process_video, args=args,
                   dp_duration_penalty_weight=dp_dpw,
                   dp_gap_penalty_weight=dp_gpw,
                   dp_window_size=dp_ws,
                   dp_max_gap=dp_mg,
                   similarity_weight=similarity_weight,
                   output_dir=output_dir,
                   save_elan=save_elan,
                   seg_model=seg_model,
                   seg_sign_b=seg_sign_b,
                   seg_sign_o=seg_sign_o,
                   pr_subs_delta_bias_start=pr_subs_start,
                   pr_subs_delta_bias_end=pr_subs_end,
                   post_subs_delta_bias_start=post_subs_start,
                   post_subs_delta_bias_end=post_subs_end)
    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(func, video_ids), total=len(video_ids), desc="Processing videos"):
                pass
    else:
        for vid in tqdm(video_ids, desc="Processing videos"):
            func(vid)

def main():
    parser = argparse.ArgumentParser(
        description="Shift subtitle timings, align cues to SIGN segments, merge CSLR signs (if set), and write an updated ELAN file."
    )
    parser.add_argument("--video_ids", type=str,
                        default="/users/zifan/subtitle_align/data/bobsl_align_test.txt",
                        help="Path to text file with video ids (one per line).")
    parser.add_argument("--pr_sub_path", type=str,
                        default="/users/zifan/BOBSL/v1.4/automatic_annotations/signing_aligned_subtitles/audio_aligned_heuristic_correction",
                        help="Directory where subtitle (VTT) files are stored.")
    parser.add_argument("--gt_sub_path", type=str,
                        default="/users/zifan/BOBSL/v1.4/manual_annotations/signing_aligned_subtitles",
                        help="Directory where ground truth subtitle (VTT) files are stored.")
    parser.add_argument("--save_dir", type=str,
                        default="/users/zifan/subtitle_align/alternative/aligned_subtitles",
                        help="Directory to store aligned subtitle VTT files.")
    parser.add_argument("--segmentation_dir", type=str,
                        default="/scratch/shared/beegfs/zifan/bobsl/segmentation",
                        help="Directory with segmentation ELAN (.eaf) files.")
    parser.add_argument("--similarity_measure", type=str, default="none", choices=["none", "cslr_subtitle", "cslr_text_embedding", "sign_clip_embedding"],
                        help="Similarity measure to use.")
    parser.add_argument("--subtitle_embedding_dir", type=str,
                        default="/scratch/shared/beegfs/zifan/bobsl/subtitle_embedding/sign_clip",
                        help="Directory containing subtitle cue embeddings (NPY files) for sign_clip_embedding.")
    parser.add_argument("--segmentation_embedding_dir", type=str,
                        default="/scratch/shared/beegfs/zifan/bobsl/segmentation_embedding/E4s-1_30_50/sign_clip",
                        help="Directory containing sign segment embeddings (NPY files) for sign_clip_embedding.")
    parser.add_argument("--visualize_similarity", action="store_true")
    parser.add_argument("--cslr", action="store_true",
                        help="Merge CSLR CSV sign annotations with ELAN signs before DP alignment.")
    # New flag added after --cslr:
    parser.add_argument("--cslr_partial_eval", action="store_true",
                        help="If set, filter subtitle cues to those overlapping with CSLR signs only (partial evaluation).")
    parser.add_argument("--cslr_dir", type=str,
                        default="/users/zifan/BOBSL/v1.4/manual_annotations/continuous_sign_sequences/cslr-raw",
                        help="Directory with CSLR CSV files (searched recursively).")
    parser.add_argument("--fps", type=int, default=25)
    # Now pr_subs_delta_bias_* accept multiple values.
    parser.add_argument("--pr_subs_delta_bias_start", type=float, nargs='+', default=[2.7],
                        help="Delta bias (seconds) added to the start time of each subtitle cue (pre-alignment).")
    parser.add_argument("--pr_subs_delta_bias_end", type=float, nargs='+', default=[2.7],
                        help="Delta bias (seconds) added to the end time of each subtitle cue (pre-alignment).")
    # New post_subs_delta_bias arguments.
    parser.add_argument("--post_subs_delta_bias_start", type=float, nargs='+', default=[0.0],
                        help="Delta bias (seconds) added to the start time of each subtitle cue (post-alignment).")
    parser.add_argument("--post_subs_delta_bias_end", type=float, nargs='+', default=[1.0],
                        help="Delta bias (seconds) added to the end time of each subtitle cue (post-alignment).")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of processes for parallel processing.")
    parser.add_argument("--overwrite", action='store_true',
                        help="Overwrite existing files if set.")
    parser.add_argument("--live_segmentation", action="store_true",
                        help="Run live segmentation before alignment.")
    parser.add_argument("--mode", type=str, default="inference", choices=["inference", "training"],
                        help="Mode: inference (default) or training (parameter search).")
    parser.add_argument("--num_search", type=int, default=10,
                        help="Number of random search iterations in training mode.")
    parser.add_argument("--dp_duration_penalty_weight", type=float, nargs='+', default=[5.0],
                        help="Duration penalty weight(s) for DP alignment.")
    parser.add_argument("--dp_gap_penalty_weight", type=float, nargs='+', default=[10.0],
                        help="Gap penalty weight(s) for DP alignment.")
    parser.add_argument("--dp_window_size", type=int, nargs='+', default=[50],
                        help="Window size(s) for DP alignment.")
    parser.add_argument("--dp_max_gap", type=float, nargs='+', default=[8.0],
                        help="Max gap(s) allowed between SIGN segments for DP alignment.")
    parser.add_argument("--similarity_weight", type=float, nargs='+', default=[30.0],
                        help="Similarity weight(s) for DP alignment.")
    # New segmentation parameters.
    parser.add_argument("--segmentation_model", nargs='+', default=["model_E4s-1.pth"], type=str, help="Path(s) to segmentation model file")
    parser.add_argument("--sign-b-threshold", nargs='+', default=[30], type=int, help="Threshold(s) for sign B")
    parser.add_argument("--sign-o-threshold", nargs='+', default=[70], type=int, help="Threshold(s) for sign O")
    parser.add_argument("--debug", action="store_true",
                        help="If set, only use the first 20-30 seconds of cues and segments for debugging.")
    args = parser.parse_args()

    with open(args.video_ids, "r") as f:
        video_ids = [line.strip() for line in f if line.strip()]

    if args.mode == "inference":
        dp_dpw = args.dp_duration_penalty_weight[0]
        dp_gpw = args.dp_gap_penalty_weight[0]
        dp_ws  = args.dp_window_size[0]
        dp_mg  = args.dp_max_gap[0]
        sim_w  = args.similarity_weight[0]
        seg_model = args.segmentation_model[0]
        seg_sign_b = args.sign_b_threshold[0]
        seg_sign_o = args.sign_o_threshold[0]
        pr_subs_start = args.pr_subs_delta_bias_start[0]
        pr_subs_end   = args.pr_subs_delta_bias_end[0]
        post_subs_start = args.post_subs_delta_bias_start[0]
        post_subs_end   = args.post_subs_delta_bias_end[0]
        process_all_videos(video_ids, args, dp_dpw, dp_gpw, dp_ws, dp_mg, sim_w, args.save_dir, save_elan=True,
                           seg_model=seg_model, seg_sign_b=seg_sign_b, seg_sign_o=seg_sign_o,
                           pr_subs_start=pr_subs_start, pr_subs_end=pr_subs_end,
                           post_subs_start=post_subs_start, post_subs_end=post_subs_end)
        eval_output = eval_subtitle_alignment(Path(args.save_dir), Path(args.gt_sub_path), video_ids, args.fps, 0, 0, filter_gt_by_pred=args.cslr_partial_eval)
        print(eval_output)
    elif args.mode == "training":
        training_base = f"{args.save_dir}_training"
        os.makedirs(training_base, exist_ok=True)
        best_score = -1.0
        best_params = None
        scores = {}
        for i in range(args.num_search):
            dp_dpw = random.choice(args.dp_duration_penalty_weight)
            dp_gpw = random.choice(args.dp_gap_penalty_weight)
            dp_ws  = random.choice(args.dp_window_size)
            dp_mg  = random.choice(args.dp_max_gap)
            sim_w  = random.choice(args.similarity_weight)
            seg_model = random.choice(args.segmentation_model)
            seg_sign_b = random.choice(args.sign_b_threshold)
            seg_sign_o = random.choice(args.sign_o_threshold)
            pr_subs_start = random.choice(args.pr_subs_delta_bias_start)
            pr_subs_end   = random.choice(args.pr_subs_delta_bias_end)
            post_subs_start = random.choice(args.post_subs_delta_bias_start)
            post_subs_end   = random.choice(args.post_subs_delta_bias_end)
            comb_str = f"dpd_{dp_dpw}_dpg_{dp_gpw}_ws_{dp_ws}_mg_{dp_mg}_sim_{sim_w}_{seg_model}_{seg_sign_b}_{seg_sign_o}_{pr_subs_start}_{pr_subs_end}_{post_subs_start}_{post_subs_end}"
            output_dir = os.path.join(training_base, comb_str)
            os.makedirs(output_dir, exist_ok=True)
            process_all_videos(video_ids, args, dp_dpw, dp_gpw, dp_ws, dp_mg, sim_w, output_dir, save_elan=False,
                               seg_model=seg_model, seg_sign_b=seg_sign_b, seg_sign_o=seg_sign_o,
                               pr_subs_start=pr_subs_start, pr_subs_end=pr_subs_end,
                               post_subs_start=post_subs_start, post_subs_end=post_subs_end)
            eval_output = eval_subtitle_alignment(Path(output_dir), Path(args.gt_sub_path), video_ids, args.fps, 0, 0, filter_gt_by_pred=args.cslr_partial_eval)
            f1_score = extract_f1_score(eval_output)
            scores[comb_str] = f1_score
            print(f"Trial {i+1}/{args.num_search}, Params: {comb_str}, F1@0.50: {f1_score}")
            if f1_score > best_score:
                best_score = f1_score
                best_params = (dp_dpw, dp_gpw, dp_ws, dp_mg, sim_w, seg_model, seg_sign_b, seg_sign_o, pr_subs_start, pr_subs_end, post_subs_start, post_subs_end)
                print("New best found!")
                print(f"New Best F1@0.50: {best_score} with parameters: dp_duration_penalty_weight={best_params[0]}, "
                      f"dp_gap_penalty_weight={best_params[1]}, dp_window_size={best_params[2]}, "
                      f"dp_max_gap={best_params[3]}, similarity_weight={best_params[4]}, "
                      f"segmentation_model={best_params[5]}, sign-b-threshold={best_params[6]}, sign-o-threshold={best_params[7]}, "
                      f"pr_subs_delta_bias_start={best_params[8]}, pr_subs_delta_bias_end={best_params[9]}, "
                      f"post_subs_delta_bias_start={best_params[10]}, post_subs_delta_bias_end={best_params[11]}")
        print("----- All Trials -----")
        for comb_str, score in scores.items():
            print(f"{comb_str} => F1@0.50: {score}")
        print("----- Best Parameters -----")
        print(f"Best F1@0.50: {best_score} with parameters: dp_duration_penalty_weight={best_params[0]}, "
              f"dp_gap_penalty_weight={best_params[1]}, dp_window_size={best_params[2]}, "
              f"dp_max_gap={best_params[3]}, similarity_weight={best_params[4]}, "
              f"segmentation_model={best_params[5]}, sign-b-threshold={best_params[6]}, sign-o-threshold={best_params[7]}, "
              f"pr_subs_delta_bias_start={best_params[8]}, pr_subs_delta_bias_end={best_params[9]}, "
              f"post_subs_delta_bias_start={best_params[10]}, post_subs_delta_bias_end={best_params[11]}")
        process_all_videos(video_ids, args, best_params[0], best_params[1], best_params[2],
                           best_params[3], best_params[4], args.save_dir, save_elan=True,
                           seg_model=best_params[5], seg_sign_b=best_params[6], seg_sign_o=best_params[7],
                           pr_subs_start=best_params[8], pr_subs_end=best_params[9],
                           post_subs_start=best_params[10], post_subs_end=best_params[11])
        final_eval = eval_subtitle_alignment(Path(args.save_dir), Path(args.gt_sub_path), video_ids, args.fps, 0, 0, filter_gt_by_pred=args.cslr_partial_eval)
        print(final_eval)

if __name__ == '__main__':
    main()
