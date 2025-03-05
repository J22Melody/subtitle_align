#!/usr/bin/env python3
import os
import sys
import argparse
import xml.etree.ElementTree as ET
import bisect
import numpy as np
from fastdtw import fastdtw
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import itertools
import re
import csv
import glob

# Add the ../misc directory to sys.path to import the evaluation function.
current_dir = os.path.dirname(os.path.abspath(__file__))
misc_dir = os.path.join(current_dir, "../misc")
if misc_dir not in sys.path:
    sys.path.append(misc_dir)
from evaluate_sub_alignment import eval_subtitle_alignment

from align_dp import dp_align_subtitles_to_signs, dp_align_subtitles_to_signs_dtw

def parse_time(time_str):
    """Convert a time string (HH:MM:SS.mmm) to total seconds (float)."""
    parts = time_str.strip().split(':')
    if len(parts) != 3:
        raise ValueError(f"Invalid time format: {time_str}")
    hours = int(parts[0])
    minutes = int(parts[1])
    sec_parts = parts[2].split('.')
    seconds = int(sec_parts[0])
    millis = int(sec_parts[1]) if len(sec_parts) > 1 else 0
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0

def format_time(total_seconds):
    """Convert total seconds back to a time string (HH:MM:SS.mmm)."""
    if total_seconds < 0:
        total_seconds = 0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def shift_subtitle_timing(vtt_content, delta_start, delta_end):
    """Shift timings in a VTT file by the specified deltas."""
    output_lines = []
    for line in vtt_content.splitlines():
        if " --> " in line:
            try:
                parts = line.split(" --> ")
                start_time = parts[0]
                end_parts = parts[1].split(maxsplit=1)
                end_time = end_parts[0]
                new_start = format_time(parse_time(start_time) + delta_start)
                new_end = format_time(parse_time(end_time) + delta_end)
                if len(end_parts) > 1:
                    extra = end_parts[1]
                    new_line = f"{new_start} --> {new_end} {extra}"
                else:
                    new_line = f"{new_start} --> {new_end}"
                output_lines.append(new_line)
            except Exception as e:
                output_lines.append(line)
        else:
            output_lines.append(line)
    return "\n".join(output_lines)

def get_subtitle_cues(vtt_content):
    """Parse VTT content and return header lines and a list of cues (each with start, end, mid, text)."""
    cues = []
    lines = vtt_content.splitlines()
    header_lines = []
    i = 0
    while i < len(lines) and (lines[i].strip() == "" or "WEBVTT" in lines[i]):
        header_lines.append(lines[i])
        i += 1
    while i < len(lines):
        if lines[i].strip() == "":
            i += 1
            continue
        if "-->" in lines[i]:
            try:
                parts = lines[i].strip().split(" --> ")
                start = parse_time(parts[0])
                end = parse_time(parts[1].split()[0])
                mid = (start + end) / 2
            except Exception as e:
                i += 1
                continue
            i += 1
            cue_text_lines = []
            while i < len(lines) and lines[i].strip() != "":
                cue_text_lines.append(lines[i])
                i += 1
            cue_text = "\n".join(cue_text_lines)
            cues.append({'start': start, 'end': end, 'mid': mid, 'text': cue_text})
        else:
            i += 1
    return header_lines, cues

def reconstruct_vtt(header_lines, cues):
    """Reconstruct a VTT file from header lines and cues."""
    output_lines = header_lines + [""]
    for cue in cues:
        output_lines.append(f"{format_time(cue['start'])} --> {format_time(cue['end'])}")
        if cue['text']:
            output_lines.append(cue['text'])
        output_lines.append("")
    return "\n".join(output_lines)

def get_sign_segments(segmentation_file, video_id):
    """Parse an ELAN (.eaf) file and return all segments from the SIGN tier."""
    segments = []
    try:
        tree = ET.parse(segmentation_file)
        root = tree.getroot()
    except Exception as e:
        return segments
    time_order = root.find("TIME_ORDER")
    time_slots = {}
    if time_order is not None:
        for ts in time_order.findall("TIME_SLOT"):
            ts_id = ts.get("TIME_SLOT_ID")
            ts_value = ts.get("TIME_VALUE")
            if ts_value is not None:
                try:
                    time_slots[ts_id] = float(ts_value) / 1000.0
                except ValueError:
                    time_slots[ts_id] = None
    else:
        return segments
    sign_tier = None
    for tier in root.findall("TIER"):
        if tier.get("TIER_ID") == "SIGN":
            sign_tier = tier
            break
    if sign_tier is None:
        return segments
    for annotation in sign_tier.findall("ANNOTATION"):
        annotation_elem = None
        for child in annotation:
            annotation_elem = child
            break
        if annotation_elem is None:
            continue
        text_elem = annotation_elem.find("ANNOTATION_VALUE")
        text = text_elem.text if text_elem is not None else ""
        start_time = None
        end_time = None
        if "TIME_SLOT_REF1" in annotation_elem.attrib and "TIME_SLOT_REF2" in annotation_elem.attrib:
            ts1 = annotation_elem.attrib["TIME_SLOT_REF1"]
            ts2 = annotation_elem.attrib["TIME_SLOT_REF2"]
            start_time = time_slots.get(ts1, None)
            end_time = time_slots.get(ts2, None)
        if start_time is not None and end_time is not None:
            mid = (start_time + end_time) / 2
        else:
            mid = None
        segments.append({'start': start_time, 'end': end_time, 'mid': mid, 'text': text})
    return segments

def get_cslr_signs(video_id, cslr_dir):
    """
    Recursively search for a CSV file named <video_id>.csv under cslr_dir.
    Read all sign-level annotations. Expected header: 
      start_sub, end_sub, english sentence, approx gloss sequence.
    In the "approx gloss sequence" field, signs are annotated as: word[start-end].
    This function returns a list of signs, each with keys: 'start', 'end', 'mid', 'text', and 'subtitle'
    (where 'text' is the sign word and 'subtitle' is the english sentence).
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
            for m in matches:
                word, start_str, end_str = m
                start = float(start_str)
                end = float(end_str)
                mid = (start + end) / 2
                signs.append({'start': start, 'end': end, 'mid': mid, 'text': word, 'subtitle': english_sentence})
    signs.sort(key=lambda s: s['start'])
    return signs

def merge_signs(elan_signs, cslr_signs):
    """
    Merge CSLR signs into the base ELAN sign list.
    For each CSLR sign, remove all overlapping ELAN segments and insert the CSLR sign.
    The final merged list is sorted by start time and returned.
    """
    elan_signs = sorted(elan_signs, key=lambda s: s['start'])
    cslr_signs = sorted(cslr_signs, key=lambda s: s['start'])
    merged = elan_signs[:]
    for cs in cslr_signs:
        new_merged = []
        replaced = False
        for seg in merged:
            if seg['end'] > cs['start'] and seg['start'] < cs['end']:
                replaced = True
            else:
                new_merged.append(seg)
        if replaced:
            new_merged.append(cs)
            new_merged = sorted(new_merged, key=lambda s: s['start'])
        merged = new_merged
    return merged

def write_updated_eaf(eaf_file, cues, video_id, merged_signs=None):
    """
    Write one updated ELAN file that contains two new tiers:
      - SIGN_MERGED: the merged sign annotations (if merged_signs is provided)
      - SUBTITLE_SHIFTED: the DP-aligned subtitle cues.
    The updated file is saved with the suffix "_updated.eaf".
    """
    try:
        tree = ET.parse(eaf_file)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing ELAN file {eaf_file}: {e}")
        return
    time_order = root.find("TIME_ORDER")
    if time_order is None:
        print(f"No TIME_ORDER element found in {eaf_file}")
        return
    if merged_signs is not None and len(merged_signs) > 0:
        sign_tier = ET.Element("TIER", {"TIER_ID": "SIGN_MERGED", "LINGUISTIC_TYPE_REF": "default-lt"})
        for i, sign in enumerate(merged_signs):
            ts1 = f"SIGN_MERGED_TS_{video_id}_{i}_1"
            ts2 = f"SIGN_MERGED_TS_{video_id}_{i}_2"
            new_ts1 = ET.Element("TIME_SLOT", {"TIME_SLOT_ID": ts1, "TIME_VALUE": str(int(sign['start'] * 1000))})
            new_ts2 = ET.Element("TIME_SLOT", {"TIME_SLOT_ID": ts2, "TIME_VALUE": str(int(sign['end'] * 1000))})
            time_order.append(new_ts1)
            time_order.append(new_ts2)
            annotation = ET.Element("ANNOTATION")
            alignable = ET.Element("ALIGNABLE_ANNOTATION", {
                "ANNOTATION_ID": f"a_sign_merged_{video_id}_{i}",
                "TIME_SLOT_REF1": ts1,
                "TIME_SLOT_REF2": ts2
            })
            annotation_value = ET.Element("ANNOTATION_VALUE")
            annotation_value.text = sign['text']
            # Optionally, you could also store the english sentence in a separate field or as a comment.
            alignable.append(annotation_value)
            annotation.append(alignable)
            sign_tier.append(annotation)
        root.append(sign_tier)
    subtitle_tier = ET.Element("TIER", {"TIER_ID": "SUBTITLE_SHIFTED", "LINGUISTIC_TYPE_REF": "default-lt"})
    for i, cue in enumerate(cues):
        ts1 = f"SUBTITLE_TS_{video_id}_{i}_1"
        ts2 = f"SUBTITLE_TS_{video_id}_{i}_2"
        new_ts1 = ET.Element("TIME_SLOT", {"TIME_SLOT_ID": ts1, "TIME_VALUE": str(int(cue['start'] * 1000))})
        new_ts2 = ET.Element("TIME_SLOT", {"TIME_SLOT_ID": ts2, "TIME_VALUE": str(int(cue['end'] * 1000))})
        time_order.append(new_ts1)
        time_order.append(new_ts2)
        annotation = ET.Element("ANNOTATION")
        alignable = ET.Element("ALIGNABLE_ANNOTATION", {
            "ANNOTATION_ID": f"a_subtitle_{video_id}_{i}",
            "TIME_SLOT_REF1": ts1,
            "TIME_SLOT_REF2": ts2
        })
        annotation_value = ET.Element("ANNOTATION_VALUE")
        annotation_value.text = cue['text']
        alignable.append(annotation_value)
        annotation.append(alignable)
        subtitle_tier.append(annotation)
    root.append(subtitle_tier)
    output_eaf = os.path.splitext(eaf_file)[0] + "_updated.eaf"
    tree.write(output_eaf, encoding="utf-8", xml_declaration=True)
    print(f"Written updated ELAN file to {output_eaf}")

def process_video(video_id, args, dp_duration_penalty_weight, dp_gap_penalty_weight, dp_window_size, dp_max_gap, output_dir, save_elan):
    """Process one video: shift timings, merge CSLR signs (if --cslr), run DP using merged signs, and write output."""
    print(f"\nProcessing video: {video_id}")
    segmentation_file = os.path.join(args.segmentation_dir, f"{video_id}.eaf")
    elan_signs = []
    if os.path.exists(segmentation_file):
        elan_signs = get_sign_segments(segmentation_file, video_id)
    merged_signs = elan_signs
    if args.cslr:
        cslr_signs = get_cslr_signs(video_id, args.cslr_dir)
        if cslr_signs:
            merged_signs = merge_signs(elan_signs, cslr_signs)
    input_file = os.path.join(args.input_dir, f"{video_id}.vtt")
    if not os.path.exists(input_file):
        print(f"Subtitle file for video {video_id} not found in {args.input_dir}. Skipping.")
        return
    try:
        with open(input_file, "r", encoding="utf-8") as fin:
            vtt_content = fin.read()
    except Exception as e:
        return
    shifted_content = shift_subtitle_timing(vtt_content, args.pr_subs_delta_bias_start, args.pr_subs_delta_bias_end)
    header_lines, cues = get_subtitle_cues(shifted_content)
    if not cues:
        return
    dp_align_subtitles_to_signs(cues, merged_signs,
       duration_penalty_weight=dp_duration_penalty_weight,
       gap_penalty_weight=dp_gap_penalty_weight,
       window_size=dp_window_size,
       max_gap=dp_max_gap)
    # dp_align_subtitles_to_signs_dtw(cues, merged_signs)
    updated_vtt = reconstruct_vtt(header_lines, cues)
    output_vtt = os.path.join(output_dir, f"{video_id}.vtt")
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(output_vtt, "w", encoding="utf-8") as fout:
            fout.write(updated_vtt)
    except Exception as e:
        pass
    if save_elan and os.path.exists(segmentation_file):
        write_updated_eaf(segmentation_file, cues, video_id, merged_signs)

def extract_f1_score(eval_output):
    """Extract F1@0.50 score from evaluation output."""
    m = re.search(r"F1@0\.50:\s*([\d.]+)", eval_output)
    if m:
        return float(m.group(1))
    return 0.0

def process_all_videos(video_ids, args, dp_dpw, dp_gpw, dp_ws, dp_mg, output_dir, save_elan):
    from functools import partial
    func = partial(process_video, args=args,
                   dp_duration_penalty_weight=dp_dpw,
                   dp_gap_penalty_weight=dp_gpw,
                   dp_window_size=dp_ws,
                   dp_max_gap=dp_mg,
                   output_dir=output_dir,
                   save_elan=save_elan)
    if args.num_workers > 1:
        from multiprocessing import Pool
        with Pool(args.num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(func, video_ids), total=len(video_ids), desc="Processing videos"):
                pass
    else:
        for vid in tqdm(video_ids, desc="Processing videos"):
            func(vid)

def main():
    parser = argparse.ArgumentParser(
        description="Shift subtitle timings, align cues to SIGN segments, merge CSLR signs (if --cslr is set), and write one updated ELAN file."
    )
    parser.add_argument("--mode", type=str, default="inference", choices=["inference", "training"],
                        help="Mode: inference (default) or training (parameter search).")
    parser.add_argument("--video_ids", type=str,
                        default="/users/zifan/subtitle_align/data/bobsl_align_test.txt",
                        help="Path to text file with video ids (one per line).")
    parser.add_argument("--input_dir", type=str,
                        default="/users/zifan/BOBSL/v1.4/automatic_annotations/signing_aligned_subtitles/audio_aligned_heuristic_correction",
                        help="Directory where subtitle (VTT) files are stored.")
    parser.add_argument("--save_dir", type=str,
                        default="/users/zifan/subtitle_align/alternative/aligned_subtitles",
                        help="Directory to store aligned subtitle VTT files.")
    parser.add_argument("--segmentation_dir", type=str,
                        default="/scratch/shared/beegfs/zifan/bobsl/video_features/segmentation",
                        help="Directory with segmentation ELAN (.eaf) files.")
    parser.add_argument("--cslr_dir", type=str,
                        default="/users/zifan/BOBSL/v1.4/manual_annotations/continuous_sign_sequences/cslr-raw",
                        help="Directory with CSLR CSV files (searched recursively).")
    parser.add_argument("--cslr", action="store_true",
                        help="If set, merge CSLR CSV sign annotations with ELAN signs before DP alignment.")
    parser.add_argument("--overwrite", action='store_true',
                        help="Overwrite existing files if set.")
    parser.add_argument("--pr_subs_delta_bias_start", type=float, default=2.7,
                        help="Delta bias (seconds) added to the start time of each subtitle cue.")
    parser.add_argument("--pr_subs_delta_bias_end", type=float, default=2.7,
                        help="Delta bias (seconds) added to the end time of each subtitle cue.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of processes for parallel processing.")
    parser.add_argument("--gt_sub_path", type=str,
                        default="/users/zifan/BOBSL/v1.4/manual_annotations/signing_aligned_subtitles",
                        help="Ground truth subtitle directory for evaluation.")
    parser.add_argument("--dp_duration_penalty_weight", type=float, nargs='+', default=[1.0],
                        help="Duration penalty weight(s) for DP alignment.")
    parser.add_argument("--dp_gap_penalty_weight", type=float, nargs='+', default=[2.0],
                        help="Gap penalty weight(s) for DP alignment.")
    parser.add_argument("--dp_window_size", type=int, nargs='+', default=[40],
                        help="Window size(s) for DP alignment.")
    parser.add_argument("--dp_max_gap", type=float, nargs='+', default=[8.0],
                        help="Max gap(s) allowed between SIGN segments for DP alignment.")
    
    args = parser.parse_args()
    try:
        with open(args.video_ids, "r") as f:
            video_ids = [line.strip() for line in f if line.strip()]
    except Exception as e:
        return
    if args.mode == "inference":
        dp_dpw = args.dp_duration_penalty_weight[0]
        dp_gpw = args.dp_gap_penalty_weight[0]
        dp_ws  = args.dp_window_size[0]
        dp_mg  = args.dp_max_gap[0]
        process_all_videos(video_ids, args, dp_dpw, dp_gpw, dp_ws, dp_mg, args.save_dir, save_elan=True)
        eval_output = eval_subtitle_alignment(Path(args.save_dir), Path(args.gt_sub_path), video_ids, 25, 0, 0)
        print(eval_output)
    else:
        training_base = f"{args.save_dir}_training"
        os.makedirs(training_base, exist_ok=True)
        best_score = -1.0
        best_params = None
        scores = {}
        combinations = list(itertools.product(args.dp_duration_penalty_weight,
                                              args.dp_gap_penalty_weight,
                                              args.dp_window_size,
                                              args.dp_max_gap))
        for comb in itertools.product(args.dp_duration_penalty_weight,
                                        args.dp_gap_penalty_weight,
                                        args.dp_window_size,
                                        args.dp_max_gap):
            dp_dpw, dp_gpw, dp_ws, dp_mg = comb
            comb_str = f"dpd_{dp_dpw}_dpg_{dp_gpw}_ws_{dp_ws}_mg_{dp_mg}"
            output_dir = os.path.join(training_base, comb_str)
            os.makedirs(output_dir, exist_ok=True)
            process_all_videos(video_ids, args, dp_dpw, dp_gpw, dp_ws, dp_mg, output_dir, save_elan=False)
            eval_output = eval_subtitle_alignment(Path(output_dir), Path(args.gt_sub_path), video_ids, 25, 0, 0)
            f1_score = extract_f1_score(eval_output)
            scores[comb_str] = f1_score
            if f1_score > best_score:
                best_score = f1_score
                best_params = (dp_dpw, dp_gpw, dp_ws, dp_mg)
        if best_params is not None:
            process_all_videos(video_ids, args, best_params[0], best_params[1], best_params[2], best_params[3],
                                args.save_dir, save_elan=True)
            final_eval = eval_subtitle_alignment(Path(args.save_dir), Path(args.gt_sub_path), video_ids, 25, 0, 0)
            print(final_eval)

if __name__ == '__main__':
    main()
