"""Compute subtitle alignment metrics.

This module provides code to evaluate the quality of a set of subtitle alignments (i.e.
how well their start and end times match those of a set of ground-truth subtitle times).

Note: several of the metrics are derived from the following code:
    https://github.com/yabufarha/ms-tcn/blob/
        c6eab71ddd7b4190ffb3bf6f1b57f3517454939b/eval.py#L15

Example usage:
python misc/sub_align/evaluate_sub_alignment.py \
    --pred_subtitle_dir /scratch/shared/beegfs/albanie/shared-datasets/bbcsl_raw/subtitles/subtitles-vtt-text-normalized-aligned/heuristic-aligned-subs-05_01_2021-mouth-padding4_all
"""
import sys
sys.path.append('/athenahomes/zifan/subtitle_align') #hack

import argparse
from pickle import SHORT_BINSTRING
from typing import List, Tuple
from pathlib import Path
import multiprocessing

import os 
import tqdm
from statistics import mean, median
import numpy as np
import webvtt
from beartype import beartype

if __name__ == "__main__":
    from config.config import *
    opts = load_opts()

@beartype
def get_labels_start_end_time(
        frame_wise_labels: List[int],
        bg_class: List[int]
) -> Tuple[List[int], List[int], List[int]]:
    """Given a single sequence of frame level labels, find: (i) the start index,
    (ii) the end index and (iii) the label, of each contiguous subsequence of labels.

    Args:
        frame_wise_labels: a single sequence of frame-level labels
        bg_class: if given, skip labels that fall within this list of background classes.

    Returns:
        A tuple consisting of three elements:
            the label associated with each subsequence
            the start index associated with each subsequence
            the end index associated with each subsequence
    """
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


@beartype
def f_score(
        recognized: List[int],
        ground_truth: List[int],
        overlap: float,
        bg_class: List[int],
) -> Tuple[float, float, float]:
    """Compute the f-score of a sequence of predicted sequences against a set of ground
    truth annotations (this is the F1 metric used in https://arxiv.org/abs/1903.01945).

    Args:
        recognized: a list of frame-level sequence label predictions
        ground_truth: a list of frame-level sequence label ground truth
        overlap: the F1 overlap threshold
        bg_class: a list of classes that should be excluded from the evaluation

    Returns:
        A tuple containing:
            (i) the total number of true positives
            (i) the total number of false positives
            (i) the total number of false negatives
    """
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0
    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x]
                                               for x in range(len(y_label))])

        # Get the best scoring segment
        idx = np.array(IoU).argmax()
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


@beartype
def subs2frames(
        subs: List[webvtt.Caption],
        max_time: float,
        fps: int,
        exclude_subs: List[int],
        background_label: int,
) -> List[int]:
    """Convert subtitles into a single sequence of frames in which each subtitle is
    assigned a unique integer and the frames covered by that subtitle are assigned this
    integer.

    Args:
        subs: a list of webvtt caption objects, each of which represnts a subtitle caption
           with an associated start and end time as well as text.
        max_time: the maximum duration of the frame sequence to be created (in seconds).
        fps: the frame rate of the videos
        exclude_subs: the indices of subtitles that should be excluded from the evaluation
        background_label: the value to be assigned to frames that are not covered by any
           subtitle (or frames covered by subtitles that are excluded).

    Returns:
        A list of frame-level sequence labels, each indicating the index of the subtitle
        which covered the current frame.
    """
    frames = [background_label for _ in range(round(fps * max_time))]
    for sub_idx, caption in enumerate(subs):
        if sub_idx in exclude_subs:
            continue

        # Ensure that predicted caption alignment time falls within the bounds o fevaluation
        start_time = min(caption._start, max_time)
        end_time = min(caption._end, max_time)

        start_idx = round(fps * start_time)
        end_idx = round(fps * end_time)
        frames[start_idx:end_idx] = [sub_idx for _ in range(end_idx - start_idx)]
    return frames

def _process_video(pred_path, gt_path, vid_id, shift_start, shift_end, fps, MAX_TIME_PAD_SECS, overlaps, BACKGROUND_LABEL, ext_gt, ext_pred):
    """
    Process a single video's subtitles and compute metrics.
    """
    # Initialize local metrics
    video_correct = 0
    video_total_frames = 0
    video_total_subs = 0
    video_all_offset_start = []
    video_all_offset_end = []
    video_all_offset_start_abs = []
    video_all_offset_end_abs = []
    video_tp = np.zeros(len(overlaps))
    video_fp = np.zeros(len(overlaps))
    video_fn = np.zeros(len(overlaps))
    video_msg = 'no susbtitle to evaluate, skipping'
    
    # if 'natural' in str(pred_path):
    #     continue

    gt_subs = list(webvtt.from_srt(gt_path) if ext_gt == '.srt' else webvtt.read(gt_path))
    pred_subs = list(webvtt.from_srt(pred_path) if ext_pred == '.srt' else webvtt.read(pred_path))

    # suppose pred_subs already exclude [NON-SIGN], etc.
    if len(gt_subs) != len(pred_subs):
        gt_subs = [sub for sub in gt_subs if not ("[" in sub.text and "]" in sub.text)]
        pred_subs = [sub for sub in pred_subs if not ("[" in sub.text and "]" in sub.text)]
    else:
        # Get indices in pred_subs where the exclusion tag appears
        excluded_indices = [i for i, sub in enumerate(gt_subs) if '[' in sub.text and ']' in sub.text]
        # Filter out entries with those indices from both lists
        gt_subs = [sub for i, sub in enumerate(gt_subs) if i not in excluded_indices]
        pred_subs = [sub for i, sub in enumerate(pred_subs) if i not in excluded_indices]

    # Get indices in pred_subs where the exclusion tag appears
    excluded_indices = [i for i, sub in enumerate(pred_subs) if '{CSLR_EXCLUDED}' in sub.text]
    # Filter out entries with those indices from both lists
    gt_subs = [sub for i, sub in enumerate(gt_subs) if i not in excluded_indices]
    pred_subs = [sub for i, sub in enumerate(pred_subs) if i not in excluded_indices]

    # for s in gt_subs:
    #     print(s)
    # for s in pred_subs:
    #     print(s)

    for sub_idx in range(len(pred_subs)): 
        pred_subs[sub_idx]._start += shift_start
        pred_subs[sub_idx]._end += shift_end

    msg = (f"Expected num. preds {len(pred_subs)} to match num. gt {len(gt_subs)} for"
           f" {pred_path}")
    assert len(pred_subs) == len(gt_subs), msg

    if len(gt_subs) > 0:
        video_total_subs += len(gt_subs)

        # We pick the maximum time for the evaluation to be a fixed offset (10 seconds)
        # beyond the last ground truth subtitle.
        max_time = gt_subs[-1]._end + MAX_TIME_PAD_SECS

        # If an annotator is unable to align the subtitle with the signing, they leave
        # a comment in the content of the subtitle itself, which looks like this:
        # "<subtitle-text> [NOT SURE WHERE]"
        # For other subtitles they leave a comment indicating that they do not agree
        # with the interpretation e.g.
        # "<subtitle-text> [INCORRECT]"
        # or that the signing itself is inappropriate/might be offensive
        # "<subtitle-text> [INAPPROPRIATE SIGN]"
        # We exclude these subtitles from the evaluation
        exclude_subs = []
        for sub_idx, sub in enumerate(gt_subs):
            if "[" in sub.text and "]" in sub.text:
                exclude_subs.append(sub_idx)
            else:
                video_all_offset_start.append(sub._start - pred_subs[sub_idx]._start)
                video_all_offset_end.append(sub._end - pred_subs[sub_idx]._end)
                video_all_offset_start_abs.append(abs(sub._start - pred_subs[sub_idx]._start))
                video_all_offset_end_abs.append(abs(sub._end - pred_subs[sub_idx]._end))
        video_total_subs -= len(exclude_subs)

        # Convert subtitles into a sequence of frame-level labels.
        pred_frames = subs2frames(
            subs=pred_subs,
            max_time=float(max_time),
            # exclude_subs=exclude_subs,
            exclude_subs=[],
            fps=fps,
            background_label=BACKGROUND_LABEL,
        )
        gt_frames = subs2frames(
            subs=gt_subs,
            max_time=float(max_time),
            # exclude_subs=exclude_subs,
            exclude_subs=[],
            fps=fps,
            background_label=BACKGROUND_LABEL,
        )

        # Compute frame-level accuracy
        for pred, gt in zip(pred_frames, gt_frames):
            video_total_frames += 1
            if pred == gt:
                video_correct += 1

        # Compute f-scores at various overlaps over frame sequences
        for ii, overlap in enumerate(overlaps):
            tp1, fp1, fn1 = f_score(
                recognized=pred_frames,
                ground_truth=gt_frames,
                overlap=overlap,
                bg_class=[BACKGROUND_LABEL],
            )
            video_tp[ii] += tp1
            video_fp[ii] += fp1
            video_fn[ii] += fn1

        # Compute evaluation message for this video
        video_msg = (
            f"Mean and median start offset: {mean(video_all_offset_start):.2f} / {median(video_all_offset_start):.2f} \n"
            f"Mean and median end offset: {mean(video_all_offset_end):.2f} / {median(video_all_offset_end):.2f} \n"
            f"Mean and median start offset (abs): {mean(video_all_offset_start_abs):.2f} / {median(video_all_offset_start_abs):.2f} \n"
            f"Mean and median end offset (abs): {mean(video_all_offset_end_abs):.2f} / {median(video_all_offset_end_abs):.2f} \n"
            f"Computed over {video_total_frames} frames, {video_total_subs} sentences - "
            f"Frame-level accuracy: {100 * float(video_correct)/video_total_frames:.2f}"
        )
        for ii, overlap in enumerate(overlaps):
            precision = video_tp[ii] / float(video_tp[ii] + video_fp[ii])
            recall = video_tp[ii] / float(video_tp[ii] + video_fn[ii])
            f1 = 2.0 * (precision * recall) / (precision + recall)
            f1 = np.nan_to_num(f1) * 100
            video_msg += f" F1@{overlap:0.2f}: {f1:.2f}"
        
    return {
        'correct': video_correct,
        'total_frames': video_total_frames,
        'total_subs': video_total_subs,
        'all_offset_start': video_all_offset_start,
        'all_offset_end': video_all_offset_end,
        'all_offset_start_abs': video_all_offset_start_abs,
        'all_offset_end_abs': video_all_offset_end_abs,
        'tp': video_tp,
        'fp': video_fp,
        'fn': video_fn,
        'msg': video_msg,  # add the per-video evaluation message
    }

def eval_subtitle_alignment(
        pred_path_root: "Path",
        gt_anno_path_root: "Path",
        list_videos: list,
        fps: int, 
        shift_start=0,
        shift_end=0,
        num_workers=1,
        debug=False,  # new debug parameter, default False
):
    if os.path.exists(os.path.join(gt_anno_path_root, list_videos[0]+'.vtt')): 
        ext_gt = '.vtt'
    elif os.path.exists(os.path.join(gt_anno_path_root, list_videos[0]+'.srt')): 
        ext_gt = '.srt'
    else: 
        ext_gt = '/signhd.vtt'

    if os.path.exists(os.path.join(pred_path_root, list_videos[0]+'.vtt')): 
        ext_pred = '.vtt'
    elif os.path.exists(os.path.join(pred_path_root, list_videos[0]+'.srt')): 
        ext_pred = '.srt'
    else: 
        ext_pred = '/signhd.vtt'
    gt_anno_paths = [f'{gt_anno_path_root}/{p}{ext_gt}' for p in list_videos]
    pred_paths = [f'{pred_path_root}/{p}{ext_pred}' for p in list_videos]

    """Evaluate subtitle alignment quality.

    Args:
        pred_paths: the locations of subtitle timing predictions (in .vtt format)
        gt_anno_paths: the locations of subtitle ground truth timings (in .vtt format)
        fps: the frame rate of the videos
    """
    correct = 0
    total = 0
    total_subs = 0
    all_offset_start = []
    all_offset_end = []
    all_offset_start_abs = []
    all_offset_end_abs = []
    BACKGROUND_LABEL = -1
    MAX_TIME_PAD_SECS = 10
    overlaps = [0.1, 0.25, 0.5]
    tp = np.zeros(3)
    fp = np.zeros(3)
    fn = np.zeros(3)

    # Process each video either sequentially or in parallel based on num_workers.
    results = []
    if num_workers > 1:
        # Use multiprocessing to parallelize the processing of videos.
        args_list = [
            (
                pred_path,
                gt_path,
                vid_id,
                shift_start,
                shift_end,
                fps,
                MAX_TIME_PAD_SECS,
                overlaps,
                BACKGROUND_LABEL,
                ext_gt,
                ext_pred,
            )
            for pred_path, gt_path, vid_id in zip(pred_paths, gt_anno_paths, list_videos)
        ]
        with multiprocessing.Pool(num_workers) as pool:
            # Using starmap to pass multiple arguments to _process_video.
            for res in tqdm.tqdm(pool.starmap(_process_video, args_list), total=len(args_list)):
                results.append(res)
    else:
        # Sequential processing as before.
        for pred_path, gt_path, vid_id in tqdm.tqdm(zip(pred_paths, gt_anno_paths, list_videos)):
            res = _process_video(
                pred_path,
                gt_path,
                vid_id,
                shift_start,
                shift_end,
                fps,
                MAX_TIME_PAD_SECS,
                overlaps,
                BACKGROUND_LABEL,
                ext_gt,
                ext_pred,
            )
            results.append(res)
            if debug:
                print(f"Video {vid_id} evaluation:\n{res['msg']}\n")

    # If running in parallel and debug is True, print each video's evaluation message
    if num_workers > 1 and debug:
        for idx, res in enumerate(results):
            print(f"Video {list_videos[idx]} evaluation:\n{res['msg']}\n")

    # Aggregate results from all videos
    for res in results:
        correct += res['correct']
        total += res['total_frames']
        total_subs += res['total_subs']
        all_offset_start.extend(res['all_offset_start'])
        all_offset_end.extend(res['all_offset_end'])
        all_offset_start_abs.extend(res['all_offset_start_abs'])
        all_offset_end_abs.extend(res['all_offset_end_abs'])
        tp += res['tp']
        fp += res['fp']
        fn += res['fn']

    # Provide a summary of the computed metrics
    print('total ', total, 'subs', total_subs)
    msg = ( 
            f"Mean and median start offset: {mean(all_offset_start):.2f} / {median(all_offset_start):.2f} \n"
            f"Mean and median end offset: {mean(all_offset_end):.2f} / {median(all_offset_end):.2f} \n"
            f"Mean and median start offset (abs): {mean(all_offset_start_abs):.2f} / {median(all_offset_start_abs):.2f} \n"
            f"Mean and median end offset (abs): {mean(all_offset_end_abs):.2f} / {median(all_offset_end_abs):.2f} \n"
            f"Computed over {total} frames, {total_subs} sentences - "
            f"Frame-level accuracy: {100 * float(correct)/total:.2f}"            
           )
    for ii, overlap in enumerate(overlaps):
        precision = tp[ii] / float(tp[ii] + fp[ii])
        recall = tp[ii] / float(tp[ii] + fn[ii])
        f1 = 2.0 * (precision * recall) / (precision + recall)
        f1 = np.nan_to_num(f1) * 100
        f1_msg = (f"F1@{overlap:0.2f}: {f1:.2f}")
        msg = f'{msg} {f1_msg}'

    # print(msg)
    return msg

def parse_args():
    # pylint: disable=line-too-long
    # flake8: noqa: E501
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_subtitle_dir", type=Path, default="/scratch/shared/beegfs/albanie/shared-datasets/bobsl/public_dataset_release/subtitles/audio-aligned")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument( "--gt_anno_dir", type=Path, default="/scratch/shared/beegfs/albanie/shared-datasets/bobsl/public_dataset_release/subtitles/manually-aligned")
    return parser.parse_args()


def main():
    # args = parse_args()
    # gt_anno_paths = list(args.gt_anno_dir.glob("**/*.vtt"))
    # print(f"Found {len(gt_anno_paths)} ground truth annotation files in {args.gt_anno_dir}")
    # pred_paths = [args.pred_subtitle_dir / path.relative_to(args.gt_anno_dir) for path
    #               in gt_anno_paths]

    # #import ipdb; ipdb.set_trace(context=20)

    # eval_subtitle_alignment(
    #     pred_paths=pred_paths,
    #     fps=args.fps,
    #     gt_anno_paths=gt_anno_paths,
    # )

    test_files = open(opts.test_videos_txt, "r").read().split('\n')
    eval_str = eval_subtitle_alignment(
        pred_path_root=Path(f'{opts.pred_path_root}'),
        gt_anno_path_root=Path(f'{opts.gt_sub_path}'),
        list_videos=test_files,
        fps=opts.fps,
        shift_start=opts.pr_subs_delta_bias_start,
        shift_end=opts.pr_subs_delta_bias_end,
    )
    print(eval_str)


if __name__ == "__main__":
    main()
