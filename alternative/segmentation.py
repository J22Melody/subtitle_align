#!/usr/bin/env python3
import argparse
import os
import subprocess
from tqdm import tqdm
import multiprocessing
from functools import partial

def process_video(vid, args):
    pose_file = os.path.join(args.pose_dir, f"{vid}.pose")
    elan_file = os.path.join(args.save_dir, f"{vid}.eaf")

    # Skip processing if output already exists and overwrite is not set.
    if not args.overwrite and os.path.exists(elan_file):
        return f"Skipping {vid}: output already exists at {elan_file}"

    # Build the pose_to_segments command.
    cmd = f"pose_to_segments --no-pose-link --model=model_E4s-1.pth --pose={pose_file} --elan={elan_file}"

    # Check for the video file.
    video_file = os.path.join(args.video_dir, f"{vid}.mp4")
    if os.path.exists(video_file):
        # As per instructions, pass a relative path for the video.
        cmd += f" --video=./{vid}.mp4"

    # Check for the automatic subtitles file.
    subtitle_file = os.path.join(args.subtitle_dir, f"{vid}.vtt")
    if os.path.exists(subtitle_file):
        cmd += f" --subtitles={subtitle_file}"

    # Check for the manually corrected subtitles file.
    subtitle_corrected_file = os.path.join(args.subtitle_dir_corrected, f"{vid}.vtt")
    if os.path.exists(subtitle_corrected_file):
        cmd += f" --subtitles-corrected={subtitle_corrected_file}"

    # Run the command.
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        return f"Error processing video id {vid} (return code {result.returncode})"
    return f"Processed {vid}"

def main():
    parser = argparse.ArgumentParser(
        description="Segment videos based on their pose files and save results."
    )
    parser.add_argument(
        "--video_ids",
        type=str,
        default="/users/zifan/subtitle_align/data/bobsl_align.txt",
        help="Path to text file containing video ids (one per line)."
    )
    parser.add_argument(
        "--pose_dir",
        type=str,
        default="/scratch/shared/beegfs/zifan/bobsl/video_features/mediapipe_v2_refine_face_complexity_2",
        help="Directory where pose files are stored."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/scratch/shared/beegfs/zifan/bobsl/video_features/segmentation",
        help="Directory to store segmentation results."
    )
    parser.add_argument(
        "--overwrite",
        action='store_true',
        help="Overwrite existing feature files if set"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/users/zifan/BOBSL/derivatives/original_videos",
        help="Directory containing original videos."
    )
    parser.add_argument(
        "--subtitle_dir",
        type=str,
        default="/users/zifan/BOBSL/v1.4/automatic_annotations/signing_aligned_subtitles/audio_aligned_heuristic_correction",
        help="Directory containing automatically aligned subtitles."
    )
    parser.add_argument(
        "--subtitle_dir_corrected",
        type=str,
        default="/users/zifan/BOBSL/v1.4/manual_annotations/signing_aligned_subtitles",
        help="Directory containing manually corrected subtitles."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers to process videos. Default is 1 (sequential processing)."
    )
    args = parser.parse_args()

    # Ensure that the save directory exists.
    os.makedirs(args.save_dir, exist_ok=True)

    # Read video ids from the provided file.
    with open(args.video_ids, "r") as file:
        video_ids = [line.strip() for line in file if line.strip()]

    if args.num_workers > 1:
        # Parallel processing with multiple workers.
        worker_func = partial(process_video, args=args)
        with multiprocessing.Pool(args.num_workers) as pool:
            for res in tqdm(pool.imap_unordered(worker_func, video_ids),
                            total=len(video_ids), desc="Processing videos"):
                tqdm.write(res)
    else:
        # Sequential processing.
        for vid in tqdm(video_ids, desc="Processing videos"):
            res = process_video(vid, args)
            tqdm.write(res)

if __name__ == "__main__":
    main()
