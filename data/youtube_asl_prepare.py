import os
import sys
import argparse

def curate_youtube_asl(input_dir):
    metadata_file = os.path.join(input_dir, "youtube-asl_youtube_asl_video_ids.txt")
    downloads_dir = os.path.join(input_dir, "downloads")
    
    if not os.path.exists(metadata_file):
        sys.exit(f"ERROR: Metadata file {metadata_file} does not exist.")
    
    with open(metadata_file, "r") as f:
        video_ids = [line.strip() for line in f if line.strip()]
    
    found_count = 0
    found_count_specific_json = 0
    
    for vid in video_ids:
        video_download_dir = os.path.join(downloads_dir, vid)
        
        if not os.path.exists(video_download_dir):
            print(f"Video {vid} directory does not exist.")
            continue
        
        mp4_files = [f for f in os.listdir(video_download_dir) if f.endswith('.mp4')]
        pose_files = [f for f in os.listdir(video_download_dir) if f.endswith('.pose')]
        json_files = [f for f in os.listdir(video_download_dir) if f.endswith('.json')]
        specific_json_files = [f for f in json_files if f in ["English en.json", "en.json"]]
        
        if len(mp4_files) != 1:
            print(f"Video {vid} has {'no' if not mp4_files else 'multiple'} mp4 files.")
        if len(pose_files) != 1:
            print(f"Video {vid} has {'no' if not pose_files else 'multiple'} pose files.")
        if len(json_files) == 0:
            print(f"Video {vid} has no json files.")
        elif len(json_files) > 1:
            print(f"Video {vid} has multiple json files: {', '.join(json_files)}")
        
        if len(mp4_files) == 1 and len(pose_files) == 1 and len(json_files) >= 1:
            found_count += 1
        
        if len(mp4_files) == 1 and len(pose_files) == 1 and len(specific_json_files) >= 1:
            found_count_specific_json += 1
    
    print(f"Found {found_count} videos with all required files out of {len(video_ids)} total IDs.")
    print(f"Found {found_count_specific_json} videos considering only 'English en.json' and 'en.json'.")

def main():
    parser = argparse.ArgumentParser(description="Curate the YouTube-ASL video subtitle dataset.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="/scratch/shared/beegfs/zifan/YouTube-ASL",
        help="Input directory for the YouTube-ASL dataset (default: %(default)s)"
    )
    args = parser.parse_args()
    curate_youtube_asl(args.input_dir)

if __name__ == "__main__":
    main()